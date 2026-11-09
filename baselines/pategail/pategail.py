from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common
import pg_accounting as acc
import pg_official

CFG = json.loads((HERE / "frozen.json").read_text())
H = CFG["hyper"]
K_FROZEN = int(CFG["owner_decisions"]["D-K"]["K"])
assert K_FROZEN == H["K"]
ROUNDS = int(H["rounds"])
BETA, DELTA_W = float(H["beta"]), float(H["delta_w"])
TRAIN_ITER = int(H["train_iter"])
TIME_SCALE = int(H["time_scale"])
SLOT_MIN = int(H["slot_minutes"])
CELL_DEG = float(H["cell_deg"])
MAX_RANK, EPR_ALPHA = int(H["max_closest_rank"]), float(H["epr_alpha"])
DISC_ITER, BATCH = int(H["disc_iter"]), int(H["batch_size"])
PPO_UPDATES, POLICY_ITER, VALUE_ITER = int(H["ppo_updates"]), int(H["policy_iter"]), int(H["value_iter"])
CLIP_EPS, ENT_W = float(H["clip_eps"]), float(H["entropy_weight"])
GAMMA, LAM = float(H["gamma"]), float(H["lam"])
LR, WD, CLIP_GRAD = float(H["lr"]), float(H["weight_decay_policy_value"]), float(H["clip_grad"])
DELTA = float(CFG["privacy"]["delta"])
AUX_FRAC, N_AUX, PILOT_SHARE = float(CFG["privacy"]["aux_frac"]), int(CFG["privacy"]["n_aux_hists"]), \
    float(CFG["privacy"]["pilot_share"])
LEN_MAX = common.LEN_MAX
TRAJ_LEN_SPECIAL = int(H["traj_length_special_case"])
assert TRAJ_LEN_SPECIAL == LEN_MAX
TD = H["teacher"]
PD = H["policy"]
N_ACTIONS = 4
STREAMS = ["aux", "start_cell", "pate_noise", "teachers", "teacher_init", "train_episodes", "train_actions", "ppo",
           "release", "venues"]
FORBIDDEN = {"gw_austin", "gw_nyc"}
CODE_FILES = ["pategail.py", "pg_accounting.py", "pg_official.py", "run_pategail.py", "frozen.json"]

SIDECAR_TOP = {"method", "dataset", "eps", "delta", "seed", "user_level", "n_trails", "n_checkins", "budget",
               "accounting", "aux", "venue_prior", "start_cell", "empty_cell_fallbacks", "config", "official_commit",
               "code_sha256", "runtime_s", "python", "argv", "host_pid", "private_diagnostics"}
SIDECAR_BUDGET = {"eps_total", "rho_total", "eps_h", "rho_aux", "rho_pate", "rounds", "pilot_share", "K"}
SIDECAR_ACCOUNTING = {"rho_spent", "rho_spent_aux", "rho_spent_pate", "eps_spent", "eps_target", "delta",
                      "conversion", "per_round", "eps_q_min", "eps_q_median", "eps_q_max", "n_pate_releases",
                      "mechanisms"}
PRIVATE_OK = {"jakarta"}


def log(msg: str) -> None:
    common.log(msg)


def sidecar_problems(side: dict, dataset: str) -> list:
    P = [f"sidecar key {k!r} not in the allow-list" for k in side if k not in SIDECAR_TOP]
    P += [f"budget key {k!r} not in the allow-list" for k in side.get("budget", {}) if k not in SIDECAR_BUDGET]
    P += [f"accounting key {k!r} not in the allow-list" for k in side.get("accounting", {})
          if k not in SIDECAR_ACCOUNTING]
    if "private_diagnostics" in side and dataset not in PRIVATE_OK:
        P.append(f"private_diagnostics present on {dataset} (allowed only on {sorted(PRIVATE_OK)})")
    return P


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def streams(seed: int) -> dict:
    return dict(zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS))))


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563


def vincenty(lat1, lon1, lat2, lon2, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
    a, f = WGS84_A, WGS84_F
    b = a * (1.0 - f)
    p1, l1, p2, l2 = np.broadcast_arrays(*(np.radians(np.asarray(x, np.float64)) for x in (lat1, lon1, lat2, lon2)))
    L = l2 - l1
    U1, U2 = np.arctan((1.0 - f) * np.tan(p1)), np.arctan((1.0 - f) * np.tan(p2))
    sU1, cU1, sU2, cU2 = np.sin(U1), np.cos(U1), np.sin(U2), np.cos(U2)
    lam = L.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            sl, cl = np.sin(lam), np.cos(lam)
            ss = np.sqrt((cU2 * sl) ** 2 + (cU1 * sU2 - sU1 * cU2 * cl) ** 2)
            cs = sU1 * sU2 + cU1 * cU2 * cl
            sg = np.arctan2(ss, cs)
            sa = np.where(ss > 0, cU1 * cU2 * sl / np.where(ss > 0, ss, 1.0), 0.0)
            c2a = 1.0 - sa * sa
            c2sm = np.where(c2a > 0, cs - 2.0 * sU1 * sU2 / np.where(c2a > 0, c2a, 1.0), 0.0)
            C = f / 16.0 * c2a * (4.0 + f * (4.0 - 3.0 * c2a))
            lam_new = L + (1.0 - C) * f * sa * (sg + C * ss * (c2sm + C * cs * (-1.0 + 2.0 * c2sm * c2sm)))
            conv = np.all(np.abs(lam_new - lam) < tol)
            lam = lam_new
            if conv:
                break
        u2 = c2a * (a * a - b * b) / (b * b)
        A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
        B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
        dsg = B * ss * (c2sm + B / 4.0 * (cs * (-1.0 + 2.0 * c2sm * c2sm)
                                          - B / 6.0 * c2sm * (-3.0 + 4.0 * ss * ss) * (-3.0 + 4.0 * c2sm * c2sm)))
        s = b * A * (sg - dsg)
    return np.where(ss > 0, s, 0.0)


def explore_cdf(n_rank: int, alpha: float) -> np.ndarray:
    normalization = sum([i ** -alpha for i in range(1, n_rank + 1)])
    k_prob = [(x ** (-alpha)) / normalization for x in range(1, n_rank + 1)]
    out, acc_ = [], 0
    for p_k in k_prob:
        acc_ += p_k
        out.append(acc_)
    return np.array(out, np.float64)


def rank_list(lat: np.ndarray, lon: np.ndarray, n_rank: int, threads: int = 4, chunk: int = 128) -> np.ndarray:
    n = len(lat)
    out = np.empty((n, n_rank), np.int64)

    def work(s):
        e = min(s + chunk, n)
        d = vincenty(lat[s:e, None], lon[s:e, None], lat[None, :], lon[None, :])
        out[s:e] = np.argsort(d, axis=1, kind="stable")[:, :n_rank]

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        list(ex.map(work, range(0, n, chunk)))
    if not np.array_equal(out[:, 0], np.arange(n)):
        raise RuntimeError("rank list: a cell is not its own nearest cell")
    return out


@dataclass
class Geo:
    cell_ij: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    venue_cell: np.ndarray
    rects: np.ndarray
    rank: np.ndarray
    kcdf: np.ndarray

    @property
    def n(self) -> int:
        return len(self.lat)


def lattice(pois: pd.DataFrame) -> tuple:
    li = np.round(pois["lat"].to_numpy(np.float64) / CELL_DEG).astype(np.int64)
    lj = np.round(pois["lon"].to_numpy(np.float64) / CELL_DEG).astype(np.int64)
    ij, venue_cell = np.unique(np.c_[li, lj], axis=0, return_inverse=True)
    return ij, venue_cell.reshape(-1).astype(np.int64)


def build_geo(pois: pd.DataFrame, threads: int = 4) -> Geo:
    ij, venue_cell = lattice(pois)
    lat_c, lon_c = ij[:, 0] * CELL_DEG, ij[:, 1] * CELL_DEG
    h = CELL_DEG / 2.0
    rects = np.c_[lon_c - h, lon_c + h, lat_c - h, lat_c + h]
    n_rank = min(MAX_RANK, len(ij))
    return Geo(ij, lat_c, lon_c, venue_cell, rects,
               rank_list(lat_c, lon_c, n_rank, threads), explore_cdf(n_rank, EPR_ALPHA))


@dataclass
class Private:
    ds: common.Dataset
    trail_ids: np.ndarray
    offsets: np.ndarray
    cells: np.ndarray
    slots: np.ndarray


def load_private(name: str, geo: Geo | None = None, ds: common.Dataset | None = None) -> tuple:
    if name in FORBIDDEN:
        raise SystemExit(f"{name} must never be read")
    ds = ds or common.load_dataset(name)
    geo = geo or build_geo(ds.pois)
    df = pd.read_parquet(common.DATA / name / "train.parquet", columns=["trail_id", "pos", "t_local", "venue_id"])
    df = df.sort_values(["trail_id", "pos"], kind="stable").reset_index(drop=True)
    codes, uniq = pd.factorize(df["trail_id"], sort=False)
    lengths = np.bincount(codes)
    if not np.array_equal(lengths, ds.lengths):
        raise RuntimeError("trail order differs from common.load_dataset")
    vi = pd.Index(ds.pois["venue_id"]).get_indexer(df["venue_id"].astype(str))
    if not np.array_equal(vi, ds.checkin_venue):
        raise RuntimeError("check-in venues differ from common.load_dataset")
    tmin = df["t_local"].to_numpy("datetime64[m]").astype(np.int64)
    slots = ((tmin % 1440) // SLOT_MIN).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    cells = geo.venue_cell[ds.checkin_venue]
    return Private(ds, np.asarray(uniq).astype(str), offsets, cells, slots), geo


def teacher_of(trail_ids, K: int) -> np.ndarray:
    return np.array([int.from_bytes(hashlib.blake2b(f"pategail|{t}".encode(), digest_size=8).digest(), "big") % K
                     for t in trail_ids], dtype=np.int64)


class TrailEnv:

    def __init__(self, c0: np.ndarray, geo: Geo):
        self.geo = geo
        c0 = np.asarray(c0, np.int64)
        self.hist = np.zeros((len(c0), LEN_MAX), np.int64)
        self.hist[:, 0] = c0
        self.h = 1
        self.cur = c0.copy()
        self.home = c0.copy()

    def phi(self, idx: np.ndarray) -> tuple:
        H = self.hist[idx, :self.h]
        c = self.cur[idx]
        s = np.sort(H, axis=1)
        ppc = 1 + np.count_nonzero(s[:, 1:] != s[:, :-1], axis=1)
        stay = np.cumprod(H[:, ::-1] == c[:, None], axis=1).sum(1)
        return ppc.astype(np.int64), stay.astype(np.int64)

    def label(self, idx: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        c, home = self.cur[idx], self.home[idx]
        seen = (self.hist[idx, :self.h] == nxt[:, None]).any(1)
        return np.where(nxt == c, 0, np.where(nxt == home, 2, np.where(seen, 3, 1))).astype(np.int64)

    def transition(self, idx: np.ndarray, actions: np.ndarray, u: np.ndarray) -> np.ndarray:
        g = self.geo
        c, home = self.cur[idx], self.home[idx]
        H1 = np.concatenate([self.hist[idx, :self.h], c[:, None]], axis=1)
        nxt = c.copy()
        m = np.flatnonzero(actions == 1)
        if len(m):
            i = np.searchsorted(g.kcdf, u[m], side="left")
            ok = i < len(g.kcdf)
            nxt[m[ok]] = g.rank[c[m[ok]], i[ok]]
        m = np.flatnonzero(actions == 2)
        if len(m):
            nxt[m] = home[m]
            if H1.shape[1] == TRAJ_LEN_SPECIAL - 1:
                sp = m[(H1[m] == H1[m, :1]).all(1)]
                nxt[sp] = g.rank[home[sp], 1]
        m = np.flatnonzero(actions == 3)
        if len(m):
            mask = H1[m] != home[m, None]
            cnt = mask.sum(1)
            has = cnt > 0
            j = np.minimum(np.floor(u[m] * cnt).astype(np.int64), np.maximum(cnt - 1, 0))
            pick = np.argmax(np.cumsum(mask, axis=1) > j[:, None], axis=1)
            nxt[m[has]] = H1[m[has], pick[has]]
        return nxt

    def advance(self, idx: np.ndarray, nxt: np.ndarray) -> None:
        self.hist[idx, self.h] = self.cur[idx]
        self.cur[idx] = nxt
        self.h += 1


PAIR_FIELDS = ("pos", "slot", "act", "ppc", "stay")


def real_pairs(cells: np.ndarray, slots: np.ndarray, offsets: np.ndarray, geo: Geo) -> dict:
    L = np.diff(offsets)
    env = TrailEnv(cells[offsets[:-1]], geo)
    rec = {f: [] for f in PAIR_FIELDS + ("trail", "t")}
    for t in range(LEN_MAX - 1):
        idx = np.flatnonzero(L - 1 > t)
        if not len(idx):
            break
        ppc, stay = env.phi(idx)
        nxt = cells[offsets[idx] + t + 1]
        rec["pos"].append(env.cur[idx].copy())
        rec["slot"].append(slots[offsets[idx] + t])
        rec["act"].append(env.label(idx, nxt))
        rec["ppc"].append(ppc)
        rec["stay"].append(stay)
        rec["trail"].append(idx)
        rec["t"].append(np.full(len(idx), t, np.int64))
        env.advance(idx, nxt)
    out = {k: np.concatenate(v).astype(np.int64) if v else np.zeros(0, np.int64) for k, v in rec.items()}
    order = np.lexsort((out["t"], out["trail"]))
    return {k: v[order] for k, v in out.items()}


def torch_setup(device: str, threads: int):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def make_policy_value(n_loc: int, seed: int, device):
    import torch
    net = pg_official.net()
    torch.manual_seed(seed)
    args = (n_loc, TIME_SCALE, None, PD["loc_emb"], PD["tim_emb"], PD["pre_pos_emb"], PD["stay_emb"], PD["hidden"],
            False, "checkins", None, "zero", None)
    policy = net.ATNetwork(*args, return_prob=True).to(device)
    value = net.ATNetwork(*args, return_prob=True).to(device)
    popt = torch.optim.Adam(policy.parameters(), lr=LR, weight_decay=WD)
    vopt = torch.optim.Adam(value.parameters(), lr=LR, weight_decay=WD)
    return policy, value, popt, vopt


def _lt(x, device):
    import torch
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.long, device=device)


def policy_probs(policy, pos, slot, ppc, stay, device, chunk: int = 8192) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(pos), chunk):
            e = s + chunk
            p = policy(_lt(pos[s:e], device), _lt(slot[s:e], device), _lt(ppc[s:e], device), _lt(stay[s:e], device))
            out.append(p.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_ACTIONS), np.float32)


def value_means(value, pos, slot, ppc, stay, device) -> np.ndarray:
    import torch
    with torch.no_grad():
        v = value(_lt(pos, device), _lt(slot, device), _lt(ppc, device), _lt(stay, device)).mean(-1)
    return v.float().cpu().numpy().astype(np.float64)


def sample_actions(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    p = probs.astype(np.float64)
    cdf = np.cumsum(p / p.sum(1, keepdims=True), axis=1)
    return np.minimum((cdf <= u[:, None]).sum(1), N_ACTIONS - 1).astype(np.int64)


def rollout(policy, geo: Geo, lengths: np.ndarray, slots: np.ndarray, c0: np.ndarray, rng_act, rng_env, device,
            value=None, keep_steps: bool = True) -> dict:
    lengths = np.asarray(lengths, np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    cells = np.empty(int(offsets[-1]), np.int64)
    cells[offsets[:-1]] = c0
    env = TrailEnv(c0, geo)
    rec = {f: [] for f in PAIR_FIELDS + ("bprob", "value", "trail", "t")} if keep_steps else None
    for t in range(LEN_MAX - 1):
        idx = np.flatnonzero(lengths - 1 > t)
        if not len(idx):
            break
        ppc, stay = env.phi(idx)
        pos = env.cur[idx].copy()
        slot = slots[offsets[idx] + t]
        probs = policy_probs(policy, pos, slot, ppc, stay, device)
        a = sample_actions(probs, rng_act.random(len(idx)))
        nxt = env.transition(idx, a, rng_env.random(len(idx)))
        if keep_steps:
            rec["pos"].append(pos)
            rec["slot"].append(slot)
            rec["act"].append(a)
            rec["ppc"].append(ppc)
            rec["stay"].append(stay)
            rec["bprob"].append(probs[np.arange(len(idx)), a].astype(np.float32))
            rec["value"].append(value_means(value, pos, slot, ppc, stay, device) if value is not None
                                else np.full(len(idx), 0.25))
            rec["trail"].append(idx)
            rec["t"].append(np.full(len(idx), t, np.int64))
        env.advance(idx, nxt)
        cells[offsets[idx] + t + 1] = nxt
    out = {"cells": cells, "offsets": offsets}
    if keep_steps:
        st = {k: np.concatenate(v) for k, v in rec.items()}
        order = np.lexsort((st["t"], st["trail"]))
        st = {k: v[order] for k, v in st.items()}
        st["done"] = (st["t"] == lengths[st["trail"]] - 2).astype(np.float64)
        out["steps"] = st
    return out


class TeacherEnsemble:

    def __init__(self, K: int, n_loc: int, gen, device):
        import torch
        self.K, self.n_loc, self.device = K, n_loc, device
        T = TIME_SCALE
        emb = [("loc", n_loc, TD["loc_emb"]), ("tim", T, TD["tim_emb"]), ("act", N_ACTIONS, TD["act_emb"]),
               ("stay", T, TD["stay_emb"]), ("pre", T, TD["pre_pos_emb"])]
        names, init = [], []
        for name, n, d in emb:
            names.append(name)
            init.append(torch.randn(K, n, d, generator=gen))
        d_in = TD["loc_emb"] + TD["tim_emb"] + TD["act_emb"] + TD["stay_emb"] + TD["pre_pos_emb"]
        if d_in != TD["mlp_in"]:
            raise RuntimeError("teacher input width")
        hid = TD["hidden"]
        for i, (fi, fo) in enumerate(zip([d_in, hid, hid, hid], [hid, hid, hid, 1])):
            bound = 1.0 / math.sqrt(fi)
            names += [f"W{i}", f"b{i}"]
            init.append((torch.rand(K, fo, fi, generator=gen) * 2.0 - 1.0) * bound)
            init.append((torch.rand(K, fo, generator=gen) * 2.0 - 1.0) * bound)
        self.params = []
        while init:
            self.params.append(torch.nn.Parameter(init.pop(0).to(device)))
        self.p = dict(zip(names, self.params))
        self.opt = torch.optim.Adam(self.params, lr=LR)
        self._ar = torch.arange(K, device=device)[:, None]

    def load_official(self, modules: list) -> None:
        import torch
        with torch.no_grad():
            for k, m in enumerate(modules):
                self.p["loc"][k].copy_(m.loc_embedding.weight)
                self.p["tim"][k].copy_(m.tim_embedding.weight)
                self.p["act"][k].copy_(m.act_embedding.weight)
                self.p["stay"][k].copy_(m.stay_time_embedding.weight)
                self.p["pre"][k].copy_(m.pre_pos_count_embedding.weight)
                for i, j in enumerate((0, 2, 4, 6)):
                    self.p[f"W{i}"][k].copy_(m.mlp_layer[j].weight)
                    self.p[f"b{i}"][k].copy_(m.mlp_layer[j].bias)

    def forward(self, pos, slot, act, ppc, stay):
        import torch
        import torch.nn.functional as F
        K = self.K

        def emb(name, idx):
            t = self.p[name]
            return F.embedding(idx + self._ar * t.shape[1], t.view(K * t.shape[1], t.shape[2]))

        x = torch.cat([emb("loc", pos), emb("tim", slot), emb("act", act), emb("stay", stay), emb("pre", ppc)], -1)
        for i in range(4):
            x = torch.baddbmm(self.p[f"b{i}"].unsqueeze(1), x, self.p[f"W{i}"].transpose(1, 2))
            x = torch.relu(x) if i < 3 else torch.sigmoid(x)
        return x.squeeze(-1)

    def predict(self, pairs: dict) -> np.ndarray:
        import torch
        n = len(pairs["pos"])
        with torch.no_grad():
            args = [_lt(pairs[f], self.device)[None, :].expand(self.K, n) for f in PAIR_FIELDS]
            out = self.forward(*args)
        return out.double().cpu().numpy()

    def train(self, real: tuple, gen: tuple, mask, n_iter: int = DISC_ITER, clip: float = CLIP_GRAD,
              diag: bool = False) -> dict:
        import torch
        import torch.nn.functional as F
        K = self.K
        mask = mask.to(self.device, torch.float32)
        ones = torch.ones(real[0].shape, device=self.device)
        zeros = torch.zeros(gen[0].shape, device=self.device)
        info = {}
        for it in range(n_iter):
            out_r, out_g = self.forward(*real), self.forward(*gen)
            l_r = F.binary_cross_entropy(out_r, ones, reduction="none").mean(1)
            l_g = F.binary_cross_entropy(out_g, zeros, reduction="none").mean(1)
            loss_k = (l_r + l_g) / 2.0
            self.opt.zero_grad(set_to_none=True)
            (loss_k * mask).sum().backward()
            clip_per_teacher(self.params, clip, K)
            self.opt.step()
            if diag and it in (0, n_iter - 1):
                with torch.no_grad():
                    sel = mask > 0
                    info[f"loss_{'first' if it == 0 else 'last'}"] = float(loss_k[sel].mean()) if sel.any() else None
                    if it == n_iter - 1 and sel.any():
                        info["acc_real"] = float((out_r[sel] > 0.5).float().mean())
                        info["acc_gen"] = float((out_g[sel] < 0.5).float().mean())
                        info["mean_D_real"] = float(out_r[sel].mean())
                        info["mean_D_gen"] = float(out_g[sel].mean())
        return info


def clip_per_teacher(params: list, max_norm: float, K: int) -> None:
    import torch
    grads = [p.grad for p in params]
    norms = torch.stack([torch.linalg.vector_norm(g.reshape(K, -1), 2.0, dim=1) for g in grads])
    total = torch.linalg.vector_norm(norms, 2.0, dim=0)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for g in grads:
        g.mul_(coef.view(K, *([1] * (g.dim() - 1))))


def gae(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, gamma: float = GAMMA, lam: float = LAM) -> tuple:
    n = len(rewards)
    ret, adv = np.zeros(n), np.zeros(n)
    R = 0.0
    Adv = 0.0
    Value_previous = 0.0
    for i in reversed(range(n)):
        d, v = float(dones[i]), float(values[i])
        R = gamma * R * (1 - d) + v
        delta = float(rewards[i]) + Value_previous * gamma * (1 - d) - v
        Adv = delta + (1 - d) * Adv * gamma * lam
        Value_previous = v
        ret[i], adv[i] = R, Adv
    return ret, adv


def ppo_update(policy, value, popt, vopt, b: dict, device, entropy_weight: float = ENT_W) -> dict:
    import torch
    pos, times, ppc, stay = (_lt(b[f], device) for f in ("pos", "slot", "ppc", "stay"))
    advantages = torch.as_tensor(b["adv"], dtype=torch.float32, device=device).unsqueeze(1)
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages = advantages.detach()
    returns = torch.as_tensor(b["ret"], dtype=torch.float32, device=device).unsqueeze(1).detach()
    for _ in range(VALUE_ITER):
        values = value(pos, times, ppc, stay)
        value_loss = (returns - values).pow(2).mean()
        vopt.zero_grad()
        value_loss.backward()
        vopt.step()
    actions_d = _lt(b["act"], device).unsqueeze(1)
    old_probs = torch.as_tensor(b["bprob"], dtype=torch.float32, device=device).unsqueeze(1)
    info = {}
    for _ in range(POLICY_ITER):
        probs_all = policy(pos, times, ppc, stay)
        probs = probs_all.gather(1, actions_d)
        ratio = probs / old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        entropy = torch.distributions.Categorical(probs_all).entropy().unsqueeze(1)
        policy_loss = (-torch.min(surr1, surr2) - entropy_weight * entropy).mean()
        popt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP_GRAD)
        popt.step()
        r = ratio.detach()
        info = {"ratio_min": float(r.min()), "ratio_max": float(r.max()),
                "clipped": float(((r < 1 - CLIP_EPS) | (r > 1 + CLIP_EPS)).float().mean()),
                "entropy": float(entropy.detach().mean()), "policy_loss": float(policy_loss.detach())}
    return info


def fit_start_cells(first_cells: np.ndarray, n_cells: int, eps_h: float, rng, fp: float) -> tuple:
    h = np.bincount(first_cells, minlength=n_cells).astype(np.float64)
    b = 1.0 / eps_h if math.isfinite(eps_h) else 0.0
    noisy = h + rng.laplace(0.0, b, n_cells)
    tau = max(2.0 * b, b * math.log(n_cells / (2.0 * fp)))
    keep = np.where(noisy > tau, noisy, 0.0)
    passing = int((keep > 0).sum())
    p = keep / keep.sum() if passing else np.full(n_cells, 1.0 / n_cells)
    info = {"mechanism": "laplace", "sensitivity": 1.0, "eps": eps_h, "threshold": tau, "fp_target": fp,
            "cells": int(n_cells), "cells_passing": passing, "uniform_fallback": not passing}
    return p, info


@dataclass
class TrainResult:
    policy: object
    ledger: object
    per_round: list
    eps_q_all: np.ndarray
    diag: dict


def train(priv: Private, geo: Geo, aux, p_c0: np.ndarray, budget: dict, arm: str, K: int, rounds: int,
          ss: dict, device, seed: int, diag: bool, log_private=None) -> TrainResult:
    import torch
    finite = math.isfinite(budget["eps_total"])
    rho_round = budget["rho_pate"] / rounds if finite else math.inf
    ledger = acc.Ledger(budget["rho_total"], budget["rho_aux"]) if finite else None

    teach = teacher_of(priv.trail_ids, K)
    pairs = real_pairs(priv.cells, priv.slots, priv.offsets, geo)
    tp = teach[pairs["trail"]]
    order = np.argsort(tp, kind="stable")
    pairs = {k: v[order] for k, v in pairs.items()}
    counts = np.bincount(tp, minlength=K)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nonempty = counts > 0
    t_rngs = [np.random.Generator(np.random.PCG64(s)) for s in ss["teachers"].spawn(K)]
    gen = torch.Generator().manual_seed(int(ss["teacher_init"].generate_state(1)[0]))
    ens = TeacherEnsemble(K, geo.n, gen, device)
    mask = torch.as_tensor(nonempty)
    D = {"n_train_trails": int(len(priv.trail_ids)), "n_train_pairs": int(len(tp)),
         "n_empty_teachers": int((~nonempty).sum()), "teacher_pairs_min_med_max":
             [int(counts.min()), float(np.median(counts)), int(counts.max())],
         "real_action_mix": np.bincount(pairs["act"], minlength=N_ACTIONS).tolist(), "rounds": []} if diag else {}
    if diag and log_private:
        log_private(f"[private] {D['n_train_trails']} trails, {D['n_train_pairs']} pairs, "
                    f"{D['n_empty_teachers']} empty teachers of {K}, real action mix {D['real_action_mix']}")

    policy, value, popt, vopt = make_policy_value(geo.n, seed, device)
    rng_ep = np.random.Generator(np.random.PCG64(ss["train_episodes"]))
    rng_act = np.random.Generator(np.random.PCG64(ss["train_actions"]))
    rng_ppo = np.random.Generator(np.random.PCG64(ss["ppo"]))
    rng_noise = np.random.Generator(np.random.PCG64(ss["pate_noise"]))
    per_round, eps_q_all = [], []
    t0 = time.time()
    for r in range(1, rounds + 1):
        Ls = aux.sample_lengths(TRAIN_ITER, rng_ep)
        m = int(np.searchsorted(np.cumsum(Ls - 1), TRAIN_ITER, side="left")) + 1
        L = Ls[:m]
        tmin = aux.sample_times(L, rng_ep)
        slots = ((tmin % 1440) // SLOT_MIN).astype(np.int64)
        c0 = rng_ep.choice(geo.n, size=m, p=p_c0)
        buf = rollout(policy, geo, L, slots, c0, rng_act, rng_ep, device, value=value)["steps"]
        n_r = len(buf["pos"])
        Dk = ens.predict(buf)
        Dk[~nonempty] = 0.5
        mbar = Dk.mean(0)
        var = Dk.var(0)
        rec = {"round": r, "n_r": n_r}
        if finite:
            if arm == "main":
                e_p = acc.pilot_eps(rho_round, n_r, PILOT_SHARE)
                ledger.charge("M7", r, np.full(n_r, e_p))
                pilot = mbar + rng_noise.laplace(0.0, (1.0 / K) / e_p, n_r)
                eps_q = acc.main_eps(acc.weights(pilot, DELTA_W), rho_round, PILOT_SHARE)
                rec["eps_p"] = e_p
            elif arm == "uniform":
                eps_q = acc.uniform_eps(rho_round, n_r)
                rec["eps_p"] = None
            else:
                raise ValueError(arm)
            ledger.charge("M8", r, eps_q)
            R_q = mbar + rng_noise.laplace(0.0, (1.0 / K) / eps_q)
            ledger.charge("M9", r, eps_q)
            xi = np.sqrt(np.abs(var + rng_noise.laplace(0.0, (1.0 / K) / eps_q)))
            reward = R_q - BETA * xi
            rec["sum_eps_q_sq"] = math.fsum((eps_q * eps_q).tolist())
            eps_q_all.append(eps_q)
        else:
            R_q, xi = mbar, np.sqrt(np.abs(var))
            reward = R_q - BETA * xi
        ret, adv = gae(reward, buf["done"], buf["value"])
        buf["ret"], buf["adv"] = ret, adv
        real_idx = np.zeros((K, BATCH), np.int64)
        gen_idx = np.tile(np.arange(BATCH, dtype=np.int64), (K, 1))
        for k in np.flatnonzero(nonempty):
            g = t_rngs[k]
            real_idx[k] = starts[k] + g.integers(0, counts[k], size=BATCH)
            gen_idx[k] = g.choice(n_r, size=BATCH, replace=False)
        real = tuple(_lt(pairs[f][real_idx], device) for f in PAIR_FIELDS)
        genb = tuple(_lt(buf[f][gen_idx], device) for f in PAIR_FIELDS)
        tinfo = ens.train(real, genb, mask, diag=diag)
        pinfo = []
        for _ in range(PPO_UPDATES):
            idx = rng_ppo.choice(n_r, size=BATCH, replace=False)
            pinfo.append(ppo_update(policy, value, popt, vopt, {k: v[idx] for k, v in buf.items()}, device))
        mix = np.bincount(buf["act"], minlength=N_ACTIONS).tolist()
        per_round.append(rec)
        if diag:
            D["rounds"].append({"round": r, "n_r": n_r, **tinfo, "mean_teacher_mean": float(mbar.mean()),
                                "mean_teacher_var": float(var.mean()), "mean_reward": float(np.mean(reward)),
                                "buffer_action_mix": mix,
                                "ppo_clipped_frac_last": pinfo[-1]["clipped"]})
        if r == 1 or r % 10 == 0 or r == rounds:
            spent = f" rho {ledger.rho:.6g}/{ledger.rho_total:.6g}" if ledger else ""
            log(f"round {r}/{rounds}: n_r {n_r}, mean released reward {float(np.mean(reward)):+.4f}, "
                f"generated action mix {mix}{spent}, ppo ratio [{pinfo[-1]['ratio_min']:.3f}, "
                f"{pinfo[-1]['ratio_max']:.3f}], {time.time() - t0:.0f}s")
        if diag and log_private and (r == 1 or r % 10 == 0 or r == rounds):
            log_private(f"[private] round {r}: teacher loss {tinfo.get('loss_first')} -> {tinfo.get('loss_last')}, "
                        f"acc real {tinfo.get('acc_real')}, acc gen {tinfo.get('acc_gen')}, "
                        f"exact mean D {float(mbar.mean()):.4f}")
    eq = np.concatenate(eps_q_all) if eps_q_all else np.zeros(0)
    return TrainResult(policy, ledger, per_round, eq, D)


def mechanisms_table(budget: dict, K: int, arm: str, n_venues: int, n_cats: int, n_cells: int) -> list:
    eh = budget["eps_h"]
    rows = [("M1", "start week-hour histogram (common.AuxModel)", common.N_WEEK_HOURS),
            ("M2", "gap histogram, normalised mode (common.AuxModel)", common.N_GAP_BINS),
            ("M3", "trail-length histogram 2..20 (common.AuxModel)", len(common.LEN_VALUES)),
            ("M4", "venue popularity, 1/L per check-in, thresholded (common.VenuePrior)", n_venues),
            ("M5", "category popularity, 1/L per check-in (common.VenuePrior)", n_cats),
            ("M6", "start-cell histogram over the public cells, thresholded", n_cells)]
    out = [{"id": i, "release": r, "domain": int(d), "sensitivity": "L1 1", "noise": "Laplace(1/eps_h)", "eps": eh,
            "rho": eh * eh / 2.0 if math.isfinite(eh) else None} for i, r, d in rows]
    sens = f"1/K = {1.0 / K:.6g}"
    if arm == "main":
        out.append({"id": "M7", "release": "pilot teacher mean per query (sensitivity-aware allocation)",
                    "sensitivity": sens, "noise": "Laplace((1/K)/eps_p)", "eps": "eps_p per round (per_round)",
                    "rho": f"{PILOT_SHARE} * rho_round per round"})
    out.append({"id": "M8", "release": "teacher mean per query (paper Eq. 6)", "sensitivity": sens,
                "noise": "Laplace((1/K)/eps_q)", "eps": "eps_q per query (min/median/max below)",
                "rho": "sum_q eps_q^2 / 2 per round"})
    out.append({"id": "M9", "release": "teacher population variance per query (paper Eq. 7)",
                "sensitivity": f"(K-1)/K^2 = {(K - 1) / K ** 2:.6g} <= 1/K (noise calibrated to 1/K)",
                "noise": "Laplace((1/K)/eps_q)", "eps": "eps_q per query (same as M8)",
                "rho": "sum_q eps_q^2 / 2 per round"})
    return out


def run(dataset: str, eps: float, seed: int, arm: str, out: Path, device: str = "auto", threads: int = 4,
        n_out: int = common.N_OUT, K: int | None = None, rounds: int | None = None) -> dict:
    t_start = time.time()
    if dataset in FORBIDDEN:
        raise SystemExit(f"{dataset} must never be read")
    if not math.isfinite(eps) and dataset != "jakarta":
        raise SystemExit("eps = inf (noise-free reference) is for Jakarta only")
    if arm not in ("main", "uniform"):
        raise SystemExit(f"unknown arm {arm}")
    K = K or K_FROZEN
    rounds = rounds or ROUNDS
    diag = dataset in PRIVATE_OK
    priv_lines = []

    def log_private(msg):
        if diag:
            priv_lines.append(msg)
            log(msg)

    dev = torch_setup(device, threads)
    log(f"PateGAIL++ ({arm}) {dataset} eps={eps:g} seed={seed} K={K} R={rounds} device={dev}")
    ss = streams(seed)
    budget = acc.calibrate(eps, K, rounds, AUX_FRAC, N_AUX, PILOT_SHARE, DELTA)
    eps_h = budget["eps_h"]

    ds = common.load_dataset(dataset)
    geo = build_geo(ds.pois, threads)
    priv, geo = load_private(dataset, geo, ds)
    log(f"public cells |C| = {geo.n}, rank list {geo.rank.shape[1]} ({time.time() - t_start:.0f}s)")

    rng_aux = np.random.Generator(np.random.PCG64(ss["aux"]))
    aux = common.AuxModel.fit_central(ds, eps_h, eps_h, eps_h, rng_aux, gap_mode="normalized")
    prior = common.VenuePrior.fit_central(ds, eps_h, eps_h, rng_aux, float(H["venue_floor"]), float(H["venue_fp"]))
    rng_m6 = np.random.Generator(np.random.PCG64(ss["start_cell"]))
    p_c0, m6 = fit_start_cells(priv.cells[priv.offsets[:-1]], geo.n, eps_h, rng_m6, float(H["start_cell_fp"]))

    res = train(priv, geo, aux, p_c0, budget, arm, K, rounds, ss, dev, seed, diag, log_private)
    t_train = time.time()
    log(f"training done ({t_train - t_start:.0f}s)")

    rng_rel = np.random.Generator(np.random.PCG64(ss["release"]))
    L = aux.sample_lengths(n_out, rng_rel)
    tmin = aux.sample_times(L, rng_rel)
    slots = ((tmin % 1440) // SLOT_MIN).astype(np.int64)
    c0 = rng_rel.choice(geo.n, size=n_out, p=p_c0)
    roll = rollout(res.policy, geo, L, slots, c0, rng_rel, rng_rel, dev, value=None, keep_steps=False)
    cells = roll["cells"]
    rng_v = np.random.Generator(np.random.PCG64(ss["venues"]))
    vidx, n_empty = common.VenueIndex(ds.pois, ds.bbox).sample_in_rects(geo.rects[cells], rng_v,
                                                                         weights=prior.weights)
    prefix = CFG["arms"][arm]["job_prefix"]
    df = common.build_frame(vidx, L, tmin, ds.pois, prefix)

    if res.ledger is not None:
        rho_spent = res.ledger.rho
        accounting = {
            "rho_spent": rho_spent, "rho_spent_aux": res.ledger.rho_aux,
            "rho_spent_pate": math.fsum(res.ledger.charges[1:]),
            "eps_spent": acc.cdp_eps(rho_spent, DELTA), "eps_target": eps, "delta": DELTA,
            "conversion": acc.CONVERSION, "per_round": res.per_round,
            "eps_q_min": float(res.eps_q_all.min()), "eps_q_median": float(np.median(res.eps_q_all)),
            "eps_q_max": float(res.eps_q_all.max()), "n_pate_releases": res.ledger.n_releases,
            "mechanisms": mechanisms_table(budget, K, arm, len(ds.venue_cat), len(ds.categories), geo.n)}
        if accounting["eps_spent"] > eps + 1e-9:
            raise RuntimeError(f"accounting: eps spent {accounting['eps_spent']} > target {eps}")
    else:
        accounting = {"eps_target": eps, "delta": DELTA, "conversion": "noise-free reference: no DP guarantee",
                      "per_round": res.per_round,
                      "mechanisms": mechanisms_table(budget, K, arm, len(ds.venue_cat), len(ds.categories), geo.n)}
    label = CFG["arms"][arm]["label"] + ("" if math.isfinite(eps) else " (noise-free reference)")
    config = {"arm": arm, "K": K, "rounds": rounds, "beta": BETA, "delta_w": DELTA_W, "train_iter": TRAIN_ITER,
              "disc_iter": DISC_ITER, "batch_size": BATCH, "ppo_updates": PPO_UPDATES, "clip_eps": CLIP_EPS,
              "entropy_weight": ENT_W, "gamma": GAMMA, "lam": LAM, "lr": LR, "weight_decay_policy_value": WD,
              "clip_grad": CLIP_GRAD, "epr_alpha": EPR_ALPHA, "max_closest_rank": MAX_RANK,
              "n_rank": int(geo.rank.shape[1]), "time_scale": TIME_SCALE, "cell_deg": CELL_DEG, "n_cells": geo.n,
              "teacher_dims": TD, "policy_dims": PD, "aux_frac": AUX_FRAC, "pilot_share": PILOT_SHARE,
              "venue_floor": H["venue_floor"], "venue_fp": H["venue_fp"], "start_cell_fp": H["start_cell_fp"],
              "n_out": n_out, "device": dev.type, "threads": threads, "streams": STREAMS,
              "frozen": "baselines/pategail/frozen.json", "owner_decisions": CFG["owner_decisions"]}
    side = {"method": label, "dataset": dataset, "eps": eps, "delta": DELTA, "seed": seed, "user_level": False,
            "budget": budget, "accounting": accounting, "aux": aux.to_json(), "venue_prior": prior.info,
            "start_cell": m6, "empty_cell_fallbacks": int(n_empty), "config": config,
            "official_commit": pg_official.COMMIT,
            "code_sha256": {f: sha256_file(HERE / f) for f in CODE_FILES} | {"common.py": sha256_file(
                HERE.parent / "common.py")},
            "runtime_s": time.time() - t_start}
    if diag:
        side["private_diagnostics"] = {**res.diag, "release_cells_unique": int(len(np.unique(cells))),
                                       "log": priv_lines}
    probs = sidecar_problems(side | {"n_trails": 0, "n_checkins": 0, "python": "", "argv": [], "host_pid": 0},
                             dataset)
    if probs:
        raise RuntimeError(f"sidecar allow-list: {probs}")
    common.write_outputs(df, Path(out), side)
    try:
        import torch
        if dev.type == "cuda":
            log(f"peak GPU memory: allocated {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB, reserved "
                f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GiB (a function of K and |C| only)")
    except Exception:
        pass
    log(f"done in {time.time() - t_start:.0f}s (training {t_train - t_start:.0f}s); empty-cell fallbacks {n_empty}")
    return side
