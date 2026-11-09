from __future__ import annotations
import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "processed"
SRC = Path(f"{_TP_WORK}/baselines_src")
RUNS = Path(f"{_TP_WORK}/runs/baselines")
VENV_SITE = SRC / "_venv" / "lib" / "python3.10" / "site-packages"

REF_MONDAY = pd.Timestamp("2012-07-02")
GAP_EDGES = np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240, 360, 480])
N_GAP_BINS = len(GAP_EDGES) - 1
N_WEEK_HOURS = 7 * 24
LEN_MIN, LEN_MAX = 2, 20
LEN_VALUES = np.arange(LEN_MIN, LEN_MAX + 1)
OUT_COLS = ["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"]
N_OUT = 50_000
CITIES = ["nyc", "istanbul", "tokyo", "sao_paulo", "kuwait_city", "jakarta"]
CITIES += ["tsmc_nyc", "petaling_jaya", "moscow", "bandung", "sydney", "melbourne", "tangerang", "palembang", "shanghai",
           "beijing", "ms_tokyo", "gw_stockholm", "gw_san_francisco"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Dataset:
    name: str
    bbox: dict
    trails: list
    lengths: np.ndarray
    start_wh: np.ndarray
    gaps: list
    pois: pd.DataFrame
    checkin_venue: np.ndarray
    venue_cat: np.ndarray
    categories: list
    meta: dict = field(default_factory=dict)


def one_trail_per_user(trail_ids: np.ndarray, user_ids: np.ndarray, seed: int) -> np.ndarray:
    h = np.array([int.from_bytes(hashlib.blake2b(f"{seed}|{u}|{t}".encode(), digest_size=8).digest(), "big")
                  for t, u in zip(trail_ids, user_ids)], dtype=np.uint64)
    order = np.lexsort((h, user_ids))
    first = np.ones(len(order), bool)
    first[1:] = user_ids[order][1:] != user_ids[order][:-1]
    return trail_ids[order][first]


def load_dataset(name: str, user_level: bool = False, user_seed: int = 0) -> Dataset:
    d = DATA / name
    meta = json.loads((d / "meta.json").read_text())
    df = pd.read_parquet(d / "train.parquet",
                         columns=["trail_id", "user_id", "venue_id", "lat", "lon", "t_local", "pos"])
    if user_level:
        tu = df.drop_duplicates("trail_id")[["trail_id", "user_id"]].astype(str)
        keep = one_trail_per_user(tu["trail_id"].to_numpy(), tu["user_id"].to_numpy(), user_seed)
        df = df[df["trail_id"].astype(str).isin(set(keep))]
    df = df.sort_values(["trail_id", "pos"], kind="stable").reset_index(drop=True)
    codes, _ = pd.factorize(df["trail_id"], sort=False)
    lengths = np.bincount(codes)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    lonlat = df[["lon", "lat"]].to_numpy(np.float64)
    trails = [lonlat[offsets[i]:offsets[i + 1]] for i in range(len(lengths))]
    t = df["t_local"].to_numpy("datetime64[m]").astype(np.int64)
    first = offsets[:-1]
    t0 = df["t_local"].iloc[first]
    start_wh = (t0.dt.dayofweek * 24 + t0.dt.hour).to_numpy(np.int64)
    gaps = [np.diff(t[offsets[i]:offsets[i + 1]]) for i in range(len(lengths))]
    pois = pd.read_parquet(d / "pois.parquet").reset_index(drop=True)
    pois["venue_id"] = pois["venue_id"].astype(str)
    venue_cat, cats = pd.factorize(pois["category"].astype(str), sort=True)
    checkin_venue = pd.Index(pois["venue_id"]).get_indexer(df["venue_id"].astype(str))
    if (checkin_venue < 0).any():
        raise ValueError(f"{(checkin_venue < 0).sum()} train check-ins reference venues outside the public catalog")
    return Dataset(name, meta["bbox"], trails, lengths, start_wh, gaps, pois, checkin_venue.astype(np.int64),
                   venue_cat.astype(np.int64), list(cats), meta)


def split_budget(eps: float, aux_frac: float = 0.1, use_len_hist: bool = True, use_venue_prior: bool = False) -> dict:
    if not 0.0 <= aux_frac < 1.0:
        raise ValueError("aux_frac must be in [0, 1)")
    aux = aux_frac * eps
    k = 2 + int(use_len_hist) + 2 * int(use_venue_prior)
    share = aux / k
    return {"eps_total": eps, "aux_frac": aux_frac, "eps_gen": eps - aux, "n_aux_hists": k,
            "eps_start": share, "eps_gap": share, "eps_len": share if use_len_hist else 0.0,
            "eps_venue": share if use_venue_prior else 0.0, "eps_cat": share if use_venue_prior else 0.0}


def _clip_normalize(h: np.ndarray) -> np.ndarray:
    h = np.clip(np.asarray(h, np.float64), 0.0, None)
    s = h.sum()
    return h / s if s > 0 else np.full(len(h), 1.0 / len(h))


def _norm_sub(est: np.ndarray, n: float) -> np.ndarray:
    est = np.asarray(est, np.float64)
    lo, hi = est.min() - n, est.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.clip(est - mid, 0.0, None).sum() > n:
            lo = mid
        else:
            hi = mid
    return _clip_normalize(est - hi)


def gap_bin(gaps_min: np.ndarray) -> np.ndarray:
    b = np.searchsorted(GAP_EDGES, gaps_min, side="right") - 1
    return np.clip(b, 0, N_GAP_BINS - 1)


def oue_estimate(values: np.ndarray, domain: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return np.zeros(domain)
    p, q = 0.5, 1.0 / (np.exp(eps) + 1.0)
    true = np.bincount(values, minlength=domain)
    ones = rng.binomial(true, p) + rng.binomial(n - true, q)
    return (ones - n * q) / (p - q)


@dataclass
class AuxModel:
    p_start: np.ndarray
    p_gap: np.ndarray
    p_len: np.ndarray | None
    mode: str
    info: dict
    n_hat: float = float("nan")

    @classmethod
    def fit_central(cls, ds: Dataset, eps_start: float, eps_gap: float, eps_len: float,
                    rng: np.random.Generator, gap_mode: str = "normalized") -> "AuxModel":
        h_start = np.bincount(ds.start_wh, minlength=N_WEEK_HOURS).astype(np.float64)
        if gap_mode == "normalized":
            b = gap_bin(np.concatenate(ds.gaps))
            w = np.repeat(1.0 / (ds.lengths - 1), ds.lengths - 1)
            h_gap = np.bincount(b, weights=w, minlength=N_GAP_BINS)
        elif gap_mode == "first":
            h_gap = np.bincount(gap_bin(np.array([g[0] for g in ds.gaps])), minlength=N_GAP_BINS).astype(float)
        else:
            raise ValueError(gap_mode)
        h_len = np.bincount(np.clip(ds.lengths, LEN_MIN, LEN_MAX) - LEN_MIN, minlength=len(LEN_VALUES)).astype(float)
        n_start = h_start + rng.laplace(0.0, 1.0 / eps_start, h_start.shape)
        n_gap = h_gap + rng.laplace(0.0, 1.0 / eps_gap, h_gap.shape)
        p_len = (_clip_normalize(h_len + rng.laplace(0.0, 1.0 / eps_len, h_len.shape))
                 if eps_len > 0 else None)
        n_hat = float(max(n_gap.sum(), 1.0))
        info = {"mechanism": "laplace", "sensitivity": 1.0, "gap_mode": gap_mode,
                "eps_start": eps_start, "eps_gap": eps_gap, "eps_len": eps_len, "n_hat": n_hat}
        return cls(_clip_normalize(n_start), _clip_normalize(n_gap), p_len, "central", info, n_hat)

    @classmethod
    def fit_ldp(cls, ds: Dataset, eps_start: float, eps_gap: float, eps_len: float,
                rng: np.random.Generator) -> "AuxModel":
        gpick = np.array([g[rng.integers(len(g))] for g in ds.gaps])
        e_start = oue_estimate(ds.start_wh, N_WEEK_HOURS, eps_start, rng)
        e_gap = oue_estimate(gap_bin(gpick), N_GAP_BINS, eps_gap, rng)
        info = {"mechanism": "OUE (local DP) + Norm-Sub", "gap_mode": "one random gap per trail",
                "eps_start": eps_start, "eps_gap": eps_gap, "eps_len": eps_len}
        n = float(len(ds.start_wh))
        p_len = None
        if eps_len > 0:
            e_len = oue_estimate(np.clip(ds.lengths, LEN_MIN, LEN_MAX) - LEN_MIN, len(LEN_VALUES), eps_len, rng)
            p_len = _norm_sub(e_len, n)
        return cls(_norm_sub(e_start, n), _norm_sub(e_gap, n), p_len, "ldp", info)

    def mean_length(self) -> float:
        return float(np.dot(self.p_len, LEN_VALUES)) if self.p_len is not None else float("nan")

    def sample_lengths(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(LEN_VALUES, size=n, p=self.p_len)

    def sample_times(self, lengths: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(lengths)
        wh = rng.choice(N_WEEK_HOURS, size=n, p=self.p_start)
        t0 = wh * 60 + rng.integers(0, 60, size=n)
        n_gaps = int(np.sum(lengths - 1))
        gb = rng.choice(N_GAP_BINS, size=n_gaps, p=self.p_gap)
        lo, hi = GAP_EDGES[gb], GAP_EDGES[gb + 1]
        hi = np.where(gb == N_GAP_BINS - 1, hi + 1, hi)
        g = lo + np.floor(rng.random(n_gaps) * (hi - lo)).astype(np.int64)
        tmin = np.empty(int(lengths.sum()), np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        goff = np.concatenate([[0], np.cumsum(lengths - 1)])
        for i in range(n):
            s, e = offsets[i], offsets[i + 1]
            tmin[s] = t0[i]
            tmin[s + 1:e] = t0[i] + np.cumsum(g[goff[i]:goff[i + 1]])
        return tmin

    def to_json(self) -> dict:
        return {"mode": self.mode, **self.info,
                "mean_length": self.mean_length() if self.p_len is not None else None}


def venue_weights(pop: np.ndarray, cat: np.ndarray, venue_cat: np.ndarray, floor_share: float) -> np.ndarray:
    pop = np.maximum(pop, 0.0)
    total = pop.sum()
    floor_mass = floor_share / (1.0 - floor_share) * total if total > 0 else 1.0
    cs = cat.sum()
    cat_share = cat / cs if cs > 0 else np.full(len(cat), 1.0 / len(cat))
    n_in_cat = np.bincount(venue_cat, minlength=len(cat)).astype(np.float64)
    floor = (cat_share / np.maximum(n_in_cat, 1.0))[venue_cat]
    floor = floor_mass * floor / floor.sum() if floor.sum() > 0 else np.full(len(pop), floor_mass / len(pop))
    return pop + floor + 1e-12


@dataclass
class VenuePrior:
    weights: np.ndarray | None
    info: dict

    @classmethod
    def uniform(cls) -> "VenuePrior":
        return cls(None, {"venue_prior": "uniform"})

    @classmethod
    def fit_central(cls, ds: Dataset, eps_venue: float, eps_cat: float, rng: np.random.Generator,
                    floor_share: float = 0.6, fp_target: float = 100.0) -> "VenuePrior":
        D, C = len(ds.venue_cat), len(ds.categories)
        w = np.repeat(1.0 / ds.lengths, ds.lengths)
        pop = np.bincount(ds.checkin_venue, weights=w, minlength=D)
        cat = np.bincount(ds.venue_cat[ds.checkin_venue], weights=w, minlength=C)
        b = 1.0 / eps_venue
        tau = max(2.0 * b, b * np.log(D / (2.0 * fp_target)))
        pop = pop + rng.laplace(0.0, b, D)
        pop = np.where(pop > tau, pop, 0.0)
        cat = np.maximum(cat + rng.laplace(0.0, 1.0 / eps_cat, C), 0.0)
        info = {"venue_prior": "dp", "mechanism": "laplace", "eps_venue": eps_venue, "eps_cat": eps_cat,
                "floor_share": floor_share, "fp_target": fp_target, "threshold": tau,
                "venues_passing": int((pop > 0).sum()), "pop_mass_passing": float(pop.sum())}
        return cls(venue_weights(pop, cat, ds.venue_cat, floor_share), info)

    @classmethod
    def fit_ldp(cls, ds: Dataset, eps_venue: float, eps_cat: float, rng: np.random.Generator,
                floor_share: float = 0.6, fp_target: float = 100.0) -> "VenuePrior":
        from scipy.stats import norm
        D, C = len(ds.venue_cat), len(ds.categories)
        n = len(ds.lengths)
        offsets = np.concatenate([[0], np.cumsum(ds.lengths)[:-1]])
        pick = offsets + (rng.random(len(ds.lengths)) * ds.lengths).astype(np.int64)
        v = ds.checkin_venue[pick]
        est_v = oue_estimate(v, D, eps_venue, rng)
        est_c = oue_estimate(ds.venue_cat[v], C, eps_cat, rng)
        p, q = 0.5, 1.0 / (np.exp(eps_venue) + 1.0)
        sigma = np.sqrt(n * q * (1 - q)) / (p - q)
        tau = sigma * max(2.0, float(norm.isf(fp_target / D)))
        pop = np.where(est_v > tau, est_v, 0.0)
        cat = _norm_sub(est_c, float(n))
        info = {"venue_prior": "dp", "mechanism": "OUE (local DP)", "eps_venue": eps_venue, "eps_cat": eps_cat,
                "floor_share": floor_share, "fp_target": fp_target, "threshold": tau,
                "venues_passing": int((pop > 0).sum()), "pop_mass_passing": float(pop.sum())}
        return cls(venue_weights(pop, cat, ds.venue_cat, floor_share), info)


def make_venue_prior(args, ds: Dataset, bud: dict, rng: np.random.Generator, local: bool = False) -> VenuePrior:
    if args.venue_prior == "uniform":
        return VenuePrior.uniform()
    fit = VenuePrior.fit_ldp if local else VenuePrior.fit_central
    return fit(ds, bud["eps_venue"], bud["eps_cat"], rng, args.venue_floor, args.venue_fp)


class VenueIndex:

    def __init__(self, pois: pd.DataFrame, bbox: dict):
        self.pois = pois
        self.lon = pois["lon"].to_numpy(np.float64)
        self.lat = pois["lat"].to_numpy(np.float64)
        lat0 = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
        self.kx, self.ky = 111.320 * np.cos(np.radians(lat0)), 110.574
        self.tree = cKDTree(np.c_[self.lon * self.kx, self.lat * self.ky])
        self.order = np.argsort(self.lon, kind="stable")
        self.lon_sorted = self.lon[self.order]
        self._rect_cache: dict = {}

    def nearest(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(np.c_[np.asarray(lon) * self.kx, np.asarray(lat) * self.ky])
        return idx

    def in_rect(self, minx: float, maxx: float, miny: float, maxy: float) -> np.ndarray:
        key = (minx, maxx, miny, maxy)
        hit = self._rect_cache.get(key)
        if hit is None:
            a = np.searchsorted(self.lon_sorted, minx, side="left")
            b = np.searchsorted(self.lon_sorted, maxx, side="right")
            cand = self.order[a:b]
            la = self.lat[cand]
            hit = cand[(la >= miny) & (la <= maxy)]
            self._rect_cache[key] = hit
        return hit

    def sample_in_rects(self, rects: np.ndarray, rng: np.random.Generator,
                        weights: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        rects = np.asarray(rects, np.float64)
        out = np.empty(len(rects), np.int64)
        uniq, inv = np.unique(rects, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(inv, minlength=len(uniq)))])
        n_empty = 0
        for u in range(len(uniq)):
            idx = order[bounds[u]:bounds[u + 1]]
            minx, maxx, miny, maxy = uniq[u]
            cand = self.in_rect(minx, maxx, miny, maxy)
            if len(cand) and weights is not None:
                pw = weights[cand]
                out[idx] = rng.choice(cand, size=len(idx), p=pw / pw.sum())
            elif len(cand):
                out[idx] = cand[rng.integers(0, len(cand), size=len(idx))]
            else:
                out[idx] = self.nearest([0.5 * (minx + maxx)], [0.5 * (miny + maxy)])[0]
                n_empty += len(idx)
        return out, n_empty


def realize_lengths(k: np.ndarray, aux: AuxModel, mode: str, rng: np.random.Generator) -> np.ndarray:
    k = np.asarray(k)
    if mode == "native":
        return np.clip(k, LEN_MIN, LEN_MAX)
    if mode == "hist":
        return aux.sample_lengths(len(k), rng)
    raise ValueError(mode)


def expand_runs(seq: np.ndarray, L: int, rng: np.random.Generator) -> np.ndarray:
    k = len(seq)
    if L == k:
        return np.asarray(seq)
    reps = 1 + rng.multinomial(L - k, np.full(k, 1.0 / k))
    return np.repeat(np.asarray(seq), reps)


def fit_path(seq, L: int, rng: np.random.Generator) -> np.ndarray:
    seq = np.asarray(seq)
    k = len(seq)
    if k <= L:
        return expand_runs(seq, L, rng)
    inner = np.sort(rng.choice(np.arange(1, k - 1), size=L - 2, replace=False)) if L > 2 else np.zeros(0, int)
    return seq[np.concatenate([[0], inner, [k - 1]]).astype(np.int64)]


def build_frame(venue_idx: np.ndarray, lengths: np.ndarray, tmin: np.ndarray, pois: pd.DataFrame,
                prefix: str) -> pd.DataFrame:
    n = len(lengths)
    trail_of = np.repeat(np.arange(n), lengths)
    pos = np.arange(len(venue_idx)) - np.repeat(np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    names = np.array([f"{prefix}_{i}" for i in range(n)], dtype=object)
    df = pd.DataFrame({
        "trail_id": names[trail_of],
        "user_id": names[trail_of],
        "venue_id": pois["venue_id"].to_numpy()[venue_idx].astype(str),
        "lat": pois["lat"].to_numpy(np.float64)[venue_idx],
        "lon": pois["lon"].to_numpy(np.float64)[venue_idx],
        "category": pois["category"].to_numpy()[venue_idx].astype(str),
        "t_local": REF_MONDAY + pd.to_timedelta(tmin, unit="m"),
        "pos": pos.astype(np.int64),
    })
    df["t_local"] = df["t_local"].astype("datetime64[ns]")
    return df[OUT_COLS]


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_outputs(df: pd.DataFrame, out: Path, sidecar: dict) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    sidecar = {**sidecar, "n_trails": int(df["trail_id"].nunique()), "n_checkins": int(len(df)),
               "python": sys.version.split()[0], "argv": sys.argv, "host_pid": os.getpid()}
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=_json_default))
    log(f"wrote {out} ({sidecar['n_trails']} trails, {sidecar['n_checkins']} check-ins) + sidecar .json")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def add_common_args(ap, default_length_mode: str = "hist", default_venue_prior: str = "uniform") -> None:
    ap.add_argument("--dataset", required=True, choices=CITIES)
    ap.add_argument("--eps", type=float, required=True, help="total privacy budget (generator + auxiliary)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=N_OUT, help="#synthetic trails (public constant, default 50000)")
    ap.add_argument("--user_level", action="store_true",
                    help="user-level DP: keep one trail per user (hash rule, --user_seed) before running")
    ap.add_argument("--user_seed", type=int, default=0, help="fixed seed of the one-trail-per-user rule")
    ap.add_argument("--out", type=str, required=True, help="output parquet path")
    ap.add_argument("--aux_frac", type=float, default=0.1,
                    help="fraction of eps for the auxiliary DP histograms (start week-hour, gap and, when used, "
                         "length; split evenly). The generator gets (1 - aux_frac) * eps.")
    ap.add_argument("--length_mode", choices=["hist", "native"], default=default_length_mode,
                    help="hist: #check-ins from the DP length histogram; the generated cell path is inflated "
                         "(dwell visits) or subsampled (endpoints kept) to that length. native: one check-in "
                         "per generated cell (clipped to [2, 20]).")
    ap.add_argument("--gap_mode", choices=["normalized", "first"], default="normalized",
                    help="contribution bounding for the central-DP gap histogram")
    ap.add_argument("--venue_prior", choices=["uniform", "dp"], default=default_venue_prior,
                    help="venue inside a generated cell: uniform over public catalog venues, or proportional to a "
                         "DP prior (thresholded noisy venue popularity + category-share floor; costs two more "
                         "auxiliary histograms out of the same aux_frac)")
    ap.add_argument("--venue_floor", type=float, default=0.6, help="share of the DP venue-weight mass in the floor")
    ap.add_argument("--venue_fp", type=float, default=100.0,
                    help="expected #pure-noise venues passing the popularity threshold")


def seeds(seed: int, n: int) -> list:
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]
