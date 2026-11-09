from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np

import common
from accounting import budget
from encoding import Decoder, PublicDomain, Spec, spatial_tilt

FROZEN = HERE / "frozen.json"
DEFAULTS = {"mech": "aim", "k_sub": 16, "grid": 16, "od_grid": 8, "max_pos": 10, "time_split": True,
            "strata": "2,3,4,5,6-20", "workload": "seq", "init": "oneway", "rounds": 4.0, "max_model_size": 80.0, "max_iters": 1000,
            "prior_frac": 0.1, "venue_floor": 0.6, "venue_fp": 100.0, "venue_prior": "dp", "tilt": 0}


class Tee(io.StringIO):

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        return super().write(s)


def parse_strata(s: str) -> tuple:
    out = []
    for part in str(s).split(","):
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return tuple(out)


def frozen_defaults(eps: float, path: Path = FROZEN) -> dict:
    if not path.exists():
        return {}
    fz = json.loads(path.read_text())
    out = dict(fz.get("defaults", {}))
    by = fz.get("by_eps", {})
    if by:
        key = min(by, key=lambda k: abs(np.log(float(k)) - np.log(eps)))
        out.update(by[key])
    return out


def allocate(n_out: int, totals: list) -> np.ndarray:
    t = np.array([x if np.isfinite(x) and x > 0 else 0.0 for x in totals])
    if t.sum() <= 0:
        t = np.ones(len(totals))
    q = n_out * t / t.sum()
    n = np.floor(q).astype(np.int64)
    n[np.argsort(-(q - n), kind="stable")[: n_out - n.sum()]] += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="directory under data/processed (any city with train/pois/meta)")
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=common.N_OUT)
    ap.add_argument("--user_level", action="store_true", help="bound every user to one trail (user-level DP)")
    ap.add_argument("--user_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mech", choices=["aim", "mst"])
    ap.add_argument("--k_sub", type=int, help="leaves per root with their own class (public catalog counts)")
    ap.add_argument("--grid", type=int, help="start-cell grid (default 16)")
    ap.add_argument("--od_grid", type=int, help="last-cell grid (default 8)")
    ap.add_argument("--max_pos", type=int, help="positions 2..max_pos get their own attributes")
    ap.add_argument("--time_split", type=int, choices=[0, 1], help="start time as (dow, hod) (1) or week-hour (0)")
    ap.add_argument("--strata", type=str, help="length strata, e.g. '2,3,4,5,6-20' or '2-20' (one table)")
    ap.add_argument("--workload", choices=["seq", "official2"], help="seq: hand-built (2026-10-21); official2: all 2-way marginals with <= 10,000 cells, weight 1 (official aim.py defaults; frozen_v2.json)")
    ap.add_argument("--init", choices=["oneway", "presence"], help="AIM initial cliques: one-way marginals (official "
                    "default) or one presence-linking clique per attribute (mech.presence_init)")
    ap.add_argument("--rounds", type=float, help="AIM rounds T per stratum; < 50 means a multiple of the stratum's "
                                                 "#attributes (0 = official default 16 * #attributes)")
    ap.add_argument("--max_model_size", type=float, help="AIM max junction-tree size in MB (official 80)")
    ap.add_argument("--max_iters", type=int, help="AIM mirror-descent iterations per round (official 1000)")
    ap.add_argument("--prior_frac", type=float, help="share of eps for the DP venue + category prior (0.1)")
    ap.add_argument("--venue_prior", choices=["dp", "uniform"])
    ap.add_argument("--venue_floor", type=float)
    ap.add_argument("--venue_fp", type=float)
    ap.add_argument("--tilt", type=int, choices=[0, 1], help="decoder: spatial tilt of hop venues toward the "
                    "synthetic start-cell distribution (encoding.spatial_tilt; post-processing)")
    ap.add_argument("--save_records", action="store_true", help="also save the synthetic records (<out>.records.npz)")
    ap.add_argument("--no_frozen", action="store_true", help="ignore frozen.json (built-in defaults)")
    ap.add_argument("--config", type=Path, default=FROZEN, help="settings file read instead of frozen.json, e.g. "
                    "frozen_v2.json (the no-tuning baseline configuration, README section v2)")
    args = ap.parse_args(argv)
    cfg = dict(DEFAULTS)
    if not args.no_frozen:
        cfg.update(frozen_defaults(args.eps, args.config))
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    cfg["time_split"] = bool(cfg["time_split"])
    if cfg["venue_prior"] == "uniform":
        cfg["prior_frac"] = 0.0

    t_start = time.time()
    s_prior, s_mech, s_synth, s_dec = common.seeds(args.seed, 4)
    ds = common.load_dataset(args.dataset, args.user_level, args.user_seed)
    bud = budget(args.eps, args.delta, cfg["prior_frac"])
    common.log(f"AIM-seq {args.dataset}: eps={args.eps} delta={args.delta} user_level={args.user_level} "
               f"rho_total={bud['rho_total']:.5f} rho_mech={bud['rho_mech']:.5f} cfg={cfg}")
    if cfg["venue_prior"] == "dp":
        prior = common.VenuePrior.fit_central(ds, bud["eps_venue"], bud["eps_cat"], np.random.default_rng(s_prior),
                                              cfg["venue_floor"], cfg["venue_fp"])
    else:
        prior = common.VenuePrior.uniform()
    spec = Spec(grid=cfg["grid"], od_grid=cfg["od_grid"], k_sub=cfg["k_sub"], max_pos=cfg["max_pos"],
                time_split=cfg["time_split"], strata=parse_strata(cfg["strata"]))
    dom = PublicDomain(ds.pois, ds.bbox, spec)
    assert dom.categories == ds.categories

    import mech as M
    strat = spec.stratum_of(ds.lengths)
    trail_of = np.repeat(np.arange(len(ds.lengths)), ds.lengths)
    gaps_all = np.concatenate(ds.gaps)
    gap_trail = np.repeat(np.arange(len(ds.lengths)), ds.lengths - 1)
    mseeds = common.seeds(s_mech, len(spec.strata))
    sseeds = common.seeds(s_synth, len(spec.strata))
    fits, infos, ledgers = [], [], []
    t0 = time.time()
    log_buf = Tee(sys.stdout)
    for j, st in enumerate(spec.strata):
        keep = strat == j
        attrs = dom.attributes(st)
        rec = dom.encode(ds.checkin_venue[keep[trail_of]], ds.lengths[keep], ds.start_wh[keep],
                         gaps_all[keep[gap_trail]], st)
        data = M.make_dataset(rec, attrs)
        rounds = cfg["rounds"]
        rounds = int(round(rounds * len(attrs))) if 0 < rounds < 50 else int(rounds)
        common.log(f"stratum {st}: {len(attrs)} attributes, T={rounds or 16 * len(attrs)}")
        ts = time.time()
        with contextlib.redirect_stdout(log_buf):
            if cfg["mech"] == "aim":
                workload = M.build_workload(attrs, cfg["workload"])
                init = M.presence_init(attrs) if cfg["init"] == "presence" else None
                sample, total, ledger, info = M.run_aim(data, workload, bud["rho_mech"], mseeds[j],
                                                        rounds=rounds or None, max_model_size=cfg["max_model_size"],
                                                        max_iters=cfg["max_iters"], initial_cliques=init)
            else:
                sample, total, ledger, info = M.run_mst(data, bud["rho_mech"], mseeds[j])
        info["stratum"], info["n_attributes"], info["sec"] = list(st), len(attrs), time.time() - ts
        info["dp_total"] = total
        if ledger.rho > bud["rho_mech"] * (1 + 1e-9):
            raise RuntimeError(f"stratum {st}: privacy ledger exceeds the budget: {ledger.rho} > {bud['rho_mech']}")
        common.log(f"stratum {st}: {info['sec']:.0f}s, rho spent {ledger.rho:.6f} / {bud['rho_mech']:.6f}, "
                   f"DP total {total:.0f}")
        fits.append(sample)
        infos.append(info)
        ledgers.append(ledger.summary())
    t_mech = time.time() - t0
    mlog = log_buf.getvalue().splitlines()
    n_per = allocate(args.n_out, [i["dp_total"] for i in infos])

    t1 = time.time()
    synths, saved = {}, {}
    for j, st in enumerate(spec.strata):
        if n_per[j] <= 0:
            continue
        attrs = dom.attributes(st)
        synth = fits[j](int(n_per[j]), sseeds[j])
        srec = {a: np.asarray(synth.to_dict()[a], np.int64) for a in attrs}
        assert len(next(iter(srec.values()))) == n_per[j]
        synths[j] = srec
        if args.save_records:
            saved.update({f"{st[0]}-{st[1]}/{a}": v for a, v in srec.items()})
    tilt = (spatial_tilt(dom, prior.weights, np.concatenate([r["sc"] for r in synths.values()]))
            if cfg["tilt"] else None)
    dec = Decoder(dom, prior.weights, tilt)
    rng = np.random.default_rng(s_dec)
    dstats: dict = {}
    parts = []
    for j, st in enumerate(spec.strata):
        if j in synths:
            parts.append(dec.decode(synths[j], rng, spec.min_gap, st, dstats)[:3])
    off_parts = [np.concatenate([[0], np.cumsum(p[1])]) for p in parts]
    trails = [(p[0][o[i]:o[i + 1]], p[2][o[i]:o[i + 1]]) for p, o in zip(parts, off_parts) for i in range(len(p[1]))]
    perm = rng.permutation(len(trails))
    vidx = np.concatenate([trails[i][0] for i in perm])
    tmin = np.concatenate([trails[i][1] for i in perm])
    lengths = np.array([len(trails[i][0]) for i in perm], np.int64)
    t_dec = time.time() - t1
    df = common.build_frame(vidx, lengths, tmin, ds.pois, prefix="aimseq")
    assert df["trail_id"].nunique() == args.n_out
    if args.save_records:
        np.savez_compressed(Path(args.out).with_suffix(".records.npz"), **saved)
    runtime = time.time() - t_start
    rho_spent = max(led["rho_spent"] for led in ledgers)
    common.write_outputs(df, Path(args.out), {
        "method": f"AIM-seq ({cfg['mech'].upper()})", "source": "https://github.com/ryan112358/private-pgm (mbi)",
        "commit": M.git_commit(), "jax_version": M.jax.__version__, "dataset": args.dataset, "eps": args.eps,
        "delta": args.delta, "user_level": bool(args.user_level), "user_seed": args.user_seed, "seed": args.seed,
        "config": cfg, "config_file": None if args.no_frozen else str(args.config), "budget": bud,
        "privacy": {"composition": "Laplace prior (pure DP -> zCDP) + one mechanism per disjoint length stratum "
                                   "(parallel composition): rho = rho_prior + max over strata of rho_spent",
                    "rho_spent_max_stratum": rho_spent, "rho_total_spent": bud["rho_prior"] + rho_spent},
        "strata": [{"stratum": i["stratum"], "dp_total": i["dp_total"], "n_out": int(n)} for i, n in zip(infos, n_per)],
        "ledgers": ledgers, "mech_info": infos,
        "measured": [ln.split("Measuring Clique", 1)[1].strip() for ln in mlog if "Measuring Clique" in ln],
        "sigma_reductions": sum("Reducing sigma" in ln for ln in mlog),
        "venue_prior": prior.info, "domain": dom.summary(), "decode": dstats,
        "runtime_sec": runtime, "mech_sec": t_mech, "decode_sec": t_dec})
    common.log(f"done in {runtime:.0f}s (mech {t_mech:.0f}s, decode {t_dec:.0f}s); strata n_out {n_per.tolist()}; "
               f"decode stats {dstats}")


if __name__ == "__main__":
    main()
