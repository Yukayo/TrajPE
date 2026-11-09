import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "baselines" / "dplora"))

import pandas as pd

import hybrid as HY
import run_pe as RP
import textfmt
from trajpe.api_pool import PoolAPI
from trajpe.dp import eps_for
from trajpe.paths import RESULTS, RUNS


def load_generator(work: Path, eps: float, texts_name: str = "", texts_required: bool = True):
    cal = json.loads((work / "calib.json").read_text())
    tj = json.loads((work / "train.json").read_text())
    if cal.get("method") != "hybrid" or abs(cal["eps_total"] - eps) > 1e-12:
        raise SystemExit(f"{work}/calib.json is not a hybrid calibration for eps={eps}")
    if (tj.get("private_stats") or {}).get("oom_retries", 0) or tj.get("complete") is False:
        raise SystemExit(f"refusing {work}: training release gate (oom_retries / incomplete)")
    for k_t, k_c in (("noise_multiplier", "sigma_sgd"), ("sample_rate", "sample_rate"), ("steps", "steps")):
        if abs(float(tj[k_t]) - float(cal[k_c])) > 1e-12:
            raise SystemExit(f"train.json {k_t}={tj[k_t]} does not match calib {k_c}={cal[k_c]}")
    if abs(float(tj["eps_target"]) - eps) > 1e-12 or tj.get("eps_prv", 0) > eps:
        raise SystemExit("training record does not match eps or its PRV epsilon exceeds it")
    cands = sorted(work.glob(texts_name or "texts*.parquet"))
    if not texts_required:
        if cands:
            raise SystemExit(f"--generator_failed but {work} holds texts {[c.name for c in cands]}")
        return cal, tj, None
    if len(cands) != 1:
        raise SystemExit(f"expected exactly one texts parquet in {work}, found {[c.name for c in cands]}")
    return cal, tj, cands[0]


def parse_texts(path: Path, categories, n: int):
    texts = pd.read_parquet(path)
    cats = set(categories)
    recs = []
    for text, fin in zip(texts["text"], texts["finished"]):
        res, why = textfmt.parse(text, cats, bool(fin))
        if res is None:
            raise SystemExit(f"unparsable text in {path}: {why}")
        recs.append(res)
        if n and len(recs) == n:
            break
    if n and len(recs) < n:
        raise SystemExit(f"{path} has {len(recs)} texts < pool_n {n}")
    return recs


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True, help="TOTAL epsilon (DP-SGD + PE)")
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--work", default="", help="generator work dir (calib.json, train.json, texts*.parquet)")
    ap.add_argument("--diag_texts", default="",
                    help="DIAGNOSTIC ONLY: texts of a model that already spent its own budget; PE then spends --eps "
                         "on top, so the output is NOT a release (marked in result.json)")
    ap.add_argument("--texts", default="", help="texts parquet name inside --work (default: the only texts*.parquet)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pool_n", type=int, default=0, help="texts used (first n in sampling order; 0 = all)")
    ap.add_argument("--pool_frac", type=float, default=1.0)
    ap.add_argument("--pool_refresh", type=float, default=0.0)
    ap.add_argument("--pool_refresh_share", type=float, default=0.0,
                    help="v5 (default off): share of EVERY variation call's children replaced by pool samples "
                         "(unused first), independent of the degree (trajpe.api_pool.PoolAPI refresh_share)")
    ap.add_argument("--texts_extra", default="",
                    help="v5 (default off): comma list of further texts parquets sampled from the SAME adapter "
                         "(sample_vllm.py, post-processing), parsed after --work's texts; --pool_n counts them all")
    ap.add_argument("--ground_k", type=int, default=10)
    ap.add_argument("--ground_scale_km", type=float, default=0.05)
    ap.add_argument("--pop_radius_km", type=float, default=1.0)
    ap.add_argument("--api", default="sim", choices=["sim"], help="fixed: the pool API wraps the simulator API")
    ap.add_argument("--generator_failed", default="",
                    help="freeze 3.1 (default off): <work>/generator_failed.json written by scripts/v4_job.sh when the "
                         "generator failed sample_vllm.py's frozen validity gate (exit 3); PE then runs with the "
                         "simulator as the only candidate source (no model sample: pool_frac 0, pool_refresh 0), every "
                         "other flag and mu_PE unchanged, and the DP-SGD part stays charged")
    args, rest = ap.parse_known_args()
    if any(a in ("--eps", "--delta", "--tag", "--dataset", "--api") or a.startswith(("--eps=", "--api="))
           for a in rest):
        raise SystemExit("pass --eps/--delta/--tag/--dataset/--api once")
    diag = bool(args.diag_texts)
    if diag == bool(args.work):
        raise SystemExit("give exactly one of --work and --diag_texts")
    gen_failed, configured = None, None
    if diag:
        if args.generator_failed:
            raise SystemExit("--generator_failed needs --work (a generator that spent its budget), not --diag_texts")
        work, cal, tj, texts = None, None, {}, Path(args.diag_texts)
        eps_pe = args.eps
    else:
        work = Path(args.work)
        if args.generator_failed:
            gen_failed = json.loads(Path(args.generator_failed).read_text())
            if gen_failed.get("status") != "generator_failed" or gen_failed.get("sampler_exit") != 3 or \
                    Path(gen_failed.get("work", "")).resolve() != work.resolve():
                raise SystemExit(f"{args.generator_failed} is not a generator_failed marker of {work}")
        cal, tj, texts = load_generator(work, args.eps, args.texts, texts_required=gen_failed is None)
        if abs(cal["delta"] - args.delta) > 0:
            raise SystemExit("delta mismatch")
        eps_pe = eps_for(cal["mu_pe"] * (1.0 - HY.MU_SAFETY), args.delta)

    from trajpe.catalog import Catalog
    extra = [Path(x) for x in args.texts_extra.split(",") if x]
    if gen_failed is None and extra:
        cats = Catalog(args.dataset).categories
        recs = parse_texts(texts, cats, 0)
        for x in extra:
            recs += parse_texts(x, cats, 0)
        if args.pool_n:
            if len(recs) < args.pool_n:
                raise SystemExit(f"{len(recs)} texts < pool_n {args.pool_n}")
            recs = recs[:args.pool_n]
    elif gen_failed is None:
        recs = parse_texts(texts, Catalog(args.dataset).categories, args.pool_n)
    else:
        configured = {"pool_n": args.pool_n, "pool_frac": args.pool_frac, "refresh": args.pool_refresh}
        if args.pool_refresh_share or extra:
            configured.update(refresh_share=args.pool_refresh_share, texts_extra=[str(x) for x in extra])
        recs, args.pool_frac, args.pool_refresh, args.pool_refresh_share = [], 0.0, 0.0, 0.0
    base_build = RP.build_api

    def build_pool_api(name, catalog, a):
        base = base_build("sim", catalog, a)
        return PoolAPI(catalog, recs, base.random, base.variation, pool_frac=args.pool_frac,
                       refresh=args.pool_refresh, ground_k=args.ground_k, ground_scale_km=args.ground_scale_km,
                       pop_radius_km=args.pop_radius_km, refresh_share=args.pool_refresh_share)
    RP.build_api = build_pool_api
    sys.argv = [str(ROOT / "scripts" / "run_pe.py"), "--dataset", args.dataset, "--api", "sim", "--eps", repr(eps_pe),
                "--delta", repr(args.delta), "--tag", args.tag] + rest
    try:
        RP.main()
    finally:
        RP.build_api = base_build

    out = RUNS / args.dataset / args.tag / "result.json"
    res = json.loads(out.read_text())
    pool = {"n_texts": len(recs), "pool_frac": args.pool_frac, "refresh": args.pool_refresh, "ground_k": args.ground_k,
            "ground_scale_km": args.ground_scale_km, "pop_radius_km": args.pop_radius_km}
    if args.pool_refresh_share > 0:
        pool["refresh_share"] = args.pool_refresh_share
    if extra and gen_failed is None:
        pool["texts_extra"] = [str(x) for x in extra]
    if gen_failed is not None:
        pool["generator_failed"] = gen_failed
        pool["configured"] = configured
    if diag:
        res["hybrid"] = {"diagnostic": "NOT A RELEASE: the generator spent its own budget and PE spent eps on top",
                         "texts": str(texts), "pool": pool}
        out.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({"tag": args.tag, "diagnostic": True}))
        return
    mu_spent = float(res["privacy"]["mu"])
    chk = HY.check(cal, mu_spent)
    res["hybrid"] = {"eps_total": args.eps, "delta": args.delta, "eps_pe_arg": eps_pe, "calib": cal,
                     "generator": {"work": str(work), "texts": None if texts is None else str(texts),
                                   "model": tj.get("model"),
                                   "noise_multiplier": tj["noise_multiplier"], "sample_rate": tj["sample_rate"],
                                   "steps": tj["steps"], "eps_prv_composed": tj.get("eps_prv"),
                                   "eps_rdp_composed": tj.get("eps_rdp"), "eps_pld_composed": tj.get("eps_pld")},
                     "pool": pool,
                     "check": chk,
                     "guarantee": ("(eps_total, delta)-DP, trail level, add/remove-one: PLD composition of DP-SGD "
                                   "(PoissonSampled(q, Gaussian(sigma))^T) with PE's Gaussian mechanisms "
                                   "(mu_PE-GDP, exact); the pool, grounding and release are post-processing")}
    if gen_failed is not None:
        res["hybrid"]["generator_failed"] = gen_failed
    out.write_text(json.dumps(res, indent=1, default=float))
    dest = RESULTS / "pe" / args.dataset / f"{args.tag}.json"
    if dest.exists():
        dest.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"tag": args.tag, "hybrid_check": chk}))


if __name__ == "__main__":
    main()
