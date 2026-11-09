from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import textfmt
from trajpe.api_llm import CITY_NAMES, WEEKDAYS, Grounder
from trajpe.catalog import Catalog
from trajpe.trails import TrailSet, to_frame

OUT_COLS = ["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"]
N_OUT = 50_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--train_json", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--n_out", type=int, default=N_OUT)
    ap.add_argument("--prior", default="", help="dploravp: vp.py prior .npz (DP venue weights + popular set)")
    ap.add_argument("--method", default="dplora")
    ap.add_argument("--ground_k", type=int, default=10)
    ap.add_argument("--ground_scale_km", type=float, default=0.5)
    ap.add_argument("--pop_radius_km", type=float, default=1.0)
    args = ap.parse_args()
    assert textfmt.WEEKDAYS == WEEKDAYS and textfmt.CITY_NAMES == CITY_NAMES
    t0 = time.time()
    if args.train_json and Path(args.train_json).exists():
        tj = json.loads(Path(args.train_json).read_text())
        oom = (tj.get("private_stats") or {}).get("oom_retries", 0)
        if oom:
            raise SystemExit(f"refusing the release: {args.train_json} has oom_retries={oom}")
        if tj.get("complete") is False:
            raise SystemExit(f"refusing the release: {args.train_json} is incomplete (debug --max_steps run)")
    texts = pd.read_parquet(args.texts)
    if len(texts) != args.n_out:
        raise SystemExit(f"{args.texts} has {len(texts)} texts, expected exactly {args.n_out}")
    cat = Catalog(args.dataset)
    cats = set(cat.categories)
    g = Grounder(cat, k=args.ground_k, scale_km=args.ground_scale_km, jitter_min=args.jitter)
    prior_info = None
    if args.prior:
        z = np.load(args.prior)
        pop = np.zeros(len(cat), np.float64)
        pop[z["pop_idx"]] = z["pop_val"]
        g.venue_w = z["venue_w"]
        g.set_popular((z["popular"], pop[z["popular"]]), radius_km=args.pop_radius_km)
        prior_info = json.loads(Path(args.prior).with_suffix(".json").read_text())
        gv2 = {k: v for k, v in [("ground_k", args.ground_k), ("ground_scale_km", args.ground_scale_km),
                                 ("pop_radius_km", args.pop_radius_km)] if v != ap.get_default(k)}
        if gv2:
            prior_info = {**prior_info, "grounding_v2": gv2}
    n_underflow = [0]
    base_choose = g._choose

    def safe_choose(xy, c_, rng_):
        try:
            return base_choose(xy, c_, rng_)
        except ValueError:
            n_underflow[0] += 1
            v = cat.nearest(xy, cat=c_, k=1) if c_ is not None else None
            if v is None:
                v = cat.nearest(xy, k=1)
            return int(np.atleast_1d(v)[0])
    g._choose = safe_choose
    rng = np.random.default_rng(args.seed)
    trails, n_ck, dist = [], 0, []
    for text, fin in zip(texts["text"], texts["finished"]):
        res, why = textfmt.parse(text, cats, bool(fin))
        if res is None:
            raise SystemExit(f"unparsable text in {args.texts}: {why}")
        day, cks = res
        poi, t = g(cks, day, rng)
        xy = cat.clip_xy(cat.to_xy(np.array([c["lat"] for c in cks]), np.array([c["lon"] for c in cks])))
        dist.append(np.linalg.norm(cat.xy[poi] - xy, axis=1))
        trails.append((poi, t))
        n_ck += len(cks)
    ts = TrailSet.from_lists(trails)
    df = to_frame(ts, cat, prefix="dplora")
    for c in ["trail_id", "user_id", "venue_id", "category"]:
        df[c] = df[c].astype(object)
    df["t_local"] = df["t_local"].astype("datetime64[ns]")
    df["pos"] = df["pos"].astype(np.int64)
    df = df[OUT_COLS]
    assert df["trail_id"].nunique() == args.n_out
    d = np.concatenate(dist)
    side = {}
    if args.train_json and Path(args.train_json).exists():
        side = json.loads(Path(args.train_json).read_text())
        side.pop("log", None)
        side.pop("private_stats", None)
        side.pop("response_tokens", None)
    sj = Path(args.texts).with_suffix(".json")
    samp = json.loads(sj.read_text()) if sj.exists() else {}
    side.update({
        "eps": side.get("eps_target"), "n_out": args.n_out,
        "sampling": {k: samp.get(k) for k in ["n_generated", "valid_rate", "reasons", "generated_tokens", "batches",
                                              "gen_batch", "max_new_tokens", "generation", "sample_gen_sec",
                                              "sample_total_sec", "gpu", "max_mem_gb"]},
        "method": args.method,
        "grounding": {"method": ("trajpe.api_llm.Grounder(Catalog(city), jitter_min=%d): nearest venue of the "
                                 "category" % args.jitter) if not args.prior else
                                ("trajpe.api_llm.Grounder(Catalog(city), jitter_min=%d) with venue_w = DP venue weights "
                                 "and set_popular((popular, pop[popular])): one of the k=10 nearest venues of the "
                                 "category or a DP-popular venue of the category within 1 km, drawn with probability "
                                 "proportional to venue_w x exp(-d / 0.5 km), as TrajPE grounds LLM output" % args.jitter),
                      "jitter_min": args.jitter,
                      "jitter_note": "0: no time jitter (as in plain DP-LoRA; the fine-tuned model learns real minutes)",
                      "dp_prior": prior_info, "seed": args.seed,
                      "checkins": int(n_ck), "dist_km_median": float(np.median(d)),
                      "dist_km_p90": float(np.quantile(d, 0.9)), "dist_km_mean": float(d.mean()),
                      "ground_sec": round(time.time() - t0, 1),
                      **({"underflow_fallbacks": n_underflow[0]} if n_underflow[0] else {})},
        "train_sec": side.get("train_loop_sec"), "train_total_sec": side.get("train_total_sec"),
        "sample_sec": samp.get("sample_gen_sec"), "sample_total_sec": samp.get("sample_total_sec"),
        "texts": str(args.texts), "n_trails": int(df["trail_id"].nunique()), "n_checkins": int(len(df)),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str))
    print(f"wrote {out}: {side['n_trails']} trails, {side['n_checkins']} check-ins; grounding distance median "
          f"{side['grounding']['dist_km_median'] * 1000:.0f} m, p90 {side['grounding']['dist_km_p90'] * 1000:.0f} m "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
