from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common

import numpy as np
import pandas as pd

DIST_BINS = np.array([0, .25, .5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, np.inf])
GAP_BINS = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 481, np.inf])
DTYPES = {"trail_id": "object", "user_id": "object", "venue_id": "object", "lat": "float64", "lon": "float64",
          "category": "object", "t_local": "datetime64[ns]", "pos": "int64"}


def jsd(p, q) -> float:
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return 1.0
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))
    return 0.5 * kl(p) + 0.5 * kl(q)


def trail_stats(df: pd.DataFrame, bbox: dict, cats: list) -> dict:
    df = df.sort_values(["trail_id", "pos"], kind="stable")
    codes, _ = pd.factorize(df["trail_id"], sort=False)
    lens = np.bincount(codes)
    first = np.concatenate([[0], np.cumsum(lens)[:-1]])
    lat0 = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
    kx, ky = 111.320 * np.cos(np.radians(lat0)), 110.574
    x = (df["lon"].to_numpy() - bbox["lon_min"]) * kx
    y = (df["lat"].to_numpy() - bbox["lat_min"]) * ky
    W, H = (bbox["lon_max"] - bbox["lon_min"]) * kx, (bbox["lat_max"] - bbox["lat_min"]) * ky
    same = codes[1:] == codes[:-1]
    hop = np.hypot(np.diff(x), np.diff(y))[same]
    travel = np.bincount(codes[1:][same], weights=hop, minlength=len(lens))
    cx = np.bincount(codes, weights=x) / lens
    cy = np.bincount(codes, weights=y) / lens
    gyr = np.sqrt(np.bincount(codes, weights=(x - cx[codes]) ** 2 + (y - cy[codes]) ** 2) / lens)
    t = df["t_local"].to_numpy("datetime64[m]").astype(np.int64)
    gaps = np.diff(t)[same]
    tt = df["t_local"]

    def cell(g, xx, yy):
        return (np.clip((yy / H * g).astype(int), 0, g - 1) * g + np.clip((xx / W * g).astype(int), 0, g - 1))
    last = first + lens - 1
    cat_idx = pd.Categorical(df["category"], categories=cats).codes
    return {
        "length": np.bincount(np.minimum(lens, 20), minlength=21),
        "hop": np.histogram(hop, DIST_BINS)[0], "travel": np.histogram(travel, DIST_BINS)[0],
        "gyration": np.histogram(gyr, DIST_BINS)[0],
        "density": np.bincount(cell(16, x, y), minlength=256),
        "od": np.bincount(cell(8, x[first], y[first]) * 64 + cell(8, x[last], y[last]), minlength=64 * 64),
        "cat": np.bincount(cat_idx[cat_idx >= 0], minlength=len(cats)),
        "hour": np.bincount(tt.dt.hour.to_numpy(), minlength=24),
        "dow": np.bincount(tt.iloc[first].dt.dayofweek.to_numpy(), minlength=7),
        "gap": np.histogram(gaps, GAP_BINS)[0],
        "mean_hop_km": float(hop.mean()) if len(hop) else float("nan"),
    }


def check(path: Path, ds_name: str, pois: pd.DataFrame, bbox: dict) -> dict:
    df = pd.read_parquet(path)
    side = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    res = {"file": str(path), "method": side.get("method", "?"), "eps": side.get("eps"),
           "runtime_sec": side.get("runtime_sec"), "problems": []}
    P = res["problems"]
    if list(df.columns) != common.OUT_COLS:
        P.append(f"columns {list(df.columns)} != {common.OUT_COLS}")
    for c, dt in DTYPES.items():
        if c in df and str(df[c].dtype) != dt and not (dt == "object" and str(df[c].dtype) in ("str", "string")):
            P.append(f"dtype {c}={df[c].dtype} (want {dt})")
    df = df.sort_values(["trail_id", "pos"], kind="stable")
    lens = df.groupby("trail_id", sort=False).size().to_numpy()
    res["n_trails"] = int(len(lens))
    exp_n = side.get("n_trails")
    if exp_n is not None and exp_n != len(lens):
        P.append(f"n_trails {len(lens)} != sidecar {exp_n}")
    if len(lens) != common.N_OUT:
        P.append(f"n_trails {len(lens)} != public release size {common.N_OUT}")
    if "user_level" in side and bool(side["user_level"]) != ("_user_" in path.name):
        P.append(f"sidecar user_level={side['user_level']} does not match file name")
    res["user_level"] = side.get("user_level")
    res["len_min"], res["len_max"], res["len_mean"] = int(lens.min()), int(lens.max()), float(lens.mean())
    res["len_hist_2_10plus"] = [round(float(v), 4) for v in
                                np.bincount(np.minimum(lens, 10), minlength=11)[2:] / len(lens)]
    if lens.min() < common.LEN_MIN or lens.max() > common.LEN_MAX:
        P.append("trail length outside [2, 20]")
    pos_ok = (df.groupby("trail_id", sort=False)["pos"].transform(lambda s: np.arange(len(s))) == df["pos"]).all()
    if not pos_ok:
        P.append("pos is not 0..L-1 within trails")
    t = df["t_local"].to_numpy("datetime64[m]").astype(np.int64)
    same = (df["trail_id"].to_numpy()[1:] == df["trail_id"].to_numpy()[:-1])
    g = np.diff(t)[same]
    if (g < 0).any():
        P.append("t_local decreasing within a trail")
    if (g > 480).any():
        P.append("gap > 8h within a trail")
    t0 = df.groupby("trail_id", sort=False)["t_local"].first()
    if not ((t0 >= common.REF_MONDAY) & (t0 < common.REF_MONDAY + pd.Timedelta(days=7))).all():
        P.append("trail start outside the reference week")
    cat = pois.set_index("venue_id")
    known = df["venue_id"].isin(cat.index)
    res["frac_known_venue"] = float(known.mean())
    if not known.all():
        P.append("venue_id not in public catalog")
    else:
        ref = cat.loc[df["venue_id"]]
        if not (np.allclose(ref["lat"].to_numpy(), df["lat"]) and np.allclose(ref["lon"].to_numpy(), df["lon"])
                and (ref["category"].to_numpy() == df["category"].to_numpy()).all()):
            P.append("lat/lon/category inconsistent with catalog")
    inb = ((df.lat >= bbox["lat_min"]) & (df.lat <= bbox["lat_max"]) & (df.lon >= bbox["lon_min"])
           & (df.lon <= bbox["lon_max"]))
    res["frac_in_bbox"] = float(inb.mean())
    res["n_unique_venues"] = int(df["venue_id"].nunique())
    res["ok"] = not P
    return res, df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--syn", nargs="+", required=True)
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--ref", default="val", choices=["train", "val", "test"])
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()
    meta = json.loads((common.DATA / args.dataset / "meta.json").read_text())
    pois = pd.read_parquet(common.DATA / args.dataset / "pois.parquet")
    pois["venue_id"] = pois["venue_id"].astype(str)
    cats = sorted(pois["category"].astype(str).unique())
    real = None
    if args.fidelity:
        rdf = pd.read_parquet(common.DATA / args.dataset / f"{args.ref}.parquet")
        rdf["venue_id"] = rdf["venue_id"].astype(str)
        real = trail_stats(rdf, meta["bbox"], cats)
        rl = rdf.groupby("trail_id").size()
        print(f"reference {args.dataset}/{args.ref}: {len(rl)} trails, mean len {rl.mean():.3f}, "
              f"mean hop {real['mean_hop_km']:.2f} km")
    rows = []
    for f in args.syn:
        res, df = check(Path(f), args.dataset, pois, meta["bbox"])
        if real is not None:
            s = trail_stats(df, meta["bbox"], cats)
            for k in ["length", "hop", "travel", "gyration", "density", "od", "cat", "hour", "dow", "gap"]:
                res[f"{k}_jsd"] = round(jsd(real[k], s[k]), 4)
            res["mean_hop_km"] = round(s["mean_hop_km"], 3)
        rows.append(res)
        status = "OK" if res["ok"] else "PROBLEMS: " + "; ".join(res["problems"])
        rt = f"{res['runtime_sec']:.1f}s" if res.get("runtime_sec") is not None else "?"
        print(f"\n{f}\n  {res['method']} eps={res['eps']} {status} | trails={res['n_trails']} "
              f"len[min/mean/max]={res['len_min']}/{res['len_mean']:.2f}/{res['len_max']} "
              f"in_bbox={res['frac_in_bbox']:.4f} known_venue={res['frac_known_venue']:.4f} "
              f"unique_venues={res['n_unique_venues']} runtime={rt}")
        print(f"  length dist (2..9, >=10): {res['len_hist_2_10plus']}")
        if real is not None:
            print("  JSD vs " + args.ref + ": " + " ".join(f"{k[:-4]}={res[k]:.4f}" for k in res if k.endswith("_jsd"))
                  + f" | mean_hop={res['mean_hop_km']:.2f}km")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
