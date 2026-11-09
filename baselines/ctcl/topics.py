from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctcl_conf as C


def load_catalog(dataset: str, data: Path | None = None):
    data = Path(data or C.DATA)
    pois_f, meta_f = data / dataset / "pois.parquet", data / dataset / "meta.json"
    pois = pd.read_parquet(pois_f, columns=["venue_id", "lat", "lon", "category", "root"])
    pois["venue_id"] = pois["venue_id"].astype(str)
    pois["category"] = pois["category"].astype(str)
    pois["root"] = pois["root"].astype(str)
    bbox = json.loads(meta_f.read_text())["bbox"]
    return pois, {k: float(bbox[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")}, \
        {"pois_sha256": C.sha256_file(pois_f)}


def cell_of(lon, lat, bbox: dict) -> np.ndarray:
    n = C.GRID_N
    dx = (bbox["lon_max"] - bbox["lon_min"]) / n
    dy = (bbox["lat_max"] - bbox["lat_min"]) / n
    i = np.clip(np.ceil((np.asarray(lon, np.float64) - bbox["lon_min"]) / dx).astype(np.int64) - 1, 0, n - 1)
    j = np.clip(np.ceil((np.asarray(lat, np.float64) - bbox["lat_min"]) / dy).astype(np.int64) - 1, 0, n - 1)
    return i * n + j


def cell_centre(g: int, bbox: dict) -> tuple:
    i, j = divmod(int(g), C.GRID_N)
    dx = (bbox["lon_max"] - bbox["lon_min"]) / C.GRID_N
    dy = (bbox["lat_max"] - bbox["lat_min"]) / C.GRID_N
    return bbox["lat_min"] + (j + 0.5) * dy, bbox["lon_min"] + (i + 0.5) * dx


def daypart(hour) -> np.ndarray:
    return np.asarray(hour, np.int64) // 6


class VenueLookup:

    def __init__(self, pois: pd.DataFrame, bbox: dict):
        bad = sorted(set(pois["root"]) - set(C.ROOTS))
        if bad:
            raise ValueError(f"catalog roots outside ROOTS: {bad[:5]}")
        self.index = pd.Index(pois["venue_id"].to_numpy())
        if not self.index.is_unique:
            raise ValueError("duplicate venue ids in the catalog")
        self.root = pois["root"].map({r: i for i, r in enumerate(C.ROOTS)}).to_numpy(np.int64)
        self.cell = cell_of(pois["lon"].to_numpy(), pois["lat"].to_numpy(), bbox)

    def lookup(self, venue_ids) -> tuple:
        idx = self.index.get_indexer(pd.Index(np.asarray(venue_ids).astype(str)))
        if (idx < 0).any():
            raise ValueError("a trail venue is not in the public catalog")
        return self.root[idx], self.cell[idx]


def mode_first(vals) -> int:
    cnt = {}
    for v in vals:
        cnt[v] = cnt.get(v, 0) + 1
    m = max(cnt.values())
    for v in vals:
        if cnt[v] == m:
            return v
    raise ValueError("empty sequence")


def topic_of(roots, cells, first_hour: int) -> int:
    return C.topic_id(int(mode_first(list(roots))), int(mode_first(list(cells))), int(first_hour) // 6)


def assign_frame(df: pd.DataFrame, lk: VenueLookup) -> np.ndarray:
    df = df.sort_values(["trail_id", "pos"], kind="stable")
    roots, cells = lk.lookup(df["venue_id"].to_numpy())
    hours = df["t_local"].dt.hour.to_numpy()
    codes = df["trail_id"].to_numpy()
    out, start, n = [], 0, len(df)
    for i in range(1, n + 1):
        if i == n or codes[i] != codes[start]:
            out.append(topic_of(roots[start:i], cells[start:i], hours[start]))
            start = i
    return np.asarray(out, np.int64)


def build_table(dataset: str, data: Path | None = None, check_adagrid: bool = True) -> dict:
    pois, bbox, files = load_catalog(dataset, data)
    lk = VenueLookup(pois, bbox)
    if check_adagrid:
        n_cells = C.GRID_N * C.GRID_N
        if not ((lk.cell >= 0) & (lk.cell < n_cells)).all():
            raise AssertionError("a venue cell lies outside the public grid")
        centres = np.array([cell_centre(g, bbox) for g in range(n_cells)])
        if not np.array_equal(cell_of(centres[:, 1], centres[:, 0], bbox), np.arange(n_cells)):
            raise AssertionError("cell_of does not invert cell_centre on the public grid")
    n_roots_per_cat = pois.groupby("category")["root"].nunique()
    if int(n_roots_per_cat.max()) != 1:
        raise ValueError("a category has more than one root in this catalog")
    key = pd.DataFrame({"r": lk.root, "g": lk.cell, "category": pois["category"].to_numpy()})
    counts = key.groupby(["r", "g", "category"]).size().reset_index(name="n")
    top = {}
    for (r, g), grp in counts.groupby(["r", "g"], sort=True):
        grp = grp.sort_values(["n", "category"], ascending=[False, True], kind="stable")
        top[(int(r), int(g))] = grp["category"].head(C.N_CAT_KEYWORDS).tolist()
    rows = []
    for r in range(len(C.ROOTS)):
        for g in range(C.GRID_N * C.GRID_N):
            lat, lon = cell_centre(g, bbox)
            centre = f"{lat:.{C.COORD_DECIMALS}f},{lon:.{C.COORD_DECIMALS}f}"
            for b in range(C.N_DAYPARTS):
                kw = [C.ROOTS[r], C.DAYPARTS[b], centre] + top.get((r, g), [])
                assert len(kw) <= C.N_KEYWORDS
                rows.append({"k": C.topic_id(r, g, b), "r": r, "g": g, "b": b, "centre": centre, "keywords": kw,
                             "condition": C.condition(kw)})
    assert [x["k"] for x in rows] == list(range(C.K_TOPICS))
    return {"dataset": dataset, "K": C.K_TOPICS, "roots": C.ROOTS, "dayparts": C.DAYPARTS,
            "grid": {"n": C.GRID_N, "bbox": bbox, "cell": "g = 6 i + j, AdaGrid.locate (i: lon index, j: lat index)"},
            "doc_type": C.DOC_TYPE, "n_keywords_max": C.N_KEYWORDS, "n_category_keywords_max": C.N_CAT_KEYWORDS,
            "catalog": {"pois_sha256": files["pois_sha256"], "n_venues": int(len(pois)),
                        "n_categories": int(pois["category"].nunique()),
                        "n_nonempty_root_cells": int(len(top))},
            "rows": rows}


def table_bytes(table: dict) -> bytes:
    raw = json.dumps(table, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as z:
        z.write(raw)
    return buf.getvalue()


def load_table(path) -> dict:
    t = json.loads(gzip.decompress(Path(path).read_bytes()))
    if t["K"] != C.K_TOPICS or len(t["rows"]) != C.K_TOPICS or t["roots"] != C.ROOTS:
        raise ValueError(f"{path}: not a CTCL topic table")
    return t


def condition_tokens(table: dict, tok) -> list:
    return [len(tok(r["condition"], add_special_tokens=False)["input_ids"]) for r in table["rows"]]


def cmd_table(args):
    import os
    os.environ.setdefault("HF_HOME", C.HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL, revision=C.MODEL_COMMIT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in args.datasets:
        C.check_dataset(d)
        t = build_table(d)
        ntok = condition_tokens(t, tok)
        if max(ntok) > C.MAX_COND_TOKENS:
            raise SystemExit(f"{d}: a condition has {max(ntok)} > {C.MAX_COND_TOKENS} Qwen tokens")
        t["condition_tokens_max"] = int(max(ntok))
        b = table_bytes(t)
        p = out_dir / f"{d}.json.gz"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(b)
        tmp.replace(p)
        print(json.dumps({"dataset": d, "file": str(p), "sha256": C.sha256_bytes(b), "bytes": len(b),
                          "condition_tokens_max": int(max(ntok)), "nonempty_root_cells":
                          t["catalog"]["n_nonempty_root_cells"]}), flush=True)


def noisy_histogram(tau: np.ndarray, seed: int) -> np.ndarray:
    h = np.bincount(np.asarray(tau, np.int64), minlength=C.K_TOPICS).astype(np.float64)
    if len(h) != C.K_TOPICS:
        raise ValueError("topic id outside the public domain")
    rng = np.random.default_rng([seed, C.HIST_STREAM])
    return h + rng.normal(0.0, C.SIGMA_TOPIC_HIST, C.K_TOPICS)


def quotas(h_noisy: np.ndarray, n_out: int = C.N_OUT) -> tuple:
    if str(C.PROJECT) not in sys.path:
        sys.path.insert(0, str(C.PROJECT))
    from trajpe.ablations import apportion
    w = np.maximum(0, np.rint(h_noisy)).astype(np.int64)
    n = apportion(w.astype(np.float64), n_out)
    assert int(n.sum()) == n_out
    return w, n


def cmd_hist(args):
    t0 = time.time()
    C.check_dataset(args.dataset)
    sys.path.insert(0, str(C.PROJECT))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: M1 is never redrawn (R13)")
    table_file = Path(args.table)
    table = load_table(table_file)
    if table["dataset"] != args.dataset:
        raise SystemExit("topic table of another dataset")
    pois, bbox, files = load_catalog(args.dataset)
    if files["pois_sha256"] != table["catalog"]["pois_sha256"] or bbox != table["grid"]["bbox"]:
        raise SystemExit("the public catalog differs from the one the topic table was built from")
    lk = VenueLookup(pois, bbox)
    del pois
    df = pd.read_parquet(C.DATA / args.dataset / "train.parquet", columns=["trail_id", "venue_id", "t_local", "pos"])
    tau = assign_frame(df, lk)
    del df
    h_noisy = noisy_histogram(tau, args.seed)
    del tau
    w, n = quotas(h_noisy, args.n_out)
    res = {"mechanism": "M1: DP topic histogram (Gaussian, every bin of the public domain)", "dataset": args.dataset,
           "seed": args.seed, "noise_stream": [args.seed, C.HIST_STREAM], "K": C.K_TOPICS,
           "sigma_topic_hist": C.SIGMA_TOPIC_HIST, "sensitivity_l2": 1.0, "neighbouring": "add/remove one trail",
           "h_noisy": [float(x) for x in h_noisy], "w": [int(x) for x in w], "quota": [int(x) for x in n],
           "n_out": args.n_out, "quota_sha256": C.sha256_bytes(np.asarray(n, np.int64).tobytes()),
           "n_topics_with_quota": int((n > 0).sum()), "table_file": str(table_file),
           "table_sha256": C.sha256_file(table_file), "catalog_sha256": files["pois_sha256"],
           "data_dir": str(C.DATA), "code_sha256": {"topics.py": C.sha256_file(HERE / "topics.py"),
                                                    "ctcl_conf.py": C.sha256_file(HERE / "ctcl_conf.py")},
           "hist_sec": round(time.time() - t0, 1)}
    sha = C.write_json_atomic(out, res)
    out.with_name(out.name + ".sha256").write_text(sha + "\n")
    print(json.dumps({"hist": str(out), "sha256": sha, "n_topics_with_quota": res["n_topics_with_quota"],
                      "hist_sec": res["hist_sec"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--datasets", nargs="+", default=C.TABLE_DATASETS)
    t.add_argument("--out_dir", default=str(C.TOPIC_DIR))
    h = sub.add_parser("hist")
    h.add_argument("--dataset", required=True)
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--table", default="")
    h.add_argument("--n_out", type=int, default=C.N_OUT)
    h.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "table":
        cmd_table(a)
    else:
        a.table = a.table or str(C.table_path(a.dataset))
        cmd_hist(a)


if __name__ == "__main__":
    main()
