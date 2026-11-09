from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROCESSED, RAW

TRAIL_GAP_HOURS = 8.0
MIN_TRAIL_LEN = 2
MAX_TRAIL_LEN = 20
SPLITS = ("train", "val", "test")

DATASETS = {
    "nyc": ("massive_steps", "massive_steps_new_york", "America/New_York"),
    "istanbul": ("massive_steps", "massive_steps_istanbul", "Europe/Istanbul"),
    "jakarta": ("massive_steps", "massive_steps_jakarta", "Asia/Jakarta"),
    "tokyo": ("tsmc", "tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_TKY.txt", "Asia/Tokyo"),
    "sao_paulo": ("massive_steps", "massive_steps_sao_paulo", "America/Sao_Paulo"),
    "kuwait_city": ("massive_steps", "massive_steps_kuwait_city", "Asia/Kuwait"),
    "tsmc_nyc": ("tsmc", "tsmc2014/extracted/dataset_tsmc2014/dataset_TSMC2014_NYC.txt", "America/New_York"),
    "petaling_jaya": ("massive_steps", "massive_steps_petaling_jaya", "Asia/Kuala_Lumpur"),
    "moscow": ("massive_steps", "massive_steps_moscow", "Europe/Moscow"),
    "bandung": ("massive_steps", "massive_steps_bandung", "Asia/Jakarta"),
    "sydney": ("massive_steps", "massive_steps_sydney", "Australia/Sydney"),
    "melbourne": ("massive_steps", "massive_steps_melbourne", "Australia/Melbourne"),
    "tangerang": ("massive_steps", "massive_steps_tangerang", "Asia/Jakarta"),
    "palembang": ("massive_steps", "massive_steps_palembang", "Asia/Jakarta"),
    "shanghai": ("massive_steps", "massive_steps_shanghai", "Asia/Shanghai"),
    "beijing": ("massive_steps", "massive_steps_beijing", "Asia/Shanghai"),
    "ms_tokyo": ("massive_steps", "massive_steps_tokyo", "Asia/Tokyo"),
    "gw_stockholm": ("gowalla", "gowalla_snap/loc-gowalla_totalCheckins.txt.gz", "Europe/Stockholm"),
    "gw_san_francisco": ("gowalla", "gowalla_snap/loc-gowalla_totalCheckins.txt.gz", "America/Los_Angeles"),
    "gw_austin": ("gowalla", "gowalla_snap/loc-gowalla_totalCheckins.txt.gz", "America/Chicago"),
    "gw_nyc": ("gowalla", "gowalla_snap/loc-gowalla_totalCheckins.txt.gz", "America/New_York"),
}
V2_DATASETS = ("nyc", "istanbul", "jakarta", "tokyo", "sao_paulo", "kuwait_city")
CATEGORY_MAP_OF = {n: ("v2" if n in V2_DATASETS else "v3_gowalla" if DATASETS[n][0] == "gowalla" else "v3_fsq")
                   for n in DATASETS}
GOWALLA_SPOTS = "gowalla_liu/gowalla_spots_subset1.csv"
GOWALLA_SPLIT_SALT = b"trajpe-data-v3/gowalla-split/"
EARTH_RADIUS_KM = 6371.0088


def _load_massive_steps(rel: str, tz: str) -> pd.DataFrame:
    parts = []
    for raw_split, split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        d = pd.read_parquet(RAW / rel / f"{raw_split}.parquet")
        t_utc = pd.to_datetime(d["timestamp"]).dt.tz_localize("UTC")
        parts.append(pd.DataFrame({
            "trail_id": d["trail_id"].astype(str),
            "user_id": d["user_id"].astype(str),
            "venue_id": d["venue_id"].astype(str),
            "lat": d["latitude"].astype(float),
            "lon": d["longitude"].astype(float),
            "category": d["venue_category"].astype(str),
            "name": d["name"],
            "t_local": t_utc.dt.tz_convert(tz).dt.tz_localize(None),
            "split": split,
        }))
    return pd.concat(parts, ignore_index=True)


def _load_tsmc(rel: str, seed: int = 0) -> pd.DataFrame:
    d = pd.read_csv(RAW / rel, sep="\t", header=None, encoding="latin-1",
                    names=["user", "venue", "cat_id", "cat", "lat", "lon", "tz_min", "utc"])
    t_utc = pd.to_datetime(d["utc"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df = pd.DataFrame({
        "user_id": d["user"].astype(str),
        "venue_id": d["venue"].astype(str),
        "lat": d["lat"].astype(float),
        "lon": d["lon"].astype(float),
        "category": d["cat"].astype(str),
        "name": None,
        "t_local": (t_utc + pd.to_timedelta(d["tz_min"], unit="m")).dt.tz_localize(None),
    })
    df = df.sort_values(["user_id", "t_local"], kind="stable").reset_index(drop=True)
    gap = df.groupby("user_id")["t_local"].diff().dt.total_seconds() / 3600
    seg = (gap.isna() | (gap > TRAIL_GAP_HOURS)).astype(int).groupby(df["user_id"]).cumsum()
    df["trail_id"] = df["user_id"] + "_" + seg.astype(str)
    trails = df["trail_id"].unique()
    rng = np.random.default_rng(seed)
    u = rng.random(len(trails))
    split_of = dict(zip(trails, np.where(u < 0.7, "train", np.where(u < 0.8, "val", "test"))))
    df["split"] = df["trail_id"].map(split_of)
    return df


_GOWALLA_CACHE: dict = {}


def gowalla_spot_categories() -> pd.Series:
    if "spots" not in _GOWALLA_CACHE:
        import ast
        sp = pd.read_csv(RAW / GOWALLA_SPOTS, usecols=["id", "spot_categories"])
        key_of = {}
        for s in pd.unique(sp["spot_categories"].astype(str)):
            cats = ast.literal_eval(s)
            assert len(cats) == 1, s
            key_of[s] = "gowalla:" + cats[0]["url"].rsplit("/", 1)[-1]
        assert sp["id"].is_unique
        _GOWALLA_CACHE["spots"] = pd.Series(sp["spot_categories"].astype(str).map(key_of).to_numpy(),
                                            index=sp["id"].to_numpy())
    return _GOWALLA_CACHE["spots"]


def gowalla_trails(rel: str) -> pd.DataFrame:
    if rel not in _GOWALLA_CACHE:
        d = pd.read_csv(RAW / rel, sep="\t", header=None, names=["user", "utc", "lat", "lon", "loc"],
                        dtype={"user": np.int64, "utc": str, "lat": np.float64, "lon": np.float64, "loc": np.int64})
        t = pd.to_datetime(d["utc"], format="%Y-%m-%dT%H:%M:%SZ", utc=True)
        d["t"] = t
        d["sec"] = (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
        d["order"] = np.arange(len(d))
        d = d.sort_values(["user", "sec", "order"], kind="stable").reset_index(drop=True)
        u, sec, loc = d["user"].to_numpy(), d["sec"].to_numpy(), d["loc"].to_numpy()
        lat, lon = np.radians(d["lat"].to_numpy()), np.radians(d["lon"].to_numpy())
        first = np.ones(len(d), dtype=bool)
        first[1:] = u[1:] != u[:-1]
        dt = np.empty(len(d))
        dt[0] = np.inf
        dt[1:] = sec[1:] - sec[:-1]
        h = np.zeros(len(d))
        h[1:] = (np.sin((lat[1:] - lat[:-1]) / 2) ** 2
                 + np.cos(lat[1:]) * np.cos(lat[:-1]) * np.sin((lon[1:] - lon[:-1]) / 2) ** 2)
        dist_m = 2 * EARTH_RADIUS_KM * 1000 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
        same_loc = np.zeros(len(d), dtype=bool)
        same_loc[1:] = loc[1:] == loc[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, dist_m / dt, np.inf)
        invalid = ~first & (same_loc | (dt < 60) | (dt <= 0) | (speed > 343))
        valid = ~first & ~invalid
        reset = first | (valid & (dt >= TRAIL_GAP_HOURS * 3600))
        same = valid & ~reset
        seg = np.cumsum(reset)
        idx = np.flatnonzero(same)
        _, k = np.unique(seg[idx], return_index=True)
        emit = same.copy()
        emit[idx[k] - 1] = True
        d["seg"] = seg - np.maximum.accumulate(np.where(first, seg, 0))
        _GOWALLA_CACHE[rel] = d[emit].reset_index(drop=True).drop(columns=["order"])
        _GOWALLA_CACHE[rel + "#raw"] = d[["lat", "lon"]]
    return _GOWALLA_CACHE[rel]


def gowalla_split(trail_ids) -> np.ndarray:
    import hashlib
    u = np.array([int.from_bytes(hashlib.sha256(GOWALLA_SPLIT_SALT + str(t).encode()).digest()[:8], "big")
                  for t in trail_ids], dtype=np.float64) / 2.0 ** 64
    return np.where(u < 0.7, "train", np.where(u < 0.8, "val", "test"))


def _load_gowalla(rel: str, tz: str, bbox: dict) -> tuple[pd.DataFrame, dict]:
    d = gowalla_trails(rel)
    raw = _GOWALLA_CACHE[rel + "#raw"]
    inside_raw = (raw["lat"].between(bbox["lat_min"], bbox["lat_max"])
                  & raw["lon"].between(bbox["lon_min"], bbox["lon_max"]))
    inside = (d["lat"].between(bbox["lat_min"], bbox["lat_max"]) & d["lon"].between(bbox["lon_min"], bbox["lon_max"]))
    d = d[inside].reset_index(drop=True)
    trail_id = d["user"].astype(str) + "_" + d["seg"].astype(str)
    trails = pd.unique(trail_id)
    split_of = dict(zip(trails, gowalla_split(trails)))
    df = pd.DataFrame({
        "trail_id": trail_id,
        "user_id": d["user"].astype(str),
        "venue_id": d["loc"].astype(str),
        "lat": d["lat"].astype(float),
        "lon": d["lon"].astype(float),
        "category": d["loc"].map(gowalla_spot_categories()),
        "name": None,
        "t_local": d["t"].dt.tz_convert(tz).dt.tz_localize(None).astype("datetime64[us]"),
        "split": trail_id.map(split_of),
    })
    stats = {"snap_checkins_in_area": int(inside_raw.sum()), "checkins_in_trails_in_area": int(len(df)),
             "share_kept_by_trail_construction": round(len(df) / max(int(inside_raw.sum()), 1), 4)}
    return df, stats


def _segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["trail_id", "t_local"], kind="stable").reset_index(drop=True)
    gap = df.groupby("trail_id")["t_local"].diff().dt.total_seconds() / 3600
    sub = (gap > TRAIL_GAP_HOURS).astype(int).groupby(df["trail_id"]).cumsum()
    df["trail_id"] = df["trail_id"] + "#" + sub.astype(str)
    df["pos"] = df.groupby("trail_id").cumcount()
    df = df[df["pos"] < MAX_TRAIL_LEN]
    n = df.groupby("trail_id")["pos"].transform("size")
    return df[n >= MIN_TRAIL_LEN].reset_index(drop=True)


def prepare(name: str) -> dict:
    from .public_catalog import PUBLIC_BBOX, load_category_map, map_checkins
    loader, rel, tz = DATASETS[name]
    bbox = PUBLIC_BBOX[name]
    extra = {}
    if loader == "massive_steps":
        raw = _load_massive_steps(rel, tz)
    elif loader == "tsmc":
        raw = _load_tsmc(rel)
    elif loader == "gowalla":
        raw, gw_stats = _load_gowalla(rel, tz, bbox)
        n_cat = int(raw["category"].notna().sum())
        extra["gowalla"] = {**gw_stats, "share_missing_category": round(1 - n_cat / max(len(raw), 1), 4)}
        raw = raw.dropna(subset=["category"])
    else:
        raise ValueError(f"unknown loader {loader!r}")
    n_raw = len(raw)
    raw = raw.dropna(subset=["lat", "lon"])
    n_with_coords = len(raw)
    out = PROCESSED / name
    pois = pd.read_parquet(out / "pois.parquet")
    mapped, map_stats = map_checkins(raw, pois, load_category_map(CATEGORY_MAP_OF[name]), bbox)
    df = _segment(mapped)
    cols = ["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"]
    meta = {"name": name, "tz": tz, "bbox": bbox, "bbox_source": "OSM administrative boundary (public)",
            "catalog": "FSQ OS Places 2024-11-19 within bbox (public)", "raw_checkins": n_raw,
            "share_missing_coords": round(1 - n_with_coords / n_raw, 4), "mapping": map_stats, "n_venues": int(len(pois)), "n_categories": int(pois["category"].nunique()),
            "trail_gap_hours": TRAIL_GAP_HOURS, "min_len": MIN_TRAIL_LEN, "max_len": MAX_TRAIL_LEN}
    if name not in V2_DATASETS:
        meta.update({"data_version": "v3", "loader": loader, "category_map": CATEGORY_MAP_OF[name],
                     "split_rule": {"massive_steps": "official Massive-STEPS split",
                                    "tsmc": "random 70/10/20 by trail, numpy default_rng(0)",
                                    "gowalla": "per-trail salted SHA-256, 70/10/20"}[loader], **extra})
    for split in SPLITS:
        d = df[df["split"] == split][cols]
        d.to_parquet(out / f"{split}.parquet", index=False)
        lens = d.groupby("trail_id").size()
        meta[split] = {"checkins": int(len(d)), "trails": int(d["trail_id"].nunique()),
                       "users": int(d["user_id"].nunique()),
                       "len_mean": round(float(lens.mean()), 3), "len_max": int(lens.max())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_split(name: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / name / f"{split}.parquet")


def load_pois(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / name / "pois.parquet")


def load_meta(name: str) -> dict:
    return json.loads((PROCESSED / name / "meta.json").read_text())
