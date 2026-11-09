from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from .paths import PROCESSED, RAW

FSQ_PATH = RAW / "fsq_os_places" / "foursquare_places.parquet"

PUBLIC_BBOX = {
    "nyc": dict(lat_min=40.4766, lat_max=40.9176, lon_min=-74.2588, lon_max=-73.7002),
    "istanbul": dict(lat_min=40.7377, lat_max=41.6710, lon_min=27.9708, lon_max=29.9588),
    "tokyo": dict(lat_min=35.4817, lat_max=35.8984, lon_min=138.9429, lon_max=139.9189),
    "jakarta": dict(lat_min=-6.3745, lat_max=-6.0870, lon_min=106.6856, lon_max=106.9740),
    "sao_paulo": dict(lat_min=-24.0079, lat_max=-23.3578, lon_min=-46.8263, lon_max=-46.3651),
    "kuwait_city": dict(lat_min=28.5244, lat_max=30.1038, lon_min=46.5527, lon_max=49.0047),
    "tsmc_nyc": dict(lat_min=40.4766, lat_max=40.9176, lon_min=-74.2588, lon_max=-73.7002),
    "petaling_jaya": dict(lat_min=3.0697, lat_max=3.2088, lon_min=101.5515, lon_max=101.6637),
    "moscow": dict(lat_min=55.5167, lat_max=55.9578, lon_min=37.3262, lon_max=37.9674),
    "bandung": dict(lat_min=-6.9697, lat_max=-6.8369, lon_min=107.5454, lon_max=107.7395),
    "sydney": dict(lat_min=-34.2598, lat_max=-33.3642, lon_min=150.2608, lon_max=151.3432),
    "melbourne": dict(lat_min=-38.4994, lat_max=-37.4017, lon_min=144.4441, lon_max=146.1925),
    "tangerang": dict(lat_min=-6.2576, lat_max=-6.0973, lon_min=106.5515, lon_max=106.7487),
    "palembang": dict(lat_min=-3.1162, lat_max=-2.8546, lon_min=104.6137, lon_max=104.8606),
    "shanghai": dict(lat_min=30.6694, lat_max=31.8720, lon_min=120.8509, lon_max=123.2257),
    "beijing": dict(lat_min=39.4416, lat_max=41.0592, lon_min=115.4168, lon_max=117.5081),
    "ms_tokyo": dict(lat_min=35.4817, lat_max=35.8984, lon_min=138.9429, lon_max=139.9189),
    "gw_stockholm": dict(lat_min=59.2273, lat_max=59.4403, lon_min=17.7607, lon_max=18.2001),
    "gw_san_francisco": dict(lat_min=37.7082, lat_max=37.9297, lon_min=-122.6162, lon_max=-122.2815),
    "gw_austin": dict(lat_min=30.0985, lat_max=30.5166, lon_min=-97.9368, lon_max=-97.5605),
    "gw_nyc": dict(lat_min=40.4766, lat_max=40.9176, lon_min=-74.2588, lon_max=-73.7002),
}
ROOTS = ["Arts and Entertainment", "Business and Professional Services", "Community and Government",
         "Dining and Drinking", "Event", "Health and Medicine", "Landmarks and Outdoors", "Retail",
         "Sports and Recreation", "Travel and Transportation"]
CATEGORY_MAP_PATH = PROCESSED / "category_map_v2_to_fsq.json"
CATEGORY_MAP_V3_PATH = PROCESSED / "category_map_v3.json"
HOME_NAMES = {"自宅", "じたく", "home", "my home", "my house", "home sweet home", "rumah", "rumahku", "evim", "ev"}
MULTI_TARGET = {"Home (private)": ["Residential Building", "Apartment or Condo"],
                "Residence": ["Residential Building", "Apartment or Condo"],
                "Residential Building (Apartment / Condo)": ["Apartment or Condo", "Residential Building"],
                "gowalla:21": ["Residential Building", "Apartment or Condo"],
                "gowalla:20": ["Apartment or Condo", "Residential Building"],
                "gowalla:29": ["Apartment or Condo", "Residential Building"],
                "gowalla:87": ["Residential Building", "Apartment or Condo"]}


def build_public_catalogs(names) -> dict:
    pf = pq.ParquetFile(FSQ_PATH)
    cols = ["fsq_place_id", "name", "latitude", "longitude", "date_closed", "fsq_category_labels"]
    parts = {n: [] for n in names}
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=cols)
        lat = t["latitude"].to_numpy(zero_copy_only=False)
        lon = t["longitude"].to_numpy(zero_copy_only=False)
        for n in names:
            b = PUBLIC_BBOX[n]
            m = (lat >= b["lat_min"]) & (lat <= b["lat_max"]) & (lon >= b["lon_min"]) & (lon <= b["lon_max"])
            if m.any():
                parts[n].append(t.filter(pa.array(m)))
    stats = {}
    for n in names:
        df = pa.concat_tables(parts[n]).to_pandas()
        label = df["fsq_category_labels"].map(lambda x: x[0] if x is not None and len(x) else None)
        df = df[label.notna()].assign(label=label[label.notna()])
        out = pd.DataFrame({
            "venue_id": df["fsq_place_id"].astype(str).to_numpy(),
            "lat": df["latitude"].astype(float).to_numpy(),
            "lon": df["longitude"].astype(float).to_numpy(),
            "category": df["label"].map(lambda s: s.split(" > ")[-1]).to_numpy(),
            "root": df["label"].map(lambda s: s.split(" > ")[0]).to_numpy(),
            "name": df["name"].to_numpy(),
            "closed": df["date_closed"].notna().to_numpy(),
        })
        nm = out["name"].fillna("").astype(str).str.strip().str.lower()
        is_home = nm.isin(HOME_NAMES) | nm.str.startswith("自宅")
        out = out[out["root"].isin(ROOTS) & ~is_home].drop_duplicates("venue_id").reset_index(drop=True)
        (PROCESSED / n).mkdir(parents=True, exist_ok=True)
        out.to_parquet(PROCESSED / n / "pois.parquet", index=False)
        stats[n] = {"venues": int(len(out)), "categories": int(out["category"].nunique()),
                    "closed_share": round(float(out["closed"].mean()), 3)}
    return stats


def load_category_map(version: str = "v2") -> dict:
    if version == "v2":
        return json.loads(CATEGORY_MAP_PATH.read_text()) if CATEGORY_MAP_PATH.exists() else {}
    section = {"v3_fsq": "fsq", "v3_gowalla": "gowalla"}[version]
    return json.loads(CATEGORY_MAP_V3_PATH.read_text())[section]


def catalog_coverage(pois: pd.DataFrame, bbox: dict, cell_km: float = 1.0) -> dict:
    lat0 = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
    kx, ky = 111.320 * np.cos(np.radians(lat0)), 110.574
    w, h = (bbox["lon_max"] - bbox["lon_min"]) * kx, (bbox["lat_max"] - bbox["lat_min"]) * ky
    nx, ny = max(int(np.ceil(w / cell_km)), 1), max(int(np.ceil(h / cell_km)), 1)
    ix = np.clip(((pois["lon"].to_numpy() - bbox["lon_min"]) * kx / cell_km).astype(int), 0, nx - 1)
    iy = np.clip(((pois["lat"].to_numpy() - bbox["lat_min"]) * ky / cell_km).astype(int), 0, ny - 1)
    occupied = len(np.unique(iy * nx + ix))
    return {"area_km2": round(w * h, 1), "venues_per_km2": round(len(pois) / (w * h), 2),
            "cells": nx * ny, "share_cells_with_venue": round(occupied / (nx * ny), 4)}


def map_checkins(df: pd.DataFrame, pois: pd.DataFrame, cat_map: dict, bbox: dict,
                 r_same_km: float = 0.3, r_cat_far_km: float = 1.0, r_any_km: float = 0.5) -> tuple[pd.DataFrame, dict]:
    lat0 = 0.5 * (bbox["lat_min"] + bbox["lat_max"])
    kx, ky = 111.320 * np.cos(np.radians(lat0)), 110.574

    def xy(lat, lon):
        return np.stack([(np.asarray(lon) - bbox["lon_min"]) * kx, (np.asarray(lat) - bbox["lat_min"]) * ky], 1)

    inside = (df["lat"].between(bbox["lat_min"], bbox["lat_max"]) & df["lon"].between(bbox["lon_min"], bbox["lon_max"]))
    df = df[inside].reset_index(drop=True)
    pxy = xy(pois["lat"], pois["lon"])
    q = xy(df["lat"], df["lon"])
    targets = {c: MULTI_TARGET.get(c, [cat_map.get(c, c)]) for c in pd.unique(df["category"])}
    pub_cat = df["category"].map(lambda c: "|".join(targets[c])).to_numpy()
    root_of = dict(zip(pois["category"], pois["root"]))
    pub_root = np.array([root_of.get(k.split("|")[0]) for k in pub_cat], dtype=object)
    choice = np.full(len(df), -1, dtype=np.int64)
    how = np.full(len(df), "drop", dtype=object)

    pois_col = {"category": pois["category"].to_numpy(), "root": pois["root"].to_numpy()}

    def assign(key_col, keys_needed, radius, label):
        for key in pd.unique(keys_needed):
            rows = np.flatnonzero((choice < 0) & (keys_needed == key))
            if len(rows) == 0 or key is None:
                continue
            idx = np.flatnonzero(np.isin(pois_col[key_col], key.split("|")))
            if len(idx) == 0:
                continue
            d, j = cKDTree(pxy[idx]).query(q[rows], distance_upper_bound=radius)
            ok = np.isfinite(d)
            choice[rows[ok]] = idx[j[ok]]
            how[rows[ok]] = label

    assign("category", pub_cat, r_same_km, "category")
    assign("category", pub_cat, r_cat_far_km, "category_1km")
    assign("root", pub_root, r_same_km, "root")
    rest = np.flatnonzero(choice < 0)
    if len(rest):
        d, j = cKDTree(pxy).query(q[rest], distance_upper_bound=r_any_km)
        ok = np.isfinite(d)
        choice[rest[ok]] = j[ok]
        how[rest[ok]] = "any"
    keep = choice >= 0
    dist = np.linalg.norm(q - pxy[np.maximum(choice, 0)], axis=1)
    stats = {"checkins_in_area": int(len(df)), "share_inside_area": round(float(inside.mean()), 4),
             **{f"share_{k}": round(float(np.mean(how == k)), 4) for k in ["category", "category_1km", "root", "any", "drop"]},
             "share_exact_10m": round(float(np.mean(keep & (dist < 0.01))), 4),
             "median_dist_m": round(float(np.median(dist[keep]) * 1000), 1)}
    sel = choice[keep]
    out = df[keep].copy()
    out["venue_id"] = pois["venue_id"].to_numpy()[sel]
    out["lat"] = pois["lat"].to_numpy()[sel]
    out["lon"] = pois["lon"].to_numpy()[sel]
    out["category"] = pois["category"].to_numpy()[sel]
    return out.reset_index(drop=True), stats
