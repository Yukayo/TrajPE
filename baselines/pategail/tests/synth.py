from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BBOX = {"lat_min": 35.60, "lat_max": 35.74, "lon_min": 139.62, "lon_max": 139.80}


def make_catalog(rng: np.random.Generator, n_venues: int = 2500, n_cats: int = 12) -> pd.DataFrame:
    lat = rng.uniform(BBOX["lat_min"], BBOX["lat_max"], n_venues)
    lon = rng.uniform(BBOX["lon_min"], BBOX["lon_max"], n_venues)
    cats = np.array([f"cat{i:02d}" for i in range(n_cats)])
    return pd.DataFrame({"venue_id": [f"v{i}" for i in range(n_venues)], "lat": lat, "lon": lon,
                         "category": cats[rng.integers(0, n_cats, n_venues)], "root": "r", "name": "n",
                         "closed": False})


def make_trails(rng: np.random.Generator, pois: pd.DataFrame, n_trails: int, n_users: int,
                id_prefix: str = "t") -> pd.DataFrame:
    lat, lon = pois["lat"].to_numpy(), pois["lon"].to_numpy()
    rows = []
    for i in range(n_trails):
        L = int(min(20, 2 + rng.geometric(0.45) - 1))
        v = int(rng.integers(len(pois)))
        home = v
        t = pd.Timestamp("2013-03-04") + pd.Timedelta(minutes=int(rng.integers(0, 7 * 1440)))
        u = f"u{int(rng.integers(n_users))}"
        for p in range(L):
            rows.append((f"{id_prefix}{i:05d}", u, pois["venue_id"].iat[v], lat[v], lon[v],
                         pois["category"].iat[v], t, p))
            r = rng.random()
            if r < 0.25:
                pass
            elif r < 0.4:
                v = home
            else:
                d = (lat - lat[v]) ** 2 + (lon - lon[v]) ** 2
                v = int(np.argsort(d)[int(rng.integers(1, 60))])
            t = t + pd.Timedelta(minutes=int(rng.integers(5, 300)))
    df = pd.DataFrame(rows, columns=["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"])
    df["t_local"] = df["t_local"].astype("datetime64[us]")
    return df


def write_dataset(root: Path, name: str, pois: pd.DataFrame, train: pd.DataFrame) -> Path:
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    pois.to_parquet(d / "pois.parquet", index=False)
    train.to_parquet(d / "train.parquet", index=False)
    (d / "meta.json").write_text(json.dumps({"name": name, "bbox": BBOX}))
    return d


def make_dataset(root: Path, name: str, seed: int = 0, n_trails: int = 300, n_users: int = 60,
                 n_venues: int = 2500) -> tuple:
    rng = np.random.default_rng(seed)
    pois = make_catalog(rng, n_venues)
    train = make_trails(rng, pois, n_trails, n_users)
    write_dataset(root, name, pois, train)
    return pois, train
