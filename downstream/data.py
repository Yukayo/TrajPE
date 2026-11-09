from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_LEN = 20
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320


@dataclass
class Catalog:
    venue_ids: np.ndarray
    cat_idx: np.ndarray
    categories: np.ndarray
    xy: np.ndarray
    lat0: float
    lon0: float
    index: pd.Index

    @property
    def n_venues(self) -> int:
        return len(self.venue_ids)

    @property
    def n_cats(self) -> int:
        return len(self.categories)

    def project(self, lat, lon) -> np.ndarray:
        lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
        x = (lon - self.lon0) * KM_PER_DEG_LON * np.cos(np.radians(self.lat0))
        y = (lat - self.lat0) * KM_PER_DEG_LAT
        return np.stack([x, y], axis=-1).astype(np.float32)


def load_catalog(dataset_dir: Path) -> Catalog:
    pois = pd.read_parquet(Path(dataset_dir) / "pois.parquet", columns=["venue_id", "lat", "lon", "category"])
    pois = pois.assign(venue_id=pois["venue_id"].astype(str)).drop_duplicates("venue_id")
    pois = pois.sort_values("venue_id").reset_index(drop=True)
    categories, cat_idx = np.unique(pois["category"].astype(str).to_numpy(), return_inverse=True)
    cat = Catalog(venue_ids=pois["venue_id"].to_numpy(), cat_idx=cat_idx.astype(np.int64),
                  categories=categories, xy=np.zeros((len(pois), 2), np.float32),
                  lat0=float(pois["lat"].median()), lon0=float(pois["lon"].median()),
                  index=pd.Index(pois["venue_id"]))
    cat.xy = cat.project(pois["lat"], pois["lon"])
    return cat


def data_version(dataset_dir: Path) -> str:
    return hashlib.md5((Path(dataset_dir) / "meta.json").read_bytes()).hexdigest()[:10]


def vocabulary(trails: "Trails") -> np.ndarray:
    return np.unique(trails.checkin_venues())


@dataclass
class Trails:
    venue: np.ndarray
    hour: np.ndarray
    dow: np.ndarray
    length: np.ndarray

    def __len__(self) -> int:
        return len(self.length)

    @property
    def n_targets(self) -> int:
        return int((self.length - 1).sum())

    def subset(self, idx) -> "Trails":
        return Trails(self.venue[idx], self.hour[idx], self.dow[idx], self.length[idx])

    def split_holdout(self, frac: float, seed: int, grouped: bool = True) -> tuple["Trails", "Trails"]:
        rng = np.random.default_rng(seed)
        n_hold = max(1, int(round(frac * len(self))))
        if not grouped:
            perm = rng.permutation(len(self))
            return self.subset(np.sort(perm[n_hold:])), self.subset(np.sort(perm[:n_hold]))
        _, group = np.unique(self.venue, axis=0, return_inverse=True)
        group = group.ravel()
        order = rng.permutation(group.max() + 1)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        by_rank = np.argsort(rank[group], kind="stable")
        hold = np.zeros(len(self), bool)
        sizes = np.bincount(group)
        cum = np.cumsum(sizes[order])
        n_groups = int(np.searchsorted(cum, n_hold)) + 1
        hold[by_rank[:int(cum[n_groups - 1])]] = True
        return self.subset(np.flatnonzero(~hold)), self.subset(np.flatnonzero(hold))

    def transitions(self) -> dict[str, np.ndarray]:
        k = np.arange(1, self.venue.shape[1])
        m = k[None, :] < self.length[:, None]
        return {"src": self.venue[:, :-1][m], "dst": self.venue[:, 1:][m],
                "hour": self.hour[:, :-1][m], "dow": self.dow[:, :-1][m],
                "k": np.broadcast_to(k, m.shape)[m]}

    def checkin_venues(self) -> np.ndarray:
        return self.venue[self.venue >= 0]


def _venue_keys(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all():
        s = s.astype("Int64")
    return s.astype(str)


def load_trails(path: Path, catalog: Catalog, max_len: int = MAX_LEN) -> tuple[Trails, dict]:
    df = pd.read_parquet(path)
    missing = {"trail_id", "venue_id", "t_local"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    t = pd.to_datetime(df["t_local"], errors="coerce")
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)
    d = pd.DataFrame({"trail": df["trail_id"].astype(str).to_numpy(),
                      "order": (df["pos"] if "pos" in df.columns else t).to_numpy(),
                      "t": t.to_numpy(), "venue_id": _venue_keys(df["venue_id"]).to_numpy()})
    has_xy = {"lat", "lon"} <= set(df.columns)
    if has_xy:
        d["lat"], d["lon"] = df["lat"].to_numpy(np.float64), df["lon"].to_numpy(np.float64)
    d = d.sort_values(["trail", "order"], kind="stable").reset_index(drop=True)

    vidx = catalog.index.get_indexer(d["venue_id"]).astype(np.int64)
    unknown = vidx < 0
    n_snapped = 0
    if unknown.any() and has_xy:
        from scipy.spatial import cKDTree
        ok = unknown & np.isfinite(d["lat"].to_numpy()) & np.isfinite(d["lon"].to_numpy())
        if ok.any():
            _, nearest = cKDTree(catalog.xy).query(catalog.project(d.loc[ok, "lat"], d.loc[ok, "lon"]))
            vidx[ok] = nearest
            n_snapped = int(ok.sum())
    invalid = (vidx < 0) | d["t"].isna().to_numpy()

    seg = pd.Series(invalid.astype(np.int64)).groupby(d["trail"]).cumsum().to_numpy()
    keep = ~invalid
    trail_key = d.loc[keep, ["trail"]].assign(seg=seg[keep])
    gid = trail_key.groupby(["trail", "seg"], sort=False).ngroup().to_numpy()
    pos = pd.Series(gid).groupby(gid).cumcount().to_numpy()
    seg_len = np.bincount(gid)
    t_keep = pd.DatetimeIndex(d.loc[keep, "t"])
    hour, dow, vidx = t_keep.hour.to_numpy(np.int64), t_keep.dayofweek.to_numpy(np.int64), vidx[keep]

    good = seg_len >= 2
    new_id = np.cumsum(good) - 1
    sel = good[gid] & (pos < max_len)
    rows, cols = new_id[gid[sel]], pos[sel]
    n = int(good.sum())
    if n == 0:
        raise ValueError(f"{path}: no trail with >= 2 valid check-ins")
    venue = np.full((n, max_len), -1, np.int64)
    hr, dw = np.zeros((n, max_len), np.int64), np.zeros((n, max_len), np.int64)
    venue[rows, cols], hr[rows, cols], dw[rows, cols] = vidx[sel], hour[sel], dow[sel]
    trails = Trails(venue, hr, dw, np.minimum(seg_len[good], max_len))

    stats = {"rows": int(len(d)), "trail_ids": int(d["trail"].nunique()),
             "unknown_venue_rows": int(unknown.sum()), "snapped_rows": n_snapped,
             "dropped_rows": int(invalid.sum()), "dropped_short_segments": int((~good).sum()),
             "truncated_trails": int((seg_len[good] > max_len).sum()),
             "trails": n, "checkins": int(trails.length.sum()), "targets": trails.n_targets,
             "venues_covered": int(np.unique(trails.checkin_venues()).size)}
    return trails, stats
