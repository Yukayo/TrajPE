from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import load_meta, load_pois
from .public_catalog import ROOTS


class Catalog:
    def __init__(self, name: str):
        pois = load_pois(name)
        meta = load_meta(name)
        self.name = name
        self.bbox = meta["bbox"]
        self.tz = meta["tz"]
        self.venue_ids = pois["venue_id"].astype(str).to_numpy()
        self.index = {v: i for i, v in enumerate(self.venue_ids)}
        self.lat = pois["lat"].to_numpy(np.float64)
        self.lon = pois["lon"].to_numpy(np.float64)
        self.names = pois["name"].to_numpy()

        lat0 = 0.5 * (self.bbox["lat_min"] + self.bbox["lat_max"])
        self.kx = 111.320 * np.cos(np.radians(lat0))
        self.ky = 110.574
        self.width_km = (self.bbox["lon_max"] - self.bbox["lon_min"]) * self.kx
        self.height_km = (self.bbox["lat_max"] - self.bbox["lat_min"]) * self.ky
        self.xy = self.to_xy(self.lat, self.lon)

        cats = pois["category"].astype(str).to_numpy()
        self.categories = sorted(set(cats))
        self.cat_index = {c: i for i, c in enumerate(self.categories)}
        self.cat = np.array([self.cat_index[c] for c in cats], dtype=np.int32)
        root_map = dict(zip(pois["category"].astype(str), pois["root"].astype(str)))
        root_of_cat = np.array([ROOTS.index(root_map[c]) if root_map.get(c) in ROOTS else len(ROOTS)
                                for c in self.categories], dtype=np.int32)
        self.n_roots = len(ROOTS) + 1
        self.root_of_cat = root_of_cat
        self.root = root_of_cat[self.cat]
        self.cat_hash = np.array([zlib.crc32(c.encode()) for c in self.categories], dtype=np.int64)

        self.tree = cKDTree(self.xy)
        self._cat_trees: dict[int, tuple[cKDTree, np.ndarray]] = {}
        self._root_trees: dict[int, tuple[cKDTree, np.ndarray]] = {}
        self.cat_freq = np.bincount(self.cat, minlength=len(self.categories)).astype(np.float64)

    def __len__(self):
        return len(self.venue_ids)

    def to_xy(self, lat, lon) -> np.ndarray:
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        return np.stack([(lon - self.bbox["lon_min"]) * self.kx, (lat - self.bbox["lat_min"]) * self.ky], -1)

    def to_latlon(self, xy) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, np.float64)
        return xy[..., 1] / self.ky + self.bbox["lat_min"], xy[..., 0] / self.kx + self.bbox["lon_min"]

    def clip_xy(self, xy) -> np.ndarray:
        return np.stack([np.clip(xy[..., 0], 0, self.width_km), np.clip(xy[..., 1], 0, self.height_km)], -1)

    def _subtree(self, store, codes, code):
        if code not in store:
            idx = np.flatnonzero(codes == code)
            store[code] = (cKDTree(self.xy[idx]) if len(idx) else None, idx)
        return store[code]

    def nearest(self, xy, cat: int | None = None, root: int | None = None, k: int = 1):
        if cat is not None:
            tree, idx = self._subtree(self._cat_trees, self.cat, cat)
        elif root is not None:
            tree, idx = self._subtree(self._root_trees, self.root, root)
        else:
            tree, idx = self.tree, None
        if tree is None:
            return None
        k = min(k, tree.n)
        _, j = tree.query(xy, k=k)
        return j if idx is None else idx[j]

    def dist_km(self, a, b) -> np.ndarray:
        return np.linalg.norm(self.xy[a] - self.xy[b], axis=-1)
