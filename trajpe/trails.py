from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEK_MIN = 7 * 24 * 60
REF_MONDAY = pd.Timestamp("2012-07-02")


@dataclass
class TrailSet:
    poi: np.ndarray
    tmin: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def trail_of(self) -> np.ndarray:
        return np.repeat(np.arange(len(self)), self.lengths)

    @property
    def first(self) -> np.ndarray:
        return self.offsets[:-1]

    @property
    def last(self) -> np.ndarray:
        return self.offsets[1:] - 1

    @property
    def pos(self) -> np.ndarray:
        return np.arange(len(self.poi)) - np.repeat(self.offsets[:-1], self.lengths)

    def trail(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.offsets[i], self.offsets[i + 1]
        return self.poi[s:e], self.tmin[s:e]

    def to_lists(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.poi[s:e].copy(), self.tmin[s:e].copy())
                for s, e in zip(self.offsets[:-1], self.offsets[1:])]

    @classmethod
    def from_lists(cls, trails) -> "TrailSet":
        if not trails:
            return cls(np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(1, np.int64))
        lens = np.fromiter((len(p) for p, _ in trails), dtype=np.int64, count=len(trails))
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        poi = np.concatenate([np.asarray(p) for p, _ in trails]).astype(np.int32)
        tmin = np.concatenate([np.asarray(t) for _, t in trails]).astype(np.int32)
        return cls(poi, tmin, offsets)

    def subset(self, idx) -> "TrailSet":
        idx = np.asarray(idx, dtype=np.int64)
        lens = self.lengths[idx]
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        starts = self.offsets[idx]
        gather = np.repeat(starts - offsets[:-1], lens) + np.arange(offsets[-1])
        return TrailSet(self.poi[gather], self.tmin[gather], offsets)

    def save(self, path) -> None:
        np.savez_compressed(path, poi=self.poi, tmin=self.tmin, offsets=self.offsets)

    @classmethod
    def load(cls, path) -> "TrailSet":
        z = np.load(path)
        return cls(z["poi"], z["tmin"], z["offsets"])


def from_frame(df: pd.DataFrame, catalog) -> tuple[TrailSet, np.ndarray]:
    df = df.copy()
    df["poi"] = df["venue_id"].astype(str).map(catalog.index)
    df = df.dropna(subset=["poi"])
    df = df.sort_values(["trail_id", "t_local"], kind="stable")
    first_t = df.groupby("trail_id", sort=False)["t_local"].transform("first")
    week_start = (first_t - pd.to_timedelta(first_t.dt.dayofweek, unit="D")).dt.normalize()
    df["tmin"] = ((df["t_local"] - week_start).dt.total_seconds() // 60).astype(np.int64)
    n = df.groupby("trail_id", sort=False)["poi"].transform("size")
    df = df[n >= 2]
    codes, _ = pd.factorize(df["trail_id"])
    lens = np.bincount(codes)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    users = (df.groupby("trail_id", sort=False)["user_id"].first().to_numpy()
             if "user_id" in df.columns else np.arange(len(lens)).astype(str))
    return TrailSet(df["poi"].to_numpy(np.int32), df["tmin"].to_numpy(np.int32), offsets), users


def to_frame(ts: TrailSet, catalog, prefix: str = "syn") -> pd.DataFrame:
    tr = ts.trail_of
    return pd.DataFrame({
        "trail_id": [f"{prefix}_{i}" for i in tr],
        "user_id": [f"{prefix}_{i}" for i in tr],
        "venue_id": catalog.venue_ids[ts.poi],
        "lat": catalog.lat[ts.poi],
        "lon": catalog.lon[ts.poi],
        "category": np.asarray(catalog.categories)[catalog.cat[ts.poi]],
        "t_local": REF_MONDAY + pd.to_timedelta(ts.tmin.astype(np.int64), unit="m"),
        "pos": ts.pos,
    })
