from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TESTS = Path(__file__).resolve().parent
CTCL = TESTS.parent
PROJECT = CTCL.parents[1]
sys.path.insert(0, str(CTCL))
import ctcl_conf as C

REAL_DATA = PROJECT / "data" / "processed"
SCRATCH = Path(os.environ.get("CTCL_TEST_TMP", tempfile.gettempdir())) / "ctcl_tests"


def public_catalog(d: str) -> pd.DataFrame:
    assert d not in C.FORBIDDEN
    p = pd.read_parquet(REAL_DATA / d / "pois.parquet", columns=["venue_id", "lat", "lon", "category", "root"])
    for c in ("venue_id", "category", "root"):
        p[c] = p[c].astype(str)
    return p


def synthetic_trails(pois: pd.DataFrame, n_trails: int, seed: int = 0, max_len: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lat, lon = pois["lat"].to_numpy(), pois["lon"].to_numpy()
    order = np.argsort(lat * 1000 + lon)
    rows = []
    base = pd.Timestamp("2012-04-02")
    for t in range(n_trails):
        L = int(min(max_len, 2 + rng.geometric(0.55) - 1))
        pos = int(rng.integers(len(pois)))
        t0 = base + pd.Timedelta(days=int(rng.integers(0, 300)), minutes=int(rng.integers(0, 24 * 60)))
        cur = t0
        for i in range(L):
            if i:
                pos = int(np.clip(pos + rng.integers(-2000, 2000), 0, len(pois) - 1))
                cur = cur + pd.Timedelta(minutes=int(rng.integers(5, 240)))
            v = order[pos]
            rows.append((f"t{t:07d}", f"u{t % 997:05d}", pois["venue_id"].iat[v], lat[v], lon[v],
                         pois["category"].iat[v], cur, i))
    df = pd.DataFrame(rows, columns=["trail_id", "user_id", "venue_id", "lat", "lon", "category", "t_local", "pos"])
    df["t_local"] = df["t_local"].astype("datetime64[ns]")
    df["pos"] = df["pos"].astype(np.int64)
    return df


def fake_dataset(root: Path, d: str, n_trails: int, seed: int = 0) -> Path:
    dd = Path(root) / d
    dd.mkdir(parents=True, exist_ok=True)
    for f in ("pois.parquet", "meta.json"):
        shutil.copy2(REAL_DATA / d / f, dd / f)
    synthetic_trails(public_catalog(d), n_trails, seed).to_parquet(dd / "train.parquet", index=False)
    return dd


def render_trail(rng, pois: pd.DataFrame, L: int) -> str:
    idx = rng.integers(0, len(pois), L)
    hhmm = list(zip(rng.integers(0, 24, L).tolist(), rng.integers(0, 60, L).tolist()))
    return C.textfmt.render(int(rng.integers(0, 7)), hhmm, pois["category"].to_numpy()[idx].tolist(),
                            pois["lat"].to_numpy()[idx], pois["lon"].to_numpy()[idx], decimals=C.COORD_DECIMALS)


class FakeGenerator:

    def __init__(self, rates, pois: pd.DataFrame, seed: int = 0):
        self.rates = rates
        self.pois = pois.iloc[:5000]
        self.rng = np.random.default_rng([seed, 99])
        self.calls = 0

    def __call__(self, reqs):
        out = []
        self.calls += 1
        for k, c in reqs:
            r = self.rates(k) if callable(self.rates) else self.rates[k]
            for _ in range(c):
                if self.rng.random() < r:
                    out.append((k, render_trail(self.rng, self.pois, int(self.rng.integers(2, 6))), True))
                else:
                    out.append((k, "Day: Someday\n99:99 | ??? | x,y", True))
        return out


def assert_close(a, b, tol, what=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{what}: {a} vs {b} (tol {tol})")
