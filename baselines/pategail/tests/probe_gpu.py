from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common
import pategail as pg
import synth


def lattice_catalog(n_cells: int, rng: np.random.Generator, per_cell: int = 2) -> tuple:
    side = int(math.ceil(math.sqrt(n_cells)))
    i0, j0 = 3000, 12000
    ij = np.array([(i0 + k // side, j0 + k % side) for k in range(n_cells)])
    lat = np.repeat(ij[:, 0] * 0.01, per_cell) + rng.uniform(-0.004, 0.004, n_cells * per_cell)
    lon = np.repeat(ij[:, 1] * 0.01, per_cell) + rng.uniform(-0.004, 0.004, n_cells * per_cell)
    n = len(lat)
    cats = np.array([f"cat{i:02d}" for i in range(30)])
    pois = pd.DataFrame({"venue_id": [f"v{i}" for i in range(n)], "lat": lat, "lon": lon,
                         "category": cats[rng.integers(0, 30, n)], "root": "r", "name": "n", "closed": False})
    bbox = {"lat_min": float(lat.min()) - 0.001, "lat_max": float(lat.max()) + 0.001,
            "lon_min": float(lon.min()) - 0.001, "lon_max": float(lon.max()) + 0.001}
    return pois, bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=7454)
    ap.add_argument("--trails", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=pg.ROUNDS)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    cuda = a.device == "cuda"
    import torch
    rng = np.random.default_rng(0)
    pois, bbox = lattice_catalog(a.cells, rng)
    synth.BBOX = bbox
    train = synth.make_trails(rng, pois, a.trails, max(1, a.trails // 9))
    tmp = Path(tempfile.mkdtemp(prefix="pategail_probe_"))
    try:
        synth.write_dataset(tmp, "synthprobe", pois, train)
        common.DATA = tmp
        res = {"cells_requested": a.cells, "K": pg.K_FROZEN, "rounds": a.rounds, "runs": []}
        hashes = []
        for rep in range(2):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            out = tmp / f"out{rep}" / "synthprobe_eps1_s0.parquet"
            side = pg.run("synthprobe", 1.0, 0, "main", out, a.device, 4, rounds=a.rounds)
            res["runs"].append({"runtime_s": time.time() - t, "n_cells": side["config"]["n_cells"],
                                "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None,
                                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30 if cuda else None,
                                "eps_spent": side["accounting"]["eps_spent"]})
            hashes.append(hashlib.sha256(out.read_bytes()).hexdigest())
        res["byte_identical_rerun"] = hashes[0] == hashes[1]
        res["torch"] = torch.__version__
        res["gpu"] = torch.cuda.get_device_name(0) if cuda else "cpu"
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
