import os as _tp_os
def _tp_root():
    p = _tp_os.path.dirname(_tp_os.path.abspath(__file__))
    while not _tp_os.path.isfile(_tp_os.path.join(p, "trajpe", "__init__.py")) and _tp_os.path.dirname(p) != p:
        p = _tp_os.path.dirname(p)
    return p
_TP_ROOT = _tp_os.environ.get("TRAJPE_ROOT", _tp_root())
_TP_WORK = _tp_os.environ.get("TRAJPE_WORK", _tp_os.path.join(_TP_ROOT, "work"))
_TP_PY = _tp_os.environ.get("TRAJPE_PYTHON", "python")
_TP_PY_DPLORA = _tp_os.environ.get("TRAJPE_PYTHON_DPLORA", _TP_PY)
_TP_PY_AIMSEQ = _tp_os.environ.get("TRAJPE_PYTHON_AIMSEQ", _TP_PY)
_TP_PY_GRID = _tp_os.environ.get("TRAJPE_PYTHON_GRID", _TP_PY)
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HOME", f"{_TP_WORK}/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import pathfmt as PF
import release as RL
import sample_pool as SP

CITY = "jakarta"
_CAT = {}


def catalog():
    if "c" not in _CAT:
        from trajpe.catalog import Catalog
        _CAT["c"] = Catalog(CITY)
    return _CAT["c"]


def synth_texts(n: int, seed: int = 0, lmax: int = 6) -> list[str]:
    c = catalog()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = int(rng.integers(2, lmax + 1))
        v = rng.integers(0, len(c), L)
        mins = np.sort(rng.integers(0, 1440, L))
        out.append(PF.textfmt.render(int(rng.integers(0, 7)), [(int(m) // 60, int(m) % 60) for m in mins],
                                     [c.categories[c.cat[i]] for i in v], c.lat[v], c.lon[v], PF.DECIMALS))
    return out


def test_truncate_rules():
    cats = {"Office", "Café", "Park"}
    ok = "Day: Monday\n09:00 | Office | -6.2000,106.8000\n12:30 | Café | -6.2100,106.8100"
    assert PF.truncate(ok, cats, True)[:2] == (ok, "ok")
    t = ok + "\n13:00 | Nope | -6.2,106.8\n14:00 | Park | -6.2200,106.8200"
    res, why, info = PF.truncate(t, cats, True)
    assert why == "ok" and res == ok and info["end"] == "invalid_category" and info["rows"] == 2
    assert PF.truncate("Day: Monday\n09:00 | Office | -6.2,106.8\n9:00 | Park | -6.2,106.8", cats, True)[1] == "too_short"
    assert PF.truncate("Day: Funday\n09:00 | Office | -6.2,106.8\n10:00 | Park | -6.2,106.8", cats, True)[1] == "bad_day"
    assert PF.truncate("09:00 | Office | -6.2,106.8", cats, True)[1] == "bad_day"
    res, why, info = PF.truncate(ok + "\n13:00 | Pa", cats, False)
    assert why == "ok" and res == ok and info["end"] == "token_limit"
    rows = [f"{h:02d}:00 | Office | -6.2000,106.8000" for h in range(23)]
    res, why, info = PF.truncate("Day: Sunday\n" + "\n".join(rows) + "\n", cats, True)
    assert why == "ok" and res.count("\n") == 20 and info["rows"] == 23 and info["end"] == "eos"
    res, _, _ = PF.truncate("Day: Sunday \n 09:00 |  Office | -6.2000,106.8000 \n10:00 | Park | -6.2,106.8", cats, True)
    assert res == "Day: Sunday\n09:00 | Office | -6.2000,106.8000\n10:00 | Park | -6.2,106.8"
    assert PF.truncate("Day: Monday\n\n09:00 | Office | -6.2,106.8\n10:00 | Park | -6.2,106.8", cats, True)[1] == "too_short"
    assert PF.textfmt.parse(ok, cats, True)[1] == "ok"


def test_sampler_stopping_rule():
    assert SP.n_requests(100_000, 0.9) == int(100_000 / 0.9 * 1.03)
    assert SP.n_requests(100_000, 0.01) == 300_000 and SP.n_requests(10, 0.9) == 256
    assert PF.MAX_ROUNDS == 20 and PF.MIN_VALID == 0.02 and PF.MIN_VALID_AFTER == 20_000
    assert PF.MAX_ROUNDS * 3 * PF.POOL_M * PF.MAX_NEW == 3_840_000_000


def test_vote_sensitivity_sqrt_k():
    rng = np.random.default_rng(1)
    cand = rng.normal(size=(500, 16)).astype(np.float32)
    priv = rng.normal(size=(200, 16)).astype(np.float32)
    h = RL.vote_counts(priv, cand)
    assert h.sum() == 10 * len(priv)
    for i in range(20):
        x = rng.normal(size=(1, 16)).astype(np.float32)
        d = RL.vote_counts(np.vstack([priv, x]), cand) - h
        assert set(np.unique(d)) <= {0.0, 1.0} and d.sum() == 10
        assert abs(np.linalg.norm(d) - math.sqrt(10)) < 1e-12


def test_selection_determinism_ties_and_noise():
    h = np.zeros(20_000)
    h[:300] = 200.0
    a_noisy, a = RL.select(h, 4.365, 1000, 0)
    b_noisy, b = RL.select(h, 4.365, 1000, 0)
    assert np.array_equal(a, b) and np.array_equal(a_noisy, b_noisy)
    _, c = RL.select(h, 4.365, 1000, 1)
    assert not np.array_equal(a, c)
    assert len(np.unique(a)) == 1000 and set(range(300)) <= set(a.tolist())
    z = (a_noisy - h)[300:]
    assert abs(z.std() / (math.sqrt(10) * 4.365) - 1) < 0.03 and abs(z.mean()) < 0.5
    cnt = np.zeros(10)
    for s in range(2000):
        cnt[RL.top_n(np.zeros(10), 3, np.random.default_rng(s))] += 1
    assert np.all(np.abs(cnt / 2000 - 0.3) < 0.05), cnt
    rng = np.random.default_rng([0, PF.STREAM_SELECT])
    noisy = RL.noisy_votes(h, 4.365, rng)
    keep = RL.top_n(noisy, 1000, rng)
    assert np.array_equal(noisy, a_noisy) and set(keep.tolist()) == set(a.tolist())
    assert np.array_equal(np.random.default_rng([0, PF.STREAM_SHUFFLE]).permutation(keep), a)


def test_ground_and_release_schema_deterministic():
    c = catalog()
    texts = synth_texts(400, seed=2, lmax=25)
    texts = [PF.truncate(t, set(c.categories), True)[0] for t in texts]
    Z, g = RL.ground(texts, c, 0)
    Z2, _ = RL.ground(texts, c, 0)
    assert np.array_equal(Z.poi, Z2.poi) and np.array_equal(Z.tmin, Z2.tmin)
    order = np.random.default_rng(0).permutation(len(Z))[:150]
    df = RL.release_frame(Z, order, c)
    assert list(df.columns) == RL.OUT_COLS
    assert str(df["t_local"].dtype) == "datetime64[ns]" and str(df["pos"].dtype) == "int64"
    assert all(str(df[k].dtype) == "object" for k in ("trail_id", "user_id", "venue_id", "category"))
    assert df["trail_id"].nunique() == 150 and (df["trail_id"] == df["user_id"]).all()
    lens = df.groupby("trail_id", sort=False).size()
    assert lens.min() >= 2 and lens.max() <= 20
    assert (df.groupby("trail_id", sort=False)["pos"].transform(lambda s: np.arange(len(s))) == df["pos"]).all()
    pois = pd.read_parquet(PF.DATA / CITY / "pois.parquet").set_index("venue_id")
    pois.index = pois.index.astype(str)
    ref = pois.loc[df["venue_id"].astype(str)]
    assert np.allclose(ref["lat"].to_numpy(), df["lat"]) and (ref["category"].astype(str).to_numpy() == df["category"].astype(str).to_numpy()).all()
    t = df["t_local"].to_numpy("datetime64[m]").astype(np.int64)
    same = df["trail_id"].to_numpy()[1:] == df["trail_id"].to_numpy()[:-1]
    gaps = np.diff(t)[same]
    assert gaps.min() >= 0 and gaps.max() <= 480
    t0 = df.groupby("trail_id", sort=False)["t_local"].first()
    assert ((t0 >= pd.Timestamp("2012-07-02")) & (t0 < pd.Timestamp("2012-07-09"))).all()
    first = [PF.textfmt.parse(texts[i], set(c.categories), True)[0][1][0]["category"] for i in order[:50]]
    assert first == df.groupby("trail_id", sort=False)["category"].first().tolist()[:50]
    assert g["checkins"] == sum(len(t.split("\n")) - 1 for t in texts)


class FakeEmbedder:
    dim = 8
    n_truncated = 0

    def __call__(self, texts):
        import hashlib
        seeds = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) for t in texts]
        return np.stack([np.random.default_rng(s).normal(size=self.dim) for s in seeds]).astype(np.float32)


def test_row_texts_and_table_embeddings_order():
    from trajpe.trails import to_frame
    c = catalog()
    texts = [PF.truncate(t, set(c.categories), True)[0] for t in synth_texts(30, seed=3)]
    Z, _ = RL.ground(texts, c, 0)
    fr = to_frame(Z, c, prefix="cand")
    rows = PF.row_texts(fr)
    assert rows[0].split(" ")[0] in PF.WEEKDAYS and len(rows) == len(fr)
    priv = fr.copy()
    priv["trail_id"] = "u" + priv["trail_id"]
    priv = priv.sample(frac=1.0, random_state=0)
    assert sorted(PF.row_texts(priv)) == sorted(rows)
    emb = FakeEmbedder()
    e, ids = RL.table_embeddings(fr, emb)
    assert list(ids) == [f"cand_{i}" for i in range(len(Z))]
    r = emb(rows)
    for i in (0, 9, 10, 29):
        m = (fr["trail_id"] == f"cand_{i}").to_numpy()
        assert np.allclose(e[i], r[m].mean(0), atol=1e-6)
    e2, ids2 = RL.table_embeddings(priv, emb)
    lookup = {k: v for k, v in zip(ids2, e2)}
    assert all(np.allclose(lookup["u" + k], v, atol=1e-6) for k, v in zip(ids, e))


def test_real_embedder_batch_independent():
    emb = RL.RowEmbedder("cpu")
    rows = ["Monday 09:39 | Office | -6.1500,106.8880", "Tuesday 13:05 | Asian Restaurant | -6.1830,106.8240",
            "Monday 09:39 | Office | -6.1500,106.8880", "Sunday 23:59 | Park | -6.2000,106.8000"]
    a = emb(rows)
    b = np.vstack([emb([r]) for r in rows])
    assert a.shape == (4, 1024) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b, atol=2e-3) and np.array_equal(a[0], a[2])
    assert emb.n_truncated == 0


def test_accounting_block_and_private_keys():
    cal = {"eps_total": 4.0, "delta": 1e-6, "pld_target": 3.985, "eps_composed_pld": 3.98, "sample_rate": PF.sample_rate(),
           "steps": PF.n_steps(), "sigma_sgd": 0.6059, "eps_sgd_only_pld": 3.88, "eps_select": 1.0,
           "delta_select": 5e-7, "sigma_sel": 4.365, "eps_sel_only_pld": 0.965, "paper_literal": {"x": 1}}
    tj = {"eps_prv": 3.99, "eps_rdp": 4.8, "eps_sgd_part_prv": 3.9, "eps_sgd_part_rdp": 4.7}
    b = RL.accounting_block(cal, tj)
    for k in ("eps_total", "delta", "unit", "relation", "composition", "pld_target", "eps_joint_pld", "eps_joint_prv",
              "eps_joint_rdp", "dp_sgd", "selection", "paper_literal"):
        assert k in b, k
    assert b["selection"]["noise_std"] == math.sqrt(10) * 4.365 and b["selection"]["sigma_unit"] == 4.365
    assert not (RL.PRIVATE_KEYS & RL.all_keys({"accounting": b}))
    assert RL.all_keys({"a": [{"private_stats": 1}]}) >= {"private_stats"}


def test_release_main_end_to_end_synthetic():
    import json
    import tempfile
    from trajpe.trails import to_frame
    c = catalog()
    old = (PF.POOL_M, PF.N_OUT, PF.DATA, RL.RowEmbedder)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        try:
            PF.POOL_M, PF.N_OUT = 200, 50
            RL.RowEmbedder = lambda dev: FakeEmbedder()
            (td / "data" / CITY).mkdir(parents=True)
            Zp, _ = RL.ground([PF.truncate(t, set(c.categories), True)[0] for t in synth_texts(120, seed=7)], c, 0)
            to_frame(Zp, c, prefix="real").to_parquet(td / "data" / CITY / "train.parquet", index=False)
            PF.DATA = td / "data"
            work = td / "work"
            (work / "adapter").mkdir(parents=True)
            (work / "adapter" / "adapter_model.safetensors").write_bytes(b"\0" * 64)
            cal = {"method": "path", "eps_total": 4.0, "delta": 1e-6, "pld_target": 3.985, "eps_composed_pld": 3.9845,
                   "sample_rate": PF.sample_rate(), "steps": PF.n_steps(), "sigma_sgd": 0.6059, "eps_sgd_only_pld": 3.88,
                   "eps_select": 1.0, "delta_select": 5e-7, "sigma_sel": 4.3652, "sigma_hist": 4.3652, "n_hist": 1,
                   "eps_sel_only_pld": 0.965, "paper_literal": {"eps_basic_reading_of_release": 5.12}}
            (work / "calib.json").write_text(json.dumps(cal))
            mk = PF.MODELS["1.5b"]
            tj = {"dataset": CITY, "model": mk["id"], "model_revision": mk["revision"], "seed": 0,
                  "noise_multiplier": 0.6059, "sample_rate": PF.sample_rate(), "steps": PF.n_steps(),
                  "steps_run": PF.n_steps(), "complete": True, "eps_prv": 3.9957, "eps_rdp": 4.88,
                  "private_stats": {"oom_retries": 0, "n_trails": 120, "train_loop_sec": 1.0}, "log": [{"loss": 1}]}
            (work / "train.json").write_text(json.dumps(tj))
            texts = [PF.truncate(t, set(c.categories), True)[0] for t in synth_texts(200, seed=8)]
            pd.DataFrame({"text": texts, "n_rows": [t.count("\n") for t in texts]}).to_parquet(work / "pool.parquet")
            (work / "pool.json").write_text(json.dumps({"n_kept": 200, "status": "ok", "model": mk["id"],
                                                        "model_revision": mk["revision"], "m": 200,
                                                        "sample_gen_sec": 1.0, "n_generated": 210}))
            out = td / "out" / "jakarta_eps4_s0.parquet"
            argv = sys.argv
            sys.argv = ["release.py", "--dataset", CITY, "--eps", "4", "--work", str(work), "--out", str(out)]
            try:
                RL.main()
            finally:
                sys.argv = argv
            df = pd.read_parquet(out)
            side = json.loads(out.with_suffix(".json").read_text())
            priv = json.loads((work / "release_private.json").read_text())
        finally:
            PF.POOL_M, PF.N_OUT, PF.DATA, RL.RowEmbedder = old
    assert list(df.columns) == RL.OUT_COLS and df["trail_id"].nunique() == 50 and side["n_trails"] == 50
    assert df["trail_id"].iloc[0].startswith("path_") and side["eps"] == 4.0 and side["method"] == "path"
    assert not (RL.PRIVATE_KEYS & RL.all_keys(side)), RL.PRIVATE_KEYS & RL.all_keys(side)
    assert side["accounting"]["selection"]["sigma_unit"] == 4.3652 and side["selection"]["n_distinct_released"] == 50
    assert set(side["timings_public_sec"]) == {"sample_load", "sample_gen", "ground", "embed_candidates"}
    assert priv["n_private_trails"] == 120 and "embed_private_sec" in priv and "knn_sec" in priv
    assert side["training"]["complete"] is True and "private_stats" not in side["training"]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name, flush=True)
    print("all release tests passed")
