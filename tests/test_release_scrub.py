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
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "dplora"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hybrid as HY
from test_v4 import _setup

CAL = dict(method="hybrid", eps_total=1.0, delta=1e-6, sigma_sgd=4.0, sample_rate=.01, steps=1, mu_pe=.1, mu_pe_run=.1,
           sgd_share=.3)
VS_TRAIN = {"density_jsd": 0.123456789, "repeat_rate_err": 0.25}
VS_TEST = {"density_jsd": 0.987654321}


def _old_run(tmp: Path):
    runs, results, work = _setup(tmp, dict(CAL))
    for tag in ("run", "rundiv"):
        p = runs / "d" / tag / "result.json"
        r = json.loads(p.read_text())
        r.update(vs_train=dict(VS_TRAIN), vs_test=dict(VS_TEST))
        p.write_text(json.dumps(r))
        (results / "pe" / "d" / f"{tag}.json").write_text(json.dumps(r))
    return runs, results, work


def _private(paths):
    found = {}
    for p in paths:
        txt = p.read_text()
        found["/".join(p.parts[-2:])] = [k for k in HY.PRIVATE_KEYS if f'"{k}"' in txt]
    return {k: v for k, v in found.items() if v}


def _files(runs, results):
    return [runs / "d" / "run" / "result.json", runs / "d" / "rundiv" / "result.json",
            results / "pe" / "d" / "run.json", results / "pe" / "d" / "rundiv.json"]


def test_scrub_private_helpers():
    clean, n = HY.scrub_private({"a": 1, "vs_train": {}, "x": [{"vs_test": 1, "b": 2}], "pe": {"vs_train": 3}})
    assert (clean, n) == ({"a": 1, "x": [{"b": 2}], "pe": {}}, 3)
    rec, n = HY.scrub_record({"vs_test": 1, "metrics": "x"})
    assert n == 1 and rec == {"metrics": HY.PRIVATE_NOTE}
    assert HY.scrub_record({"a": 1}) == ({"a": 1}, 0)


def test_sidecar_scrub_removes_everywhere():
    runs, results, _ = _old_run(Path(tempfile.mkdtemp(prefix="scrub_")))
    b = HY.sidecar("d", "rundiv", runs, results, scrub=True)
    assert _private(_files(runs, results)) == {}
    res = json.loads((runs / "d" / "rundiv" / "result.json").read_text())
    assert res["hybrid"] == json.loads(json.dumps(b)) and res["metrics"] == HY.PRIVATE_NOTE
    assert json.loads((results / "pe" / "d" / "rundiv.json").read_text()) == res
    src = json.loads((runs / "d" / "run" / "result.json").read_text())
    assert "hybrid" in src and src["metrics"] == HY.PRIVATE_NOTE
    HY.sidecar("d", "rundiv", runs, results, scrub=True)
    assert json.loads((runs / "d" / "rundiv" / "result.json").read_text()) == res


def test_sidecar_no_scrub_keeps_internal_diagnostics():
    runs, results, _ = _old_run(Path(tempfile.mkdtemp(prefix="scrub_")))
    HY.sidecar("d", "rundiv", runs, results, scrub=False)
    assert len(_private(_files(runs, results))) == 4
    assert json.loads((runs / "d" / "rundiv" / "result.json").read_text())["vs_train"] == VS_TRAIN


def test_v5_job_resumed_release_has_no_private_metrics():
    script = ROOT / "scripts" / "v5_job.sh"
    if not HY.SCRUB_PRIVATE_DEFAULT or f"{_TP_WORK}" in script.read_text():
        print("skip test_v5_job_resumed_release_has_no_private_metrics (internal tree: release build only)")
        return
    root = Path(tempfile.mkdtemp(prefix="scrub_v5_"))
    runs, results, work = _old_run(root)
    (work / "texts.parquet").touch()
    for tag in ("run", "rundiv"):
        (runs / "d" / tag / "synth.parquet").touch()
    stage = root / "stage"
    (stage / "results").mkdir(parents=True)
    os.symlink(results / "pe", stage / "results" / "pe")
    env = dict(os.environ, TRAJPE_ROOT=str(ROOT), TRAJPE_WORK=str(root), TRAJPE_PYTHON=sys.executable,
               TRAJPE_PYTHON_DPLORA=sys.executable, V5_GEN_WORK=str(work), STAGE=str(stage),
               PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(["bash", str(script), str(ROOT), "d", "1", "0", "run"], env=env, text=True,
                       capture_output=True)
    assert p.returncode == 0 and "V5_JOB_DONE" in p.stdout, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    assert _private(_files(runs, results)) == {}, _private(_files(runs, results))
    res = json.loads((runs / "d" / "rundiv" / "result.json").read_text())
    assert res["hybrid"]["eps_composed_pld"] <= 1.0 and res["metrics"] == HY.PRIVATE_NOTE
    assert "private-data metric" in p.stderr


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("PASS", k, flush=True)
