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
import os
from pathlib import Path

ROOT = Path(os.environ.get("TRAJPE_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("TRAJPE_WORK", f"{_TP_WORK}"))
RAW = WORK / "data" / "raw"
RUNS = WORK / "runs"
HF_HOME = WORK / "hf_cache"
PROCESSED = ROOT / "data" / os.environ.get("TRAJPE_DATA", "processed")
RESULTS = ROOT / "results"
