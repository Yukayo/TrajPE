from __future__ import annotations
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

import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DPLORA = PROJECT / "baselines" / "dplora"
sys.path.insert(0, str(DPLORA))
import textfmt
sys.path.remove(str(DPLORA))

DATA = PROJECT / "data" / os.environ.get("TRAJPE_DATA", "processed")
DEFAULT_DATA = PROJECT / "data" / "processed"
RUNS = Path(f"{_TP_WORK}/runs/baselines_v2/ctcl")
CALIB_CACHE = Path(os.environ.get("CTCL_CALIB_CACHE", RUNS / "_calib"))
TOPIC_DIR = HERE / "topics"
SPEC = PROJECT / "docs" / "baselines" / "CTCL_SPEC.md"

TRAJPE_PY = f"{_TP_PY}"
DPLORA_PY = f"{_TP_PY_DPLORA}"
BASE_PY = f"{_TP_PY_GRID}"

GRID_DATASETS = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc", "tsmc_nyc", "petaling_jaya", "moscow",
                 "bandung", "gw_stockholm", "gw_san_francisco", "sydney", "melbourne", "tangerang", "palembang",
                 "shanghai", "beijing", "ms_tokyo"]
TUNING_CITY = "jakarta"
TABLE_DATASETS = GRID_DATASETS + [TUNING_CITY]
FORBIDDEN = {"gw_austin", "gw_nyc"}

N_REF = 74_685
DELTA = 1e-6
SIGMA_TOPIC_HIST = 10.0
N_TOPIC_HIST = 1
PRIOR_MU2_SHARE = 0.1
N_PRIOR = 2
PRV_MARGIN = 0.015
CALIB_EPS_TOL = 0.005
CALIB_SIGMA_TOL = 1e-4
PLD_DISCRETIZATION = 1e-4
SIGMA_REF_TOL = 0.002

SIGMA_REF = {
    ("code", "plain"): {1: 0.77804, 2: 0.66283, 4: 0.54953, 8: 0.44470},
    ("code", "vp"): {1: 0.78303, 2: 0.66695, 4: 0.55492, 8: 0.45003},
    ("paper", "plain"): {1: 11.6883, 2: 5.73548, 4: 3.07673, 8: 1.77845},
}
SIGMA_PRIOR_REF = {1: 18.89334, 2: 9.97499, 4: 5.33758, 8: 2.92002}
SIGMA_EQ_REF = {1: 8.00566, 2: 5.76387, 4: 3.53111, 8: 2.02211}


def code_steps(n_ref: int = N_REF, batch: int = 128, min_steps: int = 1000) -> int:
    spe = n_ref // batch
    return math.ceil(min_steps / spe) * spe


CONFIGS = {
    "code": {"batch": 128, "steps": code_steps(), "clip": 1.0, "lr": 4e-4, "weight_decay": 0.01, "warmup_steps": 58,
             "betas": (0.9, 0.999), "adam_eps": 1e-8},
    "paper": {"batch": 4096, "steps": 2000, "clip": 1.0, "lr": 4e-4, "weight_decay": 0.1, "warmup_steps": 100,
              "betas": (0.9, 0.999), "adam_eps": 1e-8},
}
assert CONFIGS["code"]["steps"] == 1166 and CONFIGS["code"]["warmup_steps"] == int(0.05 * 1166)

METHODS = {
    "ctcl": ("plain", "code", "full"),
    "ctclvp": ("vp", "code", "full"),
    "ctclpaper": ("plain", "paper", "full"),
    "ctcl_lora": ("plain", "code", "lora"),
    "ctclvp_lora": ("vp", "code", "lora"),
}
LABELS = {"ctcl": "CTCL (authors' released code defaults)",
          "ctclvp": "CTCL + DP venue prior (authors' released code defaults)",
          "ctclpaper": "CTCL, paper B/T (R16 robustness release)",
          "ctcl_lora": "CTCL-LoRA fallback (authors' code defaults)",
          "ctclvp_lora": "CTCL-LoRA fallback + DP venue prior (authors' code defaults)"}
OUT_SUBDIR = {"ctcl": "plain", "ctclvp": "vp", "ctclpaper": "paper", "ctcl_lora": "plain_lora",
              "ctclvp_lora": "vp_lora"}

LORA = {"r": 8, "alpha": 16, "dropout": 0.0,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def lr_at(s: int, config: str = "code") -> float:
    c = CONFIGS[config]
    T, W, peak = c["steps"], c["warmup_steps"], c["lr"]
    if s < W:
        return peak * s / W
    return peak * (T - s) / (T - W)


def sample_rate(config: str) -> float:
    return CONFIGS[config]["batch"] / N_REF


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_COMMIT = "7ae557604adf67be50417f59c2c2f167def9a775"
N_PARAMS = 494_032_768
HF_HOME = f"{_TP_WORK}/hf_cache"

ROOTS = ["Arts and Entertainment", "Business and Professional Services", "Community and Government",
         "Dining and Drinking", "Event", "Health and Medicine", "Landmarks and Outdoors", "Retail",
         "Sports and Recreation", "Travel and Transportation"]
GRID_N = 6
N_DAYPARTS = 4
DAYPARTS = ["night", "morning", "afternoon", "evening"]
K_TOPICS = len(ROOTS) * GRID_N * GRID_N * N_DAYPARTS
N_KEYWORDS = 10
N_CAT_KEYWORDS = 7
DOC_TYPE = "Foursquare check-in trail"
MAX_COND_TOKENS = 128
assert K_TOPICS == 1440


def topic_id(r: int, g: int, b: int) -> int:
    return (r * GRID_N * GRID_N + g) * N_DAYPARTS + b


def condition(keywords: list) -> str:
    return f"Document Type: {DOC_TYPE}\nKeywords: " + ", ".join(keywords)


def messages(city: str, cond: str) -> list:
    m = textfmt.messages(city)
    return [m[0], {"role": m[1]["role"], "content": m[1]["content"] + "\n" + cond}]


def prompt_ids(tok, city: str, cond: str) -> list:
    text = tok.apply_chat_template(messages(city, cond), add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


COORD_DECIMALS = 4
MAX_RESP_TOKENS = 640
N_OUT = 50_000
TOP_P = 0.95
TEMPERATURE = 1.0
TOP_K = 0
MIN_P = 0.0
KV_GB = 8.0
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 1024
OVERSHOOT = 1.03
RATE_FLOOR = 0.05
MAX_ROUNDS = 20
EXTRA_ROUNDS = 20
ROUND_CAP = 3 * N_OUT
MIN_VALID = 0.02
MIN_VALID_AFTER = 20_000
V_INIT = 0.9

HIST_STREAM = 20261026
ORDER_STREAM = 20261027
NOISE_OFFSET = 7919

VP_GROUND_SCALE_KM = 0.05


def pop_radius_km(eps: float) -> float:
    return 0.3 if eps < 4 else 1.0


def eps_tag(eps: float) -> str:
    return f"{eps:g}"


def sha256_file(p, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_json_atomic(path, obj) -> str:
    path = Path(path)
    data = json.dumps(obj, indent=1, default=_json_default).encode()
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_bytes(data)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


CODE_FILES = ["ctcl_conf.py", "topics.py", "calibrate.py", "prv_ctcl.py", "train_ctcl.py", "sample_ctcl.py", "check.py",
              "pipeline_ctcl.py"]


def code_hashes() -> dict:
    return {f: sha256_file(HERE / f) for f in CODE_FILES if (HERE / f).exists()}


def table_path(dataset: str) -> Path:
    return TOPIC_DIR / f"{dataset}.json.gz"


def release_path(method: str, dataset: str, eps: float, seed: int = 0) -> Path:
    return RUNS / OUT_SUBDIR[method] / f"{dataset}_eps{eps_tag(eps)}_s{seed}.parquet"


def check_dataset(dataset: str) -> None:
    if dataset in FORBIDDEN:
        raise SystemExit(f"{dataset} is never read or run (sealed test / smoke-only set)")
    if dataset not in TABLE_DATASETS:
        raise SystemExit(f"{dataset} is not a CTCL dataset")
