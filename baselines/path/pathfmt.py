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

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


textfmt = load_module("dplora_textfmt", PROJECT / "baselines" / "dplora" / "textfmt.py")

N_REF = 74_685
BATCH = 256
EPOCHS = 3
LR = 5e-4
CLIP = 0.01
RANK = 128
ALPHA = 128
DROPOUT = 0.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WARMUP_FRAC = 0.03
ADAM = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
DELTA = 1e-6
PRV_MARGIN = 0.015
CALIB_TOL = 0.005
VOTE_K = 10
N_OUT = 50_000
POOL_M = 100_000
MAX_NEW = 640
TEMPERATURE, TOP_P, TOP_K, MIN_P = 1.0, 1.0, 0, 0.0
MAX_ROUNDS, MIN_VALID, OVERSHOOT = 20, 0.02, 1.03
MIN_VALID_AFTER = 20_000
DECIMALS = 4
MIN_LEN, MAX_LEN = textfmt.MIN_LEN, textfmt.MAX_LEN
EMBED_MAX_TOKENS = 64
SEED = 0

STREAM_WINDOWS = 220001
STREAM_GROUND = 220011
STREAM_SELECT = 220031
STREAM_SHUFFLE = 220041
NOISE_SEED_OFFSET = 7919

MODELS = {
    "1.5b": {"id": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323",
             "trainable_params": 147_718_144, "micro_max": 16, "tok_budget": 8192, "row": "path",
             "mirrors": "Gemma 3 1B (pt), PATH's 1B backbone (D1)", "licence": "apache-2.0"},
    "7b": {"id": "Qwen/Qwen2.5-7B", "revision": "d149729398750b98c0af14eb82c78cfe92750796",
           "trainable_params": 322_961_408, "micro_max": 8, "tok_budget": 4096, "row": "path_7b",
           "mirrors": "Gemma 3 4B (pt), PATH's 4B backbone (A17)", "licence": "apache-2.0"},
}
EMBEDDER = {"id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "licence": "apache-2.0"}

RUNS = Path(f"{_TP_WORK}/runs/baselines_v2")
DATA = PROJECT / "data" / "processed"
HF_HOME = f"{_TP_WORK}/hf_cache"

DATASETS = ["istanbul", "tokyo", "sao_paulo", "kuwait_city", "nyc", "tsmc_nyc", "petaling_jaya", "moscow", "bandung",
            "gw_stockholm", "gw_san_francisco", "sydney", "melbourne", "tangerang", "palembang", "shanghai",
            "beijing", "ms_tokyo"]
DATASETS_7B = ["istanbul", "tokyo", "sao_paulo", "kuwait_city"]
SMOKE = "jakarta"
NEVER = {"gw_austin", "gw_nyc"}
WEEKDAYS = textfmt.WEEKDAYS
SEP = textfmt.SEP


def check_dataset(d: str) -> str:
    if d in NEVER:
        raise SystemExit(f"{d} is never read or run (sealed test / smoke-only dataset)")
    if d not in DATASETS and d != SMOKE:
        raise SystemExit(f"{d} is not a PATH dataset")
    return d


def eps_select(eps: float) -> float:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("finite positive eps only")
    return min(1.0, max(0.25, eps / 4.0))


def sample_rate() -> float:
    return BATCH / N_REF


def n_steps() -> int:
    return int(round(EPOCHS / sample_rate()))


def header(city: str) -> str:
    m = textfmt.messages(city)
    return m[0]["content"] + "\n" + m[1]["content"] + "\n"


def window_char_bounds(text: str) -> list[int]:
    lines = text.split("\n")
    L = len(lines) - 1
    if L < 1:
        raise ValueError("a trail text has a Day line and at least one row")
    out, pos = [0], len(lines[0]) + 1
    for k in range(1, L):
        pos += len(lines[k]) + 1
        out.append(pos)
    return out


def target_starts(offsets, bounds_abs) -> tuple[list[int], int]:
    ends = np.array([e for _, e in offsets], dtype=np.int64)
    begins = np.array([b for b, _ in offsets], dtype=np.int64)
    starts, straddle = [], 0
    for b in bounds_abs:
        i = int(np.searchsorted(ends, b, side="right"))
        if i < len(begins) and begins[i] < b:
            straddle += 1
        starts.append(i)
    return starts, straddle


def parse_row(line: str, categories):
    ln = line.strip()
    if not ln:
        return None, "empty"
    parts = ln.split(SEP)
    if len(parts) != 3:
        return None, "fields"
    tm, cat, xy = (p.strip() for p in parts)
    if not textfmt._TIME.match(tm):
        return None, "time"
    if cat not in categories:
        return None, "category"
    if not textfmt._COORD.match(xy):
        return None, "coord"
    return f"{tm}{SEP}{cat}{SEP}{xy}", ""


def truncate(text: str, categories, finished: bool):
    lines = text.split("\n")
    end = "eos"
    if not finished:
        lines = lines[:-1]
        end = "token_limit"
    else:
        while lines and not lines[-1].strip():
            lines.pop()
    if not lines:
        return None, "bad_day", {"end": end, "rows": 0}
    first = lines[0].strip()
    if not first.startswith("Day: ") or first[5:].strip() not in WEEKDAYS:
        return None, "bad_day", {"end": end, "rows": 0}
    rows = []
    for ln in lines[1:]:
        row, kind = parse_row(ln, categories)
        if row is None:
            end = "invalid_" + kind
            break
        rows.append(row)
    info = {"end": end, "rows": len(rows)}
    if len(rows) < MIN_LEN:
        return None, "too_short", info
    return "\n".join([f"Day: {first[5:].strip()}"] + rows[:MAX_LEN]), "ok", info


def row_texts(frame) -> list[str]:
    t = frame["t_local"]
    wd = t.dt.dayofweek.to_numpy()
    hh, mm = t.dt.hour.to_numpy(), t.dt.minute.to_numpy()
    cats = frame["category"].astype(str).to_numpy()
    lat, lon = frame["lat"].to_numpy(np.float64), frame["lon"].to_numpy(np.float64)
    return [f"{WEEKDAYS[w]} {h:02d}:{m:02d}{SEP}{c}{SEP}{a:.4f},{o:.4f}"
            for w, h, m, c, a, o in zip(wd.tolist(), hh.tolist(), mm.tolist(), cats, lat.tolist(), lon.tolist())]


def lr_factor(step: int, total: int, warmup_frac: float = WARMUP_FRAC) -> float:
    w = max(1, int(round(warmup_frac * total)))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - w) / max(1, total - w))))


def settings(model_key: str) -> dict:
    m = MODELS[model_key]
    return {"model": m["id"], "model_revision": m["revision"], "lora_rank": RANK, "lora_alpha": ALPHA,
            "lora_dropout": DROPOUT, "lora_targets": TARGETS, "trainable_params": m["trainable_params"],
            "lr": LR, "schedule": f"linear warm-up {WARMUP_FRAC:.0%} of the steps, then cosine to 0",
            "optimizer": "AdamW(betas=(0.9, 0.999), eps=1e-8, weight_decay=0)", "batch_ref": BATCH,
            "b_nom": BATCH, "epochs": EPOCHS, "n_ref": N_REF, "clip": CLIP, "sampling": "Poisson",
            "window": "k ~ Uniform{0..L-1} per draw (A5)", "pool_m": POOL_M, "n_out": N_OUT, "vote_k": VOTE_K,
            "max_new_tokens": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "max_rounds": MAX_ROUNDS, "min_valid": MIN_VALID, "overshoot": OVERSHOOT,
            "coord_decimals": DECIMALS, "embedder": EMBEDDER["id"], "embedder_revision": EMBEDDER["revision"],
            "embed_max_tokens": EMBED_MAX_TOKENS, "seed": SEED}
