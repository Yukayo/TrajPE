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

import ast
import hashlib
import types
from pathlib import Path

REPO = Path(f"{_TP_WORK}/baselines_src/pategail/repo")
COMMIT = "fe8f78c64db2424d52178d608d3294f57646d980"
URL = "https://github.com/yingjie1234/PATEGAIL-PlusPlus-Utility-Optimized-Private-Trajectory-Generation"
LICENSE = "MIT License, Copyright (c) 2022 FIB LAB, Tsinghua University"
SHA256 = {
    "models/net.py": "0774bb5e82aed4868b46bb51436f81465e1a4309b9e08c0c402433a952463bd2",
    "models/mp_gail.py": "a7fe7def8e4330e49fd83dba69e3ebeeb2d4da3c4fc7e0df547007b56a1000d3",
    "models/replay_buffer.py": "dad262668f59183ef1ce40ba083f6298549e20d17b2c1ba94581527b73aa9343",
    "env/env.py": "cd3e866bac8d135c654eb251afb04a87a2e33a17b24a8ab0a0b6eafd6cf671c9",
    "config/config.yaml": "1d1601229ba4348016852bc0336cf33f8ffede617e63294b0ad6d90aae13d748",
    "utils/gps_process.py": "86d88eba5e11c3c31b336420484bc35a002c43367976a95c24cbd1889aeea43b",
    "run.py": "a7dfe7f75df2960dee364d0cd3b2470f56d7489dad92bc0648a0a46c646f4de2",
    "LICENSE": "52c033e904777c4ba31398a7b840f78a58f61b3de4848761ad9b206829d8eec3",
}


def source(rel: str) -> str:
    b = (REPO / rel).read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if h != SHA256[rel]:
        raise RuntimeError(f"official {rel}: sha256 {h} != pinned {SHA256[rel]}")
    return b.decode()


def _module_from_source(rel: str, name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / rel)
    exec(compile(source(rel), str(REPO / rel), "exec"), mod.__dict__)
    return mod


_cache: dict = {}


def net() -> types.ModuleType:
    if "net" not in _cache:
        _cache["net"] = _module_from_source("models/net.py", "pategail_official_net")
    return _cache["net"]


def replay_buffer() -> types.ModuleType:
    if "rb" not in _cache:
        _cache["rb"] = _module_from_source("models/replay_buffer.py", "pategail_official_replay_buffer")
    return _cache["rb"]


def env_module() -> types.ModuleType:
    if "env" not in _cache:
        _cache["env"] = _module_from_source("env/env.py", "pategail_official_env")
    return _cache["env"]


def method(rel: str, cls: str, name: str, namespace: dict):
    tree = ast.parse(source(rel))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    code = compile(ast.Module(body=[sub], type_ignores=[]), str(REPO / rel), "exec")
                    exec(code, namespace)
                    return namespace[name]
    raise KeyError(f"{cls}.{name} not found in {rel}")
