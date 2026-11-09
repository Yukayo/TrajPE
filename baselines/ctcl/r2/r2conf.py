from __future__ import annotations

import sys
from pathlib import Path

R2 = Path(__file__).resolve().parent
CTCL = R2.parent
for p in (str(CTCL), str(CTCL / "r1"), str(R2)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ctcl_conf as C

REV = "r2"
COORD_DECIMALS = 3
MAX_RESP_TOKENS = 640
DECODE = "reject"
TOPIC_DIR = R2 / "topics"
OUT_SUBDIR = {"ctcl": "plain_r2", "ctclvp": "vp_r2", "ctclpaper": "paper_r2", "ctcl_lora": "plain_lora_r2",
              "ctclvp_lora": "vp_lora_r2"}
V1 = {"COORD_DECIMALS": C.COORD_DECIMALS, "MAX_RESP_TOKENS": C.MAX_RESP_TOKENS, "TOPIC_DIR": C.TOPIC_DIR,
      "OUT_SUBDIR": dict(C.OUT_SUBDIR)}

C.COORD_DECIMALS = COORD_DECIMALS
C.MAX_RESP_TOKENS = MAX_RESP_TOKENS
C.TOPIC_DIR = TOPIC_DIR
C.OUT_SUBDIR = dict(OUT_SUBDIR)
C.R2_PATCHED = True

R2_FILES = ("r2/r2conf.py", "r2/topics_r2.py", "r2/train_ctcl_r2.py", "r2/sample_ctcl_r2.py", "r2/pipeline_ctcl_r2.py",
            "r1/sample_ctcl_r1.py")


def code_hashes() -> dict:
    return {f: C.sha256_file(CTCL / f) for f in R2_FILES}


def summary() -> dict:
    return {"rev": REV, "coord_decimals": COORD_DECIMALS, "max_resp_tokens": MAX_RESP_TOKENS, "decode": DECODE,
            "topic_tables": str(TOPIC_DIR), "sampler_gates": "r1 (first-round broken-model gate; global-surplus fill)",
            "v1_values": {"coord_decimals": V1["COORD_DECIMALS"], "max_resp_tokens": V1["MAX_RESP_TOKENS"],
                          "decode": "repair"},
            "why": "owner 2026-10-28 'unify': the DP-LoRA author-default format (frozen_default.json) for every "
                   "LLM-based baseline; 4 decimals + repair had been selected on Jakarta results (review C3)"}
