from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2conf
import train_ctcl

if __name__ == "__main__":
    assert train_ctcl.C.COORD_DECIMALS == r2conf.COORD_DECIMALS
    assert train_ctcl.C.MAX_RESP_TOKENS == r2conf.MAX_RESP_TOKENS
    assert train_ctcl.TP.C is train_ctcl.C
    train_ctcl.main()
