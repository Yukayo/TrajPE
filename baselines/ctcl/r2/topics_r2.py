from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2conf
import topics

if __name__ == "__main__":
    assert topics.C.COORD_DECIMALS == r2conf.COORD_DECIMALS and topics.C.TOPIC_DIR == r2conf.TOPIC_DIR
    topics.main()
