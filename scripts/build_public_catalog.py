import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from trajpe.paths import PROCESSED
from trajpe.public_catalog import PUBLIC_BBOX, build_public_catalogs, catalog_coverage

if __name__ == "__main__":
    names = sys.argv[1:] or list(PUBLIC_BBOX)
    stats = build_public_catalogs(names)
    for n in names:
        stats[n]["coverage"] = catalog_coverage(pd.read_parquet(PROCESSED / n / "pois.parquet", columns=["lat", "lon"]),
                                                PUBLIC_BBOX[n])
    print(json.dumps(stats, indent=1))
