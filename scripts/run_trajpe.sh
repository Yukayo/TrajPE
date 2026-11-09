#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ds=$1; eps=$2; seed=${3:-0}
exec bash "$ROOT/scripts/v5_job.sh" "$ROOT" "$ds" "$eps" "$seed" "trajpe_e${eps}_s${seed}"
