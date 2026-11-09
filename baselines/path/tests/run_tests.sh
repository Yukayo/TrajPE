#!/usr/bin/env bash
if [ -z "${TRAJPE_ROOT:-}" ]; then
  TRAJPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f trajpe/__init__.py ] && [ "$PWD" != / ]; do cd ..; done; pwd)"
fi
: "${TRAJPE_WORK:=$TRAJPE_ROOT/work}"
: "${TRAJPE_PYTHON:=python}"
: "${TRAJPE_PYTHON_DPLORA:=$TRAJPE_PYTHON}"
: "${TRAJPE_PYTHON_AIMSEQ:=$TRAJPE_PYTHON}"
: "${TRAJPE_PYTHON_GRID:=$TRAJPE_PYTHON}"
export TRAJPE_ROOT TRAJPE_WORK TRAJPE_PYTHON TRAJPE_PYTHON_DPLORA TRAJPE_PYTHON_AIMSEQ TRAJPE_PYTHON_GRID
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=${TRAJPE_WORK}/hf_cache HF_HUB_OFFLINE=1
cd /tmp
${TRAJPE_PYTHON} -I -B "$D/test_release.py"
${TRAJPE_PYTHON_DPLORA} -I -B "$D/test_dpsgd.py"
${TRAJPE_PYTHON} -I -B "$D/test_accounting.py"
echo "all PATH tests passed"
