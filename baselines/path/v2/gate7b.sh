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
set -u
T=${TRAJPE_ROOT}
PY=${TRAJPE_PYTHON}
Q=${TRAJPE_WORK}/gpuq
N=path7bv2_jakarta_eps4_s0
cd /tmp
for _ in $(seq 1 288); do
  if [ -e "$Q/done/$N" ] || [ -e "$Q/failed/$N" ]; then break; fi
  sleep 600
done
if ! { [ -e "$Q/done/$N" ] || [ -e "$Q/failed/$N" ]; }; then
  echo "$(date -u '+%F %T') $N not terminal after 48 h; nothing queued"; exit 1
fi
REP="$($PY -I -B "$T/scripts/queue_path.py" smoke_report --rows 7b)" || { echo "smoke_report failed"; exit 1; }
echo "$REP"
MEM="$(printf '%s\n' "$REP" | $PY -I -c "import json,sys
for ln in sys.stdin:
    r = json.loads(ln)
    if r.get('name') == '$N':
        print(r.get('declared_mem_rule') or '')")"
if [ -z "$MEM" ] || [ "$MEM" = "None" ]; then
  echo "$(date -u '+%F %T') no memory measurement from $N (training did not complete): the PATH-7B row is not queued"
  exit 3
fi
if $PY -I -B "$T/scripts/queue_path.py" queue --part 7b --mem_7b "$MEM"; then
  echo "$(date -u '+%F %T') PATH-7B row queued with --mem $MEM"; exit 0
fi
echo "$(date -u '+%F %T') queue --part 7b refused (O3 gate or freeze mismatch): the PATH-7B row is not queued"
exit 3
