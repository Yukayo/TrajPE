#!/bin/bash
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
SNAP=$(readlink -f "$1"); ds=$2; eps=$3; seed=$4; tag=$5
STAGE=${STAGE:-${TRAJPE_WORK}/v5_staging/root}
T=${TRAJPE_ROOT}
PY=${TRAJPE_PYTHON}
DPY=${TRAJPE_PYTHON_DPLORA}
case "$ds" in gw_austin|gw_nyc) echo "refusing $ds (sealed test / smoke only)"; exit 2;; esac
if [ -n "${V4_TEST_EXTRA_SAMPLE:-}" ] && [ "$ds" != jakarta ]; then
  echo "refusing V4_TEST_EXTRA_SAMPLE on $ds (verification hook, Jakarta only)"; exit 2
fi
mkdir -p "$STAGE/results"
[ -e "$STAGE/data" ] || ln -s "$T/data" "$STAGE/data"
if (( $(echo "$eps >= 4" | bc -l) )); then sz=large; else sz=small; fi
cfg=$SNAP/configs/v5_${sz}_eps.json; cfg4=$SNAP/configs/v4_${sz}_eps.json
[ -z "${V4_GEN_WORK:-}" ] || { echo "V4_GEN_WORK is not read by v5_job.sh (use V5_GEN_WORK)"; exit 2; }
W=${V5_GEN_WORK:-${TRAJPE_WORK}/runs/hybrid/$ds/_work/v4_e${eps}_s${seed}}
PARSED=$($PY -I - "$cfg" "$eps" "$W" "$cfg4" <<'EOF'
import json, shlex, sys
c = json.load(open(sys.argv[1])); eps, work = sys.argv[2], sys.argv[3]
if c.get("_accepted") is not True or not c.get("_frozen"):
    sys.exit(f"{sys.argv[1]} is not the frozen, accepted v5 configuration")
c4 = json.load(open(sys.argv[4]))
assert c["_hybrid"]["generator"] == c4["_hybrid"]["generator"] and c["_hybrid"]["sgd_share"] == c4["_hybrid"]["sgd_share"] \
    and c["pool_n"] == c4["pool_n"], "v5 generator recipe differs from v4's: the v4 work dir cannot be reused"
assert c["release_union_raw"] == 1 and c["vote_cluster_kappa"] == 3.0
h = c["_hybrid"]; g = h["generator"]
cal_s, pipe_s = g["script"].split(", then ")
cal = cal_s.split("hybrid.py calibrate", 1)[1].split()
pipe = pipe_s.replace("(calib.json in <work>)", "").split("pipeline.py", 1)[1].split()
cal = [eps if t == "E" else t for t in cal]
pipe = [work if t == "<work>" else t for t in pipe]
assert cal[cal.index("--sgd_share") + 1] == f"{h['sgd_share']:g}", "sgd_share mismatch"
assert pipe[pipe.index("--variant") + 1] == "vp" and pipe[pipe.index("--stage") + 1] == "gpu"
assert int(pipe[pipe.index("--n_out") + 1]) == g["texts"] == c["pool_n"]
assert g["model"] == "Qwen/Qwen2.5-0.5B-Instruct" and "--model" not in pipe
for k in ("--batch", "--epochs"):
    assert cal[cal.index(k) + 1] == pipe[pipe.index(k) + 1], k
pp = h["release_postprocess"]
assert pp["script"] == "scripts/postprocess_release.py" and pp["stream"] == "[seed, 7919]"
assert h["release_sidecar"]["script"].startswith("baselines/dplora/hybrid.py sidecar")
run = []
for k, v in c.items():
    if not k.startswith("_"):
        run += [f"--{k}", str(v)]
print(f"CAL_ARGS={shlex.quote(' '.join(cal))}")
print(f"PIPE_ARGS={shlex.quote(' '.join(pipe))}")
print(f"RUN_ARGS={shlex.quote(' '.join(run))}")
print(f"PP_P={pp['p']:g}; PP_R={pp['radius_km']:g}")
EOF
)
eval "$PARSED"
export TRAJPE_ROOT=$STAGE TRAJPE_WORK=${TRAJPE_WORK} PATH=$PATH TRAJPE_GRID_NOMETRICS=1
export OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} OPENBLAS_NUM_THREADS=${THREADS:-4}
echo "v5_job $ds eps=$eps seed=$seed tag=$tag; config $cfg; work $W"
D=$SNAP/baselines/dplora
GEN_FAILED=""
[ -f "$W/generator_failed.json" ] && GEN_FAILED="$W/generator_failed.json"
if [ -n "${V5_GEN_WORK:-}" ] || [ -n "$GEN_FAILED" ] || [ -f "$W/texts.parquet" ]; then
  echo "GEN_REUSED $W (v4's generator: not retrained, calib.json not rewritten)"
else
  echo "GEN_TRAIN $W (no v4 generator yet: training it exactly as v4_job.sh would)"
  mkdir -p "$W"
  $PY -I -B $D/hybrid.py calibrate $CAL_ARGS --out "$W/calib.json"
  LOG_OFF=$(stat -c %s "$W/pipeline.log" 2>/dev/null || echo 0)
  set +e
  (cd /tmp && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $DPY -I -B $D/pipeline.py --dataset "$ds" --eps "$eps" \
     --seed "$seed" --out "$W/unused.parquet" --mem_cap_gb 26 $PIPE_ARGS \
     ${V4_TEST_EXTRA_SAMPLE:+"--extra_sample=$V4_TEST_EXTRA_SAMPLE"}) 2>&1 | tee "$W/pipeline.out"
  GEN_RC=${PIPESTATUS[0]}
  set -e
  if [ "$GEN_RC" -ne 0 ]; then
    if ! $PY -I - "$W" "$GEN_RC" "$LOG_OFF" <<'GENFAIL'
import json, re, sys, time
from pathlib import Path
w, rc, off = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

def no(why):
    print(f"generator failure (exit {rc}) is not the frozen validity gate: {why}", flush=True)
    sys.exit(1)

if rc == 0:
    no("pipeline.py exited 0")
po = w / "pipeline.out"
lines = [ln for ln in (po.read_text(errors="replace").splitlines() if po.exists() else []) if ln.strip()]
m = re.fullmatch(r"subprocess\.CalledProcessError: Command '\[(.*)\]' returned non-zero exit status (-?\d+)\.",
                 lines[-1].strip() if lines else "")
if not m or "/sample_vllm.py'" not in m.group(1) or int(m.group(2)) != 3:
    no("pipeline.py's last line is not the CalledProcessError of sample_vllm.py with exit status 3")
pl = w / "pipeline.log"
new = pl.read_bytes()[off:].decode(errors="replace") if pl.exists() else ""
gate = re.compile(r"valid rate \d+\.\d+ < \d+(\.\d+)? after \d+ samples: model is broken|only \d+ valid texts after \d+ rounds")
msgs = [ln.strip() for ln in new.splitlines() if gate.fullmatch(ln.strip())]
if len(msgs) != 1:
    no(f"{len(msgs)} sampler gate messages in this attempt's pipeline.log (need exactly 1)")
if sorted(w.glob("texts*.parquet")):
    no("texts*.parquet present")
if not (w / "adapter").is_dir():
    no("no adapter")
tj = json.loads((w / "train.json").read_text())
cal = json.loads((w / "calib.json").read_text())
if tj.get("complete") is False or int(tj.get("steps_run", -1)) != int(tj["steps"]) or \
        (tj.get("private_stats") or {}).get("oom_retries", 0):
    no("training incomplete (complete / steps_run / oom_retries)")
if (float(tj["noise_multiplier"]), float(tj["sample_rate"]), int(tj["steps"])) != \
        (float(cal["sigma_sgd"]), float(cal["sample_rate"]), int(cal["steps"])):
    no("train.json (sigma, q, T) differ from calib.json")
tjs = w / "texts.json"
rec = None
if tjs.exists():
    t = json.loads(tjs.read_text())
    rec = {k: t.get(k) for k in ("n_out", "n_kept", "n_generated", "valid_rate", "batches")}
note = {"status": "generator_failed", "reason": msgs[0], "sampler": "sample_vllm.py", "sampler_exit": 3,
        "pipeline_exit": rc, "sampler_record": rec, "work": str(w), "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "dp_sgd_charged": True,
        "rule": "design freeze 3.1 (docs/DESIGN_FREEZE3.md): the generator failed sample_vllm.py's frozen validity gate "
                "after a complete DP-SGD training; PE runs with the simulator as the only candidate source (pool_frac 0, "
                "pool_refresh 0, every other v4 flag and mu_PE unchanged); the DP-SGD part stays charged"}
tmp = w / "generator_failed.json.tmp"
tmp.write_text(json.dumps(note, indent=1))
tmp.rename(w / "generator_failed.json")
print("GEN_FAILED " + json.dumps(note), flush=True)
GENFAIL
    then
      echo "generator failed (exit $GEN_RC): aborting"; exit "$GEN_RC"
    fi
    GEN_FAILED="$W/generator_failed.json"
  else
    echo "GEN_DONE $W"
  fi
fi
if [ -n "$GEN_FAILED" ]; then
  [ -f "$GEN_FAILED" ] && [ -f "$W/train.json" ] || { echo "generator incomplete: $W"; exit 1; }
  echo "GENERATOR_FAILED $W: PE with the simulator as the only candidate source (freeze 3.1)"
else
  [ -f "$W/texts.parquet" ] && [ -f "$W/train.json" ] || { echo "generator incomplete: $W"; exit 1; }
fi
if [ ! -f ${TRAJPE_WORK}/runs/$ds/$tag/synth.parquet ]; then
  $PY -I -B $SNAP/scripts/run_hybrid.py --dataset "$ds" --eps "$eps" --work "$W" --seed "$seed" $RUN_ARGS --tag "$tag" \
    ${GEN_FAILED:+--generator_failed "$GEN_FAILED"}
fi
if [ ! -f ${TRAJPE_WORK}/runs/$ds/${tag}div/synth.parquet ]; then
  CUDA_VISIBLE_DEVICES="" $PY -I -B $SNAP/scripts/postprocess_release.py --dataset "$ds" --tag "$tag" --p "$PP_P" \
    --radius_km "$PP_R" --out_tag "${tag}div"
fi
CUDA_VISIBLE_DEVICES="" $PY -I -B $D/hybrid.py sidecar --dataset "$ds" --tag "${tag}div"
echo "V5_JOB_DONE $ds $tag"
