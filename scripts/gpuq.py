"""Persistent job queue that keeps the GPUs (and a CPU pool) busy, independent of any interactive session.

    python -I scripts/gpuq.py add    --state DIR --name NAME --cmd "shell command" [--mem 40] [--where 0,1|cpu] [--prio 0]
    python -I scripts/gpuq.py worker --state DIR --gpu 0 [--max_jobs 2] [--reserve_gb 4] [--settle 90]
    python -I scripts/gpuq.py worker --state DIR --gpu cpu --max_jobs 6
    python -I scripts/gpuq.py status --state DIR

Run each worker detached (setsid nohup ... &). A worker launches the first pending job (lowest prio, then file order)
that may run on its device and fits: on a GPU, the declared --mem (GiB) must be below the free memory reported by
nvidia-smi minus --reserve_gb, and at most --max_jobs of its own jobs run at once; after a launch it waits --settle
seconds before the next launch on that GPU so the new job's memory shows up. Jobs are claimed atomically
(O_EXCL marker files), so several workers can share one jobs file. Files in DIR: jobs.jsonl (append-only;
re-read every loop), running/<name>, done/<name>, failed/<name> (delete it to retry), logs/<name>.log,
PAUSE (no new launches), STOP (no new launches; exit when idle). Each job runs as `bash -c CMD` in its own
session with CUDA_VISIBLE_DEVICES set to the worker's GPU ("" for cpu workers). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_jobs(state: Path) -> list[dict]:
    f = state / "jobs.jsonl"
    if not f.exists():
        return []
    jobs = []
    for i, ln in enumerate(f.read_text().splitlines()):
        if ln.strip():
            j = json.loads(ln)
            j["_order"] = i
            jobs.append(j)
    return jobs


def gpu_free_gb(gpu: int, reserved_gb: float = 0.0) -> float:
    """Free memory, counting every queue job running on this GPU at its declared size until it really uses that
    much (a job that just started has not allocated yet, and several workers may share one GPU)."""
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    total, used = [float(x) / 1024 for x in q.strip().split(",")]
    return total - max(used, reserved_gb)


def running_markers(state: Path) -> dict:
    """name -> {"dev", "pid", "mem"} for every running marker (legacy markers: "dev time" text, no pid)."""
    out = {}
    for f in (state / "running").iterdir():
        txt = f.read_text().strip()
        try:
            m = json.loads(txt)
        except json.JSONDecodeError:
            m = {"dev": txt.split()[0] if txt else "", "pid": None, "mem": None}
        out[f.name] = m
    return out


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def cmd_add(a):
    state = Path(a.state)
    for d in ("running", "done", "failed", "logs"):
        (state / d).mkdir(parents=True, exist_ok=True)
    names = {j["name"] for j in read_jobs(state)}
    if a.name in names:
        print(f"exists {a.name}")
        return
    where = a.where.split(",") if a.where else ["0", "1"]
    with open(state / "jobs.jsonl", "a") as f:
        f.write(json.dumps({"name": a.name, "cmd": a.cmd, "mem": a.mem, "where": where, "prio": a.prio}) + "\n")
    print(f"added {a.name}")


def cmd_status(a):
    state = Path(a.state)
    jobs = read_jobs(state)
    st = {"done": [], "failed": [], "running": [], "pending": []}
    for j in jobs:
        n = j["name"]
        k = ("done" if (state / "done" / n).exists() else "failed" if (state / "failed" / n).exists()
             else "running" if (state / "running" / n).exists() else "pending")
        st[k].append(n)
    for k, v in st.items():
        print(f"{k:8s} {len(v):4d}  {' '.join(v[:12])}{' ...' if len(v) > 12 else ''}")


MAX_RETRIES = 2      # transient failures (e.g. vLLM's memory profiling racing another job) are requeued


def finish(state: Path, n: str, rc: int):
    (state / "running" / n).unlink(missing_ok=True)
    if rc == 0:
        (state / "done" / n).write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        log(f"done {n}")
        return
    af = state / "attempts" / n
    af.parent.mkdir(exist_ok=True)
    try:                    # an attempts file can be empty or garbled after a disk-full event
        tries = int(af.read_text().strip() or 0) if af.exists() else 0
    except (ValueError, OSError):
        tries = 0
    if tries < MAX_RETRIES:
        af.write_text(str(tries + 1))
        log(f"retry {n} (exit {rc}, attempt {tries + 1} of {MAX_RETRIES})")
    else:
        (state / "failed" / n).write_text(f"exit {rc} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log(f"FAILED {n} (exit {rc})")


def cmd_worker(a):
    state = Path(a.state)
    for d in ("running", "done", "failed", "logs", "exit"):
        (state / d).mkdir(parents=True, exist_ok=True)
    dev = a.gpu
    is_cpu = dev == "cpu"
    mine: dict[str, subprocess.Popen] = {}
    last_launch = 0.0
    log(f"worker {dev} started (pid {os.getpid()}), max_jobs {a.max_jobs}")
    while True:
        for n, p in list(mine.items()):
            if p.poll() is not None:
                del mine[n]
                try:
                    finish(state, n, p.returncode)
                except OSError as e:          # e.g. disk full: keep the worker alive, retry the bookkeeping later
                    log(f"finish({n}) failed: {e}")
        # jobs whose worker died: reap them from their exit file once their process is gone
        markers = running_markers(state)
        for n, m in markers.items():
            if n in mine or m.get("dev") != dev or m.get("pid") is None or pid_alive(m["pid"]):
                continue
            ef = state / "exit" / n
            finish(state, n, int(ef.read_text().strip() or 1) if ef.exists() else 1)
        stop, pause = (state / "STOP").exists(), (state / "PAUSE").exists()
        if stop and not mine:
            log("STOP and idle: exiting")
            return
        jobs = sorted(read_jobs(state), key=lambda j: (j.get("prio", 0), j["_order"]))
        n_dev = sum(1 for m in running_markers(state).values() if m.get("dev") == dev)
        can_launch = (not stop and not pause and len(mine) < a.max_jobs and n_dev < a.max_dev_jobs
                      and time.time() - last_launch >= (0 if is_cpu else a.settle))
        if can_launch:
            declared = {j["name"]: float(j.get("mem", 0)) for j in jobs}
            reserved = sum(float(m["mem"]) if m.get("mem") is not None else declared.get(n, 0.0)
                           for n, m in running_markers(state).items() if m.get("dev") == dev)
            free = float("inf") if is_cpu else gpu_free_gb(int(dev), reserved)
            for j in jobs:
                n = j["name"]
                if dev not in j.get("where", []) or (state / "done" / n).exists() or (state / "failed" / n).exists():
                    continue
                if (state / "running" / n).exists():
                    continue
                if not is_cpu and float(j.get("mem", 0)) > free - a.reserve_gb:
                    continue
                try:                                   # atomic claim
                    fd = os.open(state / "running" / n, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                (state / "exit" / n).unlink(missing_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if is_cpu else str(dev))
                lf = open(state / "logs" / f"{n}.log", "a")
                lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} on {dev}: {j['cmd']}\n")
                lf.flush()
                exit_file = shlex.quote(str(state / "exit" / n))
                wrapped = f"{j['cmd']}\nrc=$?; echo $rc > {exit_file}; exit $rc"
                mine[n] = subprocess.Popen(["bash", "-c", wrapped], stdout=lf, stderr=subprocess.STDOUT, env=env,
                                           start_new_session=True)
                os.write(fd, json.dumps({"dev": dev, "pid": mine[n].pid, "mem": float(j.get("mem", 0)),
                                         "time": time.strftime("%Y-%m-%d %H:%M:%S")}).encode())
                os.close(fd)
                last_launch = time.time()
                log(f"launched {n} on {dev} (declared {j.get('mem', 0)} GiB, free {free:.1f} GiB)")
                break
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("add")
    p.add_argument("--state", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--cmd", required=True)
    p.add_argument("--mem", type=float, default=0.0, help="GiB of GPU memory the job needs")
    p.add_argument("--where", default="", help="comma list of devices: 0,1 and/or cpu (default 0,1)")
    p.add_argument("--prio", type=int, default=0, help="lower runs first")
    w = sub.add_parser("worker")
    w.add_argument("--state", required=True)
    w.add_argument("--gpu", required=True, help="GPU index or 'cpu'")
    w.add_argument("--max_jobs", type=int, default=1, help="jobs this worker runs at once")
    w.add_argument("--max_dev_jobs", type=int, default=99, help="queue jobs on this device at once, all workers")
    w.add_argument("--reserve_gb", type=float, default=4.0)
    w.add_argument("--settle", type=float, default=90.0)
    w.add_argument("--poll", type=float, default=20.0)
    s = sub.add_parser("status")
    s.add_argument("--state", required=True)
    a = ap.parse_args()
    {"add": cmd_add, "worker": cmd_worker, "status": cmd_status}[a.action](a)


if __name__ == "__main__":
    sys.exit(main())
