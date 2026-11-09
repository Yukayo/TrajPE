#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", nargs="*", default=[])
    ap.add_argument("--stage", nargs=2, action="append", metavar=("NAME", "CMD"), required=True)
    a = ap.parse_args()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            if gpu not in ("", None):
                try:
                    q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.strip().split(",")
                    samples.append((time.time(), float(q[0]), float(q[1])))
                except Exception:
                    pass
            stop.wait(0.5)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    rec = {"meta": dict(kv.split("=", 1) for kv in a.meta), "gpu": gpu, "host": socket.gethostname(),
           "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "stages": []}
    rc = 0
    for name, cmd in a.stage:
        t0 = time.time()
        rc = subprocess.run(["bash", "-c", cmd]).returncode
        t1 = time.time()
        s = [x for x in samples if t0 <= x[0] <= t1]
        rec["stages"].append({"stage": name, "sec": round(t1 - t0, 1), "rc": rc,
                              "peak_gpu_mib": max((x[1] for x in s), default=None),
                              "mean_gpu_util": round(sum(x[2] for x in s) / len(s), 1) if s else None})
        print(json.dumps(rec["stages"][-1]), flush=True)
        if rc:
            break
    stop.set()
    rec["total_sec"] = round(sum(x["sec"] for x in rec["stages"]), 1)
    rec["peak_gpu_mib"] = max((x["peak_gpu_mib"] or 0 for x in rec["stages"]), default=None)
    rec["ok"] = rc == 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
