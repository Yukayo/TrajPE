"""Release queue entry points; the only worker test runs a CPU `true` command."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_METHODS = {
    "ldptrace", "dplora_plain", "dplora_vp", "naivepe", "dplora7b_plain", "dplora7b_vp",
}
QUEUE_FILES = (
    "scripts/queue_baselines.py", "scripts/gpuq.py", "scripts/run_pe.py",
    "baselines/run_ldptrace.py", "baselines/dplora/jobs.py",
    "baselines/dplora/frozen_v2.json", "baselines/dplora/pipeline.py",
    "configs/datasets_v3.json", "frozen/datasets_v3_preregistration.sha256",
)


class TestQueueBaselines(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="trajpe queue ")
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / "release root"
        self.work = self.base / "work area"
        self.work.mkdir()
        (self.root / "trajpe").mkdir(parents=True)
        (self.root / "trajpe" / "__init__.py").write_text("")
        for rel in QUEUE_FILES:
            dest = self.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        # Test interpreter quoting as well as repository and output paths.
        self.python = self.base / "python bin" / "python"
        self.python.parent.mkdir()
        self.python.symlink_to(sys.executable)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("TRAJPE_")}
        self.env.update(
            TRAJPE_ROOT=str(self.root), TRAJPE_WORK=str(self.work),
            TRAJPE_PYTHON=str(self.python), TRAJPE_PYTHON_GRID=str(self.python),
            TRAJPE_PYTHON_DPLORA=str(self.python), PYTHONDONTWRITEBYTECODE="1",
        )
        self.queue = self.root / "scripts" / "queue_baselines.py"
        self.jobs_file = self.work / "gpuq" / "jobs.jsonl"

    def run_queue(self, command, *args):
        return subprocess.run(
            [sys.executable, "-I", "-B", str(self.queue), command,
             "--datasets", "nyc", "--eps", "1", "--levels", "trail", *args],
            env=self.env, cwd=self.base, capture_output=True, text=True, timeout=30,
        )

    def assert_success(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def plan(self):
        output = self.base / "plan.json"
        proc = self.run_queue("plan", "--json_out", str(output))
        self.assert_success(proc)
        return json.loads(output.read_text())

    def prepare_placeholders(self):
        # Enqueue validates presence only. These files must never reach training,
        # Parquet loading, a queue worker, or a GPU in this test suite.
        data = self.root / "data" / "processed" / "nyc"
        data.mkdir(parents=True)
        (data / "meta.json").write_text("{}")
        (data / "train.parquet").touch()
        (data / "pois.parquet").touch()

    def queued_jobs(self):
        return [json.loads(line) for line in self.jobs_file.read_text().splitlines() if line.strip()]

    def test_default_plan_without_data_or_existing_queue(self):
        rows = self.plan()
        self.assertEqual({row["method"] for row in rows}, PUBLIC_METHODS)
        self.assertEqual(len(rows), len(PUBLIC_METHODS))
        self.assertTrue(all(row["status"] == "missing" for row in rows), rows)
        self.assertFalse((self.root / "data").exists())
        self.assertFalse(self.jobs_file.exists())

    def test_dry_queue_without_data_does_not_enqueue(self):
        proc = self.run_queue("queue", "--dry")
        self.assert_success(proc)
        self.assertEqual(proc.stdout.count("[dry]"), len(PUBLIC_METHODS), proc.stdout)
        self.assertFalse((self.root / "data").exists())
        self.assertFalse(self.jobs_file.exists())

    def test_enqueue_requires_prepared_data(self):
        proc = self.run_queue("queue")
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("train.parquet", proc.stdout + proc.stderr)
        self.assertFalse(self.jobs_file.exists())

    def test_prepared_enqueue_is_pending_and_idempotent(self):
        self.prepare_placeholders()
        self.assert_success(self.run_queue("queue"))
        jobs = self.queued_jobs()
        rows = self.plan()
        self.assertEqual(len(jobs), len(PUBLIC_METHODS))
        self.assertEqual({j["name"] for j in jobs}, {r["name"] for r in rows})
        self.assertTrue(all(row["status"] == "pending" for row in rows), rows)
        before = self.jobs_file.read_bytes()
        proc = self.run_queue("queue")
        self.assert_success(proc)
        self.assertEqual(self.jobs_file.read_bytes(), before)
        self.assertIn("pending", proc.stdout)
        for state in ("running", "done", "failed"):
            self.assertEqual(list((self.work / "gpuq" / state).iterdir()), [])

    def test_commands_reference_shipped_scripts_and_quote_paths(self):
        self.prepare_placeholders()
        self.assert_success(self.run_queue("queue"))
        outputs = {r["name"]: r["out"] for r in self.plan()}
        seen_scripts = set()
        for job in self.queued_jobs():
            with self.subTest(job=job["name"]):
                command = job["cmd"]
                # Parsing is safe: no generated training command is executed.
                parsed = subprocess.run(["bash", "-n", "-c", command],
                                        capture_output=True, text=True, timeout=10)
                self.assert_success(parsed)
                lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
                lexer.whitespace_split = True
                tokens = list(lexer)
                self.assertIn(str(self.python), tokens)
                scripts = [Path(token) for token in tokens if token.endswith(".py")]
                self.assertTrue(scripts, command)
                for script in scripts:
                    self.assertTrue(script.is_absolute() and script.is_file(), command)
                    seen_scripts.add(script.relative_to(self.root).as_posix())
                self.assertIn(outputs[job["name"]], tokens, command)
                for i, token in enumerate(tokens):
                    if token == "--out":
                        self.assertEqual(tokens[i + 1], outputs[job["name"]], command)
                    elif token == "cd":
                        self.assertIn(tokens[i + 1], (str(self.work), "/tmp"), command)
                    elif token.startswith("TRAJPE_ROOT="):
                        self.assertEqual(token, f"TRAJPE_ROOT={self.root}", command)
        self.assertEqual(seen_scripts, {
            "baselines/run_ldptrace.py", "baselines/dplora/pipeline.py",
            "scripts/queue_baselines.py", "scripts/run_pe.py",
        })

    def test_strict_queue_requires_historical_hashes_even_when_dry(self):
        self.prepare_placeholders()
        self.assertFalse((self.root / "frozen" / "data_v3_hashes.sha256").exists())
        for extra in ((), ("--dry",)):
            with self.subTest(extra=extra):
                proc = self.run_queue("queue", "--require-frozen", *extra)
                self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn("data_v3_hashes.sha256", proc.stdout + proc.stderr)
                self.assertFalse(self.jobs_file.exists())

    def test_cpu_worker_records_exit_code_with_spaces_in_state_path(self):
        gpuq = self.root / "scripts" / "gpuq.py"
        state = self.work / "gpuq"
        added = subprocess.run(
            [sys.executable, "-I", "-B", str(gpuq), "add", "--state", str(state),
             "--name", "test_noop", "--cmd", "true", "--where", "cpu"],
            env=self.env, capture_output=True, text=True, timeout=10,
        )
        self.assert_success(added)
        worker = subprocess.Popen(
            [sys.executable, "-I", "-B", str(gpuq), "worker", "--state", str(state),
             "--gpu", "cpu", "--poll", "0.05"], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not (state / "done" / "test_noop").exists() and time.monotonic() < deadline:
                if worker.poll() is not None:
                    break
                time.sleep(0.05)
            (state / "STOP").touch()
            output, _ = worker.communicate(timeout=10)
            self.assertEqual(worker.returncode, 0, output)
            self.assertTrue((state / "done" / "test_noop").exists(), output)
            self.assertEqual((state / "exit" / "test_noop").read_text().strip(), "0")
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.communicate(timeout=10)

    def test_unshipped_methods_are_rejected_before_enqueue(self):
        for method in ("hrnet", "privtrace", "adatrace"):
            with self.subTest(method=method):
                proc = self.run_queue("queue", "--dry", "--methods", method)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn("invalid choice", proc.stderr)
                self.assertFalse(self.jobs_file.exists())


if __name__ == "__main__":
    unittest.main()
