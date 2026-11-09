import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "baselines" / "ctcl" / "tests" / "run_tests.py"
PUBLIC_TESTS = ("test_accounting.py", "test_sample.py", "test_pipeline.py", "test_train.py", "test_release.py")


class TestCTCLRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ctcl_runner_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "trajpe").mkdir()
        (self.root / "trajpe" / "__init__.py").write_text("")
        self.tests = self.root / "baselines" / "ctcl" / "tests"
        self.tests.mkdir(parents=True)
        self.runner = self.tests / "run_tests.py"
        shutil.copy2(RUNNER, self.runner)
        (self.tests.parent / "helper.py").write_text("VALUE = 1\n")
        for name in PUBLIC_TESTS:
            (self.tests / name).write_text("print('test completed')\n")
        self.env = dict(os.environ, TRAJPE_ROOT=str(self.root), TRAJPE_PYTHON=sys.executable,
                        TRAJPE_PYTHON_DPLORA=sys.executable, PYTHONDONTWRITEBYTECODE="1")

    def run_runner(self, *args):
        proc = subprocess.run([sys.executable, "-I", "-B", str(self.runner), *args],
                              env=self.env, cwd=self.root, capture_output=True, text=True, timeout=30)
        path = self.tests / "RESULTS.json"
        result = json.loads(path.read_text()) if path.exists() else None
        return proc, result

    def test_full_suite_without_internal_queue_files(self):
        proc, result = self.run_runner()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(result["all_passed"])
        self.assertTrue(result["full_suite"])
        self.assertEqual([r["test"] for r in result["tests"]], list(PUBLIC_TESTS))
        self.assertTrue(result["code_unchanged_during_run"])
        self.assertIn("baselines/ctcl/helper.py", result["code_sha256"])

    def test_successful_subset_is_not_a_full_suite(self):
        proc, result = self.run_runner("--only", "test_pipeline.py")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(result["all_passed"])
        self.assertFalse(result["full_suite"])
        self.assertEqual([r["test"] for r in result["tests"]], ["test_pipeline.py"])

    def test_explicit_complete_selection_records_full_suite(self):
        proc, result = self.run_runner("--only", *PUBLIC_TESTS)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(result["full_suite"])

    def test_unknown_and_empty_selection_are_rejected(self):
        for args in (("--only",), ("--only", "test_queue.py")):
            with self.subTest(args=args):
                proc, result = self.run_runner(*args)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIsNone(result)

    def test_missing_required_test_is_not_skipped(self):
        (self.tests / "test_sample.py").unlink()
        proc, result = self.run_runner()
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("missing required test files: test_sample.py", proc.stderr)
        self.assertIsNone(result)

    def test_failed_selected_test_returns_failure(self):
        (self.tests / "test_pipeline.py").write_text("raise SystemExit(7)\n")
        proc, result = self.run_runner("--only", "test_pipeline.py")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["tests"][0]["rc"], 7)

    def test_source_changes_fail_even_when_test_exits_zero(self):
        (self.tests / "test_pipeline.py").write_text(
            "from pathlib import Path\n"
            "(Path(__file__).resolve().parent.parent / 'helper.py').write_text('VALUE = 2\\n')\n")
        proc, result = self.run_runner("--only", "test_pipeline.py")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertFalse(result["all_passed"])
        self.assertFalse(result["code_unchanged_during_run"])
        self.assertEqual(result["tests"][0]["rc"], 0)


if __name__ == "__main__":
    unittest.main()
