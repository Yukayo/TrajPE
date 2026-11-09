from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parent
CTCL = R1.parent
sys.path.insert(0, str(CTCL))
sys.path.insert(0, str(R1))
import ctcl_conf as C
import pipeline_ctcl as PC
import sample_ctcl_r1 as S1

EXIT_BROKEN = PC.EXIT_BROKEN
R1_FILES = ("r1/sample_ctcl_r1.py", "r1/pipeline_ctcl_r1.py")


def r1_code_hashes() -> dict:
    return {f: C.sha256_file(CTCL / f) for f in R1_FILES}


class JobR1(PC.Job):
    def __init__(self, a, runner=None, log=None):
        super().__init__(a, runner=runner, log=log)
        self.failed_r1 = self.w / "texts.failed_r1.json"
        self.failed_v1_kept = self.w / "texts.failed_v1.json"

    def v1_failure(self) -> int | None:
        rec = json.loads(self.failed.read_text())
        verdict = S1.rule_i_from_rounds(rec.get("rounds") or [])
        if verdict is not None and not verdict["passed"]:
            why = (f"r1 rule (i): first-round valid rate {verdict['first_round_valid_rate']:.4g} < {C.MIN_VALID} after "
                   f"{verdict['n_samples']} samples (from the v1 failure record; not re-sampled)")
            C.write_json_atomic(self.failed_r1, {"why": why, "rev": S1.REV, "dataset": self.a.dataset,
                                                 "seed": self.a.seed, "from_v1_record": True, "r1_gate": verdict,
                                                 "v1_why": rec.get("why"), "valid_rate": rec.get("valid_rate"),
                                                 "n_generated": rec.get("n_generated"), "reasons": rec.get("reasons"),
                                                 "rounds": rec.get("rounds"), "extra_rounds": rec.get("extra_rounds")})
            self.note("r1_verdict", fails=True, why=why)
            self.say(why)
            return EXIT_BROKEN
        dst = self.failed_v1_kept
        if dst.exists():
            dst = dst.with_name(f"texts.failed_v1_{PC.stamp()}.json")
        os.replace(self.failed, dst)
        self.note("r1_verdict", fails=False, v1_why=rec.get("why"), r1_gate=verdict, v1_record=dst.name)
        self.say(f"v1 exit 3 ({rec.get('why')}); r1 rule (i) passes on its first-round valid rate "
                 f"{(verdict or {}).get('first_round_valid_rate')}: v1 marker moved to {dst.name}, re-sampling under r1")
        return None

    def do_sample(self) -> int:
        if self.failed_r1.exists():
            self.say(f"{self.failed_r1.name} exists: the sampling failed for good under r1; exit 3 without re-sampling")
            return EXIT_BROKEN
        if self.texts.exists() and self.texts.with_suffix(".json").exists():
            self.say("sampling: texts.parquet exists (reused)")
            return 0
        if self.failed.exists() and self.v1_failure() == EXIT_BROKEN:
            return EXIT_BROKEN
        if self.m2_status() == "consumed":
            raise PC.StateError("weights were deleted but the texts are missing")
        src = ["--adapter", self.wdir] if self.param == "lora" else ["--model_dir", self.wdir]
        rc = self.timed("sample", [C.TRAJPE_PY, "-I", "-B", R1 / "sample_ctcl_r1.py", "--dataset", self.a.dataset]
                        + src + ["--hist", self.hist, "--table", self.table, "--out", self.texts, "--seed", self.a.seed,
                                 "--n_out", self.a.n_out], ok_codes=(0, EXIT_BROKEN))
        return EXIT_BROKEN if rc == EXIT_BROKEN else 0

    def augment(self, rj: Path):
        super().augment(rj)
        side = json.loads(rj.read_text())
        samp = PC.read_json(self.texts.with_suffix(".json")) or {}
        rev = ((samp.get("generation") or {}).get("ctcl") or {}).get("rev", "v1")
        v1rec = sorted(p.name for p in self.w.glob("texts.failed_v1*.json"))
        blk = side["ctcl"]
        blk["spec"] = ("docs/baselines/CTCL_SPEC.md (pre-registration; frozen/ctcl_v1.sha256) + r1 amendment "
                       "(sampler validity gates; frozen/ctcl_v1r1.sha256)")
        blk["sampler_rev"] = rev
        blk["code_sha256"].update(r1_code_hashes())
        blk["r1"] = {"pipeline": "r1/pipeline_ctcl_r1.py", "texts_sampler_rev": rev,
                     "v1_failure_records": v1rec,
                     "note": "r1 changes only the sampler's validity gates (post-processing; no privacy cost)"}
        C.write_json_atomic(rj, side)

    def finish(self):
        self.note("weights_deleted", deleted=True)
        if self.wdir.is_symlink():
            tgt = self.wdir.resolve()
            if tgt.exists():
                shutil.rmtree(tgt)
            self.wdir.unlink()
        elif self.wdir.exists():
            shutil.rmtree(self.wdir)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.rel, self.out)
        os.replace(self.rel.with_suffix(".json"), self.out.with_suffix(".json"))
        self.note("released", out=str(self.out), sampler_rev="r1-pipeline",
                  time_released=time.strftime("%Y-%m-%d %H:%M:%S"))


def main(argv=None, runner=None) -> int:
    return JobR1(PC.parse(argv), runner=runner)()


if __name__ == "__main__":
    sys.exit(main())
