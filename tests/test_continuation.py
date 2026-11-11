import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from debate import prompts
from debate.analysis import analyze
from debate.config import DATASET_REVISION
from debate.dataset import select
from debate.runner import run
from debate.storage import digest, read
from test_runner import MockGenerator


def development(count=30):
    return dict(split="train", dataset_revision=DATASET_REVISION,
                questions=select([{"question": f"Unique development problem [{i}]"}
                                  for i in range(count)], "train", count))


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class ContinuationTests(unittest.TestCase):
    def test_ten_to_thirty_preserves_all_rounds_and_only_generates_twenty(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "smoke", Path(tmp) / "dev30"
            manifest = development()
            run(manifest, MockGenerator(), source, limit=10)
            before = hashes(source)
            disabled = MockGenerator(stop_at=0)
            prepared = run(manifest, disabled, output, limit=30,
                           continue_from=source, prepare_only=True)
            self.assertEqual(len(disabled.calls), 0)
            self.assertEqual(prepared["saved_responses"], 200)
            self.assertEqual(prepared["missing_responses"], 400)
            self.assertEqual(hashes(source), before)
            inherited = hashes(output / "generations")
            for path in (output / "generations").glob("*.json"):
                row = read(path)
                original = read(source / "generations" / row["source_record"])
                self.assertEqual(row["reuse_kind"], "continuation")
                self.assertEqual(row["source_record_sha256"], before["generations/" + row["source_record"]])
                self.assertEqual(row["request"], original["request"])
                for field in original["result_fields"]:
                    self.assertEqual(row[field], original[field])
            generator = MockGenerator()
            completed = run(manifest, generator, output, limit=30, resume=True, continue_from=source)
            self.assertEqual(completed["missing_responses"], 0)
            self.assertEqual(len(generator.calls), 320)
            smoke_initials = {prompts.initial(q["question"])[1]["content"] for q in manifest["questions"][:10]}
            self.assertTrue(all(messages[1]["content"] not in smoke_initials for messages, _ in generator.calls))
            self.assertEqual(hashes(source), before)
            after_generations = hashes(output / "generations")
            for name, checksum in inherited.items():
                self.assertEqual(after_generations[name], checksum)
            refs = {q["id"]: "#### 2" for q in manifest["questions"]}
            summary, _ = analyze(output, refs)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["parser_version"], "explicit-rational-v2")
            self.assertEqual(summary["actual_total_cost"]["logical_responses"], 600)
            self.assertEqual(summary["actual_total_cost"]["model_calls"], 320)
            self.assertEqual(summary["actual_total_cost"]["generation_seconds"], 320)
            self.assertEqual(summary["carried_forward_cost"]["model_calls"], 160)
            self.assertEqual(summary["distinct_workload_cost_including_reused"]["model_calls"], 480)
            self.assertEqual(summary["distinct_workload_seconds_including_reused"], 480)
            run(manifest, MockGenerator(stop_at=0), output, limit=30, resume=True, continue_from=source)

    def test_parser_only_audit_hash_does_not_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            manifest = development(2)
            run(manifest, MockGenerator(), out)
            metadata_path = out / "run.json"
            metadata = read(metadata_path)
            metadata["source_sha256"] = "previous parser/analysis source snapshot"
            metadata.pop("continuation")  # legacy smoke metadata predates this field
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            before = hashes(out)
            run(manifest, MockGenerator(stop_at=0), out, resume=True)
            self.assertEqual(hashes(out), before)

    def test_changed_runtime_prompt_or_manifest_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "smoke"
            manifest = development(2)
            run(manifest, MockGenerator(), source)
            for difference in ({"model": "different model"}, {"settings": {"temperature": 0.8}},
                               {"gpu": "different GPU"}, {"precision": "bfloat16"},
                               {"packages": {"torch": "different version"}}):
                changed = MockGenerator(stop_at=0)
                changed.identity.update(difference)
                with self.subTest(difference=difference):
                    with self.assertRaisesRegex(ValueError, "model/runtime/settings"):
                        run(manifest, changed, Path(tmp) / "changed-runtime", continue_from=source)
            with patch.object(prompts, "REVISION", prompts.REVISION + "changed"):
                with self.assertRaisesRegex(ValueError, "revision prompt"):
                    run(manifest, MockGenerator(stop_at=0), Path(tmp) / "changed-prompt", continue_from=source)
            different = development(2)
            different["selection_version"] = "different manifest"
            with self.assertRaisesRegex(ValueError, "manifest"):
                run(different, MockGenerator(stop_at=0), Path(tmp) / "changed-manifest", continue_from=source)

    def test_tampered_revision_and_incomplete_source_are_not_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "smoke"
            manifest = development(2)
            run(manifest, MockGenerator(), source)
            path = next((source / "generations").glob("*-debate3-*-r1-a0.json"))
            original = path.read_bytes()
            row = read(path)
            row["request"]["messages"][-1]["content"] += " altered peer response"
            row["request_key"] = digest(row["request"])
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "request/history mismatch"):
                run(manifest, MockGenerator(stop_at=0), Path(tmp) / "invalid", continue_from=source)
            path.write_bytes(original)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                run(manifest, MockGenerator(stop_at=0), Path(tmp) / "incomplete", continue_from=source)

    def test_completed_empty_responses_and_source_directory_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "smoke"
            manifest = development(2)
            run(manifest, MockGenerator(empty=True), source)
            before = hashes(source)
            out = Path(tmp) / "continued"
            run(manifest, MockGenerator(stop_at=0), out, continue_from=source, prepare_only=True)
            self.assertEqual(hashes(source), before)
            self.assertTrue(all(read(path)["raw_response"] == "" for path in (out / "generations").glob("*.json")))
            with self.assertRaisesRegex(ValueError, "must differ"):
                run(manifest, MockGenerator(stop_at=0), source, continue_from=source, resume=True)


if __name__ == "__main__":
    unittest.main()
