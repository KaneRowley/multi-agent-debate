import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from debate.analysis import analyze
from debate.config import DATASET_REVISION
from debate.dataset import select, validate_manifest
from debate.runner import run
from debate.prompts import revision
from debate.storage import read, save_new


def manifest():
    rows = [{"question": "Question ALPHA", "answer": "PRIVATE LABEL 999"},
            {"question": "Question BETA", "answer": "PRIVATE LABEL 888"}]
    return dict(split="train", dataset_revision=DATASET_REVISION,
                questions=select(rows, "train", 2))


class MockGenerator:
    def __init__(self, stop_at=None, empty=False):
        self.identity = {"model": "mock", "settings": {"temperature": 0.7}}
        self.calls = []
        self.stop_at = stop_at
        self.empty = empty

    def generate(self, messages, seed):
        if self.stop_at is not None and len(self.calls) == self.stop_at:
            raise RuntimeError("simulated interruption")
        self.calls.append((copy.deepcopy(messages), seed))
        content = "" if self.empty else "worked solution, unique call " + str(len(self.calls)) + r" \boxed{2}"
        return dict(raw_response=content, input_tokens=10, output_tokens=5,
                    finish_reason="eos", seconds=1.0, model_call=True)


class RunnerTests(unittest.TestCase):
    def test_histories_snapshots_association_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = manifest()
            generator = MockGenerator()
            run(m, generator, Path(tmp) / "run")
            rows = [read(p) for p in (Path(tmp) / "run/generations").glob("*.json")]
            self.assertEqual(len(rows), 40)  # 2 * (1 + 9 + 10) logical answers
            self.assertEqual(len(generator.calls), 32)  # 5 initials + 6 three-agent + 5 five-agent revisions
            for q in m["questions"]:
                subset = [r for r in rows if r["question_id"] == q["id"] and r["method"] == "debate3"]
                get = lambda r, a: next(x for x in subset if x["round"] == r and x["agent"] == a)
                for r in (1, 2):
                    for a in range(3):
                        request = get(r, a)["request"]["messages"]
                        prior = get(r - 1, a)
                        self.assertEqual(request[:-1], prior["request"]["messages"] +
                                         [{"role": "assistant", "content": prior["raw_response"]}])
                        for b in range(3):
                            if b != a:
                                self.assertIn(get(r - 1, b)["raw_response"], request[-1]["content"])
                                self.assertNotIn(get(r, b)["raw_response"], request[-1]["content"])
            for row in rows:
                content = str(row["request"]["messages"])
                self.assertNotIn("PRIVATE LABEL", content)
                self.assertNotIn("####", content)
                own = next(q for q in m["questions"] if q["id"] == row["question_id"])
                other = next(q for q in m["questions"] if q["id"] != row["question_id"])
                self.assertIn(own["question"], content)
                self.assertNotIn(other["question"], content)

    def test_resume_preserves_records_and_reuses_initial(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run(manifest(), MockGenerator(stop_at=4), out)
            existing = {p.name: p.read_bytes() for p in (out / "generations").glob("*.json")}
            self.assertTrue(existing)
            generator = MockGenerator()
            run(manifest(), generator, out, resume=True)
            self.assertEqual(len(generator.calls), 28)
            for filename, raw in existing.items():
                self.assertEqual((out / "generations" / filename).read_bytes(), raw)
            run(manifest(), MockGenerator(stop_at=0), out, resume=True)
            with self.assertRaises(FileExistsError):
                run(manifest(), MockGenerator(), out)
            with self.assertRaises(ValueError):
                run(manifest(), MockGenerator(), out, repeats=2, resume=True)

    def test_summaries_cost_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            m = manifest()
            run(m, MockGenerator(empty=True), out)
            summary, details = analyze(out, {q["id"]: "#### 2" for q in m["questions"]})
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["actual_total_cost"]["model_calls"], 32)
            self.assertEqual(summary["actual_total_cost"]["input_tokens"], 320)
            self.assertEqual(summary["actual_total_cost"]["logical_responses"], 40)
            self.assertTrue(all(not d["correct"] for d in details))
            stage = next(s for s in summary["per_round"] if s["method"] == "debate3" and s["round"] == 2)
            self.assertEqual(stage["round_cost"]["model_calls"], 6)
            self.assertEqual(stage["cumulative_cost"]["model_calls"], 16)
            self.assertEqual(stage["questions"], 2)

    def test_partial_summary_fixed_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            m = manifest()
            with self.assertRaises(RuntimeError):
                run(m, MockGenerator(stop_at=2), out)
            summary, _ = analyze(out, {q["id"]: "#### 2" for q in m["questions"]})
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["projected_test_3_repeats_gpu_hours"])
            self.assertTrue(all(s["questions"] == 2 for s in summary["per_round"]))

    def test_manifest_selection_and_storage(self):
        rows = [{"question": str(i)} for i in range(50)]
        a = select(rows, "test", 10, ["gsm8k/test/0"])
        self.assertEqual(a, select(rows, "test", 10, ["gsm8k/test/0"]))
        self.assertNotIn("gsm8k/test/0", [q["id"] for q in a])
        m = manifest()
        m["questions"][0]["answer"] = "must be rejected"
        with self.assertRaises(ValueError):
            validate_manifest(m)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "record.json"
            save_new(p, {"a": 1})
            with self.assertRaises(FileExistsError):
                save_new(p, {"a": 2})
            self.assertEqual(read(p), {"a": 1})

    def test_external_initial_reuse_requires_exact_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp) / "prior"
            m = manifest()
            run(m, MockGenerator(), prior)
            generator = MockGenerator()
            out = Path(tmp) / "next"
            run(m, generator, out, reuse_initial=prior)
            self.assertEqual(len(generator.calls), 22)  # revisions only; 10 initial calls reused
            summary, _ = analyze(out, {q["id"]: "#### 2" for q in m["questions"]})
            self.assertEqual(summary["actual_total_cost"]["model_calls"], 22)
            self.assertEqual(summary["distinct_workload_seconds_including_reused"], 32)
            changed = MockGenerator()
            changed.identity["settings"]["temperature"] = 0.8
            run(m, changed, Path(tmp) / "changed", reuse_initial=prior)
            self.assertEqual(len(changed.calls), 32)

    def test_repeat_seeds_and_round_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            m = manifest()
            m["questions"] = m["questions"][:1]
            run(m, MockGenerator(), out, repeats=3)
            rows = [read(p) for p in (out / "generations").glob("*.json")]
            self.assertEqual(len(rows), 60)
            self.assertEqual(sum(r["reused_from"] is None for r in rows), 48)
            for r in rows:
                expected = 404 + r["run"] + 100000 * m["questions"][0]["index"] + 100 * r["agent"] + r["round"]
                self.assertEqual(r["seed"], expected)
                self.assertLessEqual(r["round"], 1 if r["method"] == "debate5" else 2)

    def test_template_slots_do_not_modify_question_or_peer_text(self):
        text = revision("question with literal {peer_blocks}", ("own", "peer literal {question}"), 0)
        self.assertIn("question with literal {peer_blocks}", text)
        self.assertIn("peer literal {question}", text)


if __name__ == "__main__":
    unittest.main()
