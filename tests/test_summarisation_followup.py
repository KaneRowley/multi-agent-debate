import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from debate import summarisation_followup
from debate.runner import run
from debate.storage import read, save_new
from test_runner import MockGenerator, manifest


class FollowUpGenerator(MockGenerator):
    def __init__(self, stop_at=None):
        super().__init__(stop_at=stop_at)
        self.identity.update(
            model_revision="mock", tokenizer_revision="mock", chat_template="mock",
            effective_generation={"max_new_tokens": 1024}, precision="float16", context_limit=32768,
        )
        self.generation_config = SimpleNamespace(max_new_tokens=1024)
        self.token_limits = []

    def generate(self, messages, seed):
        self.token_limits.append(self.generation_config.max_new_tokens)
        return super().generate(messages, seed)


class SummarisationFollowUpTests(unittest.TestCase):
    def test_resume_reuses_summary_saved_before_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output, references = (root / name for name in ("baseline", "output", "references.json"))
            questions = manifest()
            run(questions, FollowUpGenerator(), source)
            save_new(references, {q["id"]: "#### 2" for q in questions["questions"]})
            argv = ["summarisation_followup", "--source", str(source), "--output", str(output),
                    "--references", str(references), "--dev-limit", "1"]

            interrupted = FollowUpGenerator(stop_at=1)
            with patch.object(sys, "argv", argv), patch.object(summarisation_followup, "Generator", return_value=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    summarisation_followup.main()
            self.assertEqual(interrupted.token_limits, [256, 1024])
            checkpoints = output / "checkpoints"
            summary_path, = checkpoints.glob("*.json")
            self.assertTrue(summary_path.name.endswith("-a0-summary.json"))
            self.assertFalse((output / "comparison.json").exists())
            saved_bytes = summary_path.read_bytes()
            saved_summary = read(summary_path)

            resumed = FollowUpGenerator()
            with patch.object(sys, "argv", argv + ["--resume"]), patch.object(summarisation_followup, "Generator", return_value=resumed):
                summarisation_followup.main()
            self.assertEqual(resumed.token_limits, [1024, 256, 1024, 256, 1024])
            self.assertEqual(summary_path.read_bytes(), saved_bytes)
            first_messages, first_seed = resumed.calls[0]
            self.assertIn(saved_summary["result"]["raw_response"], first_messages[-1]["content"])
            self.assertEqual(first_seed, saved_summary["request"]["seed"] - 1_000_000_000)
            self.assertEqual(len(list(checkpoints.glob("*.json"))), 6)
            report = read(output / "comparison.json")["summary"]
            self.assertEqual(report["parser_version"], "explicit-rational-v3")
            self.assertEqual(report["question_repeat_pairs"], 1)
            self.assertEqual(report["full"]["accuracy"], 1)
            self.assertEqual(report["summarised"]["accuracy"], 1)
