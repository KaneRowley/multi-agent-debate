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
        limit = self.generation_config.max_new_tokens
        self.token_limits.append(limit)
        result = super().generate(messages, seed)
        if limit == 256:
            result["raw_response"] = f"2\nMock calculation for seed {seed}"
        return result


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
            self.assertEqual(read(output / "run.json")["contract"]["summary_mode"], "per-recipient")

    def test_per_answer_shares_three_summaries_and_resumes_only_missing_revisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "summarisation-dev-baseline"
            output = root / "dev-pilot-per-answer"
            references = root / "references.json"
            questions = manifest()
            run(questions, FollowUpGenerator(), source)
            save_new(references, {q["id"]: "#### 2" for q in questions["questions"]})
            argv = ["summarisation_followup", "--source", str(source), "--output", str(output),
                    "--references", str(references), "--dev-limit", "1"]
            per_answer = ["--summary-mode", "per-answer"]

            interrupted = FollowUpGenerator(stop_at=4)
            with (patch.object(sys, "argv", argv + per_answer),
                  patch.object(summarisation_followup, "Generator", return_value=interrupted)):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    summarisation_followup.main()
            self.assertEqual(interrupted.token_limits, [256, 256, 256, 1024, 1024])
            checkpoints = output / "checkpoints"
            saved = {path: path.read_bytes() for path in checkpoints.glob("*.json")}
            self.assertEqual(len(saved), 4)
            self.assertFalse((output / "comparison.json").exists())
            contract = read(output / "run.json")["contract"]
            self.assertEqual(contract["summary_mode"], "per-answer")
            self.assertEqual(contract["summary_tokens"], 256)

            with (patch.object(sys, "argv", argv + ["--resume"]),
                  patch.object(summarisation_followup, "Generator") as generator):
                with self.assertRaisesRegex(ValueError, "Run inputs changed"):
                    summarisation_followup.main()
                generator.assert_not_called()

            resumed = FollowUpGenerator()
            with (patch.object(sys, "argv", argv + per_answer + ["--resume"]),
                  patch.object(summarisation_followup, "Generator", return_value=resumed)):
                summarisation_followup.main()
            self.assertEqual(resumed.token_limits, [1024, 1024])
            self.assertEqual(len(list(checkpoints.glob("*.json"))), 6)
            self.assertEqual({path: path.read_bytes() for path in saved}, saved)

            index = questions["questions"][0]["index"]
            initial = [read(source / "generations" / f"run0-debate3-q{index}-r0-a{a}.json")["raw_response"]
                       for a in range(3)]
            baseline = [read(source / "generations" / f"run0-debate3-q{index}-r1-a{a}.json")
                        for a in range(3)]
            summaries = [read(checkpoints / f"run0-q{index}-a{a}-answer-summary.json")["result"]["raw_response"]
                         for a in range(3)]
            summary_calls = interrupted.calls[:3]
            revision_calls = interrupted.calls[3:] + resumed.calls
            self.assertEqual((len(summary_calls), len(revision_calls)), (3, 3))
            for agent, (summary_call, revision_call) in enumerate(zip(summary_calls, revision_calls)):
                summary_messages, _ = summary_call
                revision_messages, seed = revision_call
                expected_summary_prompt = (
                    f"{summarisation_followup.PER_ANSWER_INSTRUCTION}\n\n"
                    f"<candidate>\n{initial[agent]}\n</candidate>"
                )
                self.assertEqual(summary_messages[-1]["content"], expected_summary_prompt)
                self.assertEqual(revision_messages[:-1], baseline[agent]["request"]["messages"][:-1])
                self.assertEqual(revision_messages[-2], {"role": "assistant", "content": initial[agent]})
                self.assertEqual(seed, baseline[agent]["seed"])
                prompt = revision_messages[-1]["content"]
                for other in range(3):
                    self.assertNotIn(initial[other], prompt)
                    block = f"Agent {other}:\n<candidate>\n{summaries[other]}\n</candidate>"
                    if other == agent:
                        self.assertNotIn(summaries[other], prompt)
                    else:
                        self.assertIn(block, prompt)

            report = read(output / "comparison.json")["summary"]
            self.assertEqual(report["summary_calls_cost"], {"input_tokens": 30, "output_tokens": 15})
            self.assertEqual(report["summarised"]["input_tokens"], 90)
            self.assertEqual(report["summarised"]["output_tokens"], 45)

            uninterrupted = FollowUpGenerator()
            fresh_args = argv.copy()
            fresh_args[fresh_args.index("--output") + 1] = str(root / "uninterrupted")
            with (patch.object(sys, "argv", fresh_args + per_answer),
                  patch.object(summarisation_followup, "Generator", return_value=uninterrupted)):
                summarisation_followup.main()
            self.assertEqual(uninterrupted.token_limits, [256, 256, 256, 1024, 1024, 1024])
            self.assertEqual(uninterrupted.calls, summary_calls + revision_calls)
