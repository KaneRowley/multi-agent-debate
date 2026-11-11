import unittest
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from debate.scoring import extract, number, plurality, reference


class ScoringTests(unittest.TestCase):
    def test_restored_actual_prediction_can_make_group_abstain(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/dev_smoke_vote_regression.json").read_text(encoding="utf-8"))
        predictions = []
        for response in fixture["responses"]:
            prediction = extract(response["raw_response"]).value
            expected = response["expected_prediction"]
            self.assertEqual(prediction, Fraction(expected) if expected is not None else None)
            predictions.append(prediction)
        self.assertEqual(plurality(predictions), (None, "tie"))

    def test_actual_rejected_smoke_responses(self):
        fixtures = json.loads((Path(__file__).parent / "fixtures/dev_smoke_rejected.json").read_text(encoding="utf-8"))
        self.assertEqual(len(fixtures["fixtures"]), 45)
        for fixture in fixtures["fixtures"]:
            with self.subTest(records=fixture["source_records"]):
                text = fixture["raw_response"]
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), fixture["response_sha256"])
                result = extract(text)
                expected = fixture["expected_prediction"]
                self.assertEqual(result.value, Fraction(expected) if expected is not None else None)

    def test_wrapped_boxes_and_introductions(self):
        cases = [r"The final numerical answer is \(\boxed{47}\).",
                 "Therefore, the final answer is:\n  \\[\n  \\boxed{23}\n  \\]",
                 "**Final Answer:**\n\\[\n\\boxed{23}\n\\]",
                 r"Final answer: $\boxed{23}$.", r"Final answer: $$\boxed{23}$$.",
                 r"Answer: \( \boxed{23} \)", r"Answer is: \boxed{23}"]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(extract(text).value, 47 if "47" in text else 23)
                self.assertTrue(extract(text).format_compliant)

    def test_format_is_separate_from_numerical_validity(self):
        cases = [(r"\(\boxed{47}\).", 47, True),
                 (r"\[\boxed{1,000}\]", 1000, False),
                 (r"\(\boxed{1/2}\)", Fraction(1, 2), False),
                 (r"\boxed{47} minutes.", 47, False),
                 (r"\boxed{800} dollars for 20 days.", 800, False),
                 (r"\boxed{47}\)", 47, False),
                 (r"\boxed{47} and again \boxed{47}", 47, False)]
        for text, expected, compliant in cases:
            with self.subTest(text=text):
                self.assertEqual(extract(text).value, expected)
                self.assertEqual(extract(text).format_compliant, compliant)

    def test_wrappers_do_not_hide_ambiguity_or_conflicts(self):
        cases = [r"Answer: \(\boxed{47}\). Final answer: 48",
                 r"Answer: \[\boxed{47}\] or 48", r"\(\boxed{47}\), 48",
                 r"\(\boxed{47}\) and 48", r"\(\boxed{47}\) + 48",
                 r"\(\boxed{47}\) then 48", r"\(\boxed{47}\) 48",
                 r"\(\boxed{1,00}\)", r"\[\boxed{2/0}\]", r"\(\boxed{2+3}\)",
                 r"\boxed{47} Answer is NaN", "Answer: 2*3", "Answer: 2**3",
                 r"Reasoning: 2+3=5, then 7+8=15", r"\(47\) and \[48\]"]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(extract(text).value)
    def test_exact_equivalence(self):
        for text in ("1000", "+1,000.00", "2000/2", r"\frac{2000}{2}"):
            self.assertEqual(number(text), Fraction(1000))
        self.assertEqual(number("-.5"), Fraction(-1, 2))
        self.assertEqual(number("-0.00"), 0)
        self.assertEqual(number("1/3"), Fraction(1, 3))

    def test_markers_and_trailing_words(self):
        for text in (r"\boxed{1,000} dollars.", "FINAL ANSWER: 1000 dollars.",
                     "Answer is 2000/2", r"\boxed{\frac{2000}{2}}"):
            self.assertEqual(extract(text).value, 1000, text)
        self.assertTrue(extract(r"work 2+2=4. \boxed{4}").format_compliant)
        self.assertFalse(extract(r"\boxed{4} dollars.").format_compliant)

    def test_ambiguity_conflicts_and_empty(self):
        invalid = ["", "work 2+3=5", "The values are 2 and 3", "FINAL ANSWER: 2 or 3",
                   "Answer: 2, 3", "Answer: 2/0", "Answer: 1,00", "Answer: NaN",
                   r"\boxed{2} or 3", r"\boxed{2}\boxed{3}",
                   r"Answer: 2 then 3", r"\boxed{2+3}", r"\boxed{2",
                   "FINAL ANSWER: 2\nFINAL ANSWER: 3", r"\boxed{2}" + "\nAnswer: 3"]
        for text in invalid:
            self.assertIsNone(extract(text).value, text)
        self.assertEqual(extract("Answer: 2\nAnswer: 2.0").value, 2)

    def test_plurality_and_labels(self):
        self.assertEqual(plurality([2, 2, 3, 4, None]), (2, "winner"))
        self.assertEqual(plurality([2, 3, None]), (None, "tie"))
        self.assertEqual(plurality([None, None]), (None, "no_valid_predictions"))
        self.assertEqual(plurality([2, None, None]), (2, "winner"))
        self.assertEqual(reference("private explanation\n#### 1,000"), 1000)
