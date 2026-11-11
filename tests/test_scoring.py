import unittest
from fractions import Fraction

from debate.scoring import extract, number, plurality, reference


class ScoringTests(unittest.TestCase):
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
