"""Conservative explicit-answer extraction; no last-number-in-reasoning fallback."""
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

PARSER_VERSION = "explicit-rational-v1"
NUMBER = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
RATIONAL = rf"{NUMBER}(?:\s*/\s*{NUMBER})?"
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
MARKER = re.compile(r"(?im)\b(?:final\s+answer|answer)\s*(?::|=|\bis\b)\s*([^\n]+)")


def number(text):
    text = text.strip()
    latex = re.fullmatch(r"([+-]?)\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", text)
    if latex:
        sign, numerator, denominator = latex.groups()
        return (-1 if sign == "-" else 1) * number(numerator) / number(denominator)
    if not re.fullmatch(RATIONAL, text):
        raise ValueError("Not one numerical value")
    parts = text.replace(",", "").split("/")
    return Fraction(parts[0].strip()) / (Fraction(parts[1].strip()) if len(parts) == 2 else 1)


@dataclass(frozen=True)
class Prediction:
    value: Fraction | None
    status: str
    format_compliant: bool


def extract(text):
    if not text.strip():
        return Prediction(None, "empty", False)
    boxes = list(BOX.finditer(text))
    # A malformed box is never repaired by grabbing an internal number.
    if text.count(r"\boxed") != len(boxes):
        return Prediction(None, "malformed_box", False)
    values = []
    try:
        values.extend(number(match.group(1)) for match in boxes)
        if boxes and re.search(r"[0-9]|\bor\b", text[boxes[-1].end():], re.I):
            raise ValueError("Ambiguous text after boxed answer")
        # Mask boxes before finding other explicit claims, but retain line positions.
        masked = BOX.sub("BOXVALUE", text)
        for match in MARKER.finditer(masked):
            claim = match.group(1).strip()
            if claim.startswith("BOXVALUE"):
                claim = claim[len("BOXVALUE"):].strip()
                # A box may be followed by words, but not competing numbers.
                if re.search(r"[0-9]", claim):
                    raise ValueError("Multiple numbers in explicit claim")
                continue
            claim = claim.strip("*$` ")
            # Permit prose suffixes/units after a clearly marked numeric answer.
            numeric = re.match(rf"^({RATIONAL})(?=$|[\s.,;!$*`])", claim)
            if not numeric:
                raise ValueError("Explicit answer is not numerical")
            suffix = claim[numeric.end():]
            if re.search(r"[0-9]|[+*/=]|\bor\b", suffix, re.I):
                raise ValueError("Ambiguous explicit answer")
            values.append(number(numeric.group(1)))
    except (ValueError, ZeroDivisionError):
        return Prediction(None, "ambiguous_or_invalid", False)
    if not values:
        # A response consisting entirely of one number is unambiguous too.
        try:
            return Prediction(number(text), "valid", False)
        except (ValueError, ZeroDivisionError):
            return Prediction(None, "no_explicit_answer", False)
    if len(set(values)) != 1:
        return Prediction(None, "conflicting_answers", False)
    compliant = (len(boxes) == 1 and bool(re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", boxes[0].group(1).strip()))
                 and not text[boxes[0].end():].strip())
    return Prediction(values[0], "valid", compliant)


def plurality(values):
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None, "no_valid_predictions"
    winners = [value for value, count in counts.items() if count == max(counts.values())]
    return (winners[0], "winner") if len(winners) == 1 else (None, "tie")


def reference(text):
    if text.count("####") != 1:
        raise ValueError("Invalid GSM8K reference")
    return number(text.split("####")[1])
