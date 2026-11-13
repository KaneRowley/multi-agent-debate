"""Conservative explicit-answer extraction; no last-number-in-reasoning fallback."""
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

PARSER_VERSION = "explicit-rational-v3"
NUMBER = r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
RATIONAL = rf"{NUMBER}(?:\s*/\s*{NUMBER})?"
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
MARKER = re.compile(r"(?im)\b(?:final\s+(?:numerical\s+)?answer|answer)\s*(?::|=|\bis\b\s*:?)\s*(?:(?:\*\*|__|`)\s*)?")
MATH_WRAPPERS = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))


def box_span(text, box):
    """Include a matching standard math wrapper around this box, if present."""
    for opening, closing in MATH_WRAPPERS:
        before = re.search(re.escape(opening) + r"\s*$", text[:box.start()])
        after = re.match(r"\s*" + re.escape(closing), text[box.end():])
        if before and after:
            return before.start(), box.end() + after.end()
    return box.start(), box.end()


def ambiguous_box_suffix(suffix):
    # Numbers in ordinary trailing prose are not additional predictions. Reject
    # direct alternatives/arithmetic instead; explicit answer markers are checked
    # independently below, wherever they occur in the response.
    return bool(re.match(rf"\s*(?:{RATIONAL}|(?:[,/+*=<>-]|and\b|then\b|equals\b)\s*{RATIONAL})", suffix, re.I)
                or re.search(r"\bor\b", suffix, re.I))


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
        spans = [box_span(text, box) for box in boxes]
        if boxes and ambiguous_box_suffix(text[spans[-1][1]:]):
            raise ValueError("Ambiguous text after boxed answer")
        # Mask entire wrapped boxes so an introduction may end on another line.
        # The sentinel must not be confused with literal text in a response.
        sentinel = "BOXVALUE"
        while sentinel in text:
            sentinel += "_"
        masked = text
        for start, end in reversed(spans):
            masked = masked[:start] + sentinel + masked[end:]
        for match in MARKER.finditer(masked):
            # A standalone heading labels the following prose; it is not a
            # separate nonnumeric claim. Only skip it when a later box exists.
            line_start = masked.rfind("\n", 0, match.start()) + 1
            line_end = masked.find("\n", match.start())
            heading = masked[line_start:line_end] if line_end >= 0 else ""
            following_line = masked[match.end():].split("\n", 1)[0].strip().strip("*$` ")
            if (sentinel in following_line and not re.match(RATIONAL, following_line)
                    and re.fullmatch(
                    r"[ \t]*(?:\#{1,6}[ \t]+|[0-9]+\.[ \t]+|[-+*][ \t]+)?"
                    r"(?:\*\*|__|`)?(?:final[ \t]+(?:numerical[ \t]+)?answer|answer)"
                    r"[ \t]*:(?:\*\*|__|`)?[ \t]*", heading, re.I)):
                continue
            claim = masked[match.end():].split("\n", 1)[0].strip().strip("*$` ")
            # A bounded introduction may qualify the immediately following box.
            # Do not skip arbitrary text or search it for an internal number.
            if re.search(r"\bis\b", match.group(), re.I):
                claim = re.sub(r"^indeed[ \t]+(?=" + re.escape(sentinel) + r")", "", claim, flags=re.I)
            if claim.startswith(sentinel):
                claim = claim[len(sentinel):].strip()
                if ambiguous_box_suffix(claim):
                    raise ValueError("Ambiguous boxed claim")
                continue
            # These describe an answer rather than asserting another value, e.g.
            # "the final answer is accurate:" before a later displayed box.
            if boxes and re.search(r"\bis\b", match.group(), re.I) and (re.match(
                    r"(?:accurate|consistent|derived)\b", claim, re.I)
                    or re.fullmatch(r"correct[.!:]?", claim, re.I)):
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
                 and bool(re.fullmatch(r"\s*[.!]?\s*", text[spans[0][1]:])))
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
