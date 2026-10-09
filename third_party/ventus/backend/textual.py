"""Normalize consumer-side LLVM IR text for the pinned Ventus LLVM 16 parser.

The Triton-side LLVM and the Ventus toolchain are different revisions, so the
text the consumer emits can contain features the Ventus parser predates. Every
such feature gets one `Rule` here, with the reason it is safe, and `normalize()`
applies them in order. This replaces the ad-hoc string edits in `compiler.py`
and is covered by `test_textual_normalizer.py` plus the end-to-end sentinel
`test_textual_boundary.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    replacement: str | Callable[[re.Match], str]
    rationale: str


# Bare special float literals (`-inf`/`+inf`/`nan` with no type on the literal)
# only appear where the instruction carries the type once, e.g. a phi:
#   %19 = phi float [ %17, %16 ], [ -inf, %3 ]
# The lexer wants a bit pattern of the instruction's type, so infer float vs
# double from the first type token on the line. A line with both types (none is
# expected for a masked load's `other`) is left for the typed rules below.
_BARE_SPECIAL = re.compile(r"(?<![\w.])[+-]?inf\b|(?<![\w.])nan\b")
# LLVM textual float/double constants are written as the 64-bit IEEE-754 bit
# pattern regardless of the operand's type (a 32-bit pattern is rejected as
# "floating point constant invalid for type"). So both tables use the double
# encoding; only NaN quietness differs and that also matches the double form.
_SPECIAL_BITS = {
    "-inf": "0xFFF0000000000000",
    "+inf": "0x7FF0000000000000",
    "inf": "0x7FF0000000000000",
    "nan": "0x7FF8000000000000",
}
_FLOAT_BITS = _SPECIAL_BITS
_DOUBLE_BITS = _SPECIAL_BITS


def _rewrite_bare_special(match: re.Match) -> str:
    line = match.string
    line_start = line.rfind("\n", 0, match.start()) + 1
    line_end = line.find("\n", match.end())
    if line_end == -1:
        line_end = len(line)
    line_text = line[line_start:line_end]
    is_float = re.search(r"\bfloat\b", line_text) is not None
    is_double = re.search(r"\bdouble\b", line_text) is not None
    if is_float == is_double:  # neither, or ambiguous: leave it to be caught loudly
        return match.group(0)
    table = _FLOAT_BITS if is_float else _DOUBLE_BITS
    return table[match.group(0)]


RULES = (
    Rule(
        name="or_disjoint",
        pattern=re.compile(r"\bor\s+disjoint\s+"),
        replacement="or ",
        rationale="`disjoint` is an LLVM 17 keyword and a promise about the operands, not part of the result; the LLVM 16 parser rejects it.",
    ),
    Rule(
        name="attr_nocreateundeforpoison",
        pattern=re.compile(r"\bnocreateundeforpoison\s+"),
        replacement="",
        rationale="A newer function attribute keyword the LLVM 16 lexer rejects; dropping it only forgoes an optimization hint.",
    ),
    Rule(
        name="float_neg_inf",
        pattern=re.compile(r"\bfloat\s+-inf\b"),
        replacement="float 0xFFF0000000000000",
        rationale="The consumer prints -inf; the LLVM 16 lexer wants the 64-bit IEEE-754 bit pattern.",
    ),
    Rule(
        name="float_pos_inf",
        pattern=re.compile(r"\bfloat\s+\+?inf\b"),
        replacement="float 0x7FF0000000000000",
        rationale="Same as float_neg_inf, for +inf.",
    ),
    Rule(
        name="float_nan",
        pattern=re.compile(r"\bfloat\s+nan\b"),
        replacement="float 0x7FF8000000000000",
        rationale="Same as float_neg_inf, for nan (quiet NaN).",
    ),
    Rule(
        name="double_neg_inf",
        pattern=re.compile(r"\bdouble\s+-inf\b"),
        replacement="double 0xFFF0000000000000",
        rationale="Same as float_neg_inf, for double.",
    ),
    Rule(
        name="double_pos_inf",
        pattern=re.compile(r"\bdouble\s+\+?inf\b"),
        replacement="double 0x7FF0000000000000",
        rationale="Same as float_neg_inf, for double +inf.",
    ),
    Rule(
        name="double_nan",
        pattern=re.compile(r"\bdouble\s+nan\b"),
        replacement="double 0x7FF8000000000000",
        rationale="Same as float_neg_inf, for double nan.",
    ),
    Rule(
        name="bare_special_float",
        pattern=_BARE_SPECIAL,
        replacement=_rewrite_bare_special,
        rationale="A phi/select carries the type once, so a bare -inf/+inf/nan needs its bit pattern "
        "inferred from the instruction's type token.",
    ),
)


def normalize(text: str) -> str:
    """Apply every rule in order; return the Ventus-LLVM-16-parseable text."""
    for rule in RULES:
        if callable(rule.replacement):
            text = rule.pattern.sub(rule.replacement, text)
        else:
            text = rule.pattern.sub(rule.replacement, text)
    return text
