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


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    replacement: str
    rationale: str


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
)


def normalize(text: str) -> str:
    """Apply every rule in order; return the Ventus-LLVM-16-parseable text."""
    for rule in RULES:
        text = rule.pattern.sub(rule.replacement, text)
    return text
