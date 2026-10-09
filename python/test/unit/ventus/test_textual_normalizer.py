"""Unit tests for the consumer-side textual IR normalizer (pure Python)."""

from triton.backends.ventus import textual


def test_or_disjoint_is_stripped():
    assert textual.normalize("%x = or disjoint i32 %a, %b\n") == "%x = or i32 %a, %b\n"


def test_plain_or_is_untouched():
    assert textual.normalize("%x = or i32 %a, %b\n") == "%x = or i32 %a, %b\n"


def test_newer_function_attribute_keywords_are_dropped():
    src = "attributes #2 = { nocallback nocreateundeforpoison nofree nosync }\n"
    out = textual.normalize(src)
    assert "nocreateundeforpoison" not in out
    assert "nocallback nofree nosync" in out


def test_special_float_literals_become_hex():
    # The consumer prints bare `-inf`/`+inf`/`nan` inside a typed instruction
    # (the phi's type precedes the literal); the LLVM 16 lexer wants the 64-bit
    # IEEE-754 bit pattern of that value.
    assert textual.normalize("%19 = phi float [ %17, %16 ], [ -inf, %3 ]\n") == \
        "%19 = phi float [ %17, %16 ], [ 0xFFF0000000000000, %3 ]\n"
    assert "0x7FF0000000000000" in textual.normalize("fadd float %a, +inf\n")
    assert "0x7FF8000000000000" in textual.normalize("fadd float %a, nan\n")
    assert "0xFFF0000000000000" in textual.normalize("%x = phi double [ -inf, %c ]\n")


def test_typed_special_float_literals_become_hex():
    assert textual.normalize("fadd float %a, -inf\n") == "fadd float %a, 0xFFF0000000000000\n"
    assert "0x7FF0000000000000" in textual.normalize("fadd double %a, +inf\n")


def test_rules_are_documented():
    assert textual.RULES, "the normalizer must have at least one rule"
    for rule in textual.RULES:
        assert rule.name and rule.rationale
