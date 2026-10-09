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


def test_rules_are_documented():
    assert textual.RULES, "the normalizer must have at least one rule"
    for rule in textual.RULES:
        assert rule.name and rule.rationale
