"""Unit tests for nlp_brain.text_utils -- pure functions, no model needed.

Use these as a template for adding new gibberish/cleanup rules: pick a
concrete input you want handled differently, add it to the parametrized
list below, run `pytest tests/test_text_utils.py -v`, and let the failure
guide the implementation change.
"""
import pytest

from nlp_brain.text_utils import clean_text, is_gibberish


class TestCleanText:
    def test_collapses_internal_whitespace(self):
        assert clean_text("dil   mein\tdard  hai") == "dil mein dard hai"

    def test_strips_leading_and_trailing_whitespace(self):
        assert clean_text("  dant mein dard hai  ") == "dant mein dard hai"

    def test_preserves_case_and_punctuation(self):
        # Case/punctuation carry signal here (e.g. "BP", "CT scan") --
        # clean_text must NOT lowercase or strip it.
        assert clean_text("BP high hai, CT scan bhi karwana hai") == "BP high hai, CT scan bhi karwana hai"

    def test_non_string_input_is_coerced(self):
        assert clean_text(None) == "None"


class TestIsGibberish:
    @pytest.mark.parametrize("text", [
        "",
        "a",
        "ab",
        "12",
        "jgjhkj",
        "asfgtqwafazfyiur",
        "zxcvbnmlkjhgfdsaqwerty",
        "qqqqqqqqq",
        "aaaaaaaaaa",
        "kjhgfdsapoiuytrewq",
    ])
    def test_flags_gibberish(self, text):
        assert is_gibberish(text) is True

    @pytest.mark.parametrize("text", [
        "dil mein dard hai",
        "Dant mein bahut dard hai, thanda garam nahi sehta",
        "sar dukh raha hai",
        "BP high hai",
        "bukhar",
    ])
    def test_does_not_flag_real_symptom_text(self, text):
        assert is_gibberish(text) is False
