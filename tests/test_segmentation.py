"""Unit tests for nlp_brain.segmentation -- splitting a sentence that names
more than one symptom into per-symptom segments.
"""
from nlp_brain.segmentation import split_segments


class TestSplitSegments:
    def test_single_symptom_sentence_stays_one_segment(self):
        assert split_segments("dant mein bahut dard ho raha hai") == [
            "dant mein bahut dard ho raha hai"
        ]

    def test_splits_on_aur(self):
        assert split_segments("pet mein dard hai aur sar bhi dukh raha hai") == [
            "pet mein dard hai",
            "sar bhi dukh raha hai",
        ]

    def test_splits_on_comma(self):
        assert split_segments("chest mein dard hai, ghutno mein bhi dikkat hai") == [
            "chest mein dard hai",
            "ghutno mein bhi dikkat hai",
        ]

    def test_splits_on_and(self):
        assert split_segments("aankhon mein jalan ho rahi hai and daant mein bhi dard hai") == [
            "aankhon mein jalan ho rahi hai",
            "daant mein bhi dard hai",
        ]

    def test_splits_on_saath_hi(self):
        assert split_segments("sar mein dard hai saath hi ulti bhi ho rahi hai") == [
            "sar mein dard hai",
            "ulti bhi ho rahi hai",
        ]

    def test_does_not_split_inside_aurat(self):
        # \b word-boundary guard -- "aurat" (woman) contains "aur" as a
        # substring but must not be treated as the joining word "aur".
        assert split_segments("aurat ko pet mein dard hai") == ["aurat ko pet mein dard hai"]

    def test_case_insensitive_split(self):
        assert split_segments("sar dard hai AND pet dard hai") == ["sar dard hai", "pet dard hai"]

    def test_short_trailing_fragment_does_not_become_its_own_segment(self):
        # A stray trailing comma/fragment shouldn't produce a near-empty
        # segment alongside the real one.
        assert split_segments("pet mein dard hai,") == ["pet mein dard hai"]

    def test_empty_string_returns_a_single_empty_segment(self):
        assert split_segments("") == [""]

    def test_too_short_input_is_not_split_away_to_nothing(self):
        # "ok" doesn't meet the >=2-words-or-4-chars bar on its own, but the
        # fallback must still return it (as the whole text) rather than [].
        assert split_segments("ok") == ["ok"]

    def test_three_way_split(self):
        assert split_segments("dil mein dard hai, dant mein bhi dard hai aur sar mein bhi dard hai") == [
            "dil mein dard hai",
            "dant mein bhi dard hai",
            "sar mein bhi dard hai",
        ]
