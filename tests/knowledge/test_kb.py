"""Tests for the KnowledgeBase seam (NullKB for v1; real KB later)."""

import pytest

from immunosense.knowledge import KnowledgeBase, KnowledgeEntry, NullKB


class TestNullKB:
    def test_satisfies_protocol(self):
        kb = NullKB()
        assert isinstance(kb, KnowledgeBase)
        assert kb.name == "null-kb"

    def test_returns_empty_list(self):
        kb = NullKB()
        assert kb.query() == []
        assert kb.query(disease="SLE") == []
        assert kb.query(tags=["hrv", "crp"]) == []
        assert kb.query(disease="RA", tags=["pollen"], limit=20) == []

    def test_query_accepts_all_keyword_args(self):
        kb = NullKB()
        # Just verify the protocol-shaped signature works.
        kb.query(disease=None, tags=None, limit=5)
        kb.query(disease="SLE")
        kb.query(tags=[])
        kb.query(limit=1)


class TestKnowledgeEntry:
    def test_creation_and_fields(self):
        e = KnowledgeEntry(
            entry_id="k1",
            text="HRV suppression has been associated with autoimmune flare onset.",
            diseases=("SLE", "RA"),
            tags=("hrv", "wearable"),
            source="literature-informed",
        )
        assert e.entry_id == "k1"
        assert "SLE" in e.diseases
        assert "hrv" in e.tags
        assert e.source

    def test_entry_is_frozen(self):
        e = KnowledgeEntry(entry_id="k", text="t")
        with pytest.raises(Exception):
            e.text = "modified"  # frozen
