"""Knowledge base seam for TFM grounding (Challenge 2).

The TFM can ground its explanations in background knowledge passed as
`kb_context` (a list of short strings). WHERE that knowledge comes from is an
implementation detail behind the `KnowledgeBase` protocol.

v1 STATUS: NullKB (returns no grounding). The TFM explains fine without it,
using the computed probability, matched patterns, and agent signals. A real
knowledge base will be built later, sourced from autoimmune disease-activity /
flare literature — the kind of knowledge that actually maps to ImmunoSense's
signals (wearables, symptoms, diet, environment, biomarkers).

NOTE ON MOLECULAR DATA (Allen Immune Health Atlas, CELLxGENE, CellTypist):
those single-cell/transcriptomic resources are candidate REFERENCE DATA for a
future molecular Agent 7 (deferred until after the data architecture and UI are
built). They are NOT TFM grounding and do not belong in this seam.

Swapping a real KB in later is one construction line: build a class implementing
KnowledgeBase.query and pass it to the Conductor instead of NullKB. Nothing else
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class KnowledgeEntry:
    """One grounding snippet with provenance.

    Fields:
        entry_id: stable id.
        text: the short grounding statement (used in the TFM prompt).
        diseases: conditions this applies to (empty = cross-disease).
        tags: signal/agent tags for retrieval (e.g. "hrv", "crp", "pollen").
        source: provenance string. For a real KB this records the citation or
            "literature-informed" / "provisional-default"; never anonymous.
    """

    entry_id: str
    text: str
    diseases: tuple = ()
    tags: tuple = ()
    source: str = "unspecified"


@runtime_checkable
class KnowledgeBase(Protocol):
    """The swap point for TFM grounding."""

    name: str

    def query(
        self,
        disease: Optional[str] = None,
        tags: Optional[list] = None,
        limit: int = 5,
    ) -> list:
        """Return up to `limit` KnowledgeEntry objects relevant to the query."""
        ...


class NullKB:
    """A knowledge base that returns nothing. The v1 default.

    Lets the Conductor depend on the KnowledgeBase protocol today while the
    real, literature-sourced KB is built later. The TFM handles empty grounding
    gracefully (it simply explains without background snippets).
    """

    name = "null-kb"

    def query(
        self,
        disease: Optional[str] = None,
        tags: Optional[list] = None,
        limit: int = 5,
    ) -> list:
        return []
