"""PatientDayEmbedding envelope — Challenge 5 (JEPA dimension mismatch).

THE PROBLEM (Challenge 5): agents emit embeddings of DIFFERENT native
dimensions — symptoms=36, wearable=29, biomarker=7, dietary=10, environment=5.
A future JEPA world-model needs to consume a patient's combined daily state, but
it cannot consume a ragged set of differently-sized vectors directly.

THE LOCKED SOLUTION: each agent keeps its NATIVE dimension. We wrap the per-
agent embeddings in a PatientDayEmbedding envelope that:
    - stores each agent's embedding under its agent_id, with its dimension,
    - records which agents were present vs absent (zero-filled),
    - can produce a deterministic CONCATENATED vector in a fixed agent order
      with fixed per-agent slot sizes, so the combined representation has a
      stable layout the JEPA model can rely on.

The JEPA MODEL ARCHITECTURE itself is deferred to v2 (concatenation vs a
transformer-style encoder is decided when real patient data exists). This
module provides the envelope + a stable concatenation so everything downstream
has a consistent contract NOW, without committing to the model design.

JEPACompatible is the protocol any embedding producer satisfies (the agents
already do, via emit_embedding / get_output_vector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Fixed agent order and native slot sizes for the concatenated layout.
# This ordering is a STABLE CONTRACT — changing it would invalidate any model
# trained on the concatenation, so it is versioned.
EMBEDDING_LAYOUT_VERSION = "pde-v1"

_AGENT_SLOT_DIMS = [
    ("agent1_biomarker", 7),
    ("agent2_dietary", 10),
    ("agent3_environment", 5),
    ("agent4_wearable", 29),
    ("agent5_symptoms_mood", 36),
]

TOTAL_CONCAT_DIM = sum(dim for _, dim in _AGENT_SLOT_DIMS)  # 87


@runtime_checkable
class JEPACompatible(Protocol):
    """Anything that can emit a fixed-size embedding for JEPA consumption."""

    def emit_embedding(self, daily_summary=None) -> np.ndarray:
        ...


@dataclass
class PatientDayEmbedding:
    """A patient's combined daily embedding state across agents.

    Fields:
        patient_id / bucket_id: identity.
        embeddings: map agent_id -> 1D np.ndarray (native dim).
        present: set of agent_ids that contributed a real (non-zero-filled)
            embedding.
        layout_version: the concatenation layout contract version.
    """

    patient_id: str
    bucket_id: str
    embeddings: dict = field(default_factory=dict)
    present: set = field(default_factory=set)
    layout_version: str = EMBEDDING_LAYOUT_VERSION

    def add(self, agent_id: str, embedding: np.ndarray, present: bool = True) -> None:
        """Add one agent's embedding to the envelope.

        The embedding is validated against the agent's expected slot dim. If a
        dimension mismatches the layout, it is rejected (loud failure beats a
        silently corrupt concatenation).
        """
        expected = dict(_AGENT_SLOT_DIMS).get(agent_id)
        arr = np.asarray(embedding, dtype=np.float64).ravel()
        if expected is not None and arr.shape[0] != expected:
            raise ValueError(
                f"{agent_id} embedding dim {arr.shape[0]} != expected slot {expected}"
            )
        self.embeddings[agent_id] = arr
        if present:
            self.present.add(agent_id)

    def get(self, agent_id: str) -> Optional[np.ndarray]:
        return self.embeddings.get(agent_id)

    def to_concat(self) -> np.ndarray:
        """Produce the deterministic fixed-layout concatenated vector.

        Agents are laid out in the fixed order and slot sizes. Any agent
        without an embedding (absent this bucket) contributes a zero block of
        its slot size, so the output is ALWAYS length TOTAL_CONCAT_DIM with a
        stable per-agent layout.
        """
        blocks = []
        for agent_id, dim in _AGENT_SLOT_DIMS:
            emb = self.embeddings.get(agent_id)
            if emb is None:
                blocks.append(np.zeros(dim, dtype=np.float64))
            else:
                # Defensive: enforce exact slot length.
                if emb.shape[0] != dim:
                    raise ValueError(
                        f"{agent_id} embedding dim {emb.shape[0]} != slot {dim}"
                    )
                blocks.append(emb)
        return np.concatenate(blocks)

    def presence_mask(self) -> np.ndarray:
        """Boolean vector (len = n agents) marking which agents were present.

        Useful for a JEPA encoder that wants to distinguish "absent" (zero by
        convention) from "genuinely measured as zero".
        """
        return np.array(
            [aid in self.present for aid, _ in _AGENT_SLOT_DIMS], dtype=bool
        )

    @property
    def n_present(self) -> int:
        return len(self.present)


def build_patient_day_embedding(
    patient_id: str,
    bucket_id: str,
    agent_outputs: dict,
) -> PatientDayEmbedding:
    """Assemble a PatientDayEmbedding from agent outputs for a bucket.

    Uses each AgentOutput.vector (the JEPA-compatible native embedding). Agents
    absent from agent_outputs are simply not added (and become zero blocks in
    to_concat()). Agents whose vector dim doesn't match their slot are skipped
    with no corruption (their slot stays zero) — robustness over strictness at
    assembly time, while add() stays strict for explicit calls.
    """
    pde = PatientDayEmbedding(patient_id=patient_id, bucket_id=bucket_id)
    slot_dims = dict(_AGENT_SLOT_DIMS)
    for agent_id, output in agent_outputs.items():
        vec = getattr(output, "vector", None)
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float64).ravel()
        expected = slot_dims.get(agent_id)
        if expected is None or arr.shape[0] != expected:
            # Unknown agent or dim mismatch -> leave its slot zero, don't corrupt.
            continue
        pde.add(agent_id, arr, present=True)
    return pde
