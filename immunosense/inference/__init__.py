"""Inference envelope for JEPA (Challenge 5). Model architecture deferred to v2."""

from immunosense.inference.patient_day_embedding import (
    EMBEDDING_LAYOUT_VERSION,
    TOTAL_CONCAT_DIM,
    JEPACompatible,
    PatientDayEmbedding,
    build_patient_day_embedding,
)

__all__ = [
    "JEPACompatible",
    "PatientDayEmbedding",
    "build_patient_day_embedding",
    "EMBEDDING_LAYOUT_VERSION",
    "TOTAL_CONCAT_DIM",
]
