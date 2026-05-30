"""Builders — translate raw UI input into the domain objects agents consume.

symptom:   SymptomLog rows      -> DailySymptomMoodSummary   (direct mapping)
biomarker: reading + profile    -> {demographics, reading}   (direct mapping)
dietary:   meal logs            -> DailyRollup               (real NHANES pipeline)
"""

from server.builders.biomarker_builder import build_biomarker_input
from server.builders.dietary_builder import DietaryPipeline, build_caches
from server.builders.symptom_builder import build_symptom_summary

__all__ = [
    "build_symptom_summary",
    "build_biomarker_input",
    "DietaryPipeline",
    "build_caches",
]
