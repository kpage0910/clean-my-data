"""Services package for business logic."""

from app.services.scanner import scan_dataframe, analyze_dataframe
from app.services.cleaner import apply_cleaning_rules, preview_cleaning
from app.services.pipeline import Pipeline, PipelineStep, apply_pipeline
from app.services.autonomous import (
    autonomous_clean,
    autonomous_scan_and_suggest,
    infer_column_type,
    infer_all_column_types,
    InferredType,
    AutonomousCleaningResult,
)

__all__ = [
    # Scanner
    "scan_dataframe",
    "analyze_dataframe",
    # Cleaner
    "apply_cleaning_rules",
    "preview_cleaning",
    # Pipeline
    "Pipeline",
    "PipelineStep",
    "apply_pipeline",
    # Autonomous
    "autonomous_clean",
    "autonomous_scan_and_suggest",
    "infer_column_type",
    "infer_all_column_types",
    "InferredType",
    "AutonomousCleaningResult",
]
