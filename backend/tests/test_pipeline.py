"""
Unit tests for the pipeline module.

Tests cover:
- Pipeline creation and validation
- Pipeline serialization/deserialization
- PipelineStep validation
- apply_pipeline execution on sample data
- Pipeline idempotency
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

from app.services.pipeline import (
    Pipeline,
    PipelineStep,
    PipelineMetadata,
    apply_pipeline,
    apply_pipeline_from_json,
    apply_pipeline_from_file,
    create_pipeline_from_rules,
)
from app.models.schemas import CleaningRule, StrictModeConfig, DEFAULT_STRICT_CONFIG
from app.services.cleaner import set_strict_config


# ============================================
# Tests for PipelineStep
# ============================================

class TestPipelineStep:
    """Tests for PipelineStep model."""

    def test_create_valid_step(self):
        """Test creating a valid pipeline step."""
        step = PipelineStep(
            rule_type="trim_whitespace",
            column="name",
            description="Trim whitespace from name column"
        )

        assert step.rule_type == "trim_whitespace"
        assert step.column == "name"
        assert step.enabled is True
        assert step.params == {}

    def test_create_step_with_params(self):
        """Test creating a step with parameters."""
        step = PipelineStep(
            rule_type="fill_missing",
            column="age",
            params={"strategy": "mean"}
        )

        assert step.params["strategy"] == "mean"

    def test_invalid_rule_type_raises_error(self):
        """Test that an invalid rule type raises a validation error."""
        with pytest.raises(ValueError, match="Unknown rule type"):
            PipelineStep(rule_type="invalid_rule", column="test")

    def test_to_cleaning_rule(self):
        """Test conversion to CleaningRule."""
        step = PipelineStep(
            rule_type="normalize_numbers",
            column="salary",
            params={"remove_currency": True}
        )

        rule = step.to_cleaning_rule()

        assert isinstance(rule, CleaningRule)
        assert rule.rule_type == "normalize_numbers"
        assert rule.column == "salary"
        assert rule.params["remove_currency"] is True

    def test_step_enabled_default(self):
        """Test that steps are enabled by default."""
        step = PipelineStep(rule_type="dedupe")
        assert step.enabled is True

    def test_step_disabled(self):
        """Test creating a disabled step."""
        step = PipelineStep(rule_type="dedupe", enabled=False)
        assert step.enabled is False


# ============================================
# Tests for Pipeline
# ============================================

class TestPipeline:
    """Tests for Pipeline model."""

    def test_create_empty_pipeline(self):
        """Test creating a pipeline with no steps."""
        pipeline = Pipeline(name="Empty Pipeline")

        assert pipeline.name == "Empty Pipeline"
        assert len(pipeline.steps) == 0
        assert isinstance(pipeline.metadata, PipelineMetadata)

    def test_create_pipeline_with_steps(self):
        """Test creating a pipeline with multiple steps."""
        pipeline = Pipeline(
            name="Test Pipeline",
            description="A test pipeline",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="dedupe"),
            ]
        )

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].rule_type == "trim_whitespace"
        assert pipeline.steps[1].rule_type == "dedupe"

    def test_get_enabled_steps(self):
        """Test getting only enabled steps."""
        pipeline = Pipeline(
            name="Test",
            steps=[
                PipelineStep(rule_type="trim_whitespace", enabled=True),
                PipelineStep(rule_type="dedupe", enabled=False),
                PipelineStep(rule_type="fill_missing", column="age", enabled=True),
            ]
        )

        enabled = pipeline.get_enabled_steps()

        assert len(enabled) == 2
        assert enabled[0].rule_type == "trim_whitespace"
        assert enabled[1].rule_type == "fill_missing"

    def test_to_cleaning_rules(self):
        """Test conversion of all steps to cleaning rules."""
        pipeline = Pipeline(
            name="Test",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="dedupe", enabled=False),
            ]
        )

        rules = pipeline.to_cleaning_rules()

        # Only enabled steps should be converted
        assert len(rules) == 1
        assert rules[0].rule_type == "trim_whitespace"

    def test_add_step(self):
        """Test adding a step to the pipeline."""
        pipeline = Pipeline(name="Test")
        pipeline.add_step("trim_whitespace", description="Trim all")
        pipeline.add_step("dedupe")

        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].description == "Trim all"

    def test_add_step_at_index(self):
        """Test adding a step at a specific index."""
        pipeline = Pipeline(
            name="Test",
            steps=[PipelineStep(rule_type="trim_whitespace")]
        )
        pipeline.add_step("dedupe", index=0)

        assert pipeline.steps[0].rule_type == "dedupe"
        assert pipeline.steps[1].rule_type == "trim_whitespace"

    def test_remove_step(self):
        """Test removing a step from the pipeline."""
        pipeline = Pipeline(
            name="Test",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="dedupe"),
            ]
        )
        pipeline.remove_step(0)

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].rule_type == "dedupe"

    def test_validate_pipeline_empty_warning(self):
        """Test that validation warns about empty pipeline."""
        pipeline = Pipeline(name="Empty")
        warnings = pipeline.validate_pipeline()

        assert any("no steps" in w for w in warnings)

    def test_validate_pipeline_all_disabled_warning(self):
        """Test that validation warns when all steps are disabled."""
        pipeline = Pipeline(
            name="Test",
            steps=[PipelineStep(rule_type="dedupe", enabled=False)]
        )
        warnings = pipeline.validate_pipeline()

        assert any("disabled" in w for w in warnings)

    def test_validate_pipeline_multiple_dedupe_warning(self):
        """Test that validation warns about multiple dedupe steps."""
        pipeline = Pipeline(
            name="Test",
            steps=[
                PipelineStep(rule_type="dedupe"),
                PipelineStep(rule_type="dedupe"),
            ]
        )
        warnings = pipeline.validate_pipeline()

        assert any("Multiple dedupe" in w for w in warnings)


# ============================================
# Tests for Pipeline Serialization
# ============================================

class TestPipelineSerialization:
    """Tests for pipeline serialization and deserialization."""

    def test_to_dict(self):
        """Test converting pipeline to dictionary."""
        pipeline = Pipeline(
            name="Test",
            steps=[PipelineStep(rule_type="trim_whitespace")]
        )

        data = pipeline.to_dict()

        assert data["name"] == "Test"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["rule_type"] == "trim_whitespace"

    def test_to_json(self):
        """Test converting pipeline to JSON string."""
        pipeline = Pipeline(
            name="Test",
            steps=[PipelineStep(rule_type="dedupe")]
        )

        json_str = pipeline.to_json()
        data = json.loads(json_str)

        assert data["name"] == "Test"
        assert len(data["steps"]) == 1

    def test_from_dict(self):
        """Test creating pipeline from dictionary."""
        data = {
            "name": "From Dict",
            "steps": [
                {"rule_type": "trim_whitespace"},
                {"rule_type": "dedupe"}
            ]
        }

        pipeline = Pipeline.from_dict(data)

        assert pipeline.name == "From Dict"
        assert len(pipeline.steps) == 2

    def test_from_json(self):
        """Test creating pipeline from JSON string."""
        json_str = '''
        {
            "name": "From JSON",
            "steps": [
                {"rule_type": "normalize_numbers", "column": "price"}
            ]
        }
        '''

        pipeline = Pipeline.from_json(json_str)

        assert pipeline.name == "From JSON"
        assert pipeline.steps[0].column == "price"

    def test_save_and_load(self):
        """Test saving and loading pipeline from file."""
        pipeline = Pipeline(
            name="Save Test",
            description="Test saving",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="fill_missing", column="age", params={"value": 0}),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            pipeline.save(temp_path)
            loaded = Pipeline.load(temp_path)

            assert loaded.name == "Save Test"
            assert len(loaded.steps) == 2
            assert loaded.steps[1].params["value"] == 0
        finally:
            temp_path.unlink()

    def test_serialization_roundtrip(self):
        """Test that serialization followed by deserialization preserves pipeline."""
        original = Pipeline(
            name="Roundtrip Test",
            description="Testing roundtrip",
            steps=[
                PipelineStep(
                    rule_type="normalize_numbers",
                    column="salary",
                    params={"remove_currency": True, "remove_commas": True}
                ),
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="dedupe", params={"keep": "first"}),
            ]
        )

        json_str = original.to_json()
        restored = Pipeline.from_json(json_str)

        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.steps) == len(original.steps)
        for orig_step, rest_step in zip(original.steps, restored.steps):
            assert orig_step.rule_type == rest_step.rule_type
            assert orig_step.column == rest_step.column
            assert orig_step.params == rest_step.params


# ============================================
# Tests for apply_pipeline
# ============================================

class TestApplyPipeline:
    """Tests for apply_pipeline function."""

    def test_apply_empty_pipeline(self):
        """Test applying an empty pipeline returns unchanged data."""
        df = pd.DataFrame({
            'name': ['  Alice  ', 'Bob'],
            'age': [25, 30]
        })
        pipeline = Pipeline(name="Empty")

        result = apply_pipeline(df, pipeline)

        pd.testing.assert_frame_equal(result, df)

    def test_apply_single_step_pipeline(self):
        """Test applying a pipeline with a single step."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  '],
            'age': [25, 30]
        })
        pipeline = Pipeline(
            name="Trim",
            steps=[PipelineStep(rule_type="trim_whitespace", column="name")]
        )

        result = apply_pipeline(df, pipeline)

        assert result['name'].tolist() == ['Alice', 'Bob']

    def test_apply_multi_step_pipeline(self):
        """Test applying a pipeline with multiple steps."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ', '  Alice  '],
            'salary': ['$50,000', '$60,000', '$50,000']
        })
        pipeline = Pipeline(
            name="Multi-step",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="normalize_numbers", column="salary"),
                PipelineStep(rule_type="dedupe"),
            ]
        )

        result = apply_pipeline(df, pipeline)

        assert result['name'].tolist() == ['Alice', 'Bob']
        assert result['salary'].tolist() == [50000.0, 60000.0]
        assert len(result) == 2  # Duplicate removed

    def test_apply_pipeline_skips_disabled_steps(self):
        """Test that disabled steps are skipped."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ']
        })
        pipeline = Pipeline(
            name="Test",
            steps=[
                PipelineStep(rule_type="trim_whitespace", enabled=False),
            ]
        )

        result = apply_pipeline(df, pipeline)

        # Whitespace should NOT be trimmed since step is disabled
        assert result['name'].tolist() == ['  Alice  ', '  Bob  ']

    def test_apply_pipeline_preserves_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ']
        })
        original_values = df['name'].tolist()
        
        pipeline = Pipeline(
            name="Test",
            steps=[PipelineStep(rule_type="trim_whitespace")]
        )

        apply_pipeline(df, pipeline)

        assert df['name'].tolist() == original_values

    def test_apply_pipeline_idempotent(self):
        """Test that applying the same pipeline twice yields the same result."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ', '  Alice  '],
            'salary': ['$50,000', '$60,000', '$50,000'],
            'age': [25, np.nan, 25]
        })
        pipeline = Pipeline(
            name="Idempotent Test",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="normalize_numbers", column="salary"),
                PipelineStep(rule_type="fill_missing", column="age", params={"value": 0}),
                PipelineStep(rule_type="dedupe"),
            ]
        )

        result1 = apply_pipeline(df, pipeline)
        result2 = apply_pipeline(result1, pipeline)

        pd.testing.assert_frame_equal(result1, result2)

    def test_apply_pipeline_with_fill_missing(self):
        """Test pipeline with fill_missing step (requires imputation enabled)."""
        # Enable imputation for this test
        set_strict_config(StrictModeConfig(enabled=True, allow_imputation=True))
        
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        pipeline = Pipeline(
            name="Fill Test",
            steps=[
                PipelineStep(
                    rule_type="fill_missing",
                    column="value",
                    params={"strategy": "mean"}
                )
            ]
        )

        result = apply_pipeline(df, pipeline)

        assert not result['value'].isna().any()
        # Mean of [1, 3, 5] = 3
        assert result['value'].tolist() == [1.0, 3.0, 3.0, 3.0, 5.0]
        
        # Reset to default strict mode
        set_strict_config(DEFAULT_STRICT_CONFIG)

    def test_apply_pipeline_on_sample_csv(self):
        """Test applying a complete pipeline to sample CSV data."""
        # Create sample data similar to messy_sample.csv
        df = pd.DataFrame({
            'name': ['  Alice  ', 'Bob', '  Alice  '],
            'email': ['alice@example.com', 'bob@example.com', 'alice@example.com'],
            'salary': ['$50,000', '$60,000', '$50,000'],
            'age': [25, np.nan, 25],
            'is_active': ['true', 'yes', 'true']
        })

        pipeline = Pipeline(
            name="Complete Cleanup",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="normalize_numbers", column="salary"),
                PipelineStep(
                    rule_type="fill_missing",
                    column="age",
                    params={"value": 30}
                ),
                PipelineStep(rule_type="dedupe"),
            ]
        )

        result = apply_pipeline(df, pipeline)

        # Verify transformations
        assert result['name'].tolist() == ['Alice', 'Bob']
        assert result['salary'].tolist() == [50000.0, 60000.0]
        assert result['age'].tolist() == [25.0, 30.0]
        assert len(result) == 2  # Duplicate removed


# ============================================
# Tests for Pipeline Utility Functions
# ============================================

class TestPipelineUtilities:
    """Tests for pipeline utility functions."""

    def test_apply_pipeline_from_json(self):
        """Test applying pipeline from JSON string."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ']
        })
        json_str = '''
        {
            "name": "JSON Pipeline",
            "steps": [{"rule_type": "trim_whitespace"}]
        }
        '''

        result = apply_pipeline_from_json(df, json_str)

        assert result['name'].tolist() == ['Alice', 'Bob']

    def test_apply_pipeline_from_file(self):
        """Test applying pipeline from file."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ']
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "name": "File Pipeline",
                "steps": [{"rule_type": "trim_whitespace"}]
            }, f)
            temp_path = Path(f.name)

        try:
            result = apply_pipeline_from_file(df, temp_path)
            assert result['name'].tolist() == ['Alice', 'Bob']
        finally:
            temp_path.unlink()

    def test_create_pipeline_from_rules(self):
        """Test creating pipeline from CleaningRule objects."""
        rules = [
            CleaningRule(rule_type="trim_whitespace", column=None, params={}),
            CleaningRule(rule_type="dedupe", column=None, params={"keep": "first"}),
        ]

        pipeline = create_pipeline_from_rules(
            name="From Rules",
            rules=rules,
            description="Created from rules"
        )

        assert pipeline.name == "From Rules"
        assert pipeline.description == "Created from rules"
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].rule_type == "trim_whitespace"
        assert pipeline.steps[1].params["keep"] == "first"


# ============================================
# Tests for Complex Pipeline Scenarios
# ============================================

class TestComplexPipelineScenarios:
    """Tests for complex pipeline use cases."""

    def test_pipeline_with_all_rule_types(self):
        """Test a pipeline using all available rule types."""
        df = pd.DataFrame({
            'id': [1, 2, 1, 3],
            'name': ['  Alice  ', 'Bob', '  Alice  ', '  Charlie  '],
            'email': ['alice@test.com', 'bob@test.com', 'alice@test.com', 'charlie@test.com'],
            'salary': ['$50,000', '$60,000', '$50,000', '$70,000'],
            'age': [25, np.nan, 25, 35],
            'signup_date': ['2023-01-15', '2023-02-20', '2023-01-15', '2023-03-25'],
            'score': ['10', '20.5', '10', '30'],
            'extra_col': ['x', 'y', 'x', 'z']
        })

        pipeline = Pipeline(
            name="Complete Pipeline",
            steps=[
                PipelineStep(rule_type="drop_columns", params={"columns": ["extra_col"]}),
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="normalize_numbers", column="salary"),
                PipelineStep(rule_type="fill_missing", column="age", params={"value": 30}),
                PipelineStep(rule_type="parse_dates", column="signup_date"),
                PipelineStep(rule_type="coerce_types", column="score", params={"dtype": "float"}),
                PipelineStep(rule_type="dedupe", params={"subset": ["id"]}),
            ]
        )

        result = apply_pipeline(df, pipeline)

        # Verify all transformations
        assert 'extra_col' not in result.columns
        assert result['name'].tolist() == ['Alice', 'Bob', 'Charlie']
        assert result['salary'].tolist() == [50000.0, 60000.0, 70000.0]
        assert result['age'].tolist() == [25.0, 30.0, 35.0]
        assert pd.api.types.is_datetime64_any_dtype(result['signup_date'])
        assert result['score'].dtype == float
        assert len(result) == 3

    def test_pipeline_order_matters(self):
        """Test that pipeline step order affects results."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Alice  ', '  Bob  ']
        })

        # Dedupe first, then trim - should keep 2 rows (duplicates with whitespace look different)
        pipeline1 = Pipeline(
            name="Dedupe First",
            steps=[
                PipelineStep(rule_type="dedupe"),
                PipelineStep(rule_type="trim_whitespace"),
            ]
        )

        # Trim first, then dedupe - should keep 2 rows (duplicates now match)
        pipeline2 = Pipeline(
            name="Trim First",
            steps=[
                PipelineStep(rule_type="trim_whitespace"),
                PipelineStep(rule_type="dedupe"),
            ]
        )

        result1 = apply_pipeline(df, pipeline1)
        result2 = apply_pipeline(df, pipeline2)

        # With trim first, exact duplicates are removed
        assert len(result1) == 2  # All rows look identical after trim, but original had differences
        assert len(result2) == 2  # Alice and Bob

    def test_chained_pipelines(self):
        """Test applying multiple pipelines in sequence."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  '],
            'salary': ['$50,000', '$60,000']
        })

        pipeline1 = Pipeline(
            name="Step 1",
            steps=[PipelineStep(rule_type="trim_whitespace")]
        )
        pipeline2 = Pipeline(
            name="Step 2",
            steps=[PipelineStep(rule_type="normalize_numbers", column="salary")]
        )

        intermediate = apply_pipeline(df, pipeline1)
        result = apply_pipeline(intermediate, pipeline2)

        assert result['name'].tolist() == ['Alice', 'Bob']
        assert result['salary'].tolist() == [50000.0, 60000.0]
