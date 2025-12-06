"""
Pipeline module for defining, serializing, and executing data cleaning pipelines.

This module provides a structured way to define reusable cleaning pipelines that can be:
- Serialized to JSON for storage and sharing
- Deserialized back to executable pipelines
- Applied to pandas DataFrames in a deterministic order

The pipeline schema guarantees that serialization followed by deserialization
yields identical cleaning behavior, making pipelines portable and reproducible.

Example usage:
    >>> pipeline = Pipeline(
    ...     name="Customer Data Cleanup",
    ...     steps=[
    ...         PipelineStep(rule_type="trim_whitespace", column="name"),
    ...         PipelineStep(rule_type="fill_missing", column="age", params={"value": 0}),
    ...     ]
    ... )
    >>> cleaned_df = apply_pipeline(df, pipeline)
    >>> pipeline.save("my_pipeline.json")
    >>> loaded_pipeline = Pipeline.load("my_pipeline.json")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from app.models.schemas import CleaningRule
from app.services.cleaner import apply_cleaning_rules, RULE_HANDLERS


# ============================================
# Pipeline Schema Definitions
# ============================================

class PipelineStep(BaseModel):
    """
    A single step in a cleaning pipeline.
    
    Each step represents one cleaning operation to be applied to the data.
    Steps are executed in the order they appear in the pipeline.
    
    Attributes:
        rule_type: The type of cleaning rule to apply. Must be one of the
                   supported rule types (drop_columns, fill_missing, coerce_types,
                   trim_whitespace, parse_dates, normalize_numbers, dedupe).
        column: The target column for column-specific operations. Can be None
                for operations that don't target a specific column (e.g., dedupe).
        params: Additional parameters for the cleaning rule. The structure
                depends on the rule_type.
        description: Optional human-readable description of what this step does.
        enabled: Whether this step should be executed. Allows temporarily
                 disabling steps without removing them from the pipeline.
    
    Example:
        >>> step = PipelineStep(
        ...     rule_type="fill_missing",
        ...     column="age",
        ...     params={"strategy": "mean"},
        ...     description="Fill missing ages with mean value"
        ... )
    """
    rule_type: str = Field(
        ...,
        description="Type of cleaning rule to apply"
    )
    column: Optional[str] = Field(
        None,
        description="Target column for column-specific operations"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the cleaning rule"
    )
    description: Optional[str] = Field(
        None,
        description="Human-readable description of this step"
    )
    enabled: bool = Field(
        True,
        description="Whether this step should be executed"
    )
    
    @field_validator('rule_type')
    @classmethod
    def validate_rule_type(cls, v: str) -> str:
        """Validate that the rule type is supported."""
        if v not in RULE_HANDLERS:
            supported = ', '.join(sorted(RULE_HANDLERS.keys()))
            raise ValueError(
                f"Unknown rule type: '{v}'. Supported types are: {supported}"
            )
        return v
    
    def to_cleaning_rule(self) -> CleaningRule:
        """
        Convert this pipeline step to a CleaningRule for execution.
        
        Returns:
            CleaningRule object that can be passed to the cleaner service.
        """
        return CleaningRule(
            rule_type=self.rule_type,
            column=self.column,
            params=self.params
        )


class PipelineMetadata(BaseModel):
    """
    Metadata about a cleaning pipeline.
    
    Stores information about the pipeline's creation, versioning, and purpose.
    
    Attributes:
        created_at: ISO format timestamp of when the pipeline was created.
        updated_at: ISO format timestamp of when the pipeline was last modified.
        version: Semantic version string for the pipeline definition.
        author: Name or identifier of the pipeline creator.
        tags: List of tags for categorizing and searching pipelines.
    """
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO format timestamp of creation"
    )
    updated_at: Optional[str] = Field(
        None,
        description="ISO format timestamp of last update"
    )
    version: str = Field(
        "1.0.0",
        description="Semantic version of the pipeline"
    )
    author: Optional[str] = Field(
        None,
        description="Author of the pipeline"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )


class Pipeline(BaseModel):
    """
    A complete data cleaning pipeline definition.
    
    A pipeline encapsulates a sequence of cleaning steps that can be applied
    to a pandas DataFrame. Pipelines are fully serializable to JSON, and
    deserialization is guaranteed to produce identical behavior.
    
    Attributes:
        name: A descriptive name for the pipeline.
        description: Detailed description of what the pipeline does and when to use it.
        steps: Ordered list of PipelineStep objects defining the cleaning operations.
        metadata: Optional metadata about the pipeline (creation date, version, etc.).
    
    Example:
        >>> pipeline = Pipeline(
        ...     name="Sales Data Cleanup",
        ...     description="Standardizes and cleans sales records",
        ...     steps=[
        ...         PipelineStep(rule_type="trim_whitespace"),
        ...         PipelineStep(rule_type="normalize_numbers", column="revenue"),
        ...         PipelineStep(rule_type="dedupe", params={"subset": ["order_id"]}),
        ...     ]
        ... )
        >>> pipeline.save("sales_pipeline.json")
    """
    name: str = Field(
        ...,
        description="Name of the pipeline"
    )
    description: Optional[str] = Field(
        None,
        description="Detailed description of the pipeline"
    )
    steps: List[PipelineStep] = Field(
        default_factory=list,
        description="Ordered list of cleaning steps"
    )
    metadata: PipelineMetadata = Field(
        default_factory=PipelineMetadata,
        description="Pipeline metadata"
    )
    
    def get_enabled_steps(self) -> List[PipelineStep]:
        """
        Get only the enabled steps from the pipeline.
        
        Returns:
            List of PipelineStep objects where enabled=True.
        """
        return [step for step in self.steps if step.enabled]
    
    def to_cleaning_rules(self) -> List[CleaningRule]:
        """
        Convert all enabled pipeline steps to CleaningRule objects.
        
        Returns:
            List of CleaningRule objects ready for execution.
        """
        return [step.to_cleaning_rule() for step in self.get_enabled_steps()]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline to a dictionary.
        
        Returns:
            Dictionary representation of the pipeline, suitable for JSON serialization.
        """
        return self.model_dump()
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the pipeline to a JSON string.
        
        Args:
            indent: Number of spaces for JSON indentation.
        
        Returns:
            JSON string representation of the pipeline.
        
        Note:
            The resulting JSON can be deserialized with Pipeline.from_json()
            to produce an identical pipeline.
        """
        return self.model_dump_json(indent=indent)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the pipeline to a JSON file.
        
        Args:
            path: File path where the pipeline should be saved.
        
        Raises:
            IOError: If the file cannot be written.
        
        Example:
            >>> pipeline.save("my_pipeline.json")
        """
        path = Path(path)
        # Update the updated_at timestamp before saving
        self.metadata.updated_at = datetime.utcnow().isoformat()
        path.write_text(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """
        Create a Pipeline from a dictionary.
        
        Args:
            data: Dictionary containing pipeline definition.
        
        Returns:
            Pipeline object.
        
        Raises:
            ValidationError: If the dictionary doesn't match the expected schema.
        """
        return cls.model_validate(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Pipeline":
        """
        Create a Pipeline from a JSON string.
        
        Args:
            json_str: JSON string containing pipeline definition.
        
        Returns:
            Pipeline object.
        
        Raises:
            ValidationError: If the JSON doesn't match the expected schema.
            JSONDecodeError: If the string is not valid JSON.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """
        Load a pipeline from a JSON file.
        
        Args:
            path: File path to load the pipeline from.
        
        Returns:
            Pipeline object.
        
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValidationError: If the file doesn't contain a valid pipeline.
        
        Example:
            >>> pipeline = Pipeline.load("my_pipeline.json")
        """
        path = Path(path)
        json_str = path.read_text()
        return cls.from_json(json_str)
    
    def add_step(
        self,
        rule_type: str,
        column: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        index: Optional[int] = None,
    ) -> "Pipeline":
        """
        Add a new step to the pipeline.
        
        Args:
            rule_type: Type of cleaning rule.
            column: Target column for the operation.
            params: Additional parameters for the rule.
            description: Human-readable description.
            index: Position to insert the step. If None, appends to end.
        
        Returns:
            Self, for method chaining.
        
        Example:
            >>> pipeline.add_step("trim_whitespace").add_step("dedupe")
        """
        step = PipelineStep(
            rule_type=rule_type,
            column=column,
            params=params or {},
            description=description,
        )
        if index is not None:
            self.steps.insert(index, step)
        else:
            self.steps.append(step)
        return self
    
    def remove_step(self, index: int) -> "Pipeline":
        """
        Remove a step from the pipeline by index.
        
        Args:
            index: Index of the step to remove.
        
        Returns:
            Self, for method chaining.
        
        Raises:
            IndexError: If the index is out of range.
        """
        del self.steps[index]
        return self
    
    def validate_pipeline(self) -> List[str]:
        """
        Validate the pipeline and return any warnings.
        
        Returns:
            List of warning messages. Empty list if pipeline is valid.
        """
        warnings = []
        
        if not self.steps:
            warnings.append("Pipeline has no steps defined.")
        
        enabled_count = len(self.get_enabled_steps())
        if enabled_count == 0 and self.steps:
            warnings.append("All steps are disabled. Pipeline will have no effect.")
        
        # Check for potentially redundant operations
        seen_dedupes = 0
        for step in self.steps:
            if step.enabled and step.rule_type == 'dedupe':
                seen_dedupes += 1
                if seen_dedupes > 1:
                    warnings.append(
                        "Multiple dedupe steps detected. "
                        "Consider consolidating into a single step."
                    )
        
        return warnings


# ============================================
# Pipeline Execution Functions
# ============================================

def apply_pipeline(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """
    Apply a cleaning pipeline to a pandas DataFrame.
    
    This function executes all enabled steps in the pipeline in order,
    transforming the DataFrame according to each step's cleaning rule.
    
    Args:
        df: The pandas DataFrame to clean.
        pipeline: The Pipeline object containing the cleaning steps.
    
    Returns:
        A new DataFrame with all cleaning rules applied.
    
    Note:
        - This function is idempotent: applying the same pipeline twice
          yields the same result (assuming deterministic operations).
        - The original DataFrame is not modified.
        - Steps with enabled=False are skipped.
        - Errors in individual steps are handled gracefully; the pipeline
          continues with subsequent steps.
    
    Example:
        >>> pipeline = Pipeline.load("my_pipeline.json")
        >>> cleaned_df = apply_pipeline(raw_df, pipeline)
    """
    cleaning_rules = pipeline.to_cleaning_rules()
    return apply_cleaning_rules(df, cleaning_rules)


def apply_pipeline_from_json(df: pd.DataFrame, json_str: str) -> pd.DataFrame:
    """
    Apply a cleaning pipeline defined as a JSON string to a DataFrame.
    
    Convenience function that combines deserialization and execution.
    
    Args:
        df: The pandas DataFrame to clean.
        json_str: JSON string containing the pipeline definition.
    
    Returns:
        A new DataFrame with all cleaning rules applied.
    
    Example:
        >>> with open("pipeline.json") as f:
        ...     cleaned_df = apply_pipeline_from_json(df, f.read())
    """
    pipeline = Pipeline.from_json(json_str)
    return apply_pipeline(df, pipeline)


def apply_pipeline_from_file(df: pd.DataFrame, path: Union[str, Path]) -> pd.DataFrame:
    """
    Apply a cleaning pipeline from a JSON file to a DataFrame.
    
    Convenience function that combines loading and execution.
    
    Args:
        df: The pandas DataFrame to clean.
        path: Path to the JSON file containing the pipeline definition.
    
    Returns:
        A new DataFrame with all cleaning rules applied.
    
    Example:
        >>> cleaned_df = apply_pipeline_from_file(df, "pipeline.json")
    """
    pipeline = Pipeline.load(path)
    return apply_pipeline(df, pipeline)


# ============================================
# Pipeline Builder Utilities
# ============================================

def create_pipeline_from_rules(
    name: str,
    rules: List[CleaningRule],
    description: Optional[str] = None,
) -> Pipeline:
    """
    Create a Pipeline from a list of CleaningRule objects.
    
    Useful for converting existing cleaning rules into a reusable pipeline.
    
    Args:
        name: Name for the new pipeline.
        rules: List of CleaningRule objects to include.
        description: Optional description for the pipeline.
    
    Returns:
        A new Pipeline object.
    
    Example:
        >>> rules = [CleaningRule(rule_type="trim_whitespace", column=None, params={})]
        >>> pipeline = create_pipeline_from_rules("My Pipeline", rules)
    """
    steps = [
        PipelineStep(
            rule_type=rule.rule_type,
            column=rule.column,
            params=rule.params,
        )
        for rule in rules
    ]
    return Pipeline(
        name=name,
        description=description,
        steps=steps,
    )


def get_supported_rule_types() -> List[str]:
    """
    Get a list of all supported cleaning rule types.
    
    Returns:
        Sorted list of supported rule type names.
    
    Example:
        >>> types = get_supported_rule_types()
        >>> print(types)
        ['coerce_types', 'dedupe', 'drop_columns', ...]
    """
    return sorted(RULE_HANDLERS.keys())


def get_rule_type_info() -> Dict[str, str]:
    """
    Get information about all supported rule types.
    
    Returns:
        Dictionary mapping rule type names to descriptions.
    """
    return {
        'drop_columns': 'Remove specified columns from the dataset',
        'fill_missing': 'Fill missing values with a specified value or strategy',
        'coerce_types': 'Convert column to a specified data type',
        'trim_whitespace': 'Remove leading/trailing whitespace from string columns',
        'parse_dates': 'Parse column as datetime',
        'normalize_numbers': 'Normalize numeric values (remove currency, commas, etc.)',
        'dedupe': 'Remove duplicate rows',
    }
