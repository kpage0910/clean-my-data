"""
AI-powered Data Quality Summary service using OpenAI API.

This module provides functions to analyze pandas DataFrames and generate
natural-language summaries of data quality issues using GPT models.

IMPORTANT: This service is READ-ONLY and does NOT modify or clean any data.
It only analyzes and reports on data quality issues.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from openai import OpenAI, OpenAIError


# ============================================
# Configuration
# ============================================

# Default model to use for generating summaries
DEFAULT_MODEL = "gpt-4.1"

# Maximum number of sample values to include in the analysis
MAX_SAMPLE_VALUES = 10

# Threshold for considering a column as having high sparsity
SPARSITY_THRESHOLD = 0.5  # 50% missing values

# Z-score threshold for outlier detection
OUTLIER_ZSCORE_THRESHOLD = 3.0


# ============================================
# Helper Functions for Data Analysis
# ============================================

def _compute_missing_value_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive missing value statistics for each column.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing missing value analysis:
        {
            "total_missing_cells": int,
            "total_cells": int,
            "overall_missing_percentage": float,
            "columns": {
                "column_name": {
                    "missing_count": int,
                    "missing_percentage": float,
                    "is_sparse": bool
                }
            },
            "sparse_columns": ["col1", "col2", ...],
            "complete_columns": ["col3", "col4", ...]
        }
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    
    if total_rows == 0:
        return {
            "total_missing_cells": 0,
            "total_cells": 0,
            "overall_missing_percentage": 0.0,
            "columns": {},
            "sparse_columns": [],
            "complete_columns": []
        }
    
    columns_stats = {}
    sparse_columns = []
    complete_columns = []
    total_missing = 0
    
    for col in df.columns:
        # Count NaN/None values
        null_count = int(df[col].isna().sum())
        
        # Also count empty strings for object columns
        empty_string_count = 0
        if df[col].dtype == object:
            empty_string_count = int((df[col] == '').sum())
        
        missing_count = null_count + empty_string_count
        missing_percentage = round((missing_count / total_rows) * 100, 2)
        is_sparse = missing_percentage >= (SPARSITY_THRESHOLD * 100)
        
        columns_stats[col] = {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "is_sparse": is_sparse
        }
        
        total_missing += missing_count
        
        if is_sparse:
            sparse_columns.append(col)
        elif missing_count == 0:
            complete_columns.append(col)
    
    return {
        "total_missing_cells": total_missing,
        "total_cells": total_cells,
        "overall_missing_percentage": round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0,
        "columns": columns_stats,
        "sparse_columns": sparse_columns,
        "complete_columns": complete_columns
    }


def _compute_duplicate_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute duplicate row statistics.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing duplicate analysis:
        {
            "total_rows": int,
            "duplicate_rows": int,
            "unique_rows": int,
            "duplicate_percentage": float,
            "has_duplicates": bool
        }
    """
    total_rows = len(df)
    
    if total_rows == 0:
        return {
            "total_rows": 0,
            "duplicate_rows": 0,
            "unique_rows": 0,
            "duplicate_percentage": 0.0,
            "has_duplicates": False
        }
    
    duplicate_count = int(df.duplicated().sum())
    unique_count = total_rows - duplicate_count
    
    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_count,
        "unique_rows": unique_count,
        "duplicate_percentage": round((duplicate_count / total_rows) * 100, 2),
        "has_duplicates": duplicate_count > 0
    }


def _compute_column_type_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute column data type statistics and detect mixed types.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing column type analysis:
        {
            "columns": {
                "column_name": {
                    "pandas_dtype": str,
                    "inferred_type": str,
                    "is_mixed_type": bool,
                    "type_distribution": {"type_name": count, ...}
                }
            },
            "mixed_type_columns": ["col1", "col2", ...],
            "dtype_summary": {"int64": 3, "object": 5, ...}
        }
    """
    columns_stats = {}
    mixed_type_columns = []
    dtype_summary = {}
    
    for col in df.columns:
        pandas_dtype = str(df[col].dtype)
        dtype_summary[pandas_dtype] = dtype_summary.get(pandas_dtype, 0) + 1
        
        # Check for mixed types in object columns
        type_distribution = {}
        is_mixed_type = False
        
        if df[col].dtype == object:
            for val in df[col].dropna():
                if val == '':
                    continue
                type_name = type(val).__name__
                type_distribution[type_name] = type_distribution.get(type_name, 0) + 1
            
            is_mixed_type = len(type_distribution) > 1
            if is_mixed_type:
                mixed_type_columns.append(col)
        
        # Infer the most likely intended type
        inferred_type = _infer_column_type(df[col])
        
        columns_stats[col] = {
            "pandas_dtype": pandas_dtype,
            "inferred_type": inferred_type,
            "is_mixed_type": is_mixed_type,
            "type_distribution": type_distribution if type_distribution else {pandas_dtype: len(df[col].dropna())}
        }
    
    return {
        "columns": columns_stats,
        "mixed_type_columns": mixed_type_columns,
        "dtype_summary": dtype_summary
    }


def _infer_column_type(series: pd.Series) -> str:
    """
    Infer the most likely intended data type for a column.
    
    Args:
        series: A pandas Series to analyze
        
    Returns:
        String describing the inferred type (e.g., "numeric", "text", "date", "boolean", "categorical")
    """
    if series.dtype in ['int64', 'int32', 'float64', 'float32']:
        return "numeric"
    
    if series.dtype == 'bool':
        return "boolean"
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # For object dtype, try to infer
    non_null = series.dropna()
    non_null = non_null[non_null != '']
    
    if len(non_null) == 0:
        return "unknown"
    
    # Sample for performance
    sample = non_null.head(100)
    
    # Check if it looks like dates
    try:
        pd.to_datetime(sample, errors='raise')
        return "datetime"
    except (ValueError, TypeError):
        pass
    
    # Check if it looks like numbers
    try:
        pd.to_numeric(sample.astype(str).str.replace(',', '').str.replace('$', '').str.replace('€', '').str.replace('£', ''), errors='raise')
        return "numeric"
    except (ValueError, TypeError):
        pass
    
    # Check if it looks like boolean
    bool_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
    if all(str(v).lower().strip() in bool_values for v in sample):
        return "boolean"
    
    # Check if categorical (low cardinality)
    unique_ratio = len(non_null.unique()) / len(non_null) if len(non_null) > 0 else 0
    if unique_ratio < 0.05 and len(non_null.unique()) < 50:
        return "categorical"
    
    return "text"


def _compute_category_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect anomalies in categorical columns (inconsistent categories).
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing category anomaly analysis:
        {
            "columns": {
                "column_name": {
                    "unique_values": int,
                    "potential_duplicates": [("value1", "value2"), ...],
                    "rare_categories": ["rare1", "rare2", ...],
                    "sample_values": ["val1", "val2", ...]
                }
            }
        }
    """
    columns_stats = {}
    
    for col in df.columns:
        if df[col].dtype != object:
            continue
        
        non_null = df[col].dropna()
        non_null = non_null[non_null != '']
        
        if len(non_null) == 0:
            continue
        
        unique_values = non_null.unique()
        value_counts = non_null.value_counts()
        
        # Skip if too many unique values (not categorical)
        if len(unique_values) > 100:
            continue
        
        # Find potential duplicate categories (case/whitespace variations)
        potential_duplicates = []
        normalized_to_original: Dict[str, List[str]] = {}
        
        for val in unique_values:
            if not isinstance(val, str):
                continue
            normalized = val.lower().strip()
            if normalized not in normalized_to_original:
                normalized_to_original[normalized] = []
            normalized_to_original[normalized].append(val)
        
        for normalized, originals in normalized_to_original.items():
            if len(originals) > 1:
                potential_duplicates.append(tuple(originals))
        
        # Find rare categories (less than 1% of total)
        threshold = len(non_null) * 0.01
        rare_categories = [str(cat) for cat, count in value_counts.items() if count < threshold and count <= 3]
        
        if potential_duplicates or rare_categories:
            columns_stats[col] = {
                "unique_values": len(unique_values),
                "potential_duplicates": potential_duplicates[:10],  # Limit to 10
                "rare_categories": rare_categories[:10],  # Limit to 10
                "sample_values": [str(v) for v in list(unique_values)[:MAX_SAMPLE_VALUES]]
            }
    
    return {"columns": columns_stats}


def _compute_outlier_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns using Z-score method.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing outlier analysis:
        {
            "columns": {
                "column_name": {
                    "outlier_count": int,
                    "outlier_percentage": float,
                    "min": float,
                    "max": float,
                    "mean": float,
                    "std": float,
                    "outlier_values": [val1, val2, ...]
                }
            },
            "columns_with_outliers": ["col1", "col2", ...]
        }
    """
    columns_stats = {}
    columns_with_outliers = []
    
    for col in df.columns:
        # Only analyze numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        non_null = df[col].dropna()
        
        if len(non_null) < 3:  # Need at least 3 values for meaningful stats
            continue
        
        mean = float(non_null.mean())
        std = float(non_null.std())
        
        if std == 0:  # All values are the same
            continue
        
        # Calculate Z-scores
        z_scores = np.abs((non_null - mean) / std)
        outlier_mask = z_scores > OUTLIER_ZSCORE_THRESHOLD
        outlier_count = int(outlier_mask.sum())
        
        if outlier_count > 0:
            outlier_values = non_null[outlier_mask].tolist()[:MAX_SAMPLE_VALUES]
            
            columns_stats[col] = {
                "outlier_count": outlier_count,
                "outlier_percentage": round((outlier_count / len(non_null)) * 100, 2),
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": round(mean, 2),
                "std": round(std, 2),
                "outlier_values": [round(v, 2) if isinstance(v, float) else v for v in outlier_values]
            }
            columns_with_outliers.append(col)
    
    return {
        "columns": columns_stats,
        "columns_with_outliers": columns_with_outliers
    }


def _compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic dataset statistics.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary containing basic stats:
        {
            "row_count": int,
            "column_count": int,
            "memory_usage_mb": float,
            "column_names": ["col1", "col2", ...]
        }
    """
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
    
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage_mb": round(memory_usage, 2),
        "column_names": list(df.columns)
    }


def _build_analysis_json(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a comprehensive JSON object containing all data quality analysis.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Complete analysis dictionary ready for OpenAI API
    """
    return {
        "basic_stats": _compute_basic_stats(df),
        "missing_values": _compute_missing_value_stats(df),
        "duplicates": _compute_duplicate_stats(df),
        "column_types": _compute_column_type_stats(df),
        "category_anomalies": _compute_category_anomalies(df),
        "outliers": _compute_outlier_stats(df)
    }


# ============================================
# OpenAI API Integration
# ============================================

def _get_openai_client() -> OpenAI:
    """
    Initialize and return an OpenAI client using the API key from environment.
    
    Returns:
        Configured OpenAI client
        
    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it with your OpenAI API key to use AI-powered summaries."
        )
    
    return OpenAI(api_key=api_key)


def _build_system_prompt() -> str:
    """
    Build the system prompt for the data quality summary generation.
    
    Returns:
        System prompt string
    """
    return """You are a data quality analyst expert. Your task is to analyze data quality statistics 
and generate a clear, actionable summary of data quality issues.

Your summary should be:
1. Concise but comprehensive
2. Prioritized by severity (most critical issues first)
3. Actionable (suggest what should be done)
4. Written in plain English for non-technical users

Structure your response with these sections:
- **Overview**: Brief summary of overall data quality
- **Critical Issues**: Problems that must be addressed
- **Warnings**: Issues that may need attention
- **Recommendations**: Prioritized list of suggested actions

Do NOT include raw numbers or statistics - interpret them meaningfully.
Do NOT use technical jargon unless necessary.
Do NOT suggest how to clean the data programmatically - just identify the issues."""


def _build_user_prompt(analysis: Dict[str, Any]) -> str:
    """
    Build the user prompt containing the data quality analysis.
    
    Args:
        analysis: The complete analysis dictionary
        
    Returns:
        User prompt string
    """
    return f"""Please analyze the following data quality statistics and provide a natural-language summary 
of the data quality issues found in this dataset.

Dataset Analysis:
```json
{json.dumps(analysis, indent=2, default=str)}
```

Generate a comprehensive but readable summary of the data quality issues."""


# ============================================
# Main Public Function
# ============================================

def generate_data_quality_summary(
    df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    include_raw_analysis: bool = False
) -> Dict[str, Any]:
    """
    Generate an AI-powered natural-language summary of data quality issues.
    
    This function analyzes a pandas DataFrame, computes comprehensive statistics
    about data quality issues, and uses the OpenAI API to generate a readable
    summary. It does NOT modify or clean any data.
    
    Args:
        df: The pandas DataFrame to analyze
        model: The OpenAI model to use (default: "gpt-4.1")
        include_raw_analysis: If True, include the raw analysis JSON in the response
        
    Returns:
        Dictionary containing:
        {
            "summary": str,  # The natural-language summary
            "analysis": Dict (optional),  # Raw analysis if include_raw_analysis=True
            "model_used": str,  # The model that generated the summary
            "success": bool,  # Whether the generation was successful
            "error": str (optional)  # Error message if success=False
        }
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
        
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("messy_data.csv")
        >>> result = generate_data_quality_summary(df)
        >>> print(result["summary"])
    """
    # Validate input
    if df is None:
        return {
            "summary": "",
            "success": False,
            "error": "DataFrame is None"
        }
    
    if len(df) == 0:
        return {
            "summary": "The dataset is empty. No data quality analysis can be performed.",
            "success": True,
            "model_used": model
        }
    
    try:
        # Step 1: Compute comprehensive analysis
        analysis = _build_analysis_json(df)
        
        # Step 2: Initialize OpenAI client
        client = _get_openai_client()
        
        # Step 3: Build prompts
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(analysis)
        
        # Step 4: Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=2000
        )
        
        # Step 5: Extract and return the summary
        summary = response.choices[0].message.content.strip()
        
        result = {
            "summary": summary,
            "success": True,
            "model_used": model
        }
        
        if include_raw_analysis:
            result["analysis"] = analysis
        
        return result
        
    except ValueError as e:
        # API key not set
        return {
            "summary": "",
            "success": False,
            "error": str(e)
        }
    except OpenAIError as e:
        # OpenAI API error
        return {
            "summary": "",
            "success": False,
            "error": f"OpenAI API error: {str(e)}"
        }
    except Exception as e:
        # Unexpected error
        return {
            "summary": "",
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def get_raw_data_quality_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get the raw data quality analysis without generating an AI summary.
    
    Useful for debugging or when you want to see the exact statistics
    that would be sent to the AI model.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Complete analysis dictionary with all computed statistics
        
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("messy_data.csv")
        >>> analysis = get_raw_data_quality_analysis(df)
        >>> print(json.dumps(analysis, indent=2))
    """
    if df is None or len(df) == 0:
        return {}
    
    return _build_analysis_json(df)


# ============================================
# Example Usage (for documentation)
# ============================================

if __name__ == "__main__":
    # Example: Generate a data quality summary
    # 
    # Prerequisites:
    # 1. Set the OPENAI_API_KEY environment variable
    # 2. Have a CSV file to analyze
    #
    # Usage:
    #   export OPENAI_API_KEY="your-api-key-here"
    #   python -m app.services.ai_summary
    
    import sys
    
    # Create sample messy data for demonstration
    sample_data = {
        "name": ["John", "JOHN", "jane", "  Mary  ", None, "Bob", "Bob"],
        "email": ["john@example.com", "invalid-email", "jane@test.com", "", "mary@example.com", "bob@test.com", "bob@test.com"],
        "age": [25, 30, "twenty-five", 35, 40, 1000, 45],  # Mixed types and outlier
        "status": ["Active", "active", "ACTIVE", "Inactive", "inactive", "Pending", "pending"],
        "score": [85.5, 90.0, 75.5, None, 88.0, 92.0, 92.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("=" * 60)
    print("AI-Powered Data Quality Summary Demo")
    print("=" * 60)
    print(f"\nSample DataFrame:\n{df}\n")
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Showing raw analysis only.\n")
        analysis = get_raw_data_quality_analysis(df)
        print("Raw Analysis:")
        print(json.dumps(analysis, indent=2, default=str))
    else:
        # Generate the AI summary
        result = generate_data_quality_summary(df, include_raw_analysis=True)
        
        if result["success"]:
            print(f"Model Used: {result['model_used']}\n")
            print("Summary:")
            print("-" * 40)
            print(result["summary"])
        else:
            print(f"Error: {result['error']}")
            
    print("\n" + "=" * 60)
