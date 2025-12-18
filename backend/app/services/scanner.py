"""
Scanner Service - Data Quality Issue Detection

This module analyzes pandas DataFrames to find data quality problems.
It is READ-ONLY—it never modifies data, only reports what it finds.

ROLE IN THE SYSTEM:
───────────────────
Scanner is the "diagnostic" step. Before cleaning anything, we need to know
what's wrong. This module answers: "What issues exist in this data?"

The scan results are shown to users so they can decide which issues to fix.
The actual fixing is done by cleaner.py (for rule-based) or suggestion_engine.py
(for the safe review workflow).

DETECTED ISSUES:
────────────────
• Missing values - NaN, None, empty strings
• Duplicate rows - Exact row-level duplicates  
• Mixed types - Column contains both strings and numbers
• Invalid formats - Emails without @, malformed dates, etc.
• Whitespace issues - Leading/trailing spaces
• Number formatting - Inconsistent use of commas, currency symbols

OUTPUT FORMAT:
──────────────
The main function scan_dataframe() returns a ScanReport containing:
- total_rows / total_columns: Basic dataset dimensions
- issues: List of DataIssue objects (one per detected problem)
- column_stats: Per-column statistics (type, nulls, unique values)
- summary: Aggregate counts and percentages
"""

import re
from typing import Any, Dict, List, Set, Union

import pandas as pd
import numpy as np

from app.models.schemas import ScanReport, DataIssue
from app.services.cleaner import is_number_phrase, parse_number_phrase


# ============================================
# Regex Patterns for Validation
# ============================================
# These patterns are used to detect invalid or inconsistent formats.
# They're intentionally simple to avoid false positives.

# Email: Must have @ and a domain with at least 2-char TLD
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# Currency: Optional symbol, optional negative, digits with commas, optional decimals
CURRENCY_PATTERN = re.compile(r'^[\$€£¥]?\s*-?\d{1,3}(,\d{3})*(\.\d+)?$')

# Numbers with thousand separators (helps detect "should this be numeric?")
NUMBER_WITH_COMMAS_PATTERN = re.compile(r'^-?\d{1,3}(,\d{3})+(\.\d+)?$')

# Boolean values we recognize (case-insensitive via .lower() before matching)
BOOLEAN_TRUE_VALUES = {'true', '1', 'yes', 'y', 't'}
BOOLEAN_FALSE_VALUES = {'false', '0', 'no', 'n', 'f'}
BOOLEAN_ALL_VALUES = BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES


# ============================================
# Individual Issue Detection Functions
# ============================================
# Each function below detects ONE type of issue. They return dicts that
# can be aggregated into the final ScanReport.

def detect_missing_values(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect missing values (NaN, None, empty strings) per column.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to missing value statistics:
        {
            "column_name": {
                "issue": "missing_values",
                "count": int,
                "percentage": float
            }
        }
    """
    result = {}
    total_rows = len(df)
    
    if total_rows == 0:
        return result
    
    for col in df.columns:
        # Count NaN/None values
        null_count = df[col].isna().sum()
        
        # Also count empty strings for string columns
        empty_string_count = 0
        if df[col].dtype == object:
            empty_string_count = (df[col] == '').sum()
        
        total_missing = null_count + empty_string_count
        
        if total_missing > 0:
            result[col] = {
                "issue": "missing_values",
                "count": int(total_missing),
                "percentage": round((total_missing / total_rows) * 100, 2)
            }
    
    return result


def detect_mixed_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect columns with inconsistent/mixed data types.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to type inconsistency info:
        {
            "column_name": {
                "issue": "mixed_types",
                "types": ["int", "str", ...],
                "type_counts": {"int": 10, "str": 5}
            }
        }
    """
    result = {}
    
    for col in df.columns:
        # Skip columns with uniform dtype that's not object
        if df[col].dtype != object:
            continue
        
        # For object columns, check actual types of non-null values
        type_counts: Dict[str, int] = {}
        
        for val in df[col].dropna():
            type_name = type(val).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # If more than one type exists, report mixed types
        if len(type_counts) > 1:
            result[col] = {
                "issue": "mixed_types",
                "types": list(type_counts.keys()),
                "type_counts": type_counts
            }
    
    return result


def detect_whitespace_issues(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect leading/trailing whitespace in string columns.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to whitespace issue info:
        {
            "column_name": {
                "issue": "whitespace",
                "count": int,
                "affected_indices": [row_indices...]
            }
        }
    """
    result = {}
    
    for col in df.columns:
        if df[col].dtype != object:
            continue
        
        affected_indices = []
        
        for idx, val in df[col].items():
            if isinstance(val, str):
                # Check for leading or trailing whitespace
                if val != val.strip():
                    affected_indices.append(int(idx))
        
        if affected_indices:
            result[col] = {
                "issue": "whitespace",
                "count": len(affected_indices),
                "affected_indices": affected_indices[:100]  # Limit to first 100
            }
    
    return result


def detect_duplicate_rows(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect exact duplicate rows in the DataFrame.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary with duplicate row information:
        {
            "duplicates": [row_indices...],
            "count": int,
            "duplicate_groups": [[indices_of_same_rows], ...]
        }
    """
    if len(df) == 0:
        return {"duplicates": [], "count": 0, "duplicate_groups": []}
    
    # Find all duplicates (keep=False marks all duplicates, not just subsequent ones)
    duplicate_mask = df.duplicated(keep=False)
    duplicate_indices = df[duplicate_mask].index.tolist()
    
    # Group duplicates by their values
    duplicate_groups = []
    if duplicate_indices:
        # Use a hash of each row to group identical rows
        seen_hashes: Dict[int, List[int]] = {}
        for idx in duplicate_indices:
            row_hash = hash(tuple(df.loc[idx].values))
            if row_hash not in seen_hashes:
                seen_hashes[row_hash] = []
            seen_hashes[row_hash].append(int(idx))
        
        duplicate_groups = [group for group in seen_hashes.values() if len(group) > 1]
    
    return {
        "duplicates": [int(i) for i in duplicate_indices],
        "count": len(duplicate_indices),
        "duplicate_groups": duplicate_groups
    }


def detect_invalid_emails(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect invalid email formats in string columns that appear to contain emails.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to invalid email info:
        {
            "column_name": {
                "issue": "invalid_email",
                "count": int,
                "invalid_indices": [row_indices...],
                "examples": ["bad@", "notanemail", ...]
            }
        }
    """
    result = {}
    
    for col in df.columns:
        if df[col].dtype != object:
            continue
        
        # Check if column likely contains emails (at least some values have @)
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            continue
        
        has_at_symbol = non_null_values.astype(str).str.contains('@', na=False)
        if has_at_symbol.sum() < len(non_null_values) * 0.3:  # Less than 30% have @
            continue
        
        invalid_indices = []
        invalid_examples = []
        
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            val_str = str(val)
            if '@' in val_str and not EMAIL_PATTERN.match(val_str):
                invalid_indices.append(int(idx))
                if len(invalid_examples) < 5:
                    invalid_examples.append(val_str)
        
        if invalid_indices:
            result[col] = {
                "issue": "invalid_email",
                "count": len(invalid_indices),
                "invalid_indices": invalid_indices[:100],
                "examples": invalid_examples
            }
    
    return result


def detect_invalid_dates(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect invalid date formats in columns that appear to contain dates.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to invalid date info:
        {
            "column_name": {
                "issue": "invalid_date",
                "count": int,
                "invalid_indices": [row_indices...],
                "examples": ["not-a-date", "13/45/2023", ...]
            }
        }
    """
    result = {}
    
    # Common date patterns to check
    date_keywords = ['date', 'time', 'created', 'updated', 'born', 'dob', 'timestamp']
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name suggests dates or if dtype is datetime
        is_date_column = any(kw in col_lower for kw in date_keywords)
        is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(df[col])
        
        if not is_date_column and not is_datetime_dtype and df[col].dtype != object:
            continue
        
        if df[col].dtype == object or is_date_column:
            invalid_indices = []
            invalid_examples = []
            valid_count = 0
            
            for idx, val in df[col].items():
                if pd.isna(val) or val == '':
                    continue
                
                try:
                    pd.to_datetime(val)
                    valid_count += 1
                except (ValueError, TypeError):
                    invalid_indices.append(int(idx))
                    if len(invalid_examples) < 5:
                        invalid_examples.append(str(val))
            
            # Only report if column seems to contain dates (some valid dates exist)
            if invalid_indices and valid_count > 0:
                result[col] = {
                    "issue": "invalid_date",
                    "count": len(invalid_indices),
                    "invalid_indices": invalid_indices[:100],
                    "examples": invalid_examples
                }
    
    return result


def detect_number_formatting_issues(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect number formatting inconsistencies like commas, currency symbols, etc.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to formatting issue info:
        {
            "column_name": {
                "issue": "number_formatting",
                "count": int,
                "has_commas": bool,
                "has_currency": bool,
                "affected_indices": [row_indices...],
                "examples": ["$1,234.56", "1,000", ...]
            }
        }
    """
    result = {}
    
    for col in df.columns:
        if df[col].dtype != object:
            continue
        
        affected_indices = []
        examples = []
        has_commas = False
        has_currency = False
        
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            
            val_str = str(val).strip()
            
            # Check for currency symbols
            if re.match(r'^[\$€£¥]', val_str):
                has_currency = True
                affected_indices.append(int(idx))
                if len(examples) < 5:
                    examples.append(val_str)
            # Check for numbers with commas as thousand separators
            elif NUMBER_WITH_COMMAS_PATTERN.match(val_str):
                has_commas = True
                affected_indices.append(int(idx))
                if len(examples) < 5:
                    examples.append(val_str)
        
        if affected_indices:
            result[col] = {
                "issue": "number_formatting",
                "count": len(affected_indices),
                "has_commas": has_commas,
                "has_currency": has_currency,
                "affected_indices": affected_indices[:100],
                "examples": examples
            }
    
    return result


def detect_number_words(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect number words that can be converted to digits.
    
    This detects both single words ("twenty", "five") and compound phrases
    ("sixty thousand", "two hundred fifty").
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to number word issue info:
        {
            "column_name": {
                "issue": "number_words",
                "count": int,
                "affected_indices": [row_indices...],
                "examples": [("sixty thousand", 60000), ...]
            }
        }
    """
    result = {}
    
    for col in df.columns:
        if df[col].dtype != object:
            continue
        
        affected_indices = []
        examples = []
        
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            
            val_str = str(val).strip().lower()
            
            # Check if this is a number phrase (single word or compound)
            if is_number_phrase(val_str):
                parsed = parse_number_phrase(val_str)
                if parsed is not None:
                    affected_indices.append(int(idx))
                    if len(examples) < 5:
                        examples.append((str(val).strip(), parsed))
        
        if affected_indices:
            result[col] = {
                "issue": "number_words",
                "count": len(affected_indices),
                "affected_indices": affected_indices[:100],
                "examples": examples
            }
    
    return result


def detect_boolean_inconsistencies(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect columns with inconsistent Boolean value formats.
    
    This function identifies columns that appear to represent True/False values
    but contain mixed formats such as: true, false, 1, 0, yes, no, y, n, t, f
    (case-insensitive).
    
    The function does NOT normalize values; it only reports the inconsistency
    so users can choose how to normalize (e.g., True/False, Yes/No, or 1/0).
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to Boolean inconsistency info:
        {
            "column_name": {
                "issue": "boolean_inconsistency",
                "count": int,
                "formats_found": {"true": 5, "yes": 3, "1": 2, ...},
                "affected_indices": [row_indices...],
                "examples": ["true", "Yes", "1", ...],
                "normalization_options": ["True/False", "Yes/No", "1/0"]
            }
        }
    """
    result = {}
    
    for col in df.columns:
        # Skip non-object columns that are already boolean
        if df[col].dtype == bool:
            continue
        
        # For numeric columns, check if they only contain 0 and 1
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            unique_vals = set(non_null.unique())
            # If it's purely 0s and 1s, it's already consistent - no issue
            if unique_vals <= {0, 1, 0.0, 1.0}:
                continue
            # If it contains other numbers, skip (not a boolean column)
            continue
        
        # For object columns, check for mixed boolean formats
        if df[col].dtype != object:
            continue
        
        formats_found: Dict[str, int] = {}
        affected_indices = []
        examples = []
        non_boolean_count = 0
        total_non_null = 0
        
        for idx, val in df[col].items():
            if pd.isna(val) or val == '':
                continue
            
            total_non_null += 1
            val_str = str(val).strip().lower()
            
            if val_str in BOOLEAN_ALL_VALUES:
                formats_found[val_str] = formats_found.get(val_str, 0) + 1
                affected_indices.append(int(idx))
                # Collect unique format examples (original case)
                original_val = str(val).strip()
                if original_val not in examples and len(examples) < 10:
                    examples.append(original_val)
            else:
                non_boolean_count += 1
        
        # Only report if:
        # 1. We found boolean-like values
        # 2. Multiple different formats are used (inconsistency)
        # 3. Most values appear to be boolean (at least 70% of non-null values)
        if len(formats_found) > 1:
            boolean_count = sum(formats_found.values())
            if total_non_null > 0 and (boolean_count / total_non_null) >= 0.7:
                # Determine which format groups are present
                has_true_false = bool(formats_found.keys() & {'true', 'false'})
                has_yes_no = bool(formats_found.keys() & {'yes', 'no'})
                has_y_n = bool(formats_found.keys() & {'y', 'n'})
                has_t_f = bool(formats_found.keys() & {'t', 'f'})
                has_1_0 = bool(formats_found.keys() & {'1', '0'})
                
                # Count distinct format groups
                format_groups = sum([has_true_false, has_yes_no, has_y_n, has_t_f, has_1_0])
                
                # Only report if multiple format groups are used (real inconsistency)
                if format_groups > 1:
                    result[col] = {
                        "issue": "boolean_inconsistency",
                        "count": boolean_count,
                        "formats_found": formats_found,
                        "affected_indices": affected_indices[:100],
                        "examples": examples,
                        "normalization_options": ["True/False", "Yes/No", "1/0"]
                    }
    
    return result


# ============================================
# Main Analysis Function
# ============================================

def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a DataFrame to detect all data quality issues.
    
    This function runs all individual detection functions and aggregates the results
    into a structured dictionary format.
    
    Args:
        df: The pandas DataFrame to analyze
        
    Returns:
        A structured dictionary summarizing all detected issues:
        {
            "column_issues": {
                "column_name": [
                    {"issue": "missing_values", "count": 5, "percentage": 10.0},
                    {"issue": "mixed_types", "types": ["int", "str"]},
                    ...
                ],
            },
            "row_issues": {
                "duplicates": [row_indices...],
                "duplicate_count": int,
                "duplicate_groups": [[indices], ...]
            },
            "summary": {
                "total_rows": int,
                "total_columns": int,
                "columns_with_issues": int,
                "total_issues_found": int
            }
        }
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'name': ['Alice', 'Bob', None, 'Charlie'],
        ...     'age': [25, '30', 35, 40]
        ... })
        >>> result = analyze_dataframe(df)
        >>> print(result['column_issues']['name'])
        [{'issue': 'missing_values', 'count': 1, 'percentage': 25.0}]
    """
    # Run all detection functions
    missing_values = detect_missing_values(df)
    mixed_types = detect_mixed_types(df)
    whitespace_issues = detect_whitespace_issues(df)
    duplicate_info = detect_duplicate_rows(df)
    invalid_emails = detect_invalid_emails(df)
    invalid_dates = detect_invalid_dates(df)
    number_formatting = detect_number_formatting_issues(df)
    number_words = detect_number_words(df)
    boolean_inconsistencies = detect_boolean_inconsistencies(df)
    
    # Aggregate column issues
    column_issues: Dict[str, List[Dict[str, Any]]] = {}
    
    # Helper to add issues to a column
    def add_column_issue(col: str, issue_dict: Dict[str, Any]):
        if col not in column_issues:
            column_issues[col] = []
        column_issues[col].append(issue_dict)
    
    # Add missing values
    for col, issue in missing_values.items():
        add_column_issue(col, issue)
    
    # Add mixed types
    for col, issue in mixed_types.items():
        add_column_issue(col, issue)
    
    # Add whitespace issues
    for col, issue in whitespace_issues.items():
        add_column_issue(col, issue)
    
    # Add invalid emails
    for col, issue in invalid_emails.items():
        add_column_issue(col, issue)
    
    # Add invalid dates
    for col, issue in invalid_dates.items():
        add_column_issue(col, issue)
    
    # Add number formatting issues
    for col, issue in number_formatting.items():
        add_column_issue(col, issue)
    
    # Add number word issues
    for col, issue in number_words.items():
        add_column_issue(col, issue)
    
    # Add boolean inconsistency issues
    for col, issue in boolean_inconsistencies.items():
        add_column_issue(col, issue)
    
    # Build row issues
    row_issues = {
        "duplicates": duplicate_info["duplicates"],
        "duplicate_count": duplicate_info["count"],
        "duplicate_groups": duplicate_info["duplicate_groups"]
    }
    
    # Calculate summary
    total_issues = sum(len(issues) for issues in column_issues.values())
    if duplicate_info["count"] > 0:
        total_issues += 1  # Count duplicates as one issue type
    
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns_with_issues": len(column_issues),
        "total_issues_found": total_issues,
        "has_duplicates": duplicate_info["count"] > 0
    }
    
    return {
        "column_issues": column_issues,
        "row_issues": row_issues,
        "summary": summary
    }


def scan_dataframe(df: pd.DataFrame) -> ScanReport:
    """
    Scan a DataFrame for data quality issues and return a ScanReport.
    
    This function wraps analyze_dataframe() and converts the results
    to the ScanReport schema format for API responses.
    
    Args:
        df: The pandas DataFrame to scan
        
    Returns:
        ScanReport containing detected issues and statistics
    """
    # Run the analysis
    analysis = analyze_dataframe(df)
    
    # Convert to DataIssue format for the ScanReport
    issues: List[DataIssue] = []
    
    for col, col_issues in analysis["column_issues"].items():
        for issue in col_issues:
            issue_type = issue["issue"]
            count = issue.get("count", 0)
            
            # Determine severity based on issue type and count
            total_rows = len(df)
            if issue_type == "missing_values":
                percentage = issue.get("percentage", 0)
                severity = "high" if percentage > 50 else "medium" if percentage > 10 else "low"
                description = f"{count} missing values ({percentage}%)"
            elif issue_type == "mixed_types":
                severity = "high"
                types = issue.get("types", [])
                description = f"Mixed types detected: {', '.join(types)}"
            elif issue_type == "whitespace":
                severity = "low"
                description = f"{count} values with leading/trailing whitespace"
            elif issue_type == "invalid_email":
                severity = "medium"
                description = f"{count} invalid email formats"
            elif issue_type == "invalid_date":
                severity = "medium"
                description = f"{count} invalid date formats"
            elif issue_type == "number_formatting":
                severity = "low"
                description = f"{count} numbers with formatting (commas/currency)"
            elif issue_type == "number_words":
                severity = "low"
                description = f"{count} number words that can be converted to digits"
            elif issue_type == "boolean_inconsistency":
                severity = "medium"
                formats_found = issue.get("formats_found", {})
                format_list = list(formats_found.keys())
                examples_list = issue.get("examples", format_list)
                normalization_options = issue.get("normalization_options", ["True/False", "Yes/No", "1/0"])
                description = (
                    f"Warning: Column contains {count} Boolean values with inconsistent formats "
                    f"(found: {', '.join(format_list)}). "
                    f"Consider normalizing to one of: {', '.join(normalization_options)}"
                )
            else:
                severity = "medium"
                description = f"{count} issues of type {issue_type}"
            
            issues.append(DataIssue(
                column=col,
                issue_type=issue_type,
                severity=severity,
                count=count,
                description=description,
                examples=issue.get("examples", issue.get("types", None))
            ))
    
    # Add duplicate row issue if exists
    if analysis["row_issues"]["duplicate_count"] > 0:
        issues.append(DataIssue(
            column="_row_",
            issue_type="duplicate_rows",
            severity="medium",
            count=analysis["row_issues"]["duplicate_count"],
            description=f"{analysis['row_issues']['duplicate_count']} duplicate rows found",
            examples=analysis["row_issues"]["duplicates"][:10]
        ))
    
    # Build column stats
    column_stats: Dict[str, Any] = {}
    for col in df.columns:
        column_stats[col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
            "issues": analysis["column_issues"].get(col, [])
        }
    
    return ScanReport(
        total_rows=len(df),
        total_columns=len(df.columns),
        issues=issues,
        column_stats=column_stats,
        summary=analysis["summary"]
    )
