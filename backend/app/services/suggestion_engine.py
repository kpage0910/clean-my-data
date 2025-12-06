"""Suggestion Engine for Safe, Non-Destructive Data Cleaning

FUNDAMENTAL PRINCIPLES (NEVER BREAK THESE):
1. NEVER invent, guess, or fabricate data
2. ONLY apply deterministic, meaning-preserving transformations
3. Automatically apply UNIVERSALLY SAFE operations (whitespace, casing, etc.)
4. NEVER apply HIGH-RISK changes without user approval
5. Missing/invalid cells remain blank unless user selects a placeholder
6. Imputation (filling missing values) is strictly opt-in and OFF by default
7. Dropping rows is strictly opt-in and OFF by default

UNIVERSALLY SAFE (automatic):
- Trim whitespace
- Normalize casing (Title Case for names, lowercase for emails)
- Convert number words to digits
- Normalize dates to ISO format
- Replace missing indicators (na, N/A, -, etc.) with null
- Standardize boolean values
- Deduplicate rows

REQUIRES USER APPROVAL:
- Drop rows
- Fill missing values
- Replace invalid formats
- Any change that could alter meaning

Workflow:
    Step 1: apply_safe_transformations() - Apply universally safe changes automatically
    Step 2: detect_issues() - Identify remaining issues
    Step 3: generate_suggestions() - Create suggested fixes for user review
    Step 4: apply_approved_actions() - Apply ONLY user-approved fixes
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from app.models.schemas import (
    SuggestedAction,
    RowDropReason,
    CellIssueSuggestion,
    RowIssueSuggestion,
    DetectedIssuesReport,
    UserApprovedAction,
    ColumnTypeInference,
    StrictModeConfig,
    DEFAULT_STRICT_CONFIG,
)
from app.services.autonomous import (
    infer_column_type,
    infer_all_column_types,
    InferredType,
    EMAIL_PATTERN,
    NUMBER_WORDS,
)
from app.services.cleaner import (
    is_number_phrase,
    parse_number_phrase,
)


# ============================================
# Universally Safe Transformations (Automatic)
# ============================================

def apply_safe_transformations(
    df: pd.DataFrame,
    strict_config: Optional[StrictModeConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply universally safe transformations automatically.
    
    These transformations are deterministic, meaning-preserving, and
    do NOT require user approval:
    
    - Trim whitespace from all string columns
    - Normalize missing indicators (na, N/A, -, etc.) to null
    - Normalize casing based on column type:
      - Names → Title Case
      - Emails → lowercase
    - Convert number words to digits (thirty → 30)
    - Normalize dates to ISO format (YYYY-MM-DD)
    - Standardize boolean values (yes/no → true/false)
    - Remove duplicate rows
    
    Args:
        df: The DataFrame to clean
        strict_config: Strict mode configuration
        
    Returns:
        Tuple of (cleaned_df, summary_of_changes)
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    from app.services.cleaner import (
        normalize_missing_indicators,
        trim_whitespace,
        normalize_capitalization,
        convert_number_words,
        parse_dates,
        coerce_types,
        dedupe,
        is_number_phrase,
        parse_number_phrase,
    )
    
    result = df.copy()
    summary = {
        "transformations_applied": [],
        "rows_before": len(df),
        "rows_after": 0,
    }
    
    # Step 1: Normalize missing indicators (na, N/A, -, etc. → null)
    result = normalize_missing_indicators(result)
    summary["transformations_applied"].append("Replaced missing indicators (na, N/A, -, etc.) with null")
    
    # Step 2: Trim whitespace from all string columns
    result = trim_whitespace(result)
    summary["transformations_applied"].append("Trimmed whitespace from all string columns")
    
    # Step 3: Infer column types and apply type-specific normalization
    column_inferences = infer_all_column_types(result)
    
    for inf in column_inferences:
        col = inf.column
        if col not in result.columns:
            continue
        
        col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
        
        # Name columns → Title Case
        if col_type == 'name':
            result = normalize_capitalization(result, col, style='title')
            summary["transformations_applied"].append(f"Normalized '{col}' to Title Case")
        
        # Email columns → lowercase
        elif col_type == 'email':
            result = normalize_capitalization(result, col, style='lower')
            summary["transformations_applied"].append(f"Normalized '{col}' to lowercase")
        
        # Numeric columns → Convert number words
        elif col_type in ('numeric', 'currency', 'percentage'):
            result = convert_number_words(result, col)
            summary["transformations_applied"].append(f"Converted number words in '{col}'")
        
        # Date columns → ISO format
        elif col_type == 'date':
            result = parse_dates(result, col)
            summary["transformations_applied"].append(f"Normalized dates in '{col}' to ISO format")
        
        # Boolean columns → standardize
        elif col_type == 'boolean':
            result = coerce_types(result, col, dtype='bool')
            summary["transformations_applied"].append(f"Standardized booleans in '{col}'")
    
    # Step 4: Deduplicate rows (last step)
    rows_before_dedupe = len(result)
    result = dedupe(result)
    rows_removed = rows_before_dedupe - len(result)
    if rows_removed > 0:
        summary["transformations_applied"].append(f"Removed {rows_removed} duplicate rows")
    
    summary["rows_after"] = len(result)
    
    return result, summary


# ============================================
# Issue Detection (Step 1 - NO CHANGES)
# ============================================

class IssueType(str, Enum):
    """Types of data quality issues that can be detected."""
    MISSING_VALUE = "missing_value"
    INVALID_FORMAT = "invalid_format"
    TYPE_MISMATCH = "type_mismatch"
    WHITESPACE = "whitespace"
    CAPITALIZATION = "capitalization"
    NUMBER_WORD = "number_word"
    CURRENCY_FORMAT = "currency_format"
    PERCENTAGE_FORMAT = "percentage_format"
    INVALID_DATE = "invalid_date"
    DUPLICATE_ROW = "duplicate_row"
    EMPTY_ROW = "empty_row"
    STRUCTURAL_ISSUE = "structural_issue"


def detect_cell_issues(
    df: pd.DataFrame,
    column_inferences: List[ColumnTypeInference],
    strict_config: Optional[StrictModeConfig] = None,
) -> List[CellIssueSuggestion]:
    """
    Detect cell-level issues in the DataFrame.
    
    Returns a list of issues with suggested actions, but DOES NOT modify data.
    
    Detected issues include:
    - Missing values
    - Invalid formats (email, date, etc.)
    - Type mismatches
    - Whitespace issues
    - Capitalization inconsistencies
    - Number words that could be converted
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    issues: List[CellIssueSuggestion] = []
    
    for inf in column_inferences:
        col = inf.column
        if col not in df.columns:
            continue
        
        col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
        
        for idx in range(len(df)):
            val = df[col].iloc[idx]
            
            # Check for missing values
            if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                issues.append(_create_missing_value_suggestion(idx, col, val, col_type, strict_config))
                continue
            
            # Type-specific issue detection
            if col_type == 'email':
                issue = _detect_email_issue(idx, col, val, strict_config)
                if issue:
                    issues.append(issue)
            
            elif col_type == 'name':
                issue = _detect_name_issue(idx, col, val, strict_config)
                if issue:
                    issues.append(issue)
            
            elif col_type in ('numeric', 'currency', 'percentage'):
                issue = _detect_numeric_issue(idx, col, val, col_type, strict_config)
                if issue:
                    issues.append(issue)
            
            elif col_type == 'date':
                issue = _detect_date_issue(idx, col, val, strict_config)
                if issue:
                    issues.append(issue)
            
            # Universal checks (for all types)
            else:
                # Check for whitespace issues
                if isinstance(val, str) and val != val.strip():
                    issues.append(_create_whitespace_suggestion(idx, col, val, strict_config))
    
    return issues


def detect_row_issues(
    df: pd.DataFrame,
    strict_config: Optional[StrictModeConfig] = None,
) -> List[RowIssueSuggestion]:
    """
    Detect row-level issues in the DataFrame.
    
    Returns a list of row issues with suggested actions.
    
    ROW DROPPING RULES (CONSERVATIVE):
    Rows may only be suggested for dropping if:
    1. The row is completely empty
    2. The row has fewer columns than the header (structural corruption)
    3. The row violates schema rules that cannot be repaired
    4. User defines a custom rule (handled separately)
    
    Otherwise, the row is ALWAYS kept by default.
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    issues: List[RowIssueSuggestion] = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        row_dict = row.to_dict()
        
        # Check for completely empty row
        is_all_null = row.isna().all()
        is_all_empty_str = False
        try:
            is_all_empty_str = all(
                pd.isna(v) or (isinstance(v, str) and v.strip() == '')
                for v in row.values
            )
        except (TypeError, ValueError):
            pass
        
        if is_all_null or is_all_empty_str:
            issues.append(RowIssueSuggestion(
                row_index=idx,
                row_data=_convert_row_values(row_dict),
                issue_type=IssueType.EMPTY_ROW.value,
                issue_description="Row is completely empty (all values are null or empty)",
                drop_reason=RowDropReason.COMPLETELY_EMPTY,
                drop_recommended=True,  # Only case where drop is recommended
                available_actions=[
                    SuggestedAction.LEAVE_AS_IS,
                    SuggestedAction.DROP_ROW
                ]
            ))
    
    # Check for duplicate rows
    duplicates = df.duplicated(keep=False)
    if duplicates.any():
        # Group duplicates
        dup_indices = df[duplicates].index.tolist()
        seen_hashes: Dict[int, List[int]] = {}
        
        for idx in dup_indices:
            row_hash = hash(tuple(str(v) for v in df.iloc[idx].values))
            if row_hash not in seen_hashes:
                seen_hashes[row_hash] = []
            seen_hashes[row_hash].append(idx)
        
        # For each group of duplicates, mark all but first
        for indices in seen_hashes.values():
            if len(indices) > 1:
                for idx in indices[1:]:  # Skip first (keep it)
                    issues.append(RowIssueSuggestion(
                        row_index=idx,
                        row_data=_convert_row_values(df.iloc[idx].to_dict()),
                        issue_type=IssueType.DUPLICATE_ROW.value,
                        issue_description=f"Row is a duplicate of row {indices[0]}",
                        drop_reason=None,  # Duplicates don't auto-qualify for drop
                        drop_recommended=False,  # Conservative: don't recommend dropping
                        available_actions=[
                            SuggestedAction.LEAVE_AS_IS,
                            SuggestedAction.DROP_ROW  # User can choose to drop
                        ]
                    ))
    
    return issues


def generate_suggestions(
    df: pd.DataFrame,
    strict_config: Optional[StrictModeConfig] = None,
) -> DetectedIssuesReport:
    """
    Generate a complete report of detected issues with suggested fixes.
    
    This is the main entry point for Steps 1 & 2 of the workflow:
    - Detects all issues WITHOUT making any changes
    - Generates suggested fixes for user review
    
    The user must then approve which actions to apply.
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    # Infer column types
    column_inferences = infer_all_column_types(df)
    
    # Convert to schema format
    col_inf_schemas = [
        ColumnTypeInference(
            column=inf.column,
            inferred_type=inf.inferred_type.value if hasattr(inf.inferred_type, 'value') else str(inf.inferred_type),
            confidence=inf.confidence,
            indicators=inf.indicators,
            is_safe=inf.is_safe,
            warning=inf.warning
        )
        for inf in column_inferences
    ]
    
    # Detect issues
    cell_issues = detect_cell_issues(df, col_inf_schemas, strict_config)
    row_issues = detect_row_issues(df, strict_config)
    
    # Generate warnings
    warnings = []
    if strict_config.enabled:
        warnings.append(
            "STRICT MODE: Only deterministic, meaning-preserving transformations are suggested. "
            "No data will be fabricated or guessed."
        )
    
    # Count issues by type
    missing_count = sum(1 for i in cell_issues if i.issue_type == IssueType.MISSING_VALUE.value)
    if missing_count > 0:
        warnings.append(
            f"Found {missing_count} missing values. These will remain blank unless you choose a placeholder."
        )
    
    return DetectedIssuesReport(
        file_id="",  # Will be set by the caller
        total_rows=len(df),
        total_columns=len(df.columns),
        total_issues=len(cell_issues) + len(row_issues),
        cell_issues_count=len(cell_issues),
        row_issues_count=len(row_issues),
        cell_issues=cell_issues,
        row_issues=row_issues,
        column_inferences=col_inf_schemas,
        warnings=warnings
    )


# ============================================
# Suggestion Helpers
# ============================================

def _create_missing_value_suggestion(
    idx: int,
    col: str,
    val: Any,
    col_type: str,
    strict_config: StrictModeConfig,
) -> CellIssueSuggestion:
    """Create a suggestion for a missing value."""
    placeholder = strict_config.unknown_placeholder
    
    return CellIssueSuggestion(
        row_index=idx,
        column=col,
        original_value=_convert_value(val),
        issue_type=IssueType.MISSING_VALUE.value,
        issue_description=f"Missing value in {col_type} column",
        available_actions=[
            SuggestedAction.LEAVE_AS_IS,
            SuggestedAction.REPLACE_WITH_BLANK,
            SuggestedAction.REPLACE_WITH_PLACEHOLDER,
        ],
        recommended_action=SuggestedAction.LEAVE_AS_IS,  # Conservative default
        action_previews={
            SuggestedAction.LEAVE_AS_IS.value: val,
            SuggestedAction.REPLACE_WITH_BLANK.value: None,
            SuggestedAction.REPLACE_WITH_PLACEHOLDER.value: placeholder,
        },
        deterministic_fix_value=None,  # No deterministic fix for missing values
        deterministic_fix_explanation=None
    )


def _detect_email_issue(
    idx: int,
    col: str,
    val: Any,
    strict_config: StrictModeConfig,
) -> Optional[CellIssueSuggestion]:
    """Detect email-related issues."""
    if not isinstance(val, str):
        return None
    
    val_str = val.strip()
    val_lower = val_str.lower()
    
    # Check for case normalization (deterministic fix)
    if val_str != val_lower and EMAIL_PATTERN.match(val_lower):
        return CellIssueSuggestion(
            row_index=idx,
            column=col,
            original_value=val,
            issue_type=IssueType.CAPITALIZATION.value,
            issue_description="Email contains uppercase letters (emails are case-insensitive)",
            available_actions=[
                SuggestedAction.LEAVE_AS_IS,
                SuggestedAction.APPLY_DETERMINISTIC_FIX,
            ],
            recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
            action_previews={
                SuggestedAction.LEAVE_AS_IS.value: val,
                SuggestedAction.APPLY_DETERMINISTIC_FIX.value: val_lower,
            },
            deterministic_fix_value=val_lower,
            deterministic_fix_explanation="Convert to lowercase (meaning-preserving: email case is not significant)"
        )
    
    # Check for invalid email format
    if not EMAIL_PATTERN.match(val_lower):
        return CellIssueSuggestion(
            row_index=idx,
            column=col,
            original_value=val,
            issue_type=IssueType.INVALID_FORMAT.value,
            issue_description="Invalid email format",
            available_actions=[
                SuggestedAction.LEAVE_AS_IS,
                SuggestedAction.REPLACE_WITH_BLANK,
                SuggestedAction.REPLACE_WITH_PLACEHOLDER,
            ],
            recommended_action=SuggestedAction.LEAVE_AS_IS,  # Conservative
            action_previews={
                SuggestedAction.LEAVE_AS_IS.value: val,
                SuggestedAction.REPLACE_WITH_BLANK.value: None,
                SuggestedAction.REPLACE_WITH_PLACEHOLDER.value: strict_config.unknown_placeholder,
            },
            deterministic_fix_value=None,  # Cannot fix invalid email deterministically
            deterministic_fix_explanation=None
        )
    
    return None


def _detect_name_issue(
    idx: int,
    col: str,
    val: Any,
    strict_config: StrictModeConfig,
) -> Optional[CellIssueSuggestion]:
    """Detect name-related issues."""
    if not isinstance(val, str):
        return None
    
    val_str = val.strip()
    
    # Check for whitespace issues
    if val != val_str:
        return CellIssueSuggestion(
            row_index=idx,
            column=col,
            original_value=val,
            issue_type=IssueType.WHITESPACE.value,
            issue_description="Name has leading/trailing whitespace",
            available_actions=[
                SuggestedAction.LEAVE_AS_IS,
                SuggestedAction.APPLY_DETERMINISTIC_FIX,
            ],
            recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
            action_previews={
                SuggestedAction.LEAVE_AS_IS.value: val,
                SuggestedAction.APPLY_DETERMINISTIC_FIX.value: val_str,
            },
            deterministic_fix_value=val_str,
            deterministic_fix_explanation="Remove leading/trailing whitespace (meaning-preserving)"
        )
    
    # Check for capitalization normalization
    val_title = val_str.title()
    if val_str != val_title and (val_str.isupper() or val_str.islower()):
        return CellIssueSuggestion(
            row_index=idx,
            column=col,
            original_value=val,
            issue_type=IssueType.CAPITALIZATION.value,
            issue_description="Name capitalization could be normalized to Title Case",
            available_actions=[
                SuggestedAction.LEAVE_AS_IS,
                SuggestedAction.APPLY_DETERMINISTIC_FIX,
            ],
            recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
            action_previews={
                SuggestedAction.LEAVE_AS_IS.value: val,
                SuggestedAction.APPLY_DETERMINISTIC_FIX.value: val_title,
            },
            deterministic_fix_value=val_title,
            deterministic_fix_explanation="Convert to Title Case (meaning-preserving: 'JOHN' and 'John' represent same name)"
        )
    
    return None


def _detect_numeric_issue(
    idx: int,
    col: str,
    val: Any,
    col_type: str,
    strict_config: StrictModeConfig,
) -> Optional[CellIssueSuggestion]:
    """Detect numeric-related issues."""
    # Already numeric, no issue
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return None
    
    if not isinstance(val, str):
        return None
    
    val_str = val.strip().lower()
    
    # Check for number words and phrases (deterministic conversion)
    # This handles both single words ("twenty") and compound phrases ("sixty thousand")
    if is_number_phrase(val_str):
        fixed_val = parse_number_phrase(val_str)
        if fixed_val is not None:
            return CellIssueSuggestion(
                row_index=idx,
                column=col,
                original_value=val,
                issue_type=IssueType.NUMBER_WORD.value,
                issue_description=f"Number word '{val}' can be converted to digit",
                available_actions=[
                    SuggestedAction.LEAVE_AS_IS,
                    SuggestedAction.APPLY_DETERMINISTIC_FIX,
                ],
                recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
                action_previews={
                    SuggestedAction.LEAVE_AS_IS.value: val,
                    SuggestedAction.APPLY_DETERMINISTIC_FIX.value: fixed_val,
                },
                deterministic_fix_value=fixed_val,
                deterministic_fix_explanation=f"Convert '{val}' to {fixed_val} (deterministic word-to-number)"
            )
    
    # Check for currency formatting
    if col_type == 'currency' and re.match(r'^[\$€£¥]', val_str):
        cleaned = re.sub(r'[$€£¥,]', '', val_str)
        try:
            fixed_val = float(cleaned)
            return CellIssueSuggestion(
                row_index=idx,
                column=col,
                original_value=val,
                issue_type=IssueType.CURRENCY_FORMAT.value,
                issue_description="Currency value has formatting symbols",
                available_actions=[
                    SuggestedAction.LEAVE_AS_IS,
                    SuggestedAction.APPLY_DETERMINISTIC_FIX,
                ],
                recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
                action_previews={
                    SuggestedAction.LEAVE_AS_IS.value: val,
                    SuggestedAction.APPLY_DETERMINISTIC_FIX.value: fixed_val,
                },
                deterministic_fix_value=fixed_val,
                deterministic_fix_explanation="Remove currency symbols and convert to number (meaning-preserving)"
            )
        except (ValueError, TypeError):
            pass
    
    # Check for percentage formatting
    if col_type == 'percentage' and '%' in val_str:
        cleaned = val_str.replace('%', '').strip()
        try:
            fixed_val = float(cleaned) / 100
            return CellIssueSuggestion(
                row_index=idx,
                column=col,
                original_value=val,
                issue_type=IssueType.PERCENTAGE_FORMAT.value,
                issue_description="Percentage value has % symbol",
                available_actions=[
                    SuggestedAction.LEAVE_AS_IS,
                    SuggestedAction.APPLY_DETERMINISTIC_FIX,
                ],
                recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
                action_previews={
                    SuggestedAction.LEAVE_AS_IS.value: val,
                    SuggestedAction.APPLY_DETERMINISTIC_FIX.value: fixed_val,
                },
                deterministic_fix_value=fixed_val,
                deterministic_fix_explanation="Convert percentage to decimal (50% → 0.5)"
            )
        except (ValueError, TypeError):
            pass
    
    return None


def _detect_date_issue(
    idx: int,
    col: str,
    val: Any,
    strict_config: StrictModeConfig,
) -> Optional[CellIssueSuggestion]:
    """Detect date-related issues."""
    if pd.isna(val):
        return None
    
    val_str = str(val).strip()
    
    # Try to parse the date
    try:
        parsed = pd.to_datetime(val_str)
        iso_format = parsed.strftime('%Y-%m-%d')
        
        # If already in ISO format, no issue
        if val_str == iso_format:
            return None
        
        return CellIssueSuggestion(
            row_index=idx,
            column=col,
            original_value=val,
            issue_type=IssueType.INVALID_DATE.value,
            issue_description="Date format could be standardized to ISO (YYYY-MM-DD)",
            available_actions=[
                SuggestedAction.LEAVE_AS_IS,
                SuggestedAction.APPLY_DETERMINISTIC_FIX,
            ],
            recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
            action_previews={
                SuggestedAction.LEAVE_AS_IS.value: val,
                SuggestedAction.APPLY_DETERMINISTIC_FIX.value: iso_format,
            },
            deterministic_fix_value=iso_format,
            deterministic_fix_explanation=f"Standardize to ISO format (meaning-preserving date format change)"
        )
    except (ValueError, TypeError):
        # Date couldn't be parsed
        return CellIssueSuggestion(
            row_index=idx,
            column=col,
            original_value=val,
            issue_type=IssueType.INVALID_DATE.value,
            issue_description="Invalid date format that cannot be parsed",
            available_actions=[
                SuggestedAction.LEAVE_AS_IS,
                SuggestedAction.REPLACE_WITH_BLANK,
                SuggestedAction.REPLACE_WITH_PLACEHOLDER,
            ],
            recommended_action=SuggestedAction.LEAVE_AS_IS,  # Conservative
            action_previews={
                SuggestedAction.LEAVE_AS_IS.value: val,
                SuggestedAction.REPLACE_WITH_BLANK.value: None,
                SuggestedAction.REPLACE_WITH_PLACEHOLDER.value: strict_config.unknown_placeholder,
            },
            deterministic_fix_value=None,
            deterministic_fix_explanation=None
        )


def _create_whitespace_suggestion(
    idx: int,
    col: str,
    val: Any,
    strict_config: StrictModeConfig,
) -> CellIssueSuggestion:
    """Create a suggestion for whitespace issues."""
    fixed_val = val.strip() if isinstance(val, str) else val
    
    return CellIssueSuggestion(
        row_index=idx,
        column=col,
        original_value=val,
        issue_type=IssueType.WHITESPACE.value,
        issue_description="Value has leading or trailing whitespace",
        available_actions=[
            SuggestedAction.LEAVE_AS_IS,
            SuggestedAction.APPLY_DETERMINISTIC_FIX,
        ],
        recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
        action_previews={
            SuggestedAction.LEAVE_AS_IS.value: val,
            SuggestedAction.APPLY_DETERMINISTIC_FIX.value: fixed_val,
        },
        deterministic_fix_value=fixed_val,
        deterministic_fix_explanation="Remove leading/trailing whitespace (meaning-preserving)"
    )


def _convert_value(val: Any) -> Any:
    """Convert numpy/pandas types to native Python types."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _convert_row_values(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all values in a row dict to native Python types."""
    return {k: _convert_value(v) for k, v in row_dict.items()}


# ============================================
# Apply Approved Actions (Step 4)
# ============================================

def apply_approved_actions(
    df: pd.DataFrame,
    approved_actions: List[UserApprovedAction],
    strict_config: Optional[StrictModeConfig] = None,
    skip_safe_transformations: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply cleaning to the DataFrame.
    
    This function:
    1. FIRST applies universally safe transformations automatically
       (whitespace, casing, missing indicators, dedupe, etc.)
    2. THEN applies user-approved actions for risky operations
    
    Args:
        df: The original DataFrame
        approved_actions: List of user-approved actions
        strict_config: Strict mode configuration
        skip_safe_transformations: If True, skip automatic safe transformations
        
    Returns:
        Tuple of (cleaned_df, summary_report)
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    # Step 1: Apply universally safe transformations automatically
    if not skip_safe_transformations:
        result, safe_summary = apply_safe_transformations(df, strict_config)
    else:
        result = df.copy()
        safe_summary = {"transformations_applied": [], "rows_before": len(df), "rows_after": len(df)}
    
    applied = []
    skipped = []
    warnings = []
    rows_to_drop = set()
    
    # Add safe transformation info to applied summary
    for transform in safe_summary.get("transformations_applied", []):
        applied.append({
            "type": "auto_safe",
            "action": transform,
            "result": "Applied automatically (universally safe)"
        })
    
    # Step 2: Apply user-approved actions
    for action in approved_actions:
        # Safety check: Validate action is allowed
        if not _is_action_allowed(action, strict_config):
            skipped.append({
                "row": action.row_index,
                "column": action.column,
                "action": action.action.value,
                "reason": "Action blocked by strict mode"
            })
            warnings.append(
                f"BLOCKED: Action '{action.action.value}' at row {action.row_index} "
                f"is not allowed in strict mode"
            )
            continue
        
        try:
            if action.action == SuggestedAction.DROP_ROW:
                # Validate row qualifies for dropping
                if _row_qualifies_for_drop(result, action.row_index):
                    rows_to_drop.add(action.row_index)
                    applied.append({
                        "row": action.row_index,
                        "action": "drop_row",
                        "result": "Row marked for deletion"
                    })
                else:
                    skipped.append({
                        "row": action.row_index,
                        "action": "drop_row",
                        "reason": "Row does not qualify for dropping under conservative rules"
                    })
                    warnings.append(
                        f"Row {action.row_index} does not qualify for dropping. "
                        "Only empty or structurally corrupt rows can be dropped."
                    )
            
            elif action.action == SuggestedAction.LEAVE_AS_IS:
                # No change needed
                applied.append({
                    "row": action.row_index,
                    "column": action.column,
                    "action": "leave_as_is",
                    "result": "No change"
                })
            
            elif action.action == SuggestedAction.REPLACE_WITH_BLANK:
                if action.column and action.column in result.columns:
                    result.at[action.row_index, action.column] = None
                    applied.append({
                        "row": action.row_index,
                        "column": action.column,
                        "action": "replace_with_blank",
                        "result": "Set to blank/null"
                    })
            
            elif action.action == SuggestedAction.REPLACE_WITH_PLACEHOLDER:
                placeholder = action.custom_placeholder or strict_config.unknown_placeholder
                if action.column and action.column in result.columns:
                    result.at[action.row_index, action.column] = placeholder
                    applied.append({
                        "row": action.row_index,
                        "column": action.column,
                        "action": "replace_with_placeholder",
                        "result": f"Set to '{placeholder}'"
                    })
            
            elif action.action == SuggestedAction.APPLY_DETERMINISTIC_FIX:
                # Re-detect the issue to get the deterministic fix
                if action.column and action.column in result.columns:
                    original_val = result.at[action.row_index, action.column]
                    fixed_val = _get_deterministic_fix(original_val, action.column, result, strict_config)
                    
                    if fixed_val is not None:
                        result.at[action.row_index, action.column] = fixed_val
                        applied.append({
                            "row": action.row_index,
                            "column": action.column,
                            "action": "apply_deterministic_fix",
                            "original": _convert_value(original_val),
                            "result": _convert_value(fixed_val)
                        })
                    else:
                        skipped.append({
                            "row": action.row_index,
                            "column": action.column,
                            "action": "apply_deterministic_fix",
                            "reason": "No deterministic fix available"
                        })
        
        except Exception as e:
            skipped.append({
                "row": action.row_index,
                "column": action.column,
                "action": action.action.value,
                "reason": f"Error applying action: {str(e)}"
            })
    
    # Drop marked rows (after all other operations)
    if rows_to_drop:
        result = result.drop(index=list(rows_to_drop)).reset_index(drop=True)
    
    summary = {
        "actions_applied": len(applied),
        "actions_skipped": len(skipped),
        "rows_modified": len(set(a.get("row", -1) for a in applied if a.get("row", -1) >= 0)),
        "rows_dropped": len(rows_to_drop),
        "applied_summary": applied,
        "skipped_summary": skipped,
        "warnings": warnings
    }
    
    return result, summary


def _is_action_allowed(
    action: UserApprovedAction,
    strict_config: StrictModeConfig,
) -> bool:
    """Check if an action is allowed under current strict mode config."""
    if not strict_config.enabled:
        return True
    
    # All actions in our enum are designed to be safe
    # But we still validate specific cases
    
    # REPLACE_WITH_PLACEHOLDER requires explicit placeholder
    if action.action == SuggestedAction.REPLACE_WITH_PLACEHOLDER:
        # Allowed - user is explicitly choosing a placeholder
        return True
    
    return True


def _row_qualifies_for_drop(df: pd.DataFrame, row_index: int) -> bool:
    """Check if a row qualifies for dropping under conservative rules."""
    if row_index < 0 or row_index >= len(df):
        return False
    
    row = df.iloc[row_index]
    
    # Rule 1: Completely empty row
    is_all_null = row.isna().all()
    is_all_empty_str = all(
        pd.isna(v) or (isinstance(v, str) and v.strip() == '')
        for v in row.values
    )
    
    return is_all_null or is_all_empty_str


def _get_deterministic_fix(
    val: Any,
    column: str,
    df: pd.DataFrame,
    strict_config: StrictModeConfig,
) -> Optional[Any]:
    """Get the deterministic fix for a value, if available."""
    if pd.isna(val):
        return None
    
    if isinstance(val, str):
        val_str = val.strip()
        val_lower = val_str.lower()
        
        # Whitespace fix
        if val != val_str:
            return val_str
        
        # Number word conversion
        if val_lower in NUMBER_WORDS:
            return NUMBER_WORDS[val_lower]
        
        # Title case for names (if all upper or all lower)
        if val_str.isupper() or val_str.islower():
            return val_str.title()
        
        # Email lowercase
        if EMAIL_PATTERN.match(val_lower):
            return val_lower
        
        # Currency symbol removal
        if re.match(r'^[\$€£¥]', val_str):
            cleaned = re.sub(r'[$€£¥,]', '', val_str)
            try:
                return float(cleaned)
            except ValueError:
                pass
        
        # Percentage conversion
        if '%' in val_str:
            cleaned = val_str.replace('%', '').strip()
            try:
                return float(cleaned) / 100
            except ValueError:
                pass
        
        # Date standardization
        try:
            parsed = pd.to_datetime(val_str)
            return parsed.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass
    
    return None
