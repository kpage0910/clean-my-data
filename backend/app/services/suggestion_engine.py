"""
Suggestion Engine - Safe, User-Controlled Data Cleaning

This module powers the "Safe Review" workflow where users approve each fix
before it's applied. Unlike the autonomous engine, nothing changes without
explicit user consent.

WORKFLOW OVERVIEW:
──────────────────
Step 1: apply_safe_transformations()
        → Apply UNIVERSALLY SAFE changes automatically (whitespace, casing)
        
Step 2: detect_issues()
        → Identify remaining problems that need user decisions
        
Step 3: generate_suggestions()
        → Create suggested fixes with "before/after" previews
        
Step 4: apply_approved_actions()
        → Apply ONLY the fixes the user explicitly approved

WHAT'S "UNIVERSALLY SAFE"? (applied automatically)
──────────────────────────────────────────────────
These transformations are always meaning-preserving:
• Trim whitespace from all strings
• Normalize casing (Title Case for names, lowercase for emails)
• Convert number words to digits ("thirty" → 30)
• Standardize dates to ISO format (YYYY-MM-DD)
• Replace missing indicators ("na", "N/A", "-") with null
• Remove exact duplicate rows

WHAT REQUIRES USER APPROVAL?
────────────────────────────
These could potentially change meaning, so we ask first:
• Dropping rows (even empty ones)
• Filling missing values with placeholders
• Replacing invalid format values
• Any change the user might disagree with

FUNDAMENTAL PRINCIPLES:
───────────────────────
1. NEVER invent, guess, or fabricate data
2. ONLY apply deterministic, meaning-preserving transformations
3. Apply safe operations automatically to reduce noise
4. ALWAYS ask before high-risk changes
5. Missing/invalid cells stay blank unless user chooses otherwise
6. Imputation (filling missing values) is OFF by default
7. Row dropping is OFF by default
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
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
# These run without user approval because they're obviously correct.

def apply_safe_transformations(
    df: pd.DataFrame,
    strict_config: Optional[StrictModeConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply universally safe transformations automatically.
    
    These transformations are deterministic and meaning-preserving.
    They reduce noise in the data without requiring user decisions.
    
    What gets applied:
    1. Normalize missing indicators ("na", "N/A", "-", etc.) → null
    2. Trim whitespace from all string columns
    3. Normalize casing based on column type:
       - Names → Title Case ("JOHN DOE" → "John Doe")
       - Emails → lowercase ("John@Email.COM" → "john@email.com")
    4. Convert number words ("thirty" → 30)
    5. Standardize dates to ISO format
    6. Remove exact duplicate rows
    
    Returns:
        (cleaned_df, summary) - the cleaned data and a log of what changed
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
        
        # Boolean columns → standardize (skipped - requires user approval for format choice)
        # This is handled via column-level issues where users can choose True/False, Yes/No, or 1/0
        elif col_type == 'boolean':
            # Don't auto-standardize booleans - user should choose the format via column issues
            pass
    
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
    BOOLEAN_INCONSISTENCY = "boolean_inconsistency"


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
            
            elif col_type == 'boolean':
                # Boolean issues are detected at column level, not cell level
                pass
            
            # Universal checks (for all types)
            else:
                # Check for whitespace issues
                if isinstance(val, str) and val != val.strip():
                    issues.append(_create_whitespace_suggestion(idx, col, val, strict_config))
    
    # Detect boolean inconsistencies at column level
    for inf in column_inferences:
        col = inf.column
        if col not in df.columns:
            continue
        
        col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
        
        if col_type == 'boolean':
            target_format = _detect_column_boolean_format(df, col)
            if target_format:
                for idx in range(len(df)):
                    val = df[col].iloc[idx]
                    if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                        continue
                    issue = _detect_boolean_issue(idx, col, val, target_format, strict_config)
                    if issue:
                        issues.append(issue)
    
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
    
    SAFE REVIEW MODE for Boolean Standardization:
    - Boolean inconsistencies are detected and placed in medium_risk_suggestions
    - No automatic standardization is performed
    - Before/after examples are provided without modifying data
    - User must explicitly confirm before any boolean transformation
    
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
    
    # Detect column-level issues (e.g., boolean standardization)
    column_issues = detect_column_issues(df, col_inf_schemas, strict_config)
    
    # Generate medium-risk suggestions (includes boolean standardization)
    # These require explicit user confirmation before applying
    medium_risk_suggestions = generate_medium_risk_suggestions(df, col_inf_schemas, strict_config)
    
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
    
    # Add warning for boolean inconsistencies requiring user confirmation
    if medium_risk_suggestions:
        bool_suggestions = [s for s in medium_risk_suggestions if s.suggestion_type == "boolean_standardization"]
        if bool_suggestions:
            cols = [s.suggestion_data.column for s in bool_suggestions]
            warnings.append(
                f"BOOLEAN STANDARDIZATION REQUIRED: Column(s) {', '.join(cols)} contain inconsistent boolean formats. "
                "Review the medium_risk_suggestions and confirm your preferred format before applying."
            )
    
    return DetectedIssuesReport(
        file_id="",  # Will be set by the caller
        total_rows=len(df),
        total_columns=len(df.columns),
        total_issues=len(cell_issues) + len(row_issues) + len(medium_risk_suggestions),
        cell_issues_count=len(cell_issues),
        row_issues_count=len(row_issues),
        cell_issues=cell_issues,
        row_issues=row_issues,
        column_inferences=col_inf_schemas,
        warnings=warnings,
        column_issues=column_issues,
        medium_risk_suggestions=medium_risk_suggestions,
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


# Boolean value mappings for normalization
BOOLEAN_TRUE_VALUES = {'true', '1', 'yes', 'y', 't', 'on', 'enabled', 'active'}
BOOLEAN_FALSE_VALUES = {'false', '0', 'no', 'n', 'f', 'off', 'disabled', 'inactive'}
BOOLEAN_ALL_VALUES = BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES


def _detect_boolean_issue(
    idx: int,
    col: str,
    val: Any,
    target_format: str,
    strict_config: StrictModeConfig,
) -> Optional[CellIssueSuggestion]:
    """
    Detect boolean formatting inconsistency.
    
    Args:
        idx: Row index
        col: Column name
        val: Cell value
        target_format: Target format ('True/False', 'Yes/No', or '1/0')
        strict_config: Strict mode configuration
    
    Returns:
        CellIssueSuggestion if inconsistency detected, None otherwise
    """
    if not isinstance(val, str):
        return None
    
    val_lower = val.strip().lower()
    
    if val_lower not in BOOLEAN_ALL_VALUES:
        return None
    
    # Determine if it's a true or false value
    is_true = val_lower in BOOLEAN_TRUE_VALUES
    
    # Get the target value based on format
    if target_format == 'True/False':
        target_val = 'True' if is_true else 'False'
    elif target_format == 'Yes/No':
        target_val = 'Yes' if is_true else 'No'
    elif target_format == '1/0':
        target_val = '1' if is_true else '0'
    else:
        target_val = 'True' if is_true else 'False'  # Default
    
    # Check if already in target format
    if val.strip() == target_val:
        return None
    
    return CellIssueSuggestion(
        row_index=idx,
        column=col,
        original_value=val,
        issue_type=IssueType.BOOLEAN_INCONSISTENCY.value,
        issue_description=f"Boolean value '{val}' can be normalized to '{target_val}'",
        available_actions=[
            SuggestedAction.LEAVE_AS_IS,
            SuggestedAction.APPLY_DETERMINISTIC_FIX,
        ],
        recommended_action=SuggestedAction.APPLY_DETERMINISTIC_FIX,
        action_previews={
            SuggestedAction.LEAVE_AS_IS.value: val,
            SuggestedAction.APPLY_DETERMINISTIC_FIX.value: target_val,
        },
        deterministic_fix_value=target_val,
        deterministic_fix_explanation=f"Normalize to '{target_format}' format (meaning-preserving)"
    )


def _detect_column_boolean_format(df: pd.DataFrame, col: str) -> Optional[str]:
    """
    Detect the most common boolean format in a column.
    
    Returns the target format to normalize to, or None if not a boolean column.
    """
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return None
    
    format_counts = {
        'True/False': 0,
        'Yes/No': 0,
        '1/0': 0,
        'Y/N': 0,
        'T/F': 0,
    }
    
    bool_count = 0
    for val in non_null:
        if not isinstance(val, str):
            continue
        val_lower = val.strip().lower()
        if val_lower in BOOLEAN_ALL_VALUES:
            bool_count += 1
            if val_lower in {'true', 'false'}:
                format_counts['True/False'] += 1
            elif val_lower in {'yes', 'no'}:
                format_counts['Yes/No'] += 1
            elif val_lower in {'1', '0'}:
                format_counts['1/0'] += 1
            elif val_lower in {'y', 'n'}:
                format_counts['Y/N'] += 1
            elif val_lower in {'t', 'f'}:
                format_counts['T/F'] += 1
    
    # Need at least 50% boolean values to consider this a boolean column
    if bool_count < len(non_null) * 0.5:
        return None
    
    # Check if multiple formats are used (inconsistency)
    used_formats = [fmt for fmt, count in format_counts.items() if count > 0]
    if len(used_formats) <= 1:
        return None  # Already consistent
    
    # Return the most common format, preferring True/False
    max_format = max(format_counts.items(), key=lambda x: (x[1], x[0] == 'True/False'))
    
    # Normalize Y/N to Yes/No and T/F to True/False
    if max_format[0] == 'Y/N':
        return 'Yes/No'
    elif max_format[0] == 'T/F':
        return 'True/False'
    
    return max_format[0]


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
        # Check if this is a column-level action (row_index is None)
        if action.row_index is None and action.column is not None:
            # Column-level action (e.g., boolean standardization)
            if action.action == SuggestedAction.APPLY_DETERMINISTIC_FIX:
                col = action.column
                if col in result.columns:
                    # Determine target format
                    target_format = action.target_format or _detect_column_boolean_format(result, col)
                    if target_format:
                        count = 0
                        for idx in range(len(result)):
                            val = result.at[idx, col]
                            if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                                continue
                            if isinstance(val, str):
                                val_lower = val.strip().lower()
                                if val_lower in BOOLEAN_ALL_VALUES:
                                    is_true = val_lower in BOOLEAN_TRUE_VALUES
                                    if target_format == 'True/False':
                                        new_val = 'True' if is_true else 'False'
                                    elif target_format == 'Yes/No':
                                        new_val = 'Yes' if is_true else 'No'
                                    elif target_format == '1/0':
                                        new_val = '1' if is_true else '0'
                                    else:
                                        new_val = 'True' if is_true else 'False'
                                    if val.strip() != new_val:
                                        result.at[idx, col] = new_val
                                        count += 1
                        applied.append({
                            "column": col,
                            "action": "standardize_booleans",
                            "target_format": target_format,
                            "result": f"Standardized {count} values to '{target_format}'"
                        })
                    else:
                        skipped.append({
                            "column": col,
                            "action": "standardize_booleans",
                            "reason": "Could not determine target format"
                        })
            elif action.action == SuggestedAction.LEAVE_AS_IS:
                applied.append({
                    "column": action.column,
                    "action": "leave_as_is",
                    "result": "No change"
                })
            continue
        
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


def detect_boolean_inconsistencies_detailed(
    df: pd.DataFrame,
    col: str,
) -> Optional[Dict[str, Any]]:
    """
    Perform detailed analysis of boolean inconsistencies in a column.
    
    SAFE REVIEW MODE: This function only analyzes and reports - it does NOT
    modify any data. All transformations require explicit user confirmation.
    
    Detects patterns:
    - true/false (case-insensitive)
    - yes/no (case-insensitive)
    - y/n (case-insensitive)
    - t/f (case-insensitive)
    - 1/0
    - Mixed casing (TRUE, True, true, etc.)
    
    Args:
        df: The DataFrame to analyze
        col: Column name to analyze
        
    Returns:
        Detailed analysis dict or None if no inconsistency detected
    """
    if col not in df.columns:
        return None
    
    non_null = df[col].dropna()
    if len(non_null) == 0:
        return None
    
    # Define format groups with their canonical representations
    format_groups = {
        'true/false': {'true', 'false'},
        'yes/no': {'yes', 'no'},
        'y/n': {'y', 'n'},
        't/f': {'t', 'f'},
        '1/0': {'1', '0'},
    }
    
    # Track what we find
    format_distribution = {
        'true/false': {'count': 0, 'values_found': set()},
        'yes/no': {'count': 0, 'values_found': set()},
        'y/n': {'count': 0, 'values_found': set()},
        't/f': {'count': 0, 'values_found': set()},
        '1/0': {'count': 0, 'values_found': set()},
    }
    
    # Track casing variations
    casing_variations = {}  # normalized -> set of original casings
    
    # Track affected rows
    affected_indices = []
    bool_count = 0
    
    for idx, val in df[col].items():
        if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
            continue
        
        if not isinstance(val, str):
            continue
        
        val_stripped = val.strip()
        val_lower = val_stripped.lower()
        
        if val_lower in BOOLEAN_ALL_VALUES:
            bool_count += 1
            affected_indices.append(int(idx))
            
            # Track casing variations
            if val_lower not in casing_variations:
                casing_variations[val_lower] = set()
            casing_variations[val_lower].add(val_stripped)
            
            # Categorize into format group
            for group_name, group_values in format_groups.items():
                if val_lower in group_values:
                    format_distribution[group_name]['count'] += 1
                    format_distribution[group_name]['values_found'].add(val_stripped)
                    break
    
    # Need at least 50% boolean values to consider this a boolean column
    if bool_count < len(non_null) * 0.5:
        return None
    
    # Check which format groups are actually used
    used_formats = [
        group for group, data in format_distribution.items() 
        if data['count'] > 0
    ]
    
    # Only report if multiple format groups are used (real inconsistency)
    # OR if there are significant casing inconsistencies
    has_casing_issues = any(len(casings) > 1 for casings in casing_variations.values())
    
    if len(used_formats) <= 1 and not has_casing_issues:
        return None  # Already consistent
    
    # Build format distribution report
    distribution_report = []
    for group_name in ['true/false', 'yes/no', '1/0', 'y/n', 't/f']:
        data = format_distribution[group_name]
        if data['count'] > 0:
            distribution_report.append({
                'format_name': group_name,
                'values_found': list(data['values_found']),
                'count': data['count'],
                'percentage': round((data['count'] / bool_count) * 100, 2) if bool_count > 0 else 0
            })
    
    # Identify casing examples
    casing_examples = []
    for normalized, casings in casing_variations.items():
        if len(casings) > 1:
            casing_examples.extend(list(casings)[:3])
    
    # Determine recommended format based on most common
    most_common_format = max(
        [(group, data['count']) for group, data in format_distribution.items()],
        key=lambda x: (x[1], x[0] == 'true/false')  # Prefer true/false on tie
    )[0]
    
    # Map to standard format names
    format_mapping = {
        'true/false': 'True/False',
        'yes/no': 'Yes/No',
        'y/n': 'Yes/No',
        't/f': 'True/False',
        '1/0': '1/0',
    }
    recommended_format = format_mapping.get(most_common_format, 'True/False')
    
    # Generate before/after examples
    before_after_examples = []
    sample_indices = affected_indices[:10]  # Limit to 10 examples
    
    for idx in sample_indices:
        val = df[col].iloc[idx]
        if not isinstance(val, str):
            continue
        
        val_stripped = val.strip()
        val_lower = val_stripped.lower()
        
        # Determine standardized value for each format
        is_true = val_lower in {'true', '1', 'yes', 'y', 't', 'on', 'enabled', 'active'}
        
        if recommended_format == 'True/False':
            standardized = 'True' if is_true else 'False'
        elif recommended_format == 'Yes/No':
            standardized = 'Yes' if is_true else 'No'
        else:  # 1/0
            standardized = '1' if is_true else '0'
        
        if val_stripped != standardized:
            before_after_examples.append({
                'row_index': idx,
                'original_value': val_stripped,
                'standardized_value': standardized
            })
    
    return {
        'detected_formats': distribution_report,
        'has_mixed_casing': has_casing_issues,
        'casing_examples': casing_examples[:5],
        'total_boolean_values': bool_count,
        'affected_rows': len([e for e in before_after_examples if e['original_value'] != e['standardized_value']]) 
                         if before_after_examples else len(affected_indices),
        'affected_row_indices': affected_indices[:100],
        'default_recommended_format': recommended_format,
        'before_after_examples': before_after_examples,
    }


def generate_medium_risk_suggestions(
    df: pd.DataFrame,
    column_inferences: Union[List, Dict],
    strict_config: Optional[StrictModeConfig] = None,
) -> list:
    """
    Generate medium-risk suggestions that require user confirmation.
    
    SAFE REVIEW MODE:
    - Detects boolean inconsistency patterns
    - Does NOT automatically standardize
    - Provides recommended standardization formats
    - Shows before/after examples without modifying data
    - Waits for explicit user confirmation before any transformation
    
    All boolean normalization suggestions are included in this function's output.
    
    Args:
        df: The DataFrame to analyze
        column_inferences: Column type inferences
        strict_config: Strict mode configuration
        
    Returns:
        List of MediumRiskSuggestion objects
    """
    from app.models.schemas import (
        MediumRiskSuggestion,
        BooleanStandardizationSuggestion,
        BooleanFormatDistribution,
        BooleanFormatExample,
    )
    
    suggestions = []
    
    # Normalize column inferences to list format
    inference_items = []
    if isinstance(column_inferences, dict):
        for col_name, inf in column_inferences.items():
            col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
            inference_items.append((col_name, col_type))
    else:
        for inf in column_inferences:
            col_name = inf.column
            col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
            inference_items.append((col_name, col_type))
    
    # Detect boolean inconsistencies for each boolean column
    for col, col_type in inference_items:
        if col_type != 'boolean' or col not in df.columns:
            continue
        
        analysis = detect_boolean_inconsistencies_detailed(df, col)
        if analysis is None:
            continue
        
        # Build format distribution objects
        format_distributions = [
            BooleanFormatDistribution(
                format_name=fmt['format_name'],
                values_found=fmt['values_found'],
                count=fmt['count'],
                percentage=fmt['percentage']
            )
            for fmt in analysis['detected_formats']
        ]
        
        # Build before/after example objects
        before_after_examples = [
            BooleanFormatExample(
                row_index=ex['row_index'],
                original_value=ex['original_value'],
                standardized_value=ex['standardized_value']
            )
            for ex in analysis['before_after_examples']
        ]
        
        # Build description
        format_names = [fmt['format_name'] for fmt in analysis['detected_formats']]
        description = (
            f"Detected inconsistent boolean formats in column '{col}': "
            f"{', '.join(format_names)}. "
        )
        if analysis['has_mixed_casing']:
            description += f"Also found mixed casing (e.g., {', '.join(analysis['casing_examples'][:3])}). "
        description += "User confirmation required before standardization."
        
        # Create the suggestion
        bool_suggestion = BooleanStandardizationSuggestion(
            column=col,
            issue_type="boolean_inconsistency",
            description=description,
            detected_formats=format_distributions,
            has_mixed_casing=analysis['has_mixed_casing'],
            casing_examples=analysis['casing_examples'],
            total_boolean_values=analysis['total_boolean_values'],
            affected_rows=analysis['affected_rows'],
            affected_row_indices=analysis['affected_row_indices'],
            recommended_formats=["True/False", "Yes/No", "1/0"],
            default_recommended_format=analysis['default_recommended_format'],
            before_after_examples=before_after_examples,
            requires_user_confirmation=True,
            user_confirmed=False,
            selected_format=None,
        )
        
        suggestions.append(MediumRiskSuggestion(
            suggestion_type="boolean_standardization",
            risk_level="medium",
            suggestion_data=bool_suggestion,
            action_required=True,
            action_taken=None,
        ))
    
    return suggestions


def detect_column_issues(
    df: pd.DataFrame,
    column_inferences: Union[List, Dict],
    strict_config: Optional[StrictModeConfig] = None,
) -> list:
    """
    Detect column-level issues (e.g., boolean standardization suggestions).
    
    Args:
        df: The DataFrame to analyze
        column_inferences: Either a list of ColumnTypeInference objects or a dict mapping column names to inferences
        strict_config: Optional strict mode configuration
    """
    from app.models.schemas import ColumnIssueSuggestion, SuggestedAction
    issues = []
    
    # Normalize to a list of (column_name, inferred_type) tuples
    inference_items = []
    if isinstance(column_inferences, dict):
        for col_name, inf in column_inferences.items():
            col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
            inference_items.append((col_name, col_type))
    else:
        for inf in column_inferences:
            col_name = inf.column
            col_type = inf.inferred_type if isinstance(inf.inferred_type, str) else inf.inferred_type.value
            inference_items.append((col_name, col_type))
    
    for col, col_type in inference_items:
        if col_type == 'boolean' and col in df.columns:
            target_format = _detect_column_boolean_format(df, col)
            if target_format:
                # Only suggest if multiple formats are present
                issues.append(ColumnIssueSuggestion(
                    column=col,
                    issue_type=IssueType.BOOLEAN_INCONSISTENCY.value,
                    description=f"Column contains boolean values with inconsistent formats.",
                    suggested_action=f"Standardize all to a consistent format",
                    available_formats=["True/False", "Yes/No", "1/0"],
                    default_format=target_format
                ))
    return issues
