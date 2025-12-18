"""
Pydantic Models for API Request/Response Schemas

This module defines the data structures used for communication between
the frontend and backend. Every API endpoint uses these schemas for:
- Request validation (ensures required fields are present)
- Response formatting (ensures consistent JSON structure)
- Documentation (auto-generates OpenAPI/Swagger docs)

ORGANIZATION:
─────────────
1. StrictModeConfig - Controls what transformations are allowed
2. Upload schemas - For /upload endpoint
3. Scan schemas - For /scan endpoint
4. Preview schemas - For /preview and /autonomous-preview
5. Apply schemas - For /apply and /autonomous-apply
6. Safe Review schemas - For /detect-issues and /apply-approved
7. AI Summary schemas - For /ai-summary endpoint

NAMING CONVENTION:
──────────────────
- *Request: Schema for incoming request body
- *Response: Schema for outgoing response
- *Info/*Report: Internal data structures returned within responses
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ============================================
# Strict Mode Configuration
# ============================================
# This is the "safety dial" that controls how aggressive the cleaner is.
# Default is STRICT mode, which only allows obviously-safe transformations.

class CleaningMode(str, Enum):
    """
    High-level cleaning mode selection.
    
    STRICT (default): Only deterministic, meaning-preserving transformations.
                      This is the safe choice for sensitive data.
                      
    LENIENT: More aggressive cleaning with heuristics and imputation.
             May fill missing values, infer categories, etc.
             Use only when you're confident the guesses will be correct.
    """
    STRICT = "strict"
    LENIENT = "lenient"


class StrictModeConfig(BaseModel):
    """
    Fine-grained control over what strict mode allows.
    
    By default, strict mode is enabled with all safe transformations allowed.
    You can selectively enable risky operations like imputation if needed.
    
    Example - allow imputation:
        StrictModeConfig(enabled=True, allow_imputation=True)
    """
    enabled: bool = Field(True, description="Whether strict mode is active")
    allow_imputation: bool = Field(
        False, 
        description="Allow imputation of missing values (only when explicitly enabled)"
    )
    unknown_placeholder: str = Field(
        "Unknown",
        description="Placeholder for invalid or unprocessable values"
    )
    preserve_blanks: bool = Field(
        True,
        description="Keep blank/missing cells blank instead of filling with placeholder"
    )
    
    # Allowed transformation categories in strict mode
    allow_capitalization_normalization: bool = Field(True, description="e.g., DANIEL → Daniel")
    allow_number_word_conversion: bool = Field(True, description="e.g., 'thirty' → 30")
    allow_date_normalization: bool = Field(True, description="Standardize date formats")
    allow_whitespace_trimming: bool = Field(True, description="Remove leading/trailing spaces")
    allow_invalid_char_removal: bool = Field(True, description="Remove non-printable characters")
    allow_type_coercion: bool = Field(True, description="Lossless type conversions only")


# Global strict mode configuration (default is strict)
DEFAULT_STRICT_CONFIG = StrictModeConfig()


# ============================================
# Upload Endpoint Schemas
# ============================================

class UploadResponse(BaseModel):
    """Response schema for file upload."""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    message: str = Field(..., description="Status message")


# ============================================
# Scan Endpoint Schemas
# ============================================

class ScanRequest(BaseModel):
    """Request schema for scanning a file."""
    file_id: str = Field(..., description="ID of the file to scan")


class DataIssue(BaseModel):
    """Schema for a single data quality issue."""
    column: str = Field(..., description="Column name where issue was found")
    issue_type: str = Field(..., description="Type of issue detected")
    severity: str = Field(..., description="Severity level: low, medium, high")
    count: int = Field(..., description="Number of affected rows")
    description: str = Field(..., description="Human-readable description")
    examples: Optional[List[Any]] = Field(None, description="Example values")


class ScanReport(BaseModel):
    """Schema for the complete scan report."""
    total_rows: int = Field(..., description="Total number of rows in file")
    total_columns: int = Field(..., description="Total number of columns")
    issues: List[DataIssue] = Field(default_factory=list, description="List of detected issues")
    column_stats: Dict[str, Any] = Field(default_factory=dict, description="Statistics per column")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Overall summary")


class ScanResponse(BaseModel):
    """Response schema for file scanning."""
    file_id: str = Field(..., description="ID of the scanned file")
    report: ScanReport = Field(..., description="Scan report with detected issues")


# ============================================
# Preview Endpoint Schemas
# ============================================

class CleaningRule(BaseModel):
    """Schema for a single cleaning rule."""
    rule_type: str = Field(..., description="Type of cleaning rule to apply")
    column: Optional[str] = Field(None, description="Target column (if applicable)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")


class PreviewRequest(BaseModel):
    """Request schema for previewing cleaning changes."""
    file_id: str = Field(..., description="ID of the file to preview")
    rules: List[CleaningRule] = Field(..., description="Cleaning rules to apply")
    n_rows: Optional[int] = Field(100, description="Number of rows to preview")
    strict_config: Optional[StrictModeConfig] = Field(
        default_factory=StrictModeConfig,
        description="Strict mode configuration for deterministic cleaning"
    )


class PreviewRow(BaseModel):
    """Schema for a single preview row showing original vs cleaned."""
    row_index: int = Field(..., description="Original row index")
    original: Dict[str, Any] = Field(..., description="Original row data")
    cleaned: Dict[str, Any] = Field(..., description="Cleaned row data")
    changes: List[str] = Field(default_factory=list, description="List of changes made")


class PreviewResponse(BaseModel):
    """Response schema for preview endpoint."""
    file_id: str = Field(..., description="ID of the file")
    preview: List[PreviewRow] = Field(..., description="Preview rows")
    total_rows: int = Field(..., description="Total rows in file")
    preview_rows: int = Field(..., description="Number of rows in preview")


class AutonomousPreviewRequest(BaseModel):
    """Request schema for autonomous cleaning preview."""
    file_id: str = Field(..., description="ID of the file to preview")
    n_rows: Optional[int] = Field(100, description="Number of rows to preview")


# ============================================
# Apply Endpoint Schemas
# ============================================

class ApplyRequest(BaseModel):
    """Request schema for applying cleaning rules."""
    file_id: str = Field(..., description="ID of the file to clean")
    rules: List[CleaningRule] = Field(..., description="Cleaning rules to apply")
    strict_config: Optional[StrictModeConfig] = Field(
        default_factory=StrictModeConfig,
        description="Strict mode configuration for deterministic cleaning"
    )


class ApplyResponse(BaseModel):
    """Response schema for apply endpoint."""
    original_file_id: str = Field(..., description="ID of the original file")
    cleaned_file_id: str = Field(..., description="ID of the new cleaned file")
    message: str = Field(..., description="Status message")
    rows_processed: int = Field(..., description="Number of rows processed")


# ============================================
# Autonomous Cleaning Endpoint Schemas
# ============================================

class AutonomousCleanRequest(BaseModel):
    """Request schema for autonomous cleaning."""
    file_id: str = Field(..., description="ID of the file to clean autonomously")
    preview_only: bool = Field(False, description="If true, only return suggestions without applying")
    strict_config: Optional[StrictModeConfig] = Field(
        default_factory=StrictModeConfig,
        description="Strict mode configuration - defaults to strict (deterministic) cleaning"
    )


class ColumnTypeInference(BaseModel):
    """Schema for a column type inference result."""
    column: str = Field(..., description="Column name")
    inferred_type: str = Field(..., description="Detected column type")
    confidence: float = Field(..., description="Confidence score 0-1")
    indicators: List[str] = Field(default_factory=list, description="Evidence for inference")
    is_safe: bool = Field(True, description="Whether inference is safe to apply")
    warning: Optional[str] = Field(None, description="Warning if inference is unsafe")


class GeneratedRuleInfo(BaseModel):
    """Schema for a generated cleaning rule."""
    rule_type: str = Field(..., description="Type of cleaning rule")
    column: Optional[str] = Field(None, description="Target column")
    params: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    reason: str = Field(..., description="Why this rule was generated")
    priority: int = Field(1, description="Execution priority")
    is_safe: bool = Field(True, description="Whether rule is safe to apply automatically")


class ValidationReportItem(BaseModel):
    """Schema for a validation report item."""
    column: str = Field(..., description="Column name")
    issue_type: str = Field(..., description="Type of issue")
    description: str = Field(..., description="Description of issue")
    fix: Optional[str] = Field(None, description="How it was fixed")
    reason: Optional[str] = Field(None, description="Why it couldn't be fixed")
    affected_rows: Optional[int] = Field(None, description="Number of affected rows")


class AutonomousCleanResponse(BaseModel):
    """Response schema for autonomous cleaning endpoint."""
    file_id: str = Field(..., description="ID of the original file")
    cleaned_file_id: Optional[str] = Field(None, description="ID of cleaned file (if not preview)")
    summary: Dict[str, Any] = Field(..., description="Summary of detected issues and fixes")
    column_inferences: List[ColumnTypeInference] = Field(..., description="Column type inferences")
    generated_rules: List[GeneratedRuleInfo] = Field(..., description="Rules that were generated")
    validation_report: Dict[str, Any] = Field(..., description="What was fixed and what couldn't be")
    warnings: List[str] = Field(default_factory=list, description="Warnings about operations")
    rows_processed: int = Field(..., description="Number of rows processed")
    columns_processed: int = Field(..., description="Number of columns processed")


# ============================================
# User Approval Workflow Schemas (SAFE DATA CLEANING)
# ============================================

class SuggestedAction(str, Enum):
    """
    Actions that can be suggested for fixing data issues.
    
    CONSERVATIVE BY DEFAULT:
    - LEAVE_AS_IS is the safest option (no change)
    - REPLACE_WITH_BLANK clears invalid data but doesn't fabricate
    - REPLACE_WITH_PLACEHOLDER uses explicit user-chosen placeholder
    - APPLY_DETERMINISTIC_FIX applies only reversible, meaning-preserving changes
    - DROP_ROW is only suggested for rows that qualify under strict rules
    """
    LEAVE_AS_IS = "leave_as_is"
    REPLACE_WITH_BLANK = "replace_with_blank"
    REPLACE_WITH_PLACEHOLDER = "replace_with_placeholder"
    APPLY_DETERMINISTIC_FIX = "apply_deterministic_fix"
    DROP_ROW = "drop_row"


class RowDropReason(str, Enum):
    """
    Valid reasons for dropping a row (conservative policy).
    
    Rows can ONLY be dropped if one of these conditions is met:
    1. COMPLETELY_EMPTY: All values are null/NaN/empty
    2. STRUCTURAL_CORRUPTION: Fewer columns than header (malformed CSV)
    3. UNREPAIRABLE_SCHEMA_VIOLATION: Violates schema in a way that cannot be fixed
    4. USER_DEFINED_RULE: User explicitly requested dropping based on custom criteria
    """
    COMPLETELY_EMPTY = "completely_empty"
    STRUCTURAL_CORRUPTION = "structural_corruption"
    UNREPAIRABLE_SCHEMA_VIOLATION = "unrepairable_schema_violation"
    USER_DEFINED_RULE = "user_defined_rule"


class CellIssueSuggestion(BaseModel):
    """
    Suggested fix for a single cell issue.
    
    Each issue comes with multiple action options for the user to choose from.
    NO ACTION IS APPLIED WITHOUT USER APPROVAL.
    """
    row_index: int = Field(..., description="Row index of the affected cell")
    column: str = Field(..., description="Column name of the affected cell")
    original_value: Any = Field(..., description="Original cell value")
    issue_type: str = Field(..., description="Type of issue detected")
    issue_description: str = Field(..., description="Human-readable issue description")
    
    # Available action options for user to choose
    available_actions: List[SuggestedAction] = Field(
        ..., 
        description="List of available actions for this issue"
    )
    
    # Recommended action (user can override)
    recommended_action: SuggestedAction = Field(
        SuggestedAction.LEAVE_AS_IS,
        description="Recommended action (conservative default: leave as-is)"
    )
    
    # Preview of what each action would produce
    action_previews: Dict[str, Any] = Field(
        default_factory=dict,
        description="Preview of result for each action option"
    )
    
    # If APPLY_DETERMINISTIC_FIX is available, what fix would be applied
    deterministic_fix_value: Optional[Any] = Field(
        None,
        description="The value that would result from applying a deterministic fix"
    )
    deterministic_fix_explanation: Optional[str] = Field(
        None,
        description="Explanation of the deterministic transformation"
    )


class RowIssueSuggestion(BaseModel):
    """
    Suggested fix for a row-level issue.
    
    Row dropping follows STRICT rules - rows are only suggested for
    dropping if they meet specific criteria.
    """
    row_index: int = Field(..., description="Index of the affected row")
    row_data: Dict[str, Any] = Field(..., description="The row data")
    issue_type: str = Field(..., description="Type of row-level issue")
    issue_description: str = Field(..., description="Human-readable description")
    
    # Why this row qualifies for dropping (if applicable)
    drop_reason: Optional[RowDropReason] = Field(
        None, 
        description="Reason row qualifies for dropping (if applicable)"
    )
    
    # Whether dropping is recommended
    drop_recommended: bool = Field(
        False,
        description="Whether dropping is recommended (conservative: usually False)"
    )
    
    # Available actions
    available_actions: List[SuggestedAction] = Field(
        default_factory=lambda: [SuggestedAction.LEAVE_AS_IS],
        description="Available actions for this row"
    )


class ColumnIssueSuggestion(BaseModel):
    """
    Suggested fix for a column-level issue (e.g., standardize all booleans).
    """
    column: str = Field(..., description="Column name of the affected column")
    issue_type: str = Field(..., description="Type of issue detected")
    description: str = Field(..., description="Human-readable issue description")
    suggested_action: str = Field(..., description="Suggested action to fix the issue")
    available_formats: List[str] = Field(
        default_factory=list,
        description="Available format options for standardization (e.g., ['True/False', 'Yes/No', '1/0'])"
    )
    default_format: Optional[str] = Field(
        None,
        description="Default/recommended format to standardize to"
    )


# ============================================
# Boolean Standardization (Safe Review Mode)
# ============================================

class BooleanFormatExample(BaseModel):
    """
    Before/after example for boolean standardization.
    Shows how a value would be transformed without modifying actual data.
    """
    row_index: int = Field(..., description="Row index of the example")
    original_value: str = Field(..., description="Original value in the data")
    standardized_value: str = Field(..., description="What it would become after standardization")
    

class BooleanFormatDistribution(BaseModel):
    """
    Distribution of boolean format variations found in a column.
    """
    format_name: str = Field(..., description="Format name (e.g., 'true/false', 'yes/no', '1/0')")
    values_found: List[str] = Field(..., description="Actual values found in this format")
    count: int = Field(..., description="Number of occurrences")
    percentage: float = Field(..., description="Percentage of total boolean values")


class BooleanStandardizationSuggestion(BaseModel):
    """
    Detailed boolean standardization suggestion for Safe Review Mode.
    
    SAFE REVIEW MODE PRINCIPLES:
    - Does NOT automatically standardize boolean values
    - Provides detailed analysis of inconsistency patterns
    - Shows before/after examples without modifying data
    - Requires explicit user confirmation before any transformation
    """
    column: str = Field(..., description="Column containing boolean inconsistencies")
    issue_type: str = Field(
        default="boolean_inconsistency",
        description="Type of issue (always 'boolean_inconsistency' for this schema)"
    )
    description: str = Field(..., description="Human-readable description of the inconsistency")
    
    # Detected patterns
    detected_formats: List[BooleanFormatDistribution] = Field(
        ...,
        description="All boolean format variations detected in the column"
    )
    has_mixed_casing: bool = Field(
        default=False,
        description="Whether mixed casing was detected (e.g., 'TRUE', 'True', 'true')"
    )
    casing_examples: List[str] = Field(
        default_factory=list,
        description="Examples of casing variations found"
    )
    
    # Affected data
    total_boolean_values: int = Field(..., description="Total number of boolean values in the column")
    affected_rows: int = Field(..., description="Number of rows that would be changed by standardization")
    affected_row_indices: List[int] = Field(
        default_factory=list,
        description="Indices of rows that would be affected (limited to first 100)"
    )
    
    # Recommended standardization formats
    recommended_formats: List[str] = Field(
        default_factory=lambda: ["True/False", "Yes/No", "1/0"],
        description="Available standardization format options"
    )
    default_recommended_format: str = Field(
        default="True/False",
        description="Default recommended format based on data analysis"
    )
    
    # Before/after examples (preview only - no actual changes)
    before_after_examples: List[BooleanFormatExample] = Field(
        default_factory=list,
        description="Sample before/after transformations (for preview only)"
    )
    
    # Action state
    requires_user_confirmation: bool = Field(
        default=True,
        description="Always True - boolean transformations require explicit user approval"
    )
    user_confirmed: bool = Field(
        default=False,
        description="Whether user has confirmed this transformation"
    )
    selected_format: Optional[str] = Field(
        default=None,
        description="Format selected by user (None until confirmed)"
    )


class MediumRiskSuggestion(BaseModel):
    """
    Container for medium-risk suggestions that require user review.
    
    Medium-risk operations are those that:
    - Are meaning-preserving but could cause downstream issues
    - Require user to choose between multiple valid options
    - Should not be applied automatically
    
    Examples:
    - Boolean standardization (user chooses format)
    - Date format standardization (user chooses format)
    - Case normalization for ambiguous columns
    """
    suggestion_type: str = Field(..., description="Type of suggestion (e.g., 'boolean_standardization')")
    risk_level: str = Field(default="medium", description="Risk level (always 'medium' for this container)")
    suggestion_data: BooleanStandardizationSuggestion = Field(
        ...,
        description="Detailed suggestion data"
    )
    
    # User action tracking
    action_required: bool = Field(
        default=True,
        description="Whether user action is required before applying"
    )
    action_taken: Optional[str] = Field(
        default=None,
        description="Action taken by user: 'approved', 'rejected', or None"
    )


class DetectedIssuesReport(BaseModel):
    """
    Complete report of detected issues with suggested fixes.
    
    Step 1 & 2 of the workflow:
    - Detects all issues WITHOUT making changes
    - Generates suggested fixes for user review
    """
    file_id: str = Field(..., description="ID of the analyzed file")
    total_rows: int = Field(..., description="Total rows in file")
    total_columns: int = Field(..., description="Total columns in file")
    
    # Summary counts
    total_issues: int = Field(..., description="Total number of issues detected")
    cell_issues_count: int = Field(..., description="Number of cell-level issues")
    row_issues_count: int = Field(..., description="Number of row-level issues")
    
    # Detailed issue lists
    cell_issues: List[CellIssueSuggestion] = Field(
        default_factory=list,
        description="Cell-level issues with suggested fixes"
    )
    row_issues: List[RowIssueSuggestion] = Field(
        default_factory=list,
        description="Row-level issues with suggested fixes"
    )
    column_issues: List[ColumnIssueSuggestion] = Field(
        default_factory=list,
        description="Column-level issues with suggested fixes"
    )
    
    # Medium-risk suggestions (require user confirmation)
    # Boolean standardization suggestions are placed here
    medium_risk_suggestions: List[MediumRiskSuggestion] = Field(
        default_factory=list,
        description="Medium-risk suggestions requiring user review (e.g., boolean standardization)"
    )
    
    # Column type inferences for context
    column_inferences: List[ColumnTypeInference] = Field(
        default_factory=list,
        description="Inferred column types"
    )
    
    # Warnings about the data
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about detected issues"
    )


class UserApprovedAction(BaseModel):
    """
    A single user-approved action for a cell, row, or column.
    
    The user must explicitly approve each action before it is applied.
    """
    row_index: Optional[int] = Field(None, description="Row index to apply action to (None for column-level actions)")
    column: Optional[str] = Field(None, description="Column (None for row-level actions, required for column-level)")
    action: SuggestedAction = Field(..., description="The approved action to take")
    custom_placeholder: Optional[str] = Field(
        None,
        description="Custom placeholder value (for REPLACE_WITH_PLACEHOLDER)"
    )
    # For column-level boolean standardization
    target_format: Optional[str] = Field(
        None,
        description="Target format for boolean standardization (e.g., 'True/False', 'Yes/No', '1/0')"
    )


class ApplyApprovedActionsRequest(BaseModel):
    """
    Request to apply user-approved actions.
    
    Step 4 of the workflow:
    - Only actions explicitly approved by the user are applied
    - Unapproved issues are left unchanged
    """
    file_id: str = Field(..., description="ID of the file to apply actions to")
    approved_actions: List[UserApprovedAction] = Field(
        ...,
        description="List of user-approved actions to apply"
    )
    strict_config: Optional[StrictModeConfig] = Field(
        default_factory=StrictModeConfig,
        description="Strict mode configuration"
    )


class ApplyApprovedActionsResponse(BaseModel):
    """
    Response after applying approved actions.
    
    Includes summary of what was applied and what was skipped.
    """
    original_file_id: str = Field(..., description="Original file ID")
    cleaned_file_id: str = Field(..., description="New cleaned file ID")
    
    # Summary
    actions_applied: int = Field(..., description="Number of actions successfully applied")
    actions_skipped: int = Field(..., description="Number of actions skipped")
    rows_modified: int = Field(..., description="Number of rows modified")
    rows_dropped: int = Field(..., description="Number of rows dropped")
    
    # Detailed reports
    applied_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of applied actions"
    )
    skipped_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of skipped actions with reasons"
    )
    
    # Warnings (e.g., if user tried to apply forbidden action)
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about the operation"
    )


# ============================================
# AI Data Quality Summary Schemas
# ============================================

class AIQualitySummaryRequest(BaseModel):
    """Request schema for AI-powered data quality summary."""
    file_id: str = Field(..., description="ID of the file to analyze")
    model: Optional[str] = Field(
        "gpt-4.1",
        description="OpenAI model to use for generating the summary"
    )
    include_raw_analysis: Optional[bool] = Field(
        False,
        description="Whether to include raw analysis statistics in the response"
    )


class AIQualitySummaryResponse(BaseModel):
    """Response schema for AI-powered data quality summary."""
    file_id: str = Field(..., description="ID of the analyzed file")
    summary: str = Field(..., description="Natural-language summary of data quality issues")
    success: bool = Field(..., description="Whether the summary generation was successful")
    ai_model: Optional[str] = Field(None, description="The OpenAI model used")
    analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw analysis statistics (if requested)"
    )
    error: Optional[str] = Field(None, description="Error message if generation failed")
