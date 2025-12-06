"""
Pydantic models for request/response schemas.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ============================================
# Strict Mode Configuration
# ============================================

class CleaningMode(str, Enum):
    """
    Cleaning mode that controls transformation behavior.
    
    STRICT: Deterministic, meaning-preserving transformations only.
            - No fabrication of data
            - No imputation of missing values (unless explicitly enabled)
            - Invalid/blank cells remain blank or become "Unknown"
            - Only reversible, lossless transformations allowed
            
    LENIENT: Standard cleaning with heuristics and imputation.
             - May fill missing values with inferred data
             - May apply non-reversible transformations
    """
    STRICT = "strict"
    LENIENT = "lenient"


class StrictModeConfig(BaseModel):
    """
    Configuration for strict (deterministic) data cleaning mode.
    
    When strict mode is enabled:
    1. No fabrication or guessing of new data
    2. Only meaning-preserving transformations allowed
    3. No imputation unless explicitly enabled
    4. Invalid/blank cells left as-is or marked "Unknown"
    5. All transformations must be deterministic and reversible
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
    A single user-approved action for a cell or row.
    
    The user must explicitly approve each action before it is applied.
    """
    row_index: int = Field(..., description="Row index to apply action to")
    column: Optional[str] = Field(None, description="Column (None for row-level actions)")
    action: SuggestedAction = Field(..., description="The approved action to take")
    custom_placeholder: Optional[str] = Field(
        None,
        description="Custom placeholder value (for REPLACE_WITH_PLACEHOLDER)"
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
