"""
Safety Guards for Data Cleaning Operations

FUNDAMENTAL PRINCIPLES (NEVER BREAK THESE):
1. NEVER invent, guess, or fabricate data
2. ONLY apply deterministic, meaning-preserving transformations
3. NEVER automatically fix or change anything without user approval
4. Keep all raw data intact until after user confirmation
5. Missing/invalid cells remain blank unless user selects a placeholder
6. Imputation is strictly opt-in and OFF by default
7. Dropping rows is strictly opt-in and OFF by default

This module provides safety validation for all cleaning operations.
It blocks forbidden actions and generates appropriate warnings.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.models.schemas import (
    SuggestedAction,
    UserApprovedAction,
    StrictModeConfig,
    DEFAULT_STRICT_CONFIG,
)


class ForbiddenAction(str, Enum):
    """Actions that are NEVER allowed unless explicitly enabled."""
    GUESS_NAME = "guess_name"
    GUESS_EMAIL = "guess_email"
    GUESS_AGE = "guess_age"
    GUESS_CATEGORY = "guess_category"
    FILL_MISSING_ASSUMPTION = "fill_missing_assumption"
    INVENT_REPLACEMENT = "invent_replacement"
    GENERATE_RANDOM = "generate_random"
    CHANGE_MEANING = "change_meaning"
    AUTO_DROP_ROW = "auto_drop_row"
    SILENT_REWRITE = "silent_rewrite"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    IMPUTE_FFILL = "impute_ffill"
    IMPUTE_BFILL = "impute_bfill"


class SafetyViolation(BaseModel):
    """A detected safety violation."""
    action: str = Field(..., description="The action that was attempted")
    reason: str = Field(..., description="Why this action is forbidden")
    suggestion: str = Field(..., description="What to do instead")


class SafetyCheckResult(BaseModel):
    """Result of a safety check."""
    is_safe: bool = Field(..., description="Whether the operation is safe")
    violations: List[SafetyViolation] = Field(
        default_factory=list,
        description="List of safety violations if any"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about the operation"
    )


# ============================================
# Forbidden Action Detection
# ============================================

FORBIDDEN_IMPUTATION_STRATEGIES = {
    'mean': ForbiddenAction.IMPUTE_MEAN,
    'median': ForbiddenAction.IMPUTE_MEDIAN,
    'mode': ForbiddenAction.IMPUTE_MODE,
    'ffill': ForbiddenAction.IMPUTE_FFILL,
    'bfill': ForbiddenAction.IMPUTE_BFILL,
}

# Keywords that suggest forbidden data fabrication
FABRICATION_KEYWORDS = [
    'guess', 'infer', 'predict', 'estimate', 'assume',
    'generate', 'create', 'synthesize', 'fabricate',
    'random', 'fill_missing', 'auto_fill', 'smart_fill',
]


def check_action_safety(
    action: UserApprovedAction,
    strict_config: Optional[StrictModeConfig] = None,
) -> SafetyCheckResult:
    """
    Check if a user action is safe to apply.
    
    Returns a SafetyCheckResult indicating whether the action is allowed
    and any violations or warnings.
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    violations = []
    warnings = []
    
    # All standard SuggestedAction values are safe by design
    # But we still validate specific cases
    
    if action.action == SuggestedAction.DROP_ROW:
        # Row dropping requires justification
        warnings.append(
            "Row dropping requested. This will only succeed if the row "
            "is completely empty or structurally corrupt."
        )
    
    if action.action == SuggestedAction.REPLACE_WITH_PLACEHOLDER:
        if not action.custom_placeholder:
            warnings.append(
                "No custom placeholder specified. Will use default: "
                f"'{strict_config.unknown_placeholder}'"
            )
    
    return SafetyCheckResult(
        is_safe=len(violations) == 0,
        violations=violations,
        warnings=warnings
    )


def check_rule_safety(
    rule_type: str,
    params: Dict[str, Any],
    strict_config: Optional[StrictModeConfig] = None,
) -> SafetyCheckResult:
    """
    Check if a cleaning rule is safe to apply.
    
    Blocks forbidden imputation and other unsafe operations.
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    violations = []
    warnings = []
    
    # Check for forbidden imputation strategies
    if rule_type == 'fill_missing':
        strategy = params.get('strategy')
        
        if strategy in FORBIDDEN_IMPUTATION_STRATEGIES:
            if strict_config.enabled and not strict_config.allow_imputation:
                violations.append(SafetyViolation(
                    action=f"fill_missing with strategy '{strategy}'",
                    reason=f"Imputation strategy '{strategy}' is forbidden in strict mode. "
                           "This would fabricate values not present in the original data.",
                    suggestion="Use 'replace_with_blank' or 'replace_with_placeholder' instead, "
                               "or enable allow_imputation in strict_config if you understand the risks."
                ))
        
        # Check for arbitrary value filling
        value = params.get('value')
        if value is not None and strategy is None:
            # Static value filling - allowed if explicit
            warnings.append(
                f"Filling missing values with static value '{value}'. "
                "Ensure this is the user's explicit choice."
            )
    
    # Check for auto row dropping
    if rule_type == 'drop_rows':
        strategy = params.get('strategy', 'conservative')
        if strategy not in ('conservative', 'empty_only', 'explicit'):
            warnings.append(
                f"Row dropping strategy '{strategy}' may drop rows that aren't completely empty. "
                "Use 'conservative' or 'empty_only' for safe row dropping."
            )
    
    return SafetyCheckResult(
        is_safe=len(violations) == 0,
        violations=violations,
        warnings=warnings
    )


def validate_transformation_is_deterministic(
    original_value: Any,
    new_value: Any,
    transformation_type: str,
) -> SafetyCheckResult:
    """
    Validate that a transformation is deterministic and meaning-preserving.
    
    A transformation is deterministic if:
    1. The same input always produces the same output
    2. The transformation can be reversed or understood
    3. No information is lost (or loss is intentional, like whitespace)
    4. The semantic meaning is preserved
    """
    violations = []
    warnings = []
    
    # Whitespace trimming - always deterministic
    if transformation_type == 'trim_whitespace':
        if isinstance(original_value, str) and isinstance(new_value, str):
            if original_value.strip() == new_value:
                return SafetyCheckResult(is_safe=True, violations=[], warnings=[])
            else:
                violations.append(SafetyViolation(
                    action="trim_whitespace",
                    reason="Whitespace trimming produced unexpected result",
                    suggestion="Verify the transformation logic"
                ))
    
    # Capitalization normalization - deterministic
    if transformation_type == 'normalize_capitalization':
        if isinstance(original_value, str) and isinstance(new_value, str):
            if original_value.lower() == new_value.lower():
                return SafetyCheckResult(is_safe=True, violations=[], warnings=[
                    "Capitalization changed - this preserves meaning but changes representation"
                ])
    
    # Number word conversion - deterministic
    if transformation_type == 'convert_number_word':
        # Validate it's a known number word
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        }
        if isinstance(original_value, str):
            if original_value.lower().strip() in number_words:
                expected = number_words[original_value.lower().strip()]
                if new_value == expected:
                    return SafetyCheckResult(is_safe=True, violations=[], warnings=[])
    
    # Date format standardization - deterministic
    if transformation_type == 'standardize_date':
        import pandas as pd
        try:
            original_parsed = pd.to_datetime(original_value)
            new_parsed = pd.to_datetime(new_value)
            if original_parsed == new_parsed:
                return SafetyCheckResult(is_safe=True, violations=[], warnings=[
                    "Date format changed - this preserves the date value"
                ])
        except (ValueError, TypeError):
            pass
    
    return SafetyCheckResult(
        is_safe=len(violations) == 0,
        violations=violations,
        warnings=warnings
    )


def generate_safety_warning(
    action_type: str,
    context: Dict[str, Any],
) -> Optional[str]:
    """
    Generate a safety warning for potentially risky operations.
    
    Returns None if no warning is needed.
    """
    if action_type == 'fill_missing':
        strategy = context.get('strategy')
        if strategy in FORBIDDEN_IMPUTATION_STRATEGIES:
            return (
                f"⚠️ SAFETY WARNING: The '{strategy}' imputation strategy "
                "fabricates values that don't exist in your data. "
                "This can introduce bias and reduce data quality. "
                "Consider using explicit placeholder values instead."
            )
    
    if action_type == 'drop_rows':
        count = context.get('count', 0)
        if count > 0:
            return (
                f"⚠️ SAFETY WARNING: {count} rows will be permanently deleted. "
                "Only empty or structurally corrupt rows should be dropped. "
                "Review the rows carefully before confirming."
            )
    
    if action_type == 'bulk_replace':
        return (
            "⚠️ SAFETY WARNING: Bulk replacement will modify multiple values. "
            "Ensure this is deterministic and meaning-preserving."
        )
    
    return None


def block_forbidden_request(
    request_description: str,
) -> Tuple[bool, str]:
    """
    Check if a request describes a forbidden operation.
    
    Returns (is_blocked, reason) tuple.
    """
    request_lower = request_description.lower()
    
    # Check for fabrication keywords
    for keyword in FABRICATION_KEYWORDS:
        if keyword in request_lower:
            return (
                True,
                f"Request contains forbidden keyword '{keyword}'. "
                "Data fabrication is not allowed in strict mode. "
                "Please use explicit, user-chosen values instead."
            )
    
    # Check for specific forbidden patterns
    forbidden_patterns = [
        ('fill.*missing.*random', "Random value filling is forbidden"),
        ('guess.*name', "Name guessing is forbidden"),
        ('guess.*email', "Email guessing is forbidden"),
        ('infer.*age', "Age inference is forbidden"),
        ('auto.*fill', "Automatic filling is forbidden"),
        ('smart.*fill', "Smart filling (inference) is forbidden"),
    ]
    
    import re
    for pattern, reason in forbidden_patterns:
        if re.search(pattern, request_lower):
            return (True, reason + ". Use explicit placeholder values instead.")
    
    return (False, "")


# ============================================
# Allowed Transformations (Safe List)
# ============================================

SAFE_TRANSFORMATIONS = {
    'trim_whitespace': {
        'description': 'Remove leading/trailing whitespace',
        'reversible': True,
        'meaning_preserving': True,
    },
    'normalize_capitalization': {
        'description': 'Standardize text capitalization (DANIEL → Daniel)',
        'reversible': True,
        'meaning_preserving': True,
    },
    'convert_number_words': {
        'description': 'Convert written numbers to digits (thirty → 30)',
        'reversible': True,
        'meaning_preserving': True,
    },
    'standardize_date_format': {
        'description': 'Convert dates to ISO format (YYYY-MM-DD)',
        'reversible': True,
        'meaning_preserving': True,
    },
    'remove_currency_symbols': {
        'description': 'Remove currency symbols ($100 → 100)',
        'reversible': False,  # Symbol is lost
        'meaning_preserving': True,  # Value is preserved
    },
    'convert_percentage': {
        'description': 'Convert percentage to decimal (50% → 0.5)',
        'reversible': True,
        'meaning_preserving': True,
    },
    'remove_invalid_characters': {
        'description': 'Remove non-printable characters',
        'reversible': False,
        'meaning_preserving': True,  # Non-printable chars have no meaning
    },
    'dedupe': {
        'description': 'Remove exact duplicate rows',
        'reversible': False,  # Duplicates are lost
        'meaning_preserving': True,  # Keeps one copy of each unique row
    },
    'validate_format': {
        'description': 'Validate format (email, phone, etc.) - detection only',
        'reversible': True,  # No change made
        'meaning_preserving': True,
    },
    'replace_with_placeholder': {
        'description': 'Replace value with explicit user-chosen placeholder',
        'reversible': False,  # Original is lost
        'meaning_preserving': False,  # Meaning changes to "unknown"
        'requires_user_approval': True,
    },
    'replace_with_blank': {
        'description': 'Clear the cell value',
        'reversible': False,
        'meaning_preserving': False,
        'requires_user_approval': True,
    },
    'drop_row': {
        'description': 'Remove entire row (only for empty/corrupt rows)',
        'reversible': False,
        'meaning_preserving': False,
        'requires_user_approval': True,
        'restricted': True,  # Only allowed for qualifying rows
    },
}


def is_safe_transformation(transformation_type: str) -> bool:
    """Check if a transformation type is in the safe list."""
    return transformation_type in SAFE_TRANSFORMATIONS


def get_transformation_info(transformation_type: str) -> Optional[Dict[str, Any]]:
    """Get information about a transformation type."""
    return SAFE_TRANSFORMATIONS.get(transformation_type)
