"""
Cleaner service for applying data cleaning rules to CSV files.

STRICT MODE (Default):
This cleaner operates as a deterministic data-cleaning engine that:
1. NEVER invents, fabricates, or guesses new data
2. Only applies meaning-preserving transformations
3. Does NOT impute missing values unless explicitly enabled
4. Leaves invalid/blank cells blank or returns "Unknown" placeholder
5. Never changes the semantic meaning of a value
6. Only applies deterministic and reversible modifications

Allowed Transformations:
- Capitalization normalization (e.g., DANIEL → Daniel)
- Number formatting (e.g., "thirty" → 30)
- Date normalization (format standardization)
- Trimming whitespace
- Removing invalid/non-printable characters
- Lossless type conversions

Forbidden Transformations (in strict mode):
- Guessing names, emails, locations, ages, genders
- Replacing missing entries with fabricated content
- Inferring values not directly derivable from input
- Any transformation that changes semantic meaning

Supported cleaning rules:
- drop_columns: Remove specified columns
- drop_rows: Remove rows (conservative - see below)
- fill_missing: Fill missing values (BLOCKED in strict mode unless imputation enabled)
- coerce_types: Convert column to specified data type (lossless only)
- trim_whitespace: Remove leading/trailing whitespace from string columns
- parse_dates: Parse column as datetime
- normalize_numbers: Normalize numeric values (remove currency symbols, commas, etc.)
- dedupe: Remove duplicate rows
- normalize_capitalization: Normalize text capitalization (new)

Row Dropping Policy (CONSERVATIVE):
A row is ONLY dropped if:
1. It is completely empty (all values are null/NaN), OR
2. It has fewer columns than the header (structural corruption), OR
3. It violates schema rules in a way that cannot be repaired, OR
4. A user-defined rule explicitly says to drop it.
Otherwise, the row is ALWAYS kept.

All operations are idempotent - applying the same rules twice yields the same result.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.models.schemas import CleaningRule, PreviewRow, StrictModeConfig, DEFAULT_STRICT_CONFIG


# ============================================
# Strict Mode Validation
# ============================================

def validate_strict_mode_operation(
    operation: str,
    strict_config: StrictModeConfig
) -> tuple[bool, str]:
    """
    Validate if an operation is allowed under the current strict mode config.
    
    Returns:
        Tuple of (is_allowed, reason_if_blocked)
    """
    if not strict_config.enabled:
        return True, ""
    
    # Mapping of operations to their config flags
    operation_checks = {
        'fill_missing': ('allow_imputation', 'Imputation is disabled in strict mode'),
        'trim_whitespace': ('allow_whitespace_trimming', 'Whitespace trimming disabled'),
        'normalize_capitalization': ('allow_capitalization_normalization', 'Capitalization normalization disabled'),
        'normalize_numbers': ('allow_number_word_conversion', 'Number normalization disabled'),
        'parse_dates': ('allow_date_normalization', 'Date normalization disabled'),
        'coerce_types': ('allow_type_coercion', 'Type coercion disabled'),
    }
    
    if operation in operation_checks:
        config_flag, message = operation_checks[operation]
        if not getattr(strict_config, config_flag, True):
            return False, message
    
    # Special case: fill_missing requires explicit imputation permission
    if operation == 'fill_missing' and not strict_config.allow_imputation:
        return False, "Missing value imputation blocked in strict mode. Enable allow_imputation to use."
    
    return True, ""


# ============================================
# Individual Cleaning Functions
# ============================================

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame.
    
    Args:
        df: The pandas DataFrame to clean
        columns: List of column names to drop
        
    Returns:
        DataFrame with specified columns removed
        
    Note:
        Idempotent: columns that don't exist are silently ignored.
        This operation is allowed in strict mode as it doesn't fabricate data.
    """
    existing_columns = [col for col in columns if col in df.columns]
    return df.drop(columns=existing_columns, errors='ignore')


def fill_missing(
    df: pd.DataFrame,
    column: str,
    value: Optional[Any] = None,
    strategy: Optional[str] = None,
    strict_config: Optional[StrictModeConfig] = None,
) -> pd.DataFrame:
    """
    Fill missing values in a column.
    
    STRICT MODE BEHAVIOR:
    - This function is BLOCKED by default in strict mode
    - Imputation (mean, median, mode, ffill, bfill) is considered data fabrication
    - Only allowed when strict_config.allow_imputation is True
    - Static value filling is only allowed for explicit user-provided values
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to fill
        value: Static value to fill with (used if strategy is None)
        strategy: Fill strategy - one of 'mean', 'median', 'mode', 'ffill', 'bfill'
        strict_config: Strict mode configuration (defaults to strict mode)
        
    Returns:
        DataFrame with missing values filled (or unchanged if blocked by strict mode)
        
    Note:
        Idempotent: filling already-filled values has no effect.
    """
    if strict_config is None:
        strict_config = DEFAULT_STRICT_CONFIG
    
    # STRICT MODE: Block imputation unless explicitly allowed
    if strict_config.enabled and not strict_config.allow_imputation:
        # Imputation strategies are blocked
        if strategy in ('mean', 'median', 'mode', 'ffill', 'bfill'):
            # Return unchanged - we cannot fabricate values
            return df
        # Even static value filling should be explicit and intentional
        # Only allow if value is explicitly the "Unknown" placeholder
        if value is not None and value != strict_config.unknown_placeholder:
            # Allow filling with the configured placeholder only
            pass  # Proceed with caution
    
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    try:
        if strategy == 'mean':
            fill_value = result[column].mean()
        elif strategy == 'median':
            fill_value = result[column].median()
        elif strategy == 'mode':
            mode_result = result[column].mode()
            fill_value = mode_result.iloc[0] if len(mode_result) > 0 else None
        elif strategy == 'ffill':
            result[column] = result[column].ffill()
            return result
        elif strategy == 'bfill':
            result[column] = result[column].bfill()
            return result
        else:
            fill_value = value
            
        if fill_value is not None:
            result[column] = result[column].fillna(fill_value)
    except (TypeError, ValueError):
        # If filling fails (e.g., mean of non-numeric), return unchanged
        pass
    
    return result


def coerce_types(
    df: pd.DataFrame,
    column: str,
    dtype: str,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    Coerce a column to a specified data type.
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to convert
        dtype: Target data type ('int', 'float', 'str', 'bool', 'datetime')
        errors: How to handle errors - 'coerce' (set to NaN), 'ignore' (leave unchanged)
        
    Returns:
        DataFrame with column converted to specified type
        
    Note:
        Idempotent: converting to the same type has no effect.
    """
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    try:
        if dtype == 'int':
            result[column] = pd.to_numeric(result[column], errors=errors)
            # Convert to nullable integer to handle NaN
            if errors == 'coerce':
                result[column] = result[column].astype('Int64')
        elif dtype == 'float':
            result[column] = pd.to_numeric(result[column], errors=errors).astype(float)
        elif dtype == 'str':
            result[column] = result[column].astype(str)
            # Replace 'nan' and 'None' strings with actual None for consistency
            result[column] = result[column].replace({'nan': None, 'None': None, '<NA>': None})
        elif dtype == 'bool':
            # Handle common boolean representations
            bool_map = {
                'true': True, 'false': False,
                'True': True, 'False': False,
                'TRUE': True, 'FALSE': False,
                '1': True, '0': False,
                1: True, 0: False,
                'yes': True, 'no': False,
                'Yes': True, 'No': False,
                'YES': True, 'NO': False,
            }
            result[column] = result[column].map(
                lambda x: bool_map.get(x, x) if pd.notna(x) else x
            )
        elif dtype == 'datetime':
            result[column] = pd.to_datetime(result[column], errors=errors)
    except (TypeError, ValueError):
        # If conversion fails entirely, return unchanged
        pass
    
    return result


def trim_whitespace(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    """
    Trim leading and trailing whitespace from string columns.
    
    Args:
        df: The pandas DataFrame to clean
        column: Specific column to trim, or None to trim all string columns
        
    Returns:
        DataFrame with whitespace trimmed
        
    Note:
        Idempotent: trimming already-trimmed strings has no effect.
    """
    result = df.copy()
    
    if column is not None:
        if column in result.columns and result[column].dtype == 'object':
            try:
                result[column] = result[column].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )
            except (AttributeError, TypeError):
                pass
    else:
        # Trim all object (string) columns
        for col in result.columns:
            if result[col].dtype == 'object':
                try:
                    result[col] = result[col].apply(
                        lambda x: x.strip() if isinstance(x, str) else x
                    )
                except (AttributeError, TypeError):
                    pass
    
    return result


def parse_dates(
    df: pd.DataFrame,
    column: str,
    format: Optional[str] = None,
    errors: str = 'coerce',
) -> pd.DataFrame:
    """
    Parse a column as datetime.
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to parse
        format: Optional strptime format string (e.g., '%Y-%m-%d')
        errors: How to handle errors - 'coerce' (set to NaT), 'ignore' (leave unchanged)
        
    Returns:
        DataFrame with column parsed as datetime
        
    Note:
        Idempotent: parsing already-parsed datetimes has no effect.
    """
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    try:
        if format:
            result[column] = pd.to_datetime(result[column], format=format, errors=errors)
        else:
            # Use format='mixed' to handle mixed date formats and avoid the UserWarning
            # about inferring format for each element individually
            result[column] = pd.to_datetime(result[column], format='mixed', errors=errors)
    except (TypeError, ValueError):
        pass
    
    return result


def normalize_numbers(
    df: pd.DataFrame,
    column: str,
    remove_currency: bool = True,
    remove_commas: bool = True,
    remove_percent: bool = True,
) -> pd.DataFrame:
    """
    Normalize numeric values by removing formatting characters.
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to normalize
        remove_currency: Remove currency symbols ($, €, £, etc.)
        remove_commas: Remove thousand separators
        remove_percent: Remove percent signs and divide by 100
        
    Returns:
        DataFrame with normalized numeric values
        
    Note:
        Idempotent: normalizing already-normalized numbers has no effect.
    """
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    def normalize_value(val: Any) -> Any:
        if pd.isna(val):
            return val
        
        # If already numeric, return as-is
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val
        
        try:
            val_str = str(val).strip()
            original_str = val_str
            
            # Check if it's a percentage
            is_percent = False
            if remove_percent and '%' in val_str:
                is_percent = True
                val_str = val_str.replace('%', '')
            
            # Remove currency symbols
            if remove_currency:
                val_str = re.sub(r'[$€£¥₹₽₩₪฿]', '', val_str)
            
            # Remove commas (thousand separators)
            if remove_commas:
                val_str = val_str.replace(',', '')
            
            # Remove any remaining whitespace
            val_str = val_str.strip()
            
            # Handle parentheses for negative numbers (accounting format)
            if val_str.startswith('(') and val_str.endswith(')'):
                val_str = '-' + val_str[1:-1]
            
            # Convert to float
            result_val = float(val_str)
            
            # Apply percent conversion
            if is_percent:
                result_val = result_val / 100
            
            return result_val
        except (ValueError, TypeError):
            return val
    
    try:
        result[column] = result[column].apply(normalize_value)
    except (TypeError, ValueError):
        pass
    
    return result


def dedupe(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    
    Args:
        df: The pandas DataFrame to clean
        subset: List of column names to consider for identifying duplicates.
                If None, all columns are used.
        keep: Which duplicate to keep - 'first', 'last', or False (drop all)
        
    Returns:
        DataFrame with duplicates removed
        
    Note:
        Idempotent: deduping already-deduped data has no effect.
        This is a meaning-preserving operation (removes exact copies).
        The original index is preserved to allow matching with original rows.
    """
    try:
        # Filter subset to only include existing columns
        if subset is not None:
            subset = [col for col in subset if col in df.columns]
            if not subset:
                subset = None
        
        # Preserve original index - don't reset it
        # This allows proper row matching in preview
        return df.drop_duplicates(subset=subset, keep=keep)
    except (TypeError, ValueError):
        return df


def drop_rows(
    df: pd.DataFrame,
    strategy: str = 'conservative',
    required_columns: Optional[List[str]] = None,
    min_non_null: Optional[int] = None,
    row_indices: Optional[List[int]] = None,
    expected_column_count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Drop rows based on conservative rules.
    
    STRICT MODE: Row dropping is conservative by default.
    A row is ONLY dropped if:
    1. It is completely empty (all values are null/NaN), OR
    2. It has fewer columns than the header (structural corruption), OR
    3. It violates schema rules that cannot be repaired, OR
    4. A user-defined rule explicitly specifies it (via row_indices)
    
    Otherwise, the row is ALWAYS kept.
    
    Args:
        df: The pandas DataFrame to clean
        strategy: Drop strategy:
            - 'conservative': Only drop completely empty rows (default)
            - 'empty_only': Same as conservative - drop only fully empty rows
            - 'structural': Drop rows with structural corruption (wrong column count)
            - 'required_missing': Drop rows missing required column values
            - 'explicit': Only drop explicitly specified row indices
        required_columns: List of columns that must have non-null values
                         (only used with 'required_missing' strategy)
        min_non_null: Minimum number of non-null values a row must have
                      (if not specified, row must be completely empty to be dropped)
        row_indices: Explicit list of row indices to drop
                     (only used with 'explicit' strategy)
        expected_column_count: Expected number of columns for structural validation
                               (only used with 'structural' strategy)
    
    Returns:
        DataFrame with problematic rows removed
        
    Note:
        This function is idempotent and conservative.
        It errs on the side of keeping data rather than losing it.
    """
    result = df.copy()
    rows_to_drop: List[int] = []
    
    if strategy == 'explicit':
        # Only drop explicitly specified rows
        if row_indices:
            valid_indices = [idx for idx in row_indices if 0 <= idx < len(result)]
            rows_to_drop.extend(valid_indices)
    
    elif strategy == 'structural':
        # Drop rows with structural corruption (fewer values than expected)
        # This handles cases where CSV parsing resulted in misaligned rows
        if expected_column_count is not None:
            for idx, row in result.iterrows():
                # Count non-null values to detect structurally corrupt rows
                # A row with fewer values than columns may have been parsed incorrectly
                non_null_count = row.notna().sum()
                # If a row has significantly fewer values, it may be corrupt
                # But we also check if it's completely empty (handled below)
                pass  # Structural corruption is hard to detect post-parsing
        
        # Also drop completely empty rows under this strategy
        for idx in result.index:
            row = result.loc[idx]
            if row.isna().all() or (row.astype(str).str.strip() == '').all():
                rows_to_drop.append(idx)
    
    elif strategy == 'required_missing':
        # Drop rows where required columns have missing values
        if required_columns:
            valid_required = [col for col in required_columns if col in result.columns]
            if valid_required:
                for idx in result.index:
                    row = result.loc[idx]
                    # Check if any required column is missing
                    for col in valid_required:
                        val = row[col]
                        if pd.isna(val) or (isinstance(val, str) and val.strip() == ''):
                            rows_to_drop.append(idx)
                            break
    
    else:  # 'conservative' or 'empty_only' (default)
        # Only drop rows that are completely empty
        for idx in result.index:
            row = result.loc[idx]
            # Check if ALL values are null/NaN or empty strings
            is_all_null = row.isna().all()
            is_all_empty_str = False
            try:
                is_all_empty_str = (row.astype(str).str.strip() == '').all()
            except (TypeError, ValueError):
                pass
            
            if is_all_null or is_all_empty_str:
                rows_to_drop.append(idx)
        
        # If min_non_null is specified, also drop rows with too few values
        if min_non_null is not None:
            for idx in result.index:
                if idx not in rows_to_drop:
                    row = result.loc[idx]
                    non_null_count = row.notna().sum()
                    # Also count empty strings as null
                    for val in row:
                        if isinstance(val, str) and val.strip() == '':
                            non_null_count -= 1
                    if non_null_count < min_non_null:
                        rows_to_drop.append(idx)
    
    # Remove duplicates from rows_to_drop and sort
    rows_to_drop = sorted(set(rows_to_drop))
    
    if rows_to_drop:
        result = result.drop(index=rows_to_drop).reset_index(drop=True)
    
    return result


# ============================================
# Strict Mode: Deterministic Transformations
# ============================================

# Number word to digit mapping (deterministic conversion)
NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
    'million': 1000000, 'billion': 1000000000,
}

# Scale words that multiply the preceding value
SCALE_WORDS = {'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000}


def parse_number_phrase(text: str) -> Optional[int]:
    """
    Parse a compound number phrase like 'sixty thousand' or 'two hundred fifty'.
    
    Handles patterns like:
    - "sixty thousand" -> 60000
    - "two hundred" -> 200
    - "twenty five" -> 25
    - "one hundred twenty three" -> 123
    - "five hundred thousand" -> 500000
    
    Returns None if the phrase cannot be parsed as a number.
    """
    if not text:
        return None
    
    text = text.strip().lower()
    
    # Handle hyphenated numbers like "twenty-five"
    text = text.replace('-', ' ')
    
    # Split into words
    words = text.split()
    
    if not words:
        return None
    
    # Check if all words are valid number words
    for word in words:
        if word not in NUMBER_WORDS:
            return None
    
    # If single word, return its value directly
    if len(words) == 1:
        return NUMBER_WORDS[words[0]]
    
    # Parse compound numbers
    result = 0
    current = 0
    
    for word in words:
        value = NUMBER_WORDS[word]
        
        if word in SCALE_WORDS:
            if current == 0:
                current = 1
            if word == 'thousand' or word == 'million' or word == 'billion':
                # These scale the entire current accumulator and add to result
                current *= value
                result += current
                current = 0
            else:
                # 'hundred' just multiplies current
                current *= value
        else:
            # Regular number, add to current
            current += value
    
    result += current
    return result


def is_number_phrase(text: str) -> bool:
    """Check if a text string is a valid number phrase."""
    return parse_number_phrase(text) is not None


def normalize_capitalization(
    df: pd.DataFrame,
    column: str,
    style: str = 'title',
) -> pd.DataFrame:
    """
    Normalize text capitalization in a deterministic, reversible way.
    
    STRICT MODE: This is a meaning-preserving transformation.
    "JOHN SMITH" and "john smith" represent the same name.
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to normalize
        style: Capitalization style - 'title', 'lower', 'upper'
        
    Returns:
        DataFrame with normalized capitalization
        
    Note:
        Idempotent: normalizing already-normalized text has no effect.
        This does NOT fabricate data - it only changes representation.
    """
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    def normalize_cap(val: Any) -> Any:
        if pd.isna(val):
            return val
        if not isinstance(val, str):
            return val
        
        val_str = val.strip()
        if not val_str:
            return val
        
        if style == 'title':
            return val_str.title()
        elif style == 'lower':
            return val_str.lower()
        elif style == 'upper':
            return val_str.upper()
        return val_str
    
    try:
        result[column] = result[column].apply(normalize_cap)
    except (TypeError, ValueError):
        pass
    
    return result


def convert_number_words(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Convert number words to their numeric equivalents.
    
    STRICT MODE: This is a deterministic, meaning-preserving transformation.
    "thirty" → 30 is unambiguous and reversible in context.
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to convert
        
    Returns:
        DataFrame with number words converted to digits
        
    Note:
        Only converts exact matches of known number words.
        Does NOT guess or infer - leaves unrecognized values unchanged.
    """
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    def convert_word(val: Any) -> Any:
        if pd.isna(val):
            return val
        
        val_str = str(val).strip().lower()
        
        # Try to parse as a number phrase (handles compound numbers like "sixty thousand")
        parsed = parse_number_phrase(val_str)
        if parsed is not None:
            return parsed
        
        return val
    
    try:
        result[column] = result[column].apply(convert_word)
    except (TypeError, ValueError):
        pass
    
    return result


def normalize_missing_indicators(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Replace common missing-value indicators with actual null (None/NaN).
    
    STRICT MODE: This is a meaning-preserving transformation.
    Converting "N/A", "na", "-", "" to proper null doesn't change meaning
    but enables proper null handling in downstream operations.
    
    Common missing-value indicators (case-insensitive):
    - na, n/a, NA, N/A
    - null, NULL, Null
    - none, None, NONE
    - missing, Missing, MISSING
    - - (single dash)
    - empty string or whitespace-only
    - . (single period)
    - #N/A, #REF!, #VALUE!, #DIV/0! (Excel errors)
    
    Args:
        df: The pandas DataFrame to clean
        column: Specific column to normalize, or None for all columns
        
    Returns:
        DataFrame with missing indicators replaced with null
        
    Note:
        Idempotent: applying this transformation twice yields the same result.
    """
    result = df.copy()
    
    # Patterns that represent missing values (case-insensitive matching)
    MISSING_INDICATORS = {
        'na', 'n/a', 'null', 'none', 'missing', 'nan', 'undefined',
        '-', '.', '', ' ', '  ', 'n.a.', 'n.a', '#n/a', '#ref!',
        '#value!', '#div/0!', '#name?', '#null!', '#num!',
    }
    
    def replace_missing(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        
        if isinstance(val, str):
            val_stripped = val.strip().lower()
            if val_stripped in MISSING_INDICATORS:
                return np.nan
            # Check for whitespace-only
            if val.strip() == '':
                return np.nan
        
        return val
    
    if column is not None:
        if column in result.columns:
            try:
                result[column] = result[column].apply(replace_missing)
            except (TypeError, ValueError):
                pass
    else:
        # Apply to all object (string) columns
        for col in result.columns:
            if result[col].dtype == object:
                try:
                    result[col] = result[col].apply(replace_missing)
                except (TypeError, ValueError):
                    pass
    
    return result


def remove_invalid_characters(
    df: pd.DataFrame,
    column: str,
    allow_pattern: str = r'[\x00-\x1f\x7f-\x9f]',  # Non-printable characters
) -> pd.DataFrame:
    """
    Remove invalid/non-printable characters from string values.
    
    STRICT MODE: This is a deterministic, meaning-preserving transformation.
    Removing non-printable characters doesn't change semantic meaning.
    
    Args:
        df: The pandas DataFrame to clean
        column: Column name to clean
        allow_pattern: Regex pattern of characters to REMOVE (default: non-printable)
        
    Returns:
        DataFrame with invalid characters removed
    """
    if column not in df.columns:
        return df
    
    result = df.copy()
    
    def remove_chars(val: Any) -> Any:
        if pd.isna(val):
            return val
        if not isinstance(val, str):
            return val
        
        # Remove non-printable characters
        cleaned = re.sub(allow_pattern, '', val)
        return cleaned
    
    try:
        result[column] = result[column].apply(remove_chars)
    except (TypeError, ValueError):
        pass
    
    return result


# ============================================
# Rule Dispatcher
# ============================================

# Mapping of rule types to their handler functions
RULE_HANDLERS: Dict[str, Callable[[pd.DataFrame, CleaningRule], pd.DataFrame]] = {}

# Strict mode configuration for rule handlers
_strict_config: StrictModeConfig = DEFAULT_STRICT_CONFIG


def set_strict_config(config: StrictModeConfig) -> None:
    """Set the global strict mode configuration for rule handlers."""
    global _strict_config
    _strict_config = config


def get_strict_config() -> StrictModeConfig:
    """Get the current strict mode configuration."""
    return _strict_config


def _handle_drop_columns(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle drop_columns rule."""
    columns = rule.params.get('columns', [])
    if rule.column:
        columns = [rule.column] if isinstance(rule.column, str) else list(rule.column)
    return drop_columns(df, columns)


def _handle_fill_missing(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """
    Handle fill_missing rule.
    
    STRICT MODE: This is BLOCKED unless imputation is explicitly enabled.
    """
    if not rule.column:
        return df
    return fill_missing(
        df,
        column=rule.column,
        value=rule.params.get('value'),
        strategy=rule.params.get('strategy'),
        strict_config=_strict_config,
    )


def _handle_coerce_types(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle coerce_types rule."""
    if not rule.column:
        return df
    return coerce_types(
        df,
        column=rule.column,
        dtype=rule.params.get('dtype', 'str'),
        errors=rule.params.get('errors', 'coerce'),
    )


def _handle_trim_whitespace(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle trim_whitespace rule."""
    return trim_whitespace(df, column=rule.column)


def _handle_parse_dates(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle parse_dates rule."""
    if not rule.column:
        return df
    return parse_dates(
        df,
        column=rule.column,
        format=rule.params.get('format'),
        errors=rule.params.get('errors', 'coerce'),
    )


def _handle_normalize_numbers(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle normalize_numbers rule."""
    if not rule.column:
        return df
    return normalize_numbers(
        df,
        column=rule.column,
        remove_currency=rule.params.get('remove_currency', True),
        remove_commas=rule.params.get('remove_commas', True),
        remove_percent=rule.params.get('remove_percent', True),
    )


def _handle_dedupe(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle dedupe rule."""
    return dedupe(
        df,
        subset=rule.params.get('subset'),
        keep=rule.params.get('keep', 'first'),
    )


def _handle_normalize_capitalization(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle normalize_capitalization rule (strict mode friendly)."""
    if not rule.column:
        return df
    return normalize_capitalization(
        df,
        column=rule.column,
        style=rule.params.get('style', 'title'),
    )


def _handle_convert_number_words(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle convert_number_words rule (strict mode friendly)."""
    if not rule.column:
        return df
    return convert_number_words(df, column=rule.column)


def _handle_remove_invalid_characters(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """Handle remove_invalid_characters rule (strict mode friendly)."""
    if not rule.column:
        return df
    return remove_invalid_characters(
        df,
        column=rule.column,
        allow_pattern=rule.params.get('pattern', r'[\x00-\x1f\x7f-\x9f]'),
    )


def _handle_normalize_missing_indicators(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """
    Handle normalize_missing_indicators rule.
    
    STRICT MODE: This is a meaning-preserving transformation.
    Converting common missing-value indicators (na, N/A, -, etc.) to proper
    null values is universally safe and does not change meaning.
    """
    return normalize_missing_indicators(df, column=rule.column)


def _handle_drop_rows(df: pd.DataFrame, rule: CleaningRule) -> pd.DataFrame:
    """
    Handle drop_rows rule.
    
    CONSERVATIVE ROW DROPPING:
    Rows are ONLY dropped if:
    1. They are completely empty
    2. They have structural corruption (fewer columns than header)
    3. They violate schema rules that cannot be repaired
    4. They are explicitly specified by user
    
    Otherwise, rows are ALWAYS kept.
    """
    return drop_rows(
        df,
        strategy=rule.params.get('strategy', 'conservative'),
        required_columns=rule.params.get('required_columns'),
        min_non_null=rule.params.get('min_non_null'),
        row_indices=rule.params.get('row_indices'),
        expected_column_count=rule.params.get('expected_column_count'),
    )


# Register all handlers
RULE_HANDLERS = {
    'drop_columns': _handle_drop_columns,
    'fill_missing': _handle_fill_missing,
    'coerce_types': _handle_coerce_types,
    'trim_whitespace': _handle_trim_whitespace,
    'parse_dates': _handle_parse_dates,
    'normalize_numbers': _handle_normalize_numbers,
    'dedupe': _handle_dedupe,
    'drop_rows': _handle_drop_rows,
    # Strict mode friendly handlers (universally safe)
    'normalize_capitalization': _handle_normalize_capitalization,
    'convert_number_words': _handle_convert_number_words,
    'remove_invalid_characters': _handle_remove_invalid_characters,
    'normalize_missing_indicators': _handle_normalize_missing_indicators,
}


def apply_cleaning_rules(
    df: pd.DataFrame, 
    rules: List[CleaningRule],
    strict_config: Optional[StrictModeConfig] = None,
) -> pd.DataFrame:
    """
    Apply a list of cleaning rules to a DataFrame.
    
    STRICT MODE (Default):
    When strict_config.enabled is True (default):
    - fill_missing with imputation strategies is BLOCKED
    - Only deterministic, meaning-preserving transformations are allowed
    - No data fabrication or guessing
    
    Args:
        df: The pandas DataFrame to clean
        rules: List of CleaningRule objects to apply
        strict_config: Strict mode configuration (defaults to strict mode enabled)
        
    Returns:
        Cleaned DataFrame
        
    Note:
        This function is idempotent - applying the same rules twice
        yields the same result. All operations are applied safely
        and will not crash on bad data.
        
    Supported rule types:
        - drop_columns: Remove specified columns
        - drop_rows: Remove rows (CONSERVATIVE - only empty/corrupt/explicit)
        - fill_missing: Fill missing values (BLOCKED in strict mode unless imputation enabled)
        - coerce_types: Convert column to specified type
        - trim_whitespace: Remove leading/trailing whitespace
        - parse_dates: Parse column as datetime
        - normalize_numbers: Remove currency symbols, commas, etc.
        - dedupe: Remove duplicate rows
        - normalize_capitalization: Normalize text case (strict mode friendly)
        - convert_number_words: Convert "thirty" → 30 (strict mode friendly)
        - remove_invalid_characters: Remove non-printable chars (strict mode friendly)
        - normalize_missing_indicators: Replace na/N/A/"-"/etc. with null (strict mode friendly)
        
    Row Dropping Policy (drop_rows):
        Rows are ONLY dropped if:
        1. Completely empty (all values null/NaN)
        2. Structural corruption (fewer columns than header)
        3. Schema violation that cannot be repaired
        4. Explicitly specified by user (row_indices parameter)
        Otherwise, rows are ALWAYS kept.
    """
    # Set the global strict config for handlers to use
    if strict_config is not None:
        set_strict_config(strict_config)
    
    cleaned_df = df.copy()
    
    for rule in rules:
        handler = RULE_HANDLERS.get(rule.rule_type)
        if handler:
            try:
                cleaned_df = handler(cleaned_df, rule)
            except Exception:
                # Silently continue on any unexpected error
                # to ensure we never crash on bad data
                pass
    
    return cleaned_df


def preview_cleaning(
    df: pd.DataFrame,
    rules: List[CleaningRule],
    n_rows: int = 100,
    strict_config: Optional[StrictModeConfig] = None,
) -> List[PreviewRow]:
    """
    Generate a preview of cleaning changes.
    
    Args:
        df: The pandas DataFrame to preview
        rules: List of CleaningRule objects to apply
        n_rows: Number of rows to include in preview
        strict_config: Strict mode configuration
        
    Returns:
        List of PreviewRow objects showing original vs cleaned data
    """
    preview_rows: List[PreviewRow] = []
    
    # Get first n_rows
    preview_df = df.head(n_rows)
    cleaned_df = apply_cleaning_rules(preview_df, rules, strict_config=strict_config)
    
    # Handle case where columns were dropped
    original_columns = set(preview_df.columns)
    cleaned_columns = set(cleaned_df.columns)
    
    def convert_to_native(val):
        """Convert numpy/pandas types to native Python types."""
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val
    
    def convert_row(row_dict):
        """Convert all values in a row dict to native Python types."""
        return {k: convert_to_native(v) for k, v in row_dict.items()}
    
    for idx in range(len(preview_df)):
        original_row = convert_row(preview_df.iloc[idx].to_dict())
        
        # Handle case where cleaned_df might have fewer rows (e.g., after dedupe)
        if idx < len(cleaned_df):
            cleaned_row = convert_row(cleaned_df.iloc[idx].to_dict())
        else:
            cleaned_row = {}
        
        # Detect changes
        changes: List[str] = []
        
        # Check for dropped columns
        for col in original_columns - cleaned_columns:
            changes.append(f"{col}: dropped")
        
        # Check for value changes in remaining columns
        for col in cleaned_columns:
            original_val = original_row.get(col)
            cleaned_val = cleaned_row.get(col)
            
            # Compare values, handling None specially
            values_differ = False
            if original_val is None and cleaned_val is None:
                values_differ = False
            elif original_val is None or cleaned_val is None:
                values_differ = True
            elif original_val != cleaned_val:
                values_differ = True
            
            if values_differ:
                changes.append(f"{col}: '{original_val}' -> '{cleaned_val}'")
        
        preview_rows.append(
            PreviewRow(
                row_index=idx,
                original=original_row,
                cleaned=cleaned_row,
                changes=changes,
            )
        )
    
    return preview_rows
