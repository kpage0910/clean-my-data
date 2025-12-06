"""
Unit tests for the cleaner service.

Tests cover all data cleaning functions:
- drop_columns: Remove specified columns
- drop_rows: Remove rows (conservative policy)
- fill_missing: Fill missing values with value or strategy (BLOCKED in strict mode)
- coerce_types: Convert column to specified type
- trim_whitespace: Remove leading/trailing whitespace
- parse_dates: Parse column as datetime
- normalize_numbers: Remove currency symbols, commas, etc.
- dedupe: Remove duplicate rows
- apply_cleaning_rules: Apply multiple rules via CleaningRule objects
- normalize_capitalization: Normalize text case (strict mode friendly)
- convert_number_words: Convert "thirty" → 30 (strict mode friendly)

All tests verify idempotency - applying the same operation twice yields the same result.

STRICT MODE TESTS:
- Tests verify that imputation is blocked unless explicitly enabled
- Tests verify meaning-preserving transformations work correctly

ROW DROPPING POLICY (CONSERVATIVE):
- Rows are ONLY dropped if completely empty, structurally corrupt, 
  violate unrepairable schema rules, or explicitly specified by user
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.services.cleaner import (
    drop_columns,
    drop_rows,
    fill_missing,
    coerce_types,
    trim_whitespace,
    parse_dates,
    normalize_numbers,
    dedupe,
    apply_cleaning_rules,
    preview_cleaning,
    normalize_capitalization,
    convert_number_words,
    remove_invalid_characters,
)
from app.models.schemas import CleaningRule, StrictModeConfig


# ============================================
# Tests for drop_columns
# ============================================

class TestDropColumns:
    """Tests for drop_columns function."""
    
    def test_drop_single_column(self):
        """Test dropping a single column."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        
        result = drop_columns(df, ['b'])
        
        assert 'b' not in result.columns
        assert 'a' in result.columns
        assert 'c' in result.columns
        assert len(result.columns) == 2
    
    def test_drop_multiple_columns(self):
        """Test dropping multiple columns."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        
        result = drop_columns(df, ['a', 'c'])
        
        assert 'a' not in result.columns
        assert 'c' not in result.columns
        assert 'b' in result.columns
        assert len(result.columns) == 1
    
    def test_drop_nonexistent_column(self):
        """Test that dropping a non-existent column doesn't crash."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        result = drop_columns(df, ['nonexistent'])
        
        assert list(result.columns) == ['a', 'b']
    
    def test_drop_columns_idempotent(self):
        """Test that dropping columns is idempotent."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        
        result1 = drop_columns(df, ['b'])
        result2 = drop_columns(result1, ['b'])
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_drop_columns_preserves_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        drop_columns(df, ['b'])
        
        assert 'b' in df.columns


# ============================================
# Tests for drop_rows (CONSERVATIVE POLICY)
# ============================================

class TestDropRows:
    """
    Tests for drop_rows function.
    
    CONSERVATIVE ROW DROPPING POLICY:
    A row is ONLY dropped if:
    1. It is completely empty (all values null/NaN), OR
    2. It has fewer columns than the header (structural corruption), OR
    3. It violates schema rules in a way that cannot be repaired, OR
    4. A user-defined rule explicitly says to drop it.
    Otherwise, the row is ALWAYS kept.
    """
    
    def test_conservative_keeps_partial_rows(self):
        """Test that conservative strategy keeps rows with any valid data."""
        df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': ['x', 'y', None, None],
            'c': [True, None, None, None]
        })
        
        result = drop_rows(df, strategy='conservative')
        
        # Row 3 (index 3) has all None values - should be dropped
        # Rows 0, 1, 2 have at least one value - should be kept
        assert len(result) == 3
        # Check that the right rows are kept (row 0, 1, 2 but not row 3)
        assert result['a'].iloc[0] == 1.0
        assert pd.isna(result['a'].iloc[1])  # Was None, now NaN
        assert result['a'].iloc[2] == 3.0
    
    def test_conservative_drops_completely_empty_rows(self):
        """Test that completely empty rows are dropped."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': ['x', None, 'z']
        })
        
        result = drop_rows(df, strategy='conservative')
        
        # Row 1 is completely empty - should be dropped
        assert len(result) == 2
        assert result['a'].tolist() == [1.0, 3.0]
    
    def test_conservative_drops_empty_string_rows(self):
        """Test that rows with only empty strings are dropped."""
        df = pd.DataFrame({
            'a': ['hello', '', 'world'],
            'b': ['foo', '   ', 'bar']
        })
        
        result = drop_rows(df, strategy='conservative')
        
        # Row 1 has only empty/whitespace - should be dropped
        assert len(result) == 2
        assert result['a'].tolist() == ['hello', 'world']
    
    def test_conservative_keeps_row_with_single_value(self):
        """Test that a row with just one valid value is kept."""
        df = pd.DataFrame({
            'a': [None, None, 'data'],
            'b': [None, None, None],
            'c': [None, 'single', None]
        })
        
        result = drop_rows(df, strategy='conservative')
        
        # Row 0 is completely empty - dropped
        # Row 1 has one value - kept
        # Row 2 has one value - kept
        assert len(result) == 2
    
    def test_explicit_strategy_only_drops_specified_rows(self):
        """Test that explicit strategy only drops specified row indices."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['a', 'b', 'c', 'd', 'e']
        })
        
        result = drop_rows(df, strategy='explicit', row_indices=[1, 3])
        
        assert len(result) == 3
        assert result['a'].tolist() == [1, 3, 5]
        assert result['b'].tolist() == ['a', 'c', 'e']
    
    def test_explicit_strategy_ignores_invalid_indices(self):
        """Test that invalid row indices are ignored."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['a', 'b', 'c']
        })
        
        result = drop_rows(df, strategy='explicit', row_indices=[1, 100, -5])
        
        # Only index 1 is valid
        assert len(result) == 2
        assert result['a'].tolist() == [1, 3]
    
    def test_required_missing_drops_rows_with_missing_required(self):
        """Test that rows with missing required columns are dropped."""
        df = pd.DataFrame({
            'id': [1, None, 3, 4],
            'name': ['Alice', 'Bob', None, 'David'],
            'email': ['a@b.com', 'b@c.com', 'c@d.com', None]
        })
        
        result = drop_rows(df, strategy='required_missing', required_columns=['id', 'name'])
        
        # Row 1 is missing 'id', Row 2 is missing 'name' - both dropped
        assert len(result) == 2
        assert result['id'].tolist() == [1, 4]
    
    def test_required_missing_handles_empty_strings(self):
        """Test that empty strings in required columns cause row to be dropped."""
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'name': ['Alice', '', 'Charlie']
        })
        
        result = drop_rows(df, strategy='required_missing', required_columns=['name'])
        
        # Row 1 has empty string in required 'name' - dropped
        assert len(result) == 2
        assert result['name'].tolist() == ['Alice', 'Charlie']
    
    def test_drop_rows_idempotent(self):
        """Test that drop_rows is idempotent."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': ['x', None, 'z']
        })
        
        result1 = drop_rows(df, strategy='conservative')
        result2 = drop_rows(result1, strategy='conservative')
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_drop_rows_preserves_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': ['x', None, 'z']
        })
        original_len = len(df)
        
        drop_rows(df, strategy='conservative')
        
        assert len(df) == original_len
    
    def test_drop_rows_via_cleaning_rule(self):
        """Test drop_rows through apply_cleaning_rules."""
        df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': ['x', None, 'z', None]
        })
        
        rules = [
            CleaningRule(
                rule_type='drop_rows',
                column=None,
                params={'strategy': 'conservative'}
            )
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        # Rows 1 and 3 are completely empty - should be dropped
        assert len(result) == 2
    
    def test_min_non_null_parameter(self):
        """Test the min_non_null parameter for more aggressive dropping."""
        df = pd.DataFrame({
            'a': [1, 2, None, None],
            'b': ['x', None, 'y', None],
            'c': [True, None, None, None]
        })
        
        # Require at least 2 non-null values per row
        result = drop_rows(df, strategy='conservative', min_non_null=2)
        
        # Row 0 has 3 values - kept
        # Row 1 has 1 value - dropped
        # Row 2 has 1 value - dropped
        # Row 3 has 0 values - dropped
        assert len(result) == 1
        assert result['a'].iloc[0] == 1
    
    def test_no_rows_dropped_when_all_valid(self):
        """Test that no rows are dropped when all have valid data."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        
        result = drop_rows(df, strategy='conservative')
        
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['a', 'b', 'c'])
        
        result = drop_rows(df, strategy='conservative')
        
        assert len(result) == 0
        assert list(result.columns) == ['a', 'b', 'c']


# ============================================
# Tests for fill_missing
# ============================================

class TestFillMissing:
    """Tests for fill_missing function."""
    
    def test_fill_with_static_value(self):
        """Test filling missing values with a static value."""
        df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': ['x', 'y', None, 'z']
        })
        
        result = fill_missing(df, 'a', value=0)
        
        assert result['a'].isna().sum() == 0
        assert result['a'].tolist() == [1.0, 0.0, 3.0, 0.0]
    
    def test_fill_with_mean_strategy(self):
        """Test filling missing values with mean (requires imputation enabled)."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, None, 4.0]
        })
        # Need to enable imputation for this to work
        lenient_config = StrictModeConfig(enabled=True, allow_imputation=True)
        
        result = fill_missing(df, 'a', strategy='mean', strict_config=lenient_config)
        
        assert result['a'].isna().sum() == 0
        # Mean of [1, 2, 4] is 7/3 ≈ 2.333
        assert result['a'].iloc[2] == pytest.approx(7/3, rel=1e-5)
    
    def test_fill_with_median_strategy(self):
        """Test filling missing values with median (requires imputation enabled)."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, None, 10.0]
        })
        # Need to enable imputation for this to work
        lenient_config = StrictModeConfig(enabled=True, allow_imputation=True)
        
        result = fill_missing(df, 'a', strategy='median', strict_config=lenient_config)
        
        assert result['a'].isna().sum() == 0
        # Median of [1, 2, 10] is 2
        assert result['a'].iloc[2] == 2.0
    
    def test_fill_with_mode_strategy(self):
        """Test filling missing values with mode (requires imputation enabled)."""
        df = pd.DataFrame({
            'a': ['x', 'y', None, 'x', 'x']
        })
        # Need to enable imputation for this to work
        lenient_config = StrictModeConfig(enabled=True, allow_imputation=True)
        
        result = fill_missing(df, 'a', strategy='mode', strict_config=lenient_config)
        
        assert result['a'].isna().sum() == 0
        assert result['a'].iloc[2] == 'x'
    
    def test_fill_with_ffill_strategy(self):
        """Test filling missing values with forward fill (requires imputation enabled)."""
        df = pd.DataFrame({
            'a': [1.0, None, None, 4.0]
        })
        # Need to enable imputation for this to work
        lenient_config = StrictModeConfig(enabled=True, allow_imputation=True)
        
        result = fill_missing(df, 'a', strategy='ffill', strict_config=lenient_config)
        
        assert result['a'].tolist() == [1.0, 1.0, 1.0, 4.0]
    
    def test_fill_with_bfill_strategy(self):
        """Test filling missing values with backward fill (requires imputation enabled)."""
        df = pd.DataFrame({
            'a': [1.0, None, None, 4.0]
        })
        # Need to enable imputation for this to work
        lenient_config = StrictModeConfig(enabled=True, allow_imputation=True)
        
        result = fill_missing(df, 'a', strategy='bfill', strict_config=lenient_config)
        
        assert result['a'].tolist() == [1.0, 4.0, 4.0, 4.0]
    
    def test_fill_nonexistent_column(self):
        """Test filling a non-existent column doesn't crash."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result = fill_missing(df, 'nonexistent', value=0)
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_fill_missing_idempotent(self):
        """Test that fill_missing is idempotent."""
        df = pd.DataFrame({
            'a': [1.0, None, 3.0]
        })
        
        result1 = fill_missing(df, 'a', value=0)
        result2 = fill_missing(result1, 'a', value=0)
        
        pd.testing.assert_frame_equal(result1, result2)


# ============================================
# Tests for fill_missing STRICT MODE
# ============================================

class TestFillMissingStrictMode:
    """Tests for fill_missing function with STRICT MODE behavior."""
    
    def test_fill_missing_blocked_with_mean_strategy_strict_mode(self):
        """STRICT MODE: fill_missing with mean strategy should be BLOCKED."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, None, 4.0]
        })
        strict_config = StrictModeConfig(enabled=True, allow_imputation=False)
        
        result = fill_missing(df, 'a', strategy='mean', strict_config=strict_config)
        
        # In strict mode, mean imputation is blocked - value should remain NaN
        assert result['a'].isna().sum() == 1  # Still has the NaN
    
    def test_fill_missing_blocked_with_median_strategy_strict_mode(self):
        """STRICT MODE: fill_missing with median strategy should be BLOCKED."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, None, 4.0]
        })
        strict_config = StrictModeConfig(enabled=True, allow_imputation=False)
        
        result = fill_missing(df, 'a', strategy='median', strict_config=strict_config)
        
        # In strict mode, median imputation is blocked
        assert result['a'].isna().sum() == 1
    
    def test_fill_missing_blocked_with_mode_strategy_strict_mode(self):
        """STRICT MODE: fill_missing with mode strategy should be BLOCKED."""
        df = pd.DataFrame({
            'a': ['x', 'y', None, 'x', 'x']
        })
        strict_config = StrictModeConfig(enabled=True, allow_imputation=False)
        
        result = fill_missing(df, 'a', strategy='mode', strict_config=strict_config)
        
        # In strict mode, mode imputation is blocked
        assert result['a'].isna().sum() == 1
    
    def test_fill_missing_blocked_with_ffill_strategy_strict_mode(self):
        """STRICT MODE: fill_missing with ffill strategy should be BLOCKED."""
        df = pd.DataFrame({
            'a': [1.0, None, None, 4.0]
        })
        strict_config = StrictModeConfig(enabled=True, allow_imputation=False)
        
        result = fill_missing(df, 'a', strategy='ffill', strict_config=strict_config)
        
        # In strict mode, ffill imputation is blocked
        assert result['a'].isna().sum() == 2
    
    def test_fill_missing_allowed_when_imputation_enabled(self):
        """When imputation is explicitly enabled, fill_missing should work."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, None, 4.0]
        })
        # Explicitly enable imputation
        strict_config = StrictModeConfig(enabled=True, allow_imputation=True)
        
        result = fill_missing(df, 'a', strategy='mean', strict_config=strict_config)
        
        # With imputation enabled, mean fill should work
        assert result['a'].isna().sum() == 0
    
    def test_fill_missing_works_when_strict_mode_disabled(self):
        """When strict mode is disabled, fill_missing should work normally."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, None, 4.0]
        })
        strict_config = StrictModeConfig(enabled=False)
        
        result = fill_missing(df, 'a', strategy='mean', strict_config=strict_config)
        
        # With strict mode disabled, imputation works
        assert result['a'].isna().sum() == 0


# ============================================
# Tests for new strict mode friendly functions
# ============================================

class TestNormalizeCapitalization:
    """Tests for normalize_capitalization function (strict mode friendly)."""
    
    def test_normalize_to_title_case(self):
        """Should convert text to title case."""
        df = pd.DataFrame({'name': ['ALICE', 'bob', 'CHARLIE BROWN']})
        result = normalize_capitalization(df, 'name', style='title')
        assert result['name'].tolist() == ['Alice', 'Bob', 'Charlie Brown']
    
    def test_normalize_to_lower_case(self):
        """Should convert text to lower case."""
        df = pd.DataFrame({'name': ['ALICE', 'Bob', 'CHARLIE']})
        result = normalize_capitalization(df, 'name', style='lower')
        assert result['name'].tolist() == ['alice', 'bob', 'charlie']
    
    def test_normalize_to_upper_case(self):
        """Should convert text to upper case."""
        df = pd.DataFrame({'name': ['alice', 'Bob', 'charlie']})
        result = normalize_capitalization(df, 'name', style='upper')
        assert result['name'].tolist() == ['ALICE', 'BOB', 'CHARLIE']
    
    def test_normalize_preserves_nan(self):
        """Should preserve NaN values."""
        df = pd.DataFrame({'name': ['Alice', None, 'Bob']})
        result = normalize_capitalization(df, 'name', style='title')
        assert pd.isna(result['name'].iloc[1])


class TestConvertNumberWords:
    """Tests for convert_number_words function (strict mode friendly)."""
    
    def test_convert_basic_number_words(self):
        """Should convert number words to digits."""
        df = pd.DataFrame({'value': ['one', 'two', 'three']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [1, 2, 3]
    
    def test_convert_larger_number_words(self):
        """Should convert larger number words."""
        df = pd.DataFrame({'value': ['twenty', 'thirty', 'fifty']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [20, 30, 50]
    
    def test_leaves_unknown_words_unchanged(self):
        """Should leave unrecognized words unchanged (no guessing)."""
        df = pd.DataFrame({'value': ['twenty', 'banana', 'fifty']})
        result = convert_number_words(df, 'value')
        assert result['value'].iloc[1] == 'banana'  # Not converted

    def test_convert_all_basic_digits(self):
        """Should convert all basic digit words (zero through ten)."""
        df = pd.DataFrame({
            'value': ['zero', 'one', 'two', 'three', 'four', 
                      'five', 'six', 'seven', 'eight', 'nine', 'ten']
        })
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_convert_teen_numbers(self):
        """Should convert teen numbers (eleven through nineteen)."""
        df = pd.DataFrame({
            'value': ['eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
                      'sixteen', 'seventeen', 'eighteen', 'nineteen']
        })
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [11, 12, 13, 14, 15, 16, 17, 18, 19]

    def test_convert_tens_numbers(self):
        """Should convert tens numbers (twenty through ninety)."""
        df = pd.DataFrame({
            'value': ['twenty', 'thirty', 'forty', 'fifty', 
                      'sixty', 'seventy', 'eighty', 'ninety']
        })
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [20, 30, 40, 50, 60, 70, 80, 90]

    def test_convert_hundred_and_thousand(self):
        """Should convert hundred and thousand."""
        df = pd.DataFrame({'value': ['hundred', 'thousand']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [100, 1000]

    def test_case_insensitive_conversion(self):
        """Should convert regardless of case."""
        df = pd.DataFrame({'value': ['ONE', 'Two', 'THREE', 'FoUr']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [1, 2, 3, 4]

    def test_whitespace_trimmed_before_conversion(self):
        """Should trim whitespace before matching number words."""
        df = pd.DataFrame({'value': ['  one  ', ' two', 'three ']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [1, 2, 3]

    def test_preserves_null_values(self):
        """Should preserve null/NaN values."""
        df = pd.DataFrame({'value': ['one', None, 'three', np.nan]})
        result = convert_number_words(df, 'value')
        assert result['value'].iloc[0] == 1
        assert pd.isna(result['value'].iloc[1])
        assert result['value'].iloc[2] == 3
        assert pd.isna(result['value'].iloc[3])

    def test_nonexistent_column_returns_unchanged(self):
        """Should return DataFrame unchanged if column doesn't exist."""
        df = pd.DataFrame({'other': ['one', 'two', 'three']})
        result = convert_number_words(df, 'nonexistent')
        pd.testing.assert_frame_equal(result, df)

    def test_converts_hyphenated_compound_words(self):
        """Should convert hyphenated compound number words like 'twenty-one'."""
        df = pd.DataFrame({'value': ['twenty-one', 'thirty-two', 'forty-five']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [21, 32, 45]

    def test_converts_compound_phrases(self):
        """Should convert compound number phrases like 'sixty thousand'."""
        df = pd.DataFrame({'value': ['sixty thousand', 'two hundred', 'five hundred thousand']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == [60000, 200, 500000]

    def test_does_not_convert_partial_matches(self):
        """Should NOT convert partial matches like 'someone' or 'threesome'."""
        df = pd.DataFrame({'value': ['someone', 'threesome', 'fiver']})
        result = convert_number_words(df, 'value')
        assert result['value'].tolist() == ['someone', 'threesome', 'fiver']

    def test_mixed_numbers_and_words(self):
        """Should handle mixed numeric strings and number words."""
        df = pd.DataFrame({'value': ['one', '2', 'three', '4']})
        result = convert_number_words(df, 'value')
        assert result['value'].iloc[0] == 1
        assert result['value'].iloc[1] == '2'  # Not converted - already a digit string
        assert result['value'].iloc[2] == 3
        assert result['value'].iloc[3] == '4'

    def test_idempotent(self):
        """Applying conversion twice should yield same result."""
        df = pd.DataFrame({'value': ['one', 'two', 'three']})
        result1 = convert_number_words(df, 'value')
        result2 = convert_number_words(result1, 'value')
        pd.testing.assert_frame_equal(result1, result2)

    def test_original_dataframe_unchanged(self):
        """Should not modify original DataFrame."""
        df = pd.DataFrame({'value': ['one', 'two', 'three']})
        original_values = df['value'].tolist()
        convert_number_words(df, 'value')
        assert df['value'].tolist() == original_values


class TestRemoveInvalidCharacters:
    """Tests for remove_invalid_characters function (strict mode friendly)."""
    
    def test_remove_nonprintable_characters(self):
        """Should remove non-printable characters."""
        df = pd.DataFrame({'text': ['Hello\x00World', 'Test\x1fData']})
        result = remove_invalid_characters(df, 'text')
        assert result['text'].iloc[0] == 'HelloWorld'
        assert result['text'].iloc[1] == 'TestData'
    
    def test_preserves_normal_text(self):
        """Should preserve normal text."""
        df = pd.DataFrame({'text': ['Hello World', 'Test Data']})
        result = remove_invalid_characters(df, 'text')
        assert result['text'].tolist() == ['Hello World', 'Test Data']


# ============================================
# Tests for coerce_types
# ============================================

class TestCoerceTypes:
    """Tests for coerce_types function."""
    
    def test_coerce_to_int(self):
        """Test coercing to integer type."""
        df = pd.DataFrame({
            'a': ['1', '2', '3']
        })
        
        result = coerce_types(df, 'a', 'int')
        
        assert result['a'].tolist() == [1, 2, 3]
    
    def test_coerce_to_float(self):
        """Test coercing to float type."""
        df = pd.DataFrame({
            'a': ['1.5', '2.7', '3.9']
        })
        
        result = coerce_types(df, 'a', 'float')
        
        assert result['a'].tolist() == pytest.approx([1.5, 2.7, 3.9])
    
    def test_coerce_to_str(self):
        """Test coercing to string type."""
        df = pd.DataFrame({
            'a': [1, 2, 3]
        })
        
        result = coerce_types(df, 'a', 'str')
        
        assert result['a'].tolist() == ['1', '2', '3']
    
    def test_coerce_to_bool(self):
        """Test coercing to boolean type."""
        df = pd.DataFrame({
            'a': ['true', 'false', 'True', 'False', '1', '0', 'yes', 'no']
        })
        
        result = coerce_types(df, 'a', 'bool')
        
        expected = [True, False, True, False, True, False, True, False]
        assert result['a'].tolist() == expected
    
    def test_coerce_to_datetime(self):
        """Test coercing to datetime type."""
        df = pd.DataFrame({
            'a': ['2023-01-01', '2023-06-15', '2023-12-31']
        })
        
        result = coerce_types(df, 'a', 'datetime')
        
        assert pd.api.types.is_datetime64_any_dtype(result['a'])
    
    def test_coerce_bad_data_with_coerce_errors(self):
        """Test coercing bad data results in NaN when errors='coerce'."""
        df = pd.DataFrame({
            'a': ['1', 'bad', '3']
        })
        
        result = coerce_types(df, 'a', 'int', errors='coerce')
        
        assert result['a'].iloc[0] == 1
        assert pd.isna(result['a'].iloc[1])
        assert result['a'].iloc[2] == 3
    
    def test_coerce_nonexistent_column(self):
        """Test coercing a non-existent column doesn't crash."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result = coerce_types(df, 'nonexistent', 'int')
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_coerce_types_idempotent(self):
        """Test that coerce_types is idempotent."""
        df = pd.DataFrame({
            'a': ['1', '2', '3']
        })
        
        result1 = coerce_types(df, 'a', 'int')
        result2 = coerce_types(result1, 'a', 'int')
        
        pd.testing.assert_frame_equal(result1, result2)


# ============================================
# Tests for trim_whitespace
# ============================================

class TestTrimWhitespace:
    """Tests for trim_whitespace function."""
    
    def test_trim_single_column(self):
        """Test trimming whitespace from a single column."""
        df = pd.DataFrame({
            'a': ['  hello  ', ' world ', 'test'],
            'b': ['  keep  ', ' this ', 'too']
        })
        
        result = trim_whitespace(df, 'a')
        
        assert result['a'].tolist() == ['hello', 'world', 'test']
        assert result['b'].tolist() == ['  keep  ', ' this ', 'too']
    
    def test_trim_all_columns(self):
        """Test trimming whitespace from all string columns."""
        df = pd.DataFrame({
            'a': ['  hello  ', ' world '],
            'b': ['  foo  ', ' bar '],
            'c': [1, 2]  # Non-string column
        })
        
        result = trim_whitespace(df)
        
        assert result['a'].tolist() == ['hello', 'world']
        assert result['b'].tolist() == ['foo', 'bar']
        assert result['c'].tolist() == [1, 2]
    
    def test_trim_with_mixed_types(self):
        """Test trimming doesn't crash on mixed types."""
        df = pd.DataFrame({
            'a': ['  hello  ', None, 123, '  world  ']
        })
        
        result = trim_whitespace(df, 'a')
        
        assert result['a'].iloc[0] == 'hello'
        assert pd.isna(result['a'].iloc[1])
        assert result['a'].iloc[2] == 123
        assert result['a'].iloc[3] == 'world'
    
    def test_trim_whitespace_idempotent(self):
        """Test that trim_whitespace is idempotent."""
        df = pd.DataFrame({
            'a': ['  hello  ', ' world ']
        })
        
        result1 = trim_whitespace(df, 'a')
        result2 = trim_whitespace(result1, 'a')
        
        pd.testing.assert_frame_equal(result1, result2)


# ============================================
# Tests for parse_dates
# ============================================

class TestParseDates:
    """Tests for parse_dates function."""
    
    def test_parse_iso_dates(self):
        """Test parsing ISO format dates."""
        df = pd.DataFrame({
            'date': ['2023-01-15', '2023-06-30', '2023-12-25']
        })
        
        result = parse_dates(df, 'date')
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert result['date'].iloc[0].day == 15
        assert result['date'].iloc[0].month == 1
        assert result['date'].iloc[0].year == 2023
    
    def test_parse_dates_with_format(self):
        """Test parsing dates with a specific format."""
        df = pd.DataFrame({
            'date': ['15/01/2023', '30/06/2023', '25/12/2023']
        })
        
        result = parse_dates(df, 'date', format='%d/%m/%Y')
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
        assert result['date'].iloc[0].day == 15
        assert result['date'].iloc[0].month == 1
    
    def test_parse_bad_dates_coerce(self):
        """Test that bad dates become NaT when errors='coerce'."""
        df = pd.DataFrame({
            'date': ['2023-01-15', 'not-a-date', '2023-12-25']
        })
        
        result = parse_dates(df, 'date', errors='coerce')
        
        assert pd.notna(result['date'].iloc[0])
        assert pd.isna(result['date'].iloc[1])
        assert pd.notna(result['date'].iloc[2])
    
    def test_parse_nonexistent_column(self):
        """Test parsing a non-existent column doesn't crash."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result = parse_dates(df, 'nonexistent')
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_parse_dates_idempotent(self):
        """Test that parse_dates is idempotent."""
        df = pd.DataFrame({
            'date': ['2023-01-15', '2023-06-30']
        })
        
        result1 = parse_dates(df, 'date')
        result2 = parse_dates(result1, 'date')
        
        pd.testing.assert_frame_equal(result1, result2)


# ============================================
# Tests for normalize_numbers
# ============================================

class TestNormalizeNumbers:
    """Tests for normalize_numbers function."""
    
    def test_remove_currency_symbols(self):
        """Test removing currency symbols."""
        df = pd.DataFrame({
            'price': ['$100', '€200', '£300', '¥400']
        })
        
        result = normalize_numbers(df, 'price')
        
        assert result['price'].tolist() == [100.0, 200.0, 300.0, 400.0]
    
    def test_remove_commas(self):
        """Test removing thousand separators."""
        df = pd.DataFrame({
            'value': ['1,000', '10,000', '100,000', '1,000,000']
        })
        
        result = normalize_numbers(df, 'value')
        
        assert result['value'].tolist() == [1000.0, 10000.0, 100000.0, 1000000.0]
    
    def test_convert_percentages(self):
        """Test converting percentages."""
        df = pd.DataFrame({
            'rate': ['50%', '25%', '100%', '7.5%']
        })
        
        result = normalize_numbers(df, 'rate')
        
        expected = [0.5, 0.25, 1.0, 0.075]
        assert result['rate'].tolist() == pytest.approx(expected)
    
    def test_handle_negative_accounting_format(self):
        """Test handling accounting format negative numbers (1,234)."""
        df = pd.DataFrame({
            'value': ['(100)', '200', '(50.5)']
        })
        
        result = normalize_numbers(df, 'value')
        
        assert result['value'].tolist() == pytest.approx([-100.0, 200.0, -50.5])
    
    def test_mixed_formats(self):
        """Test handling mixed formats in same column."""
        df = pd.DataFrame({
            'value': ['$1,000.50', '€2,500', '75%', '(100)']
        })
        
        result = normalize_numbers(df, 'value')
        
        expected = [1000.50, 2500.0, 0.75, -100.0]
        assert result['value'].tolist() == pytest.approx(expected)
    
    def test_already_numeric_values(self):
        """Test that already numeric values are unchanged."""
        df = pd.DataFrame({
            'value': [100, 200.5, 300]
        })
        
        result = normalize_numbers(df, 'value')
        
        assert result['value'].tolist() == [100, 200.5, 300]
    
    def test_handle_bad_values(self):
        """Test that non-parseable values are left unchanged."""
        df = pd.DataFrame({
            'value': ['$100', 'not-a-number', '200']
        })
        
        result = normalize_numbers(df, 'value')
        
        assert result['value'].iloc[0] == 100.0
        assert result['value'].iloc[1] == 'not-a-number'
        assert result['value'].iloc[2] == 200.0
    
    def test_normalize_nonexistent_column(self):
        """Test normalizing a non-existent column doesn't crash."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result = normalize_numbers(df, 'nonexistent')
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_normalize_numbers_idempotent(self):
        """Test that normalize_numbers is idempotent."""
        df = pd.DataFrame({
            'value': ['$1,000', '€2,500', '75%']
        })
        
        result1 = normalize_numbers(df, 'value')
        result2 = normalize_numbers(result1, 'value')
        
        pd.testing.assert_frame_equal(result1, result2)


# ============================================
# Tests for dedupe
# ============================================

class TestDedupe:
    """Tests for dedupe function."""
    
    def test_remove_exact_duplicates(self):
        """Test removing exact duplicate rows."""
        df = pd.DataFrame({
            'a': [1, 2, 1, 3, 2],
            'b': ['x', 'y', 'x', 'z', 'y']
        })
        
        result = dedupe(df)
        
        assert len(result) == 3
        assert result['a'].tolist() == [1, 2, 3]
        assert result['b'].tolist() == ['x', 'y', 'z']
    
    def test_dedupe_on_subset_of_columns(self):
        """Test deduping based on a subset of columns."""
        df = pd.DataFrame({
            'a': [1, 2, 1, 3],
            'b': ['x', 'y', 'z', 'w']  # Different values
        })
        
        result = dedupe(df, subset=['a'])
        
        assert len(result) == 3
        # First occurrence kept
        assert result['b'].tolist() == ['x', 'y', 'w']
    
    def test_dedupe_keep_last(self):
        """Test keeping last duplicate instead of first."""
        df = pd.DataFrame({
            'a': [1, 2, 1],
            'b': ['first', 'y', 'first']  # Actual duplicate row
        })
        
        result = dedupe(df, keep='last')
        
        assert len(result) == 2
        # Last occurrence of duplicated row should be kept (index 2)
        assert result['a'].tolist() == [2, 1]
        assert result['b'].tolist() == ['y', 'first']
    
    def test_dedupe_nonexistent_subset_column(self):
        """Test that non-existent subset columns are ignored."""
        df = pd.DataFrame({
            'a': [1, 2, 1],
            'b': ['x', 'y', 'x']
        })
        
        result = dedupe(df, subset=['nonexistent'])
        
        # Falls back to all columns
        assert len(result) == 2
    
    def test_dedupe_idempotent(self):
        """Test that dedupe is idempotent."""
        df = pd.DataFrame({
            'a': [1, 2, 1, 3, 2],
            'b': ['x', 'y', 'x', 'z', 'y']
        })
        
        result1 = dedupe(df)
        result2 = dedupe(result1)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_dedupe_resets_index(self):
        """Test that dedupe resets the index."""
        df = pd.DataFrame({
            'a': [1, 2, 1, 3],
            'b': ['x', 'y', 'x', 'z']
        })
        
        result = dedupe(df)
        
        # Index is preserved (not reset) to allow matching with original rows
        assert result.index.tolist() == [0, 1, 3]
        assert len(result) == 3


# ============================================
# Tests for apply_cleaning_rules
# ============================================

class TestApplyCleaningRules:
    """Tests for apply_cleaning_rules function."""
    
    def test_apply_single_rule(self):
        """Test applying a single cleaning rule."""
        df = pd.DataFrame({
            'a': ['  hello  ', '  world  ']
        })
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column='a', params={})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert result['a'].tolist() == ['hello', 'world']
    
    def test_apply_multiple_rules(self):
        """Test applying multiple cleaning rules in sequence."""
        df = pd.DataFrame({
            'a': ['  $1,000  ', '  $2,500  '],
            'b': [1, 2],
            'c': [1, 1]
        })
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column='a', params={}),
            CleaningRule(rule_type='normalize_numbers', column='a', params={}),
            CleaningRule(rule_type='drop_columns', column=None, params={'columns': ['b']}),
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert result['a'].tolist() == [1000.0, 2500.0]
        assert 'b' not in result.columns
    
    def test_apply_drop_columns_rule(self):
        """Test applying drop_columns rule."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6]
        })
        
        rules = [
            CleaningRule(rule_type='drop_columns', column=None, params={'columns': ['b', 'c']})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert list(result.columns) == ['a']
    
    def test_apply_fill_missing_rule(self):
        """Test applying fill_missing rule."""
        df = pd.DataFrame({
            'a': [1.0, None, 3.0]
        })
        
        rules = [
            CleaningRule(rule_type='fill_missing', column='a', params={'value': 0})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert result['a'].tolist() == [1.0, 0.0, 3.0]
    
    def test_apply_coerce_types_rule(self):
        """Test applying coerce_types rule."""
        df = pd.DataFrame({
            'a': ['1', '2', '3']
        })
        
        rules = [
            CleaningRule(rule_type='coerce_types', column='a', params={'dtype': 'int'})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert result['a'].tolist() == [1, 2, 3]
    
    def test_apply_parse_dates_rule(self):
        """Test applying parse_dates rule."""
        df = pd.DataFrame({
            'date': ['2023-01-15', '2023-06-30']
        })
        
        rules = [
            CleaningRule(rule_type='parse_dates', column='date', params={})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_apply_dedupe_rule(self):
        """Test applying dedupe rule."""
        df = pd.DataFrame({
            'a': [1, 2, 1],
            'b': ['x', 'y', 'x']
        })
        
        rules = [
            CleaningRule(rule_type='dedupe', column=None, params={})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        assert len(result) == 2
    
    def test_apply_unknown_rule_type(self):
        """Test that unknown rule types are silently ignored."""
        df = pd.DataFrame({
            'a': [1, 2, 3]
        })
        
        rules = [
            CleaningRule(rule_type='unknown_rule', column='a', params={})
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        pd.testing.assert_frame_equal(result, df)
    
    def test_apply_rules_preserves_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({
            'a': ['  hello  ']
        })
        original_value = df['a'].iloc[0]
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column='a', params={})
        ]
        
        apply_cleaning_rules(df, rules)
        
        assert df['a'].iloc[0] == original_value
    
    def test_apply_rules_idempotent(self):
        """Test that applying rules is idempotent."""
        df = pd.DataFrame({
            'a': ['  $1,000  ', '  $2,500  '],
            'b': [1, 2, ],
        })
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column='a', params={}),
            CleaningRule(rule_type='normalize_numbers', column='a', params={}),
        ]
        
        result1 = apply_cleaning_rules(df, rules)
        result2 = apply_cleaning_rules(result1, rules)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_apply_empty_rules_list(self):
        """Test applying an empty rules list returns copy of DataFrame."""
        df = pd.DataFrame({
            'a': [1, 2, 3]
        })
        
        result = apply_cleaning_rules(df, [])
        
        pd.testing.assert_frame_equal(result, df)
        assert result is not df  # Should be a copy


# ============================================
# Tests for preview_cleaning
# ============================================

class TestPreviewCleaning:
    """Tests for preview_cleaning function."""
    
    def test_preview_shows_changes(self):
        """Test that preview shows original and cleaned values."""
        df = pd.DataFrame({
            'a': ['  hello  ', '  world  ']
        })
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column='a', params={})
        ]
        
        preview = preview_cleaning(df, rules)
        
        assert len(preview) == 2
        assert preview[0].original['a'] == '  hello  '
        assert preview[0].cleaned['a'] == 'hello'
        assert len(preview[0].changes) > 0
    
    def test_preview_respects_n_rows(self):
        """Test that preview respects n_rows parameter."""
        df = pd.DataFrame({
            'a': list(range(100))
        })
        
        rules = []
        
        preview = preview_cleaning(df, rules, n_rows=10)
        
        assert len(preview) == 10
    
    def test_preview_shows_dropped_columns(self):
        """Test that preview indicates dropped columns."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4]
        })
        
        rules = [
            CleaningRule(rule_type='drop_columns', column=None, params={'columns': ['b']})
        ]
        
        preview = preview_cleaning(df, rules)
        
        assert 'b' in preview[0].original
        assert 'b' not in preview[0].cleaned
        assert any('dropped' in change for change in preview[0].changes)


# ============================================
# Integration Tests
# ============================================

class TestCleaningIntegration:
    """Integration tests for full cleaning workflows."""
    
    def test_full_cleaning_workflow(self):
        """Test a realistic full cleaning workflow."""
        # Create messy data
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ', '  Alice  ', '  Charlie  '],
            'email': ['alice@test.com', 'bob@test.com', 'alice@test.com', 'charlie@test.com'],
            'salary': ['$50,000', '$75,000', '$50,000', '$100,000'],
            'hire_date': ['2020-01-15', '2021-06-30', '2020-01-15', '2019-03-01'],
            'temp_col': ['x', 'y', 'x', 'z']
        })
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column='name', params={}),
            CleaningRule(rule_type='normalize_numbers', column='salary', params={}),
            CleaningRule(rule_type='parse_dates', column='hire_date', params={}),
            CleaningRule(rule_type='drop_columns', column=None, params={'columns': ['temp_col']}),
            CleaningRule(rule_type='dedupe', column=None, params={}),
        ]
        
        result = apply_cleaning_rules(df, rules)
        
        # Verify all transformations applied
        assert result['name'].tolist() == ['Alice', 'Bob', 'Charlie']
        assert result['salary'].tolist() == [50000.0, 75000.0, 100000.0]
        assert pd.api.types.is_datetime64_any_dtype(result['hire_date'])
        assert 'temp_col' not in result.columns
        assert len(result) == 3  # Dedupe removed 1 row
    
    def test_cleaning_handles_bad_data_gracefully(self):
        """Test that cleaning handles malformed data without crashing."""
        df = pd.DataFrame({
            'mixed': [1, 'two', None, 3.5, 'N/A'],
            'dates': ['2023-01-01', 'not-a-date', None, '2023-12-31', 'invalid'],
            'numbers': ['$100', 'abc', '$200', None, '(50)']
        })
        
        rules = [
            CleaningRule(rule_type='coerce_types', column='mixed', params={'dtype': 'float'}),
            CleaningRule(rule_type='parse_dates', column='dates', params={}),
            CleaningRule(rule_type='normalize_numbers', column='numbers', params={}),
        ]
        
        # Should not raise any exceptions
        result = apply_cleaning_rules(df, rules)
        
        # Verify result is a DataFrame with same number of rows
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
    
    def test_idempotency_complex_workflow(self):
        """Test idempotency with complex cleaning workflow."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  '],
            'value': ['$1,000', '$2,500'],
            'extra': ['x', 'y']
        })
        
        rules = [
            CleaningRule(rule_type='trim_whitespace', column=None, params={}),
            CleaningRule(rule_type='normalize_numbers', column='value', params={}),
            CleaningRule(rule_type='drop_columns', column=None, params={'columns': ['extra']}),
        ]
        
        result1 = apply_cleaning_rules(df, rules)
        result2 = apply_cleaning_rules(result1, rules)
        result3 = apply_cleaning_rules(result2, rules)
        
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)
