"""
Tests for the autonomous data cleaning engine.

These tests verify STRICT MODE behavior:
1. No fabrication or guessing of new data
2. Only meaning-preserving transformations
3. No imputation unless explicitly enabled
4. Invalid/blank cells left as-is (not replaced with markers)
"""

import pytest
import pandas as pd
import numpy as np

from app.services.autonomous import (
    InferredType,
    infer_column_type,
    infer_all_column_types,
    autonomous_clean,
    autonomous_scan_and_suggest,
    clean_name_column,
    clean_email_column,
    clean_numeric_column,
    clean_date_column,
    clean_categorical_column,
    generate_rules_for_column,
    ColumnInference,
)
from app.models.schemas import StrictModeConfig


# ============================================
# Type Inference Tests
# ============================================

class TestTypeInference:
    """Tests for column type inference."""
    
    def test_infer_name_column_by_name(self):
        """Should infer NAME type from column name."""
        df = pd.DataFrame({"first_name": ["Alice", "Bob", "Charlie"]})
        result = infer_column_type(df, "first_name")
        assert result.inferred_type == InferredType.NAME
        assert result.confidence >= 0.8
    
    def test_infer_email_column(self):
        """Should infer EMAIL type from values."""
        df = pd.DataFrame({"contact": ["alice@example.com", "bob@test.org", "charlie@demo.net"]})
        result = infer_column_type(df, "contact")
        assert result.inferred_type == InferredType.EMAIL
        assert result.confidence >= 0.9
    
    def test_infer_date_column_by_name(self):
        """Should infer DATE type from column name."""
        df = pd.DataFrame({"created_date": ["2023-01-01", "2023-02-15", "2023-03-20"]})
        result = infer_column_type(df, "created_date")
        assert result.inferred_type == InferredType.DATE
    
    def test_infer_numeric_column(self):
        """Should infer NUMERIC type from numeric dtype."""
        df = pd.DataFrame({"age": [25, 30, 35, 40]})
        result = infer_column_type(df, "age")
        assert result.inferred_type == InferredType.NUMERIC
    
    def test_infer_currency_column(self):
        """Should infer CURRENCY type from currency-formatted values."""
        df = pd.DataFrame({"price": ["$100.00", "$250.50", "$75.99"]})
        result = infer_column_type(df, "price")
        assert result.inferred_type == InferredType.CURRENCY
    
    def test_infer_percentage_column(self):
        """Should infer PERCENTAGE type from percentage-formatted values."""
        df = pd.DataFrame({"rate": ["10%", "25%", "50%"]})
        result = infer_column_type(df, "rate")
        assert result.inferred_type == InferredType.PERCENTAGE
    
    def test_infer_boolean_column(self):
        """Should infer BOOLEAN type from boolean-like values."""
        df = pd.DataFrame({"is_active": ["true", "false", "true"]})
        result = infer_column_type(df, "is_active")
        assert result.inferred_type == InferredType.BOOLEAN
    
    def test_infer_categorical_column(self):
        """Should infer CATEGORICAL type for low-cardinality columns."""
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"] * 25})
        result = infer_column_type(df, "status")
        assert result.inferred_type == InferredType.CATEGORICAL
    
    def test_infer_id_column(self):
        """Should infer ID type from column name pattern."""
        df = pd.DataFrame({"user_id": [1, 2, 3, 4, 5]})
        result = infer_column_type(df, "user_id")
        assert result.inferred_type == InferredType.ID
    
    def test_empty_column_warning(self):
        """Should flag empty columns as unsafe."""
        df = pd.DataFrame({"empty": [None, None, None]})
        result = infer_column_type(df, "empty")
        assert result.inferred_type == InferredType.UNKNOWN
        assert result.is_safe == False
        assert result.warning is not None


# ============================================
# Cleaning Function Tests (Strict Mode)
# ============================================

class TestCleaningFunctions:
    """Tests for individual cleaning functions with STRICT MODE behavior."""
    
    def test_clean_name_title_case(self):
        """Should convert names to title case (meaning-preserving)."""
        series = pd.Series(["ALICE", "bob", "CHARLIE BROWN"])
        cleaned, issues = clean_name_column(series)
        assert cleaned.tolist() == ["Alice", "Bob", "Charlie Brown"]
    
    def test_clean_name_whitespace(self):
        """Should strip whitespace from names (meaning-preserving)."""
        series = pd.Series(["  Alice  ", "Bob  ", "  Charlie"])
        cleaned, issues = clean_name_column(series)
        assert cleaned.tolist() == ["Alice", "Bob", "Charlie"]
    
    def test_clean_name_missing_strict_mode(self):
        """STRICT MODE: Should NOT fabricate placeholder names for missing values."""
        series = pd.Series(["Alice", None, "", "Bob"])
        cleaned, issues = clean_name_column(series)
        # In strict mode, missing values should remain as-is (not fabricated)
        assert cleaned.iloc[0] == "Alice"
        assert pd.isna(cleaned.iloc[1]) or cleaned.iloc[1] is None  # Stays None
        assert cleaned.iloc[2] == "" or pd.isna(cleaned.iloc[2])  # Stays empty
        assert cleaned.iloc[3] == "Bob"
        # Issue should be flagged but NOT fixed
        assert len(issues) == 1
        assert issues[0].fixed == False  # Not fixed in strict mode
    
    def test_clean_email_lowercase(self):
        """Should lowercase emails (meaning-preserving - email case is not significant)."""
        series = pd.Series(["ALICE@EXAMPLE.COM", "Bob@Test.Org"])
        cleaned, issues = clean_email_column(series)
        assert cleaned.tolist() == ["alice@example.com", "bob@test.org"]
    
    def test_clean_email_invalid_strict_mode(self):
        """STRICT MODE: Should NOT replace invalid emails with fabricated marker."""
        series = pd.Series(["valid@example.com", "invalid-email", "also@bad@email"])
        cleaned, issues = clean_email_column(series)
        # In strict mode, invalid emails are normalized but NOT replaced
        assert cleaned.iloc[0] == "valid@example.com"
        # Invalid emails are lowercased/trimmed but not replaced with markers
        assert cleaned.iloc[1] == "invalid-email"  # Not "INVALID_EMAIL"
        assert cleaned.iloc[2] == "also@bad@email"  # Not "INVALID_EMAIL"
        # Issue should be flagged but NOT fixed
        assert len(issues) == 1
        assert issues[0].fixed == False  # Not fixed in strict mode
    
    def test_clean_numeric_number_words(self):
        """Should convert number words to digits (deterministic transformation)."""
        series = pd.Series(["twenty", "thirty", "fifty"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [20, 30, 50]
    
    def test_clean_numeric_no_typo_correction_strict_mode(self):
        """STRICT MODE: Should NOT apply risky typo corrections (O→0, l→1)."""
        series = pd.Series(["1O", "2O", "3O"])  # O instead of 0
        cleaned, issues = clean_numeric_column(series)
        # In strict mode, these are NOT auto-corrected as they could change meaning
        # The values should remain as-is (unparseable)
        assert cleaned.iloc[0] == "1O"  # Not corrected to 10
        assert cleaned.iloc[1] == "2O"  # Not corrected to 20
        assert cleaned.iloc[2] == "3O"  # Not corrected to 30
    
    def test_clean_numeric_currency(self):
        """Should strip currency symbols (meaning-preserving formatting removal)."""
        series = pd.Series(["$100", "€200", "£300"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [100.0, 200.0, 300.0]
    
    def test_clean_numeric_percentage(self):
        """Should convert percentages to decimals (meaning-preserving)."""
        series = pd.Series(["50%", "25%", "100%"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [0.5, 0.25, 1.0]
    
    def test_clean_date_iso_format(self):
        """Should standardize dates to ISO format (meaning-preserving)."""
        series = pd.Series(["01/15/2023", "March 20, 2023", "2023-12-25"])
        cleaned, issues = clean_date_column(series)
        assert cleaned.iloc[0] == "2023-01-15"
        assert cleaned.iloc[1] == "2023-03-20"
        assert cleaned.iloc[2] == "2023-12-25"
    
    def test_clean_date_invalid_strict_mode(self):
        """STRICT MODE: Should keep unparseable dates as-is, not replace with NaT."""
        series = pd.Series(["2023-01-01", "not-a-date", "2023-12-25"])
        cleaned, issues = clean_date_column(series)
        assert cleaned.iloc[0] == "2023-01-01"
        # In strict mode, unparseable dates are kept as-is
        assert cleaned.iloc[1] == "not-a-date"  # Not NaT/None
        assert cleaned.iloc[2] == "2023-12-25"
        # Issue should be flagged but NOT fixed
        assert len(issues) == 1
        assert issues[0].fixed == False  # Not fixed in strict mode
    
    def test_clean_categorical_normalize(self):
        """Should normalize categorical capitalization (meaning-preserving)."""
        series = pd.Series(["ACTIVE", "active", "Active", "INACTIVE"])
        cleaned, issues = clean_categorical_column(series)
        assert cleaned.iloc[0] == "Active"
        assert cleaned.iloc[1] == "Active"
        assert cleaned.iloc[2] == "Active"


# ============================================
# Strict Mode Specific Tests
# ============================================

class TestStrictMode:
    """Tests specifically for strict mode behavior."""
    
    def test_strict_mode_no_fabrication(self):
        """STRICT MODE: Should never fabricate data."""
        df = pd.DataFrame({
            "name": ["Alice", None, "Charlie"],
            "email": ["alice@test.com", None, "invalid"],
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Missing name should remain missing (not "Unknown Person X")
        assert pd.isna(cleaned_df["name"].iloc[1]) or cleaned_df["name"].iloc[1] is None
        
        # Invalid email should remain (not "INVALID_EMAIL")
        assert cleaned_df["email"].iloc[2] != "INVALID_EMAIL"
        
        # Check summary indicates strict mode
        assert result.summary.get("strict_mode") == True
    
    def test_strict_mode_summary_in_result(self):
        """Result should include strict mode status."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        cleaned_df, result = autonomous_clean(df)
        
        assert "strict_mode" in result.summary
        assert result.summary["strict_mode"] == True
        assert "imputation_allowed" in result.summary
        assert result.summary["imputation_allowed"] == False
    
    def test_strict_mode_warning_included(self):
        """Result should include strict mode warning."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        cleaned_df, result = autonomous_clean(df)
        
        # Should have a warning about strict mode
        assert any("STRICT MODE" in w for w in result.warnings)


# ============================================
# Autonomous Cleaning Tests (Strict Mode)
# ============================================

class TestAutonomousCleaning:
    """Tests for the full autonomous cleaning pipeline with STRICT MODE."""
    
    def test_autonomous_clean_basic(self):
        """Should clean a simple DataFrame autonomously with strict mode rules."""
        df = pd.DataFrame({
            "name": ["  ALICE  ", "BOB", "charlie"],
            "email": ["ALICE@TEST.COM", "bob@test.com", "invalid"],
            "age": ["25", "30", "35"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Check that cleaning happened
        assert len(result.generated_rules) > 0
        assert result.rows_processed == 3
        assert result.columns_processed == 3
        
        # Check names were title-cased (meaning-preserving)
        assert cleaned_df["name"].iloc[0].strip() == "Alice"
        
        # STRICT MODE: Invalid email should NOT be replaced with marker
        assert cleaned_df["email"].iloc[2] != "INVALID_EMAIL"
    
    def test_autonomous_clean_with_duplicates(self):
        """Should remove duplicate rows (meaning-preserving - exact copies)."""
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "value": ["a", "b", "b", "c"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Dedupe should be applied (this is meaning-preserving)
        assert len(cleaned_df) == 3
    
    def test_autonomous_scan_preview(self):
        """Should return suggestions without applying."""
        df = pd.DataFrame({
            "name": ["ALICE", "BOB"],
            "age": [25, 30]
        })
        
        result = autonomous_scan_and_suggest(df)
        
        # Should have suggestions but not apply them
        assert result.validation_report.get("preview_mode") == True
        assert len(result.generated_rules) > 0
        assert len(result.column_inferences) == 2
    
    def test_autonomous_clean_date_column(self):
        """Should standardize date formats."""
        df = pd.DataFrame({
            "created_date": ["01/15/2023", "2023-03-20", "March 1, 2023"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Check date inference happened
        date_inferences = [i for i in result.column_inferences if i.inferred_type.value == "date"]
        assert len(date_inferences) == 1
    
    def test_autonomous_clean_currency_column(self):
        """Should handle currency values."""
        df = pd.DataFrame({
            "price": ["$100.00", "$250.50", "$75.99"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Check currency inference
        currency_inferences = [i for i in result.column_inferences if i.inferred_type.value == "currency"]
        assert len(currency_inferences) == 1
    
    def test_autonomous_clean_mixed_issues(self):
        """Should handle multiple issue types at once."""
        df = pd.DataFrame({
            "name": ["  JOHN  ", None, "JANE"],
            "email": ["john@test.com", "JANE@TEST.COM", "bad-email"],
            "amount": ["$1,000", "2000", "three thousand"],
            "status": ["ACTIVE", "active", "INACTIVE"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Should have processed all columns
        assert result.columns_processed == 4
        
        # Should have multiple rules
        assert len(result.generated_rules) > 4
        
        # Summary should have issue counts
        assert "issues_detected" in result.summary or "rules_applied" in result.summary


# ============================================
# Rule Generation Tests
# ============================================

class TestRuleGeneration:
    """Tests for cleaning rule generation."""
    
    def test_generate_name_rules(self):
        """Should generate appropriate rules for name columns."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        inference = ColumnInference(
            column="name",
            inferred_type=InferredType.NAME,
            confidence=0.9,
            indicators=["Column name matches"],
            is_safe=True
        )
        
        rules = generate_rules_for_column(inference, df)
        rule_types = [r.rule.rule_type for r in rules]
        
        assert "trim_whitespace" in rule_types
    
    def test_generate_numeric_rules(self):
        """Should generate appropriate rules for numeric columns."""
        df = pd.DataFrame({"amount": [100, 200, 300]})
        inference = ColumnInference(
            column="amount",
            inferred_type=InferredType.NUMERIC,
            confidence=0.9,
            indicators=["Numeric dtype"],
            is_safe=True
        )
        
        rules = generate_rules_for_column(inference, df)
        rule_types = [r.rule.rule_type for r in rules]
        
        assert "normalize_numbers" in rule_types
        assert "coerce_types" in rule_types
    
    def test_generate_date_rules(self):
        """Should generate appropriate rules for date columns."""
        df = pd.DataFrame({"date": ["2023-01-01"]})
        inference = ColumnInference(
            column="date",
            inferred_type=InferredType.DATE,
            confidence=0.9,
            indicators=["Date column name"],
            is_safe=True
        )
        
        rules = generate_rules_for_column(inference, df)
        rule_types = [r.rule.rule_type for r in rules]
        
        assert "parse_dates" in rule_types


# ============================================
# Missing Value Indicator Normalization Tests
# ============================================

class TestMissingIndicatorNormalization:
    """Tests for normalizing common missing-value indicators to null."""
    
    def test_normalize_na_indicators(self):
        """Should convert common NA indicators to null."""
        from app.services.cleaner import normalize_missing_indicators
        
        df = pd.DataFrame({
            'data': ['valid', 'na', 'N/A', 'n/a', 'NA', 'other']
        })
        result = normalize_missing_indicators(df, column='data')
        
        assert result['data'].iloc[0] == 'valid'
        assert pd.isna(result['data'].iloc[1])  # 'na' -> null
        assert pd.isna(result['data'].iloc[2])  # 'N/A' -> null
        assert pd.isna(result['data'].iloc[3])  # 'n/a' -> null
        assert pd.isna(result['data'].iloc[4])  # 'NA' -> null
        assert result['data'].iloc[5] == 'other'
    
    def test_normalize_null_none_indicators(self):
        """Should convert null/none indicators to null."""
        from app.services.cleaner import normalize_missing_indicators
        
        df = pd.DataFrame({
            'data': ['null', 'NULL', 'Null', 'none', 'None', 'NONE', 'valid']
        })
        result = normalize_missing_indicators(df, column='data')
        
        assert pd.isna(result['data'].iloc[0])  # 'null' -> null
        assert pd.isna(result['data'].iloc[1])  # 'NULL' -> null
        assert pd.isna(result['data'].iloc[2])  # 'Null' -> null
        assert pd.isna(result['data'].iloc[3])  # 'none' -> null
        assert pd.isna(result['data'].iloc[4])  # 'None' -> null
        assert pd.isna(result['data'].iloc[5])  # 'NONE' -> null
        assert result['data'].iloc[6] == 'valid'
    
    def test_normalize_special_char_indicators(self):
        """Should convert dash/period/empty string to null."""
        from app.services.cleaner import normalize_missing_indicators
        
        df = pd.DataFrame({
            'data': ['-', '.', '', '  ', 'valid']
        })
        result = normalize_missing_indicators(df, column='data')
        
        assert pd.isna(result['data'].iloc[0])  # '-' -> null
        assert pd.isna(result['data'].iloc[1])  # '.' -> null
        assert pd.isna(result['data'].iloc[2])  # '' -> null
        assert pd.isna(result['data'].iloc[3])  # '  ' -> null
        assert result['data'].iloc[4] == 'valid'
    
    def test_normalize_excel_error_indicators(self):
        """Should convert Excel error values to null."""
        from app.services.cleaner import normalize_missing_indicators
        
        df = pd.DataFrame({
            'data': ['#N/A', '#REF!', '#VALUE!', '#DIV/0!', 'valid']
        })
        result = normalize_missing_indicators(df, column='data')
        
        assert pd.isna(result['data'].iloc[0])  # '#N/A' -> null
        assert pd.isna(result['data'].iloc[1])  # '#REF!' -> null
        assert pd.isna(result['data'].iloc[2])  # '#VALUE!' -> null
        assert pd.isna(result['data'].iloc[3])  # '#DIV/0!' -> null
        assert result['data'].iloc[4] == 'valid'
    
    def test_normalize_preserves_valid_data(self):
        """Should NOT convert valid data that happens to contain na-like substrings."""
        from app.services.cleaner import normalize_missing_indicators
        
        df = pd.DataFrame({
            'data': ['banana', 'canal', 'nana', 'Diana', 'Nate']
        })
        result = normalize_missing_indicators(df, column='data')
        
        # All should be preserved - they contain 'na' but aren't missing indicators
        assert result['data'].iloc[0] == 'banana'
        assert result['data'].iloc[1] == 'canal'
        assert result['data'].iloc[2] == 'nana'
        assert result['data'].iloc[3] == 'Diana'
        assert result['data'].iloc[4] == 'Nate'
    
    def test_global_rules_include_missing_normalization(self):
        """Global rules should include normalize_missing_indicators."""
        from app.services.autonomous import generate_global_rules
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        rules = generate_global_rules(df)
        rule_types = [r.rule.rule_type for r in rules]
        
        assert 'normalize_missing_indicators' in rule_types


# ============================================
# Edge Cases Tests
# ============================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Should handle empty DataFrames gracefully."""
        df = pd.DataFrame()
        cleaned_df, result = autonomous_clean(df)
        
        assert len(cleaned_df) == 0
        assert result.rows_processed == 0
        assert result.columns_processed == 0
    
    def test_all_null_column(self):
        """Should handle columns with all null values."""
        df = pd.DataFrame({"empty": [None, None, None]})
        
        cleaned_df, result = autonomous_clean(df)
        
        # Should flag as unsafe
        assert len(result.warnings) > 0 or any(not i.is_safe for i in result.column_inferences)
    
    def test_special_characters_in_names(self):
        """Should handle special characters in name columns."""
        series = pd.Series(["O'Brien", "Mary-Jane", "José García"])
        cleaned, issues = clean_name_column(series)
        
        # Should preserve valid name characters
        assert "O'brien" in cleaned.iloc[0] or "O'Brien" in cleaned.iloc[0]
        assert "Mary-Jane" in cleaned.iloc[1] or "Mary-jane" in cleaned.iloc[1]
    
    def test_unicode_emails(self):
        """Should handle unicode in emails appropriately."""
        series = pd.Series(["test@example.com", "tëst@example.com"])
        cleaned, issues = clean_email_column(series)
        
        # First should be valid
        assert cleaned.iloc[0] == "test@example.com"
        # Second has unicode - check it's normalized but not replaced in strict mode
        assert cleaned.iloc[1] == "tëst@example.com"


# ============================================
# Number Word Conversion Tests
# ============================================

class TestNumberWordConversion:
    """
    Comprehensive tests for number word to digit conversion.
    
    These tests verify that number words are correctly converted to digits
    in various scenarios, including edge cases where the column type
    inference may not initially detect the column as numeric.
    """
    
    def test_clean_numeric_basic_number_words(self):
        """Should convert basic single-digit number words."""
        series = pd.Series(["one", "two", "three", "four", "five"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [1, 2, 3, 4, 5]
    
    def test_clean_numeric_teen_number_words(self):
        """Should convert teen number words (eleven through nineteen)."""
        series = pd.Series(["eleven", "twelve", "thirteen", "fourteen", "fifteen"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [11, 12, 13, 14, 15]
    
    def test_clean_numeric_tens_number_words(self):
        """Should convert tens number words (twenty, thirty, etc.)."""
        series = pd.Series(["twenty", "thirty", "forty", "fifty", "sixty"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [20, 30, 40, 50, 60]
    
    def test_clean_numeric_number_words_case_insensitive(self):
        """Should convert number words regardless of case."""
        series = pd.Series(["ONE", "Two", "THREE", "FoUr", "FIVE"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [1, 2, 3, 4, 5]
    
    def test_clean_numeric_number_words_with_whitespace(self):
        """Should convert number words with leading/trailing whitespace."""
        series = pd.Series(["  one  ", "two ", " three", "  four  "])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [1, 2, 3, 4]
    
    def test_clean_numeric_mixed_words_and_digits(self):
        """Should handle mixed number words and digits in same column."""
        series = pd.Series(["one", "2", "three", "4", "five"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [1, 2.0, 3, 4.0, 5]
    
    def test_clean_numeric_zero_and_hundred(self):
        """Should convert zero, hundred, and thousand."""
        series = pd.Series(["zero", "hundred", "thousand"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [0, 100, 1000]
    
    def test_clean_numeric_unknown_words_unchanged(self):
        """Should leave unknown words unchanged (no guessing)."""
        series = pd.Series(["twenty", "banana", "thirty", "apple"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.iloc[0] == 20
        assert cleaned.iloc[1] == "banana"  # Not converted
        assert cleaned.iloc[2] == 30
        assert cleaned.iloc[3] == "apple"  # Not converted
    
    def test_clean_numeric_partial_match_not_converted(self):
        """Should NOT convert words that contain number words but aren't valid number phrases."""
        series = pd.Series(["threesome", "sixpack", "someone"])
        cleaned, issues = clean_numeric_column(series)
        # These should NOT be converted as they are not valid number phrases
        assert cleaned.iloc[0] == "threesome"
        assert cleaned.iloc[1] == "sixpack"
        assert cleaned.iloc[2] == "someone"
    
    def test_clean_numeric_hyphenated_numbers_converted(self):
        """Should convert hyphenated numbers like 'twenty-one'."""
        series = pd.Series(["twenty-one", "thirty-two", "forty-five"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [21, 32, 45]
    
    def test_clean_numeric_all_supported_words(self):
        """Should convert all supported number words correctly."""
        all_words = [
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand"
        ]
        expected = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000
        ]
        series = pd.Series(all_words)
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == expected
    
    def test_column_with_only_number_words_inferred_type(self):
        """
        Test that columns containing only number words are handled correctly.
        
        This is a critical test - if a column contains ONLY number words like
        'one', 'two', 'three', it may be incorrectly inferred as TEXT or
        CATEGORICAL instead of NUMERIC, which would prevent number word
        conversion from being applied.
        """
        # Column with only number words
        df = pd.DataFrame({"quantity": ["one", "two", "three", "four", "five"] * 10})
        result = infer_column_type(df, "quantity")
        
        # The column name "quantity" matches NUMERIC_COLUMN_PATTERNS,
        # so it should be detected as NUMERIC despite containing text
        assert result.inferred_type == InferredType.NUMERIC, \
            f"Expected NUMERIC but got {result.inferred_type}. Column with number words should be detected as numeric when column name suggests numeric."
    
    def test_autonomous_clean_number_word_column(self):
        """Should convert number words during autonomous cleaning."""
        df = pd.DataFrame({
            "quantity": ["one", "two", "three", "four", "five"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # After autonomous cleaning, number words should be converted to digits
        expected_values = [1, 2, 3, 4, 5]
        actual_values = cleaned_df["quantity"].tolist()
        
        # Check if conversion happened
        assert actual_values == expected_values, \
            f"Number words were not converted. Expected {expected_values}, got {actual_values}"
    
    def test_autonomous_clean_number_words_generic_column_name(self):
        """
        Test number word conversion with a generic column name.
        
        This tests the case where the column name doesn't suggest numeric data
        (like 'data' or 'value'), but contains number words that should still
        be convertible.
        """
        df = pd.DataFrame({
            "data": ["one", "two", "three", "four", "five"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        # Even with a generic column name, number words should be detected
        # and converted since the values are recognized as number words
        expected_values = [1, 2, 3, 4, 5]
        actual_values = cleaned_df["data"].tolist()
        
        assert actual_values == expected_values, \
            f"Number words were not converted. Expected {expected_values}, got {actual_values}"
    
    def test_clean_numeric_preserves_null_values(self):
        """Should preserve null/NaN values when converting number words."""
        series = pd.Series(["one", None, "three", pd.NA, "five"])
        cleaned, issues = clean_numeric_column(series)
        
        assert cleaned.iloc[0] == 1
        assert pd.isna(cleaned.iloc[1])
        assert cleaned.iloc[2] == 3
        assert pd.isna(cleaned.iloc[3])
        assert cleaned.iloc[4] == 5
    
    def test_clean_numeric_number_words_with_currency(self):
        """Should handle columns with both number words and currency values."""
        series = pd.Series(["$100", "twenty", "$50", "thirty"])
        cleaned, issues = clean_numeric_column(series)
        
        assert cleaned.iloc[0] == 100.0
        assert cleaned.iloc[1] == 20
        assert cleaned.iloc[2] == 50.0
        assert cleaned.iloc[3] == 30

    def test_clean_numeric_seventy_eighty_ninety(self):
        """Should convert seventy, eighty, ninety correctly."""
        series = pd.Series(["seventy", "eighty", "ninety"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [70, 80, 90]

    def test_clean_numeric_sixteen_to_nineteen(self):
        """Should convert sixteen through nineteen correctly."""
        series = pd.Series(["sixteen", "seventeen", "eighteen", "nineteen"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [16, 17, 18, 19]

    def test_clean_numeric_six_to_ten(self):
        """Should convert six through ten correctly."""
        series = pd.Series(["six", "seven", "eight", "nine", "ten"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [6, 7, 8, 9, 10]

    def test_numeric_column_name_with_number_words(self):
        """
        Test that columns with numeric-suggesting names containing number words
        are correctly identified and converted.
        """
        df = pd.DataFrame({
            "amount": ["ten", "twenty", "thirty"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        assert cleaned_df["amount"].tolist() == [10, 20, 30], \
            "Columns with numeric names containing number words should be converted"

    def test_score_column_with_number_words(self):
        """Test that a 'score' column with number words gets converted."""
        df = pd.DataFrame({
            "score": ["one", "two", "three", "four", "five"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        
        assert cleaned_df["score"].tolist() == [1, 2, 3, 4, 5]

    def test_infer_column_type_numeric_pattern_check(self):
        """
        Test that columns with numeric-suggesting names and number word values
        are correctly identified as NUMERIC.
        """
        df = pd.DataFrame({"quantity": ["one", "two", "three"] * 20})
        result = infer_column_type(df, "quantity")
        
        # Should be classified as NUMERIC since column name matches pattern
        # and values are recognized number words
        assert result.inferred_type == InferredType.NUMERIC, \
            f"Expected NUMERIC but got {result.inferred_type}"
        assert any("number" in ind.lower() for ind in result.indicators), \
            f"Expected indicator about number phrases, got: {result.indicators}"

    # ============================================
    # Compound Number Phrase Tests
    # ============================================
    
    def test_clean_numeric_compound_number_sixty_thousand(self):
        """Should convert 'sixty thousand' to 60000."""
        series = pd.Series(["sixty thousand", "forty thousand", "eighty thousand"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [60000, 40000, 80000]

    def test_clean_numeric_compound_number_with_hundreds(self):
        """Should convert compound numbers with hundreds."""
        series = pd.Series(["two hundred", "five hundred", "nine hundred"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [200, 500, 900]

    def test_clean_numeric_compound_number_full(self):
        """Should convert full compound numbers like 'one hundred twenty three'."""
        series = pd.Series(["one hundred twenty three", "two hundred fifty", "three hundred"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [123, 250, 300]

    def test_clean_numeric_compound_hundreds_of_thousands(self):
        """Should convert 'five hundred thousand' correctly."""
        series = pd.Series(["five hundred thousand", "two hundred thousand"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [500000, 200000]

    def test_clean_numeric_compound_with_hyphen(self):
        """Should convert hyphenated numbers like 'twenty-five'."""
        series = pd.Series(["twenty-five", "thirty-two", "forty-eight"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [25, 32, 48]

    def test_clean_numeric_compound_mixed_with_digits(self):
        """Should handle mixed compound words and regular digits."""
        series = pd.Series(["sixty thousand", "75000", "forty thousand"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.iloc[0] == 60000
        assert cleaned.iloc[1] == 75000.0
        assert cleaned.iloc[2] == 40000

    def test_clean_numeric_million(self):
        """Should convert millions correctly."""
        series = pd.Series(["one million", "two million", "five million"])
        cleaned, issues = clean_numeric_column(series)
        assert cleaned.tolist() == [1000000, 2000000, 5000000]

    def test_autonomous_clean_salary_column_with_compound_numbers(self):
        """Should convert salary column with compound number words."""
        df = pd.DataFrame({
            "salary": ["sixty thousand", "seventy five thousand", "eighty thousand"]
        })
        
        cleaned_df, result = autonomous_clean(df)
        
        assert cleaned_df["salary"].iloc[0] == 60000
        assert cleaned_df["salary"].iloc[2] == 80000




