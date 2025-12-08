"""
Unit tests for the scanner service.

Tests cover all data quality issue detection functions:
- Missing values detection
- Mixed types detection
- Whitespace issues detection
- Duplicate rows detection
- Invalid email detection
- Invalid date detection
- Number formatting issues detection
- Boolean inconsistency detection
"""

import pytest
import pandas as pd
import numpy as np

from app.services.scanner import (
    detect_missing_values,
    detect_mixed_types,
    detect_whitespace_issues,
    detect_duplicate_rows,
    detect_invalid_emails,
    detect_invalid_dates,
    detect_number_formatting_issues,
    detect_number_words,
    detect_boolean_inconsistencies,
    analyze_dataframe,
    scan_dataframe,
)


# ============================================
# Tests for detect_missing_values
# ============================================

class TestDetectMissingValues:
    """Tests for missing value detection."""
    
    def test_missing_values_with_nan(self):
        """Test detection of NaN values."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', None, 'Charlie'],
            'age': [25, np.nan, 35, 40]
        })
        
        result = detect_missing_values(df)
        
        assert 'name' in result
        assert result['name']['issue'] == 'missing_values'
        assert result['name']['count'] == 1
        assert result['name']['percentage'] == 25.0
        
        assert 'age' in result
        assert result['age']['count'] == 1
    
    def test_missing_values_with_empty_strings(self):
        """Test detection of empty strings as missing values."""
        df = pd.DataFrame({
            'col1': ['a', '', 'c', ''],
            'col2': [1, 2, 3, 4]
        })
        
        result = detect_missing_values(df)
        
        assert 'col1' in result
        assert result['col1']['count'] == 2
        assert result['col1']['percentage'] == 50.0
        assert 'col2' not in result  # No missing values
    
    def test_no_missing_values(self):
        """Test that no issues are reported when no missing values exist."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        
        result = detect_missing_values(df)
        
        assert len(result) == 0
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = detect_missing_values(df)
        
        assert len(result) == 0


# ============================================
# Tests for detect_mixed_types
# ============================================

class TestDetectMixedTypes:
    """Tests for mixed type detection."""
    
    def test_mixed_int_and_string(self):
        """Test detection of mixed int and string types."""
        df = pd.DataFrame({
            'value': [1, '2', 3, 'four']
        })
        
        result = detect_mixed_types(df)
        
        assert 'value' in result
        assert result['value']['issue'] == 'mixed_types'
        assert 'int' in result['value']['types']
        assert 'str' in result['value']['types']
    
    def test_mixed_float_and_string(self):
        """Test detection of mixed float and string types."""
        df = pd.DataFrame({
            'price': [10.5, 'N/A', 30.0, 'unknown']
        })
        
        result = detect_mixed_types(df)
        
        assert 'price' in result
        assert 'float' in result['price']['types']
        assert 'str' in result['price']['types']
    
    def test_uniform_types(self):
        """Test that no issues are reported for uniform types."""
        df = pd.DataFrame({
            'names': ['Alice', 'Bob', 'Charlie'],
            'numbers': [1, 2, 3]
        })
        
        result = detect_mixed_types(df)
        
        assert 'names' not in result
        assert 'numbers' not in result
    
    def test_mixed_with_none_values(self):
        """Test that None values don't affect type detection."""
        df = pd.DataFrame({
            'value': [1, None, 'three', None]
        })
        
        result = detect_mixed_types(df)
        
        assert 'value' in result
        assert len(result['value']['types']) == 2


# ============================================
# Tests for detect_whitespace_issues
# ============================================

class TestDetectWhitespaceIssues:
    """Tests for whitespace detection."""
    
    def test_leading_whitespace(self):
        """Test detection of leading whitespace."""
        df = pd.DataFrame({
            'name': ['  Alice', 'Bob', '  Charlie']
        })
        
        result = detect_whitespace_issues(df)
        
        assert 'name' in result
        assert result['name']['issue'] == 'whitespace'
        assert result['name']['count'] == 2
        assert 0 in result['name']['affected_indices']
        assert 2 in result['name']['affected_indices']
    
    def test_trailing_whitespace(self):
        """Test detection of trailing whitespace."""
        df = pd.DataFrame({
            'city': ['New York  ', 'Los Angeles', 'Chicago   ']
        })
        
        result = detect_whitespace_issues(df)
        
        assert 'city' in result
        assert result['city']['count'] == 2
    
    def test_both_leading_and_trailing(self):
        """Test detection of both leading and trailing whitespace."""
        df = pd.DataFrame({
            'data': ['  value  ', 'clean', ' space ']
        })
        
        result = detect_whitespace_issues(df)
        
        assert 'data' in result
        assert result['data']['count'] == 2
    
    def test_no_whitespace_issues(self):
        """Test that no issues are reported for clean data."""
        df = pd.DataFrame({
            'clean_col': ['Alice', 'Bob', 'Charlie']
        })
        
        result = detect_whitespace_issues(df)
        
        assert len(result) == 0


# ============================================
# Tests for detect_duplicate_rows
# ============================================

class TestDetectDuplicateRows:
    """Tests for duplicate row detection."""
    
    def test_exact_duplicates(self):
        """Test detection of exact duplicate rows."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
            'age': [25, 30, 25, 35, 30]
        })
        
        result = detect_duplicate_rows(df)
        
        assert result['count'] == 4  # 2 pairs of duplicates
        assert 0 in result['duplicates']
        assert 2 in result['duplicates']
        assert 1 in result['duplicates']
        assert 4 in result['duplicates']
    
    def test_multiple_duplicates_same_row(self):
        """Test detection when same row appears multiple times."""
        df = pd.DataFrame({
            'value': [1, 1, 1, 2]
        })
        
        result = detect_duplicate_rows(df)
        
        assert result['count'] == 3  # Three rows with value 1
        assert len(result['duplicate_groups']) >= 1
    
    def test_no_duplicates(self):
        """Test that no issues are reported for unique rows."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['A', 'B', 'C', 'D']
        })
        
        result = detect_duplicate_rows(df)
        
        assert result['count'] == 0
        assert len(result['duplicates']) == 0
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = detect_duplicate_rows(df)
        
        assert result['count'] == 0


# ============================================
# Tests for detect_invalid_emails
# ============================================

class TestDetectInvalidEmails:
    """Tests for invalid email detection."""
    
    def test_invalid_email_formats(self):
        """Test detection of various invalid email formats."""
        df = pd.DataFrame({
            'email': [
                'valid@example.com',
                'invalid@',
                'also.valid@domain.org',
                '@nodomain.com',
                'spaces in@email.com'
            ]
        })
        
        result = detect_invalid_emails(df)
        
        assert 'email' in result
        assert result['email']['issue'] == 'invalid_email'
        assert result['email']['count'] >= 3
    
    def test_missing_at_symbol(self):
        """Test detection of emails missing @ symbol (mixed with valid)."""
        df = pd.DataFrame({
            'contact': [
                'user@domain.com',
                'notanemail',
                'another@valid.net',
                'user@domain.com'
            ]
        })
        
        result = detect_invalid_emails(df)
        
        # Column should be checked since it has emails
        # 'notanemail' doesn't have @ so it won't be flagged as invalid email
        # (it's simply not an email)
        assert 'contact' not in result or result['contact']['count'] == 0
    
    def test_valid_emails_only(self):
        """Test that no issues are reported for valid emails."""
        df = pd.DataFrame({
            'email': [
                'user@example.com',
                'test.user@domain.org',
                'info@company.co.uk'
            ]
        })
        
        result = detect_invalid_emails(df)
        
        assert 'email' not in result
    
    def test_non_email_column(self):
        """Test that non-email columns are skipped."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        result = detect_invalid_emails(df)
        
        assert len(result) == 0


# ============================================
# Tests for detect_invalid_dates
# ============================================

class TestDetectInvalidDates:
    """Tests for invalid date detection."""
    
    def test_invalid_date_strings(self):
        """Test detection of invalid date strings in date column."""
        df = pd.DataFrame({
            'created_date': [
                '2023-01-15',
                'not-a-date',
                '2023-06-30',
                'invalid'
            ]
        })
        
        result = detect_invalid_dates(df)
        
        assert 'created_date' in result
        assert result['created_date']['issue'] == 'invalid_date'
        assert result['created_date']['count'] == 2
    
    def test_mixed_valid_invalid_dates(self):
        """Test with mix of valid and invalid dates."""
        df = pd.DataFrame({
            'timestamp': [
                '2023-01-01 10:00:00',
                '13/45/2023',  # Invalid
                'January 15, 2023',
                'abc123'  # Invalid
            ]
        })
        
        result = detect_invalid_dates(df)
        
        assert 'timestamp' in result
        assert result['timestamp']['count'] >= 2
    
    def test_valid_dates_only(self):
        """Test that no issues are reported for valid dates."""
        df = pd.DataFrame({
            'date': [
                '2023-01-15',
                '2023-06-30',
                '2023-12-25'
            ]
        })
        
        result = detect_invalid_dates(df)
        
        assert 'date' not in result
    
    def test_non_date_column_ignored(self):
        """Test that columns without date keywords are ignored."""
        df = pd.DataFrame({
            'value': ['abc', 'def', 'ghi']
        })
        
        result = detect_invalid_dates(df)
        
        assert len(result) == 0


# ============================================
# Tests for detect_number_formatting_issues
# ============================================

class TestDetectNumberFormattingIssues:
    """Tests for number formatting issue detection."""
    
    def test_numbers_with_commas(self):
        """Test detection of numbers with comma separators."""
        df = pd.DataFrame({
            'amount': ['1,234', '5,678,901', '100', '2,500']
        })
        
        result = detect_number_formatting_issues(df)
        
        assert 'amount' in result
        assert result['amount']['issue'] == 'number_formatting'
        assert result['amount']['has_commas'] is True
        assert result['amount']['count'] >= 3
    
    def test_currency_symbols(self):
        """Test detection of currency symbols."""
        df = pd.DataFrame({
            'price': ['$100', '€50', '£75', '200']
        })
        
        result = detect_number_formatting_issues(df)
        
        assert 'price' in result
        assert result['price']['has_currency'] is True
        assert result['price']['count'] >= 3
    
    def test_mixed_formatting(self):
        """Test detection of mixed number formatting."""
        df = pd.DataFrame({
            'value': ['$1,234.56', '100', '$500', '2,000']
        })
        
        result = detect_number_formatting_issues(df)
        
        assert 'value' in result
        assert result['value']['has_commas'] is True or result['value']['has_currency'] is True
    
    def test_clean_numbers(self):
        """Test that no issues are reported for clean numbers."""
        df = pd.DataFrame({
            'count': ['100', '200', '300']
        })
        
        result = detect_number_formatting_issues(df)
        
        assert 'count' not in result


# ============================================
# Tests for detect_number_words
# ============================================

class TestDetectNumberWords:
    """Tests for number word detection."""
    
    def test_single_number_words(self):
        """Test detection of single number words."""
        df = pd.DataFrame({
            'age': ['twenty', 'thirty', '40', 'fifty']
        })
        
        result = detect_number_words(df)
        
        assert 'age' in result
        assert result['age']['issue'] == 'number_words'
        assert result['age']['count'] == 3  # twenty, thirty, fifty
    
    def test_compound_number_phrases(self):
        """Test detection of compound number phrases like 'sixty thousand'."""
        df = pd.DataFrame({
            'salary': ['sixty thousand', '75000', 'two hundred fifty', 'one million']
        })
        
        result = detect_number_words(df)
        
        assert 'salary' in result
        assert result['salary']['issue'] == 'number_words'
        assert result['salary']['count'] == 3  # sixty thousand, two hundred fifty, one million
        
        # Check examples contain the correct conversions
        examples = result['salary']['examples']
        example_dict = {ex[0].lower(): ex[1] for ex in examples}
        assert example_dict.get('sixty thousand') == 60000
        assert example_dict.get('two hundred fifty') == 250
        assert example_dict.get('one million') == 1000000
    
    def test_mixed_values(self):
        """Test detection with mixed number words and other values."""
        df = pd.DataFrame({
            'value': ['five', 'hello', '100', 'ten', None]
        })
        
        result = detect_number_words(df)
        
        assert 'value' in result
        assert result['value']['count'] == 2  # five, ten
    
    def test_no_number_words(self):
        """Test that no issues are reported when no number words present."""
        df = pd.DataFrame({
            'count': ['100', '200', '300']
        })
        
        result = detect_number_words(df)
        
        assert 'count' not in result
    
    def test_non_string_columns_ignored(self):
        """Test that non-string columns are ignored."""
        df = pd.DataFrame({
            'numbers': [1, 2, 3, 4]
        })
        
        result = detect_number_words(df)
        
        assert len(result) == 0


# ============================================
# Tests for analyze_dataframe (integration)
# ============================================

class TestAnalyzeDataframe:
    """Integration tests for the main analyze_dataframe function."""
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis with multiple issue types."""
        df = pd.DataFrame({
            'name': ['  Alice', 'Bob', None, 'Bob'],
            'email': ['alice@test.com', 'invalid@', 'bob@test.com', 'bob@test.com'],
            'age': [25, '30', 35, 35],
            'salary': ['$50,000', '60000', '$75,000', '$75,000']
        })
        
        result = analyze_dataframe(df)
        
        # Check structure
        assert 'column_issues' in result
        assert 'row_issues' in result
        assert 'summary' in result
        
        # Check that issues were detected
        assert len(result['column_issues']) > 0
        
        # Check summary
        assert result['summary']['total_rows'] == 4
        assert result['summary']['total_columns'] == 4
    
    def test_clean_dataframe(self):
        """Test analysis of clean DataFrame with no issues."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'active': [True, False, True]
        })
        
        result = analyze_dataframe(df)
        
        assert result['summary']['columns_with_issues'] == 0
        assert result['row_issues']['duplicate_count'] == 0
    
    def test_empty_dataframe(self):
        """Test analysis of empty DataFrame."""
        df = pd.DataFrame()
        
        result = analyze_dataframe(df)
        
        assert result['summary']['total_rows'] == 0
        assert result['summary']['total_columns'] == 0


# ============================================
# Tests for scan_dataframe (API format)
# ============================================

class TestScanDataframe:
    """Tests for the scan_dataframe function that returns ScanReport."""
    
    def test_returns_scan_report(self):
        """Test that scan_dataframe returns proper ScanReport format."""
        df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie'],
            'age': [25, 30, 35]
        })
        
        result = scan_dataframe(df)
        
        assert result.total_rows == 3
        assert result.total_columns == 2
        assert isinstance(result.issues, list)
        assert isinstance(result.column_stats, dict)
    
    def test_scan_report_issues_format(self):
        """Test that issues in ScanReport have correct format."""
        df = pd.DataFrame({
            'email': ['valid@test.com', 'invalid@', 'test@domain.com']
        })
        
        result = scan_dataframe(df)
        
        # Check that issues have required fields
        for issue in result.issues:
            assert hasattr(issue, 'column')
            assert hasattr(issue, 'issue_type')
            assert hasattr(issue, 'severity')
            assert hasattr(issue, 'count')
            assert hasattr(issue, 'description')
    
    def test_scan_report_column_stats(self):
        """Test that column_stats contains expected information."""
        df = pd.DataFrame({
            'value': [1, 2, None, 4]
        })
        
        result = scan_dataframe(df)
        
        assert 'value' in result.column_stats
        assert 'dtype' in result.column_stats['value']
        assert 'null_count' in result.column_stats['value']
        assert result.column_stats['value']['null_count'] == 1


# ============================================
# Tests for detect_boolean_inconsistencies
# ============================================

class TestDetectBooleanInconsistencies:
    """Tests for Boolean inconsistency detection."""
    
    def test_mixed_true_false_and_yes_no(self):
        """Test detection of mixed true/false and yes/no formats."""
        df = pd.DataFrame({
            'active': ['true', 'false', 'yes', 'no', 'true', 'yes']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'active' in result
        assert result['active']['issue'] == 'boolean_inconsistency'
        assert result['active']['count'] == 6
        assert 'true' in result['active']['formats_found']
        assert 'yes' in result['active']['formats_found']
        assert 'normalization_options' in result['active']
        assert 'True/False' in result['active']['normalization_options']
    
    def test_mixed_boolean_and_numeric(self):
        """Test detection of mixed true/false and 1/0 formats."""
        df = pd.DataFrame({
            'enabled': ['true', 'false', '1', '0', 'True', 'FALSE']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'enabled' in result
        assert result['enabled']['issue'] == 'boolean_inconsistency'
        assert '1' in result['enabled']['formats_found'] or '0' in result['enabled']['formats_found']
    
    def test_mixed_y_n_and_yes_no(self):
        """Test detection of mixed y/n and yes/no formats."""
        df = pd.DataFrame({
            'subscribed': ['y', 'n', 'yes', 'no', 'Y', 'N']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'subscribed' in result
        assert result['subscribed']['issue'] == 'boolean_inconsistency'
    
    def test_consistent_true_false_no_issue(self):
        """Test that consistent true/false format is not flagged."""
        df = pd.DataFrame({
            'active': ['true', 'false', 'true', 'false', 'TRUE', 'FALSE']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        # Should NOT be flagged because all values are in the same format group
        assert 'active' not in result
    
    def test_consistent_yes_no_no_issue(self):
        """Test that consistent yes/no format is not flagged."""
        df = pd.DataFrame({
            'confirmed': ['yes', 'no', 'Yes', 'No', 'YES', 'NO']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        # Should NOT be flagged because all values are in the same format group
        assert 'confirmed' not in result
    
    def test_consistent_1_0_no_issue(self):
        """Test that consistent 1/0 format is not flagged."""
        df = pd.DataFrame({
            'flag': ['1', '0', '1', '0', '1']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        # Should NOT be flagged because all values are in the same format group
        assert 'flag' not in result
    
    def test_non_boolean_column_ignored(self):
        """Test that non-boolean columns are not flagged."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [25, 30, 35, 40]
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'name' not in result
        assert 'age' not in result
    
    def test_column_with_null_values(self):
        """Test that null values are handled correctly."""
        df = pd.DataFrame({
            'active': ['true', None, 'yes', np.nan, 'false', '']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'active' in result
        # Count should only include non-null boolean values
        assert result['active']['count'] == 3
    
    def test_case_insensitivity(self):
        """Test that detection is case-insensitive."""
        df = pd.DataFrame({
            'status': ['TRUE', 'False', 'YES', 'no', 'true', 'NO']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'status' in result
        # Formats are stored lowercase
        assert 'true' in result['status']['formats_found']
        assert 'false' in result['status']['formats_found']
        assert 'yes' in result['status']['formats_found']
        assert 'no' in result['status']['formats_found']
    
    def test_mixed_t_f_and_true_false(self):
        """Test detection of mixed t/f and true/false formats."""
        df = pd.DataFrame({
            'valid': ['t', 'f', 'true', 'false', 'T', 'F']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'valid' in result
        assert result['valid']['issue'] == 'boolean_inconsistency'
    
    def test_mostly_non_boolean_not_flagged(self):
        """Test that columns with mostly non-boolean values are not flagged."""
        df = pd.DataFrame({
            'mixed': ['apple', 'banana', 'true', 'cherry', 'orange', 'yes', 'grape', 'melon', 'kiwi', 'mango']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        # Only 2 out of 10 values are boolean-like, should not be flagged
        assert 'mixed' not in result
    
    def test_examples_preserve_original_case(self):
        """Test that examples preserve original case."""
        df = pd.DataFrame({
            'active': ['TRUE', 'False', 'YES', 'no']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'active' in result
        examples = result['active']['examples']
        # Examples should preserve original case
        assert 'TRUE' in examples or 'False' in examples or 'YES' in examples
    
    def test_numeric_boolean_column_not_flagged(self):
        """Test that pure numeric boolean columns (0/1) are not flagged."""
        df = pd.DataFrame({
            'flag': [0, 1, 0, 1, 1, 0]
        })
        
        result = detect_boolean_inconsistencies(df)
        
        # Pure numeric 0/1 columns should not be flagged
        assert 'flag' not in result
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = detect_boolean_inconsistencies(df)
        
        assert len(result) == 0
    
    def test_normalization_options_present(self):
        """Test that normalization options are provided."""
        df = pd.DataFrame({
            'active': ['true', 'yes', '1', 'false', 'no', '0']
        })
        
        result = detect_boolean_inconsistencies(df)
        
        assert 'active' in result
        options = result['active']['normalization_options']
        assert 'True/False' in options
        assert 'Yes/No' in options
        assert '1/0' in options


class TestBooleanInconsistencyIntegration:
    """Integration tests for Boolean inconsistency detection."""
    
    def test_analyze_dataframe_includes_boolean_issues(self):
        """Test that analyze_dataframe includes Boolean inconsistency issues."""
        df = pd.DataFrame({
            'active': ['true', 'yes', 'false', 'no'],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana']
        })
        
        result = analyze_dataframe(df)
        
        assert 'active' in result['column_issues']
        issues = result['column_issues']['active']
        boolean_issue = next((i for i in issues if i['issue'] == 'boolean_inconsistency'), None)
        assert boolean_issue is not None
    
    def test_scan_dataframe_boolean_issue_format(self):
        """Test that scan_dataframe formats Boolean issues correctly."""
        df = pd.DataFrame({
            'enabled': ['true', '1', 'yes', 'false', '0', 'no']
        })
        
        result = scan_dataframe(df)
        
        boolean_issue = next((i for i in result.issues if i.issue_type == 'boolean_inconsistency'), None)
        assert boolean_issue is not None
        assert boolean_issue.severity == 'medium'
        assert 'Warning' in boolean_issue.description
        assert 'inconsistent formats' in boolean_issue.description
        assert 'normalizing' in boolean_issue.description.lower()
    
    def test_scan_dataframe_boolean_examples(self):
        """Test that scan_dataframe includes examples for Boolean issues."""
        df = pd.DataFrame({
            'status': ['True', 'YES', '1', 'False', 'No', '0']
        })
        
        result = scan_dataframe(df)
        
        boolean_issue = next((i for i in result.issues if i.issue_type == 'boolean_inconsistency'), None)
        assert boolean_issue is not None
        assert boolean_issue.examples is not None
        assert len(boolean_issue.examples) > 0


# ============================================
# Edge Case Tests
# ============================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            'col': ['value']
        })
        
        result = analyze_dataframe(df)
        
        assert result['summary']['total_rows'] == 1
    
    def test_single_column_dataframe(self):
        """Test handling of single-column DataFrame."""
        df = pd.DataFrame({
            'only_col': [1, 2, 3, 4, 5]
        })
        
        result = analyze_dataframe(df)
        
        assert result['summary']['total_columns'] == 1
    
    def test_all_null_column(self):
        """Test handling of column with all null values."""
        df = pd.DataFrame({
            'empty': [None, None, None],
            'filled': [1, 2, 3]
        })
        
        result = detect_missing_values(df)
        
        assert 'empty' in result
        assert result['empty']['percentage'] == 100.0
    
    def test_unicode_strings(self):
        """Test handling of Unicode strings."""
        df = pd.DataFrame({
            'name': ['  Müller', 'Señor  ', '日本語'],
            'email': ['münchen@test.de', 'user@example.com', 'test@日本.jp']
        })
        
        result = detect_whitespace_issues(df)
        
        assert 'name' in result
        assert result['name']['count'] == 2
    
    def test_large_numbers_with_scientific_notation(self):
        """Test handling of scientific notation in numbers."""
        df = pd.DataFrame({
            'value': ['1.5e10', '2.3e-5', '1,000,000']
        })
        
        result = detect_number_formatting_issues(df)
        
        # Scientific notation shouldn't be flagged as formatting issue
        # but comma-separated numbers should
        assert 'value' in result
        assert result['value']['count'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
