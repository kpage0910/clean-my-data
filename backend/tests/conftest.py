"""
Pytest configuration and shared fixtures for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data_dir():
    """Return path to sample data directory."""
    return Path(__file__).parent / "sample_data"


@pytest.fixture
def messy_df(sample_data_dir):
    """Load the messy sample CSV as a DataFrame."""
    csv_path = sample_data_dir / "messy_sample.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    # Fallback to inline data if file doesn't exist
    return pd.DataFrame({
        'id': [1, 2, 3, 1, 4],
        'name': ['  Alice  ', 'Bob', '  Alice  ', '  Alice  ', 'Charlie  '],
        'email': ['alice@example.com', 'bob@invalid', 'alice@example.com', 'alice@example.com', 'charlie@test.com'],
        'age': [25, 'thirty', 35, 25, 40],
        'salary': ['$50,000', '$60,000', '$50,000', '$50,000', '$80,000'],
        'is_active': ['true', 'yes', 'TRUE', 'true', 'no']
    })


@pytest.fixture
def expected_cleaned_df(sample_data_dir):
    """Load the expected cleaned CSV as a DataFrame."""
    csv_path = sample_data_dir / "cleaned_sample.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@pytest.fixture
def simple_df():
    """Create a simple test DataFrame."""
    return pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'active': [True, False, True]
    })


@pytest.fixture
def messy_simple_df():
    """Create a simple messy DataFrame for basic tests."""
    return pd.DataFrame({
        'name': ['  Alice  ', '  Bob  ', 'Charlie'],
        'value': ['$1,000', '$2,500', '$500'],
        'date': ['2023-01-15', 'invalid', '2023-03-20']
    })


@pytest.fixture
def df_with_duplicates():
    """Create a DataFrame with duplicate rows."""
    return pd.DataFrame({
        'id': [1, 2, 1, 3, 2],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
        'value': [100, 200, 100, 300, 200]
    })


@pytest.fixture
def df_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame({
        'name': ['Alice', None, 'Charlie', ''],
        'age': [25, np.nan, 35, 40],
        'email': ['alice@test.com', 'bob@test.com', None, 'dave@test.com']
    })


@pytest.fixture
def df_with_whitespace():
    """Create a DataFrame with whitespace issues."""
    return pd.DataFrame({
        'first_name': ['  Alice', 'Bob  ', '  Charlie  '],
        'last_name': ['Smith  ', '  Johnson', '  Brown  '],
        'city': ['New York', 'Los Angeles', 'Chicago']
    })


@pytest.fixture
def df_with_numbers():
    """Create a DataFrame with various number formats."""
    return pd.DataFrame({
        'amount': ['$1,234.56', '€2,500.00', '£1,000', '¥50,000'],
        'percentage': ['25%', '50%', '75%', '100%'],
        'plain': [100, 200, 300, 400]
    })


@pytest.fixture
def df_with_dates():
    """Create a DataFrame with various date formats."""
    return pd.DataFrame({
        'created_date': ['2023-01-15', '2023/02/20', '15-03-2023', 'March 1, 2023'],
        'updated_date': ['2023-04-01', 'invalid', '2023-06-15', '2023-07-20']
    })


@pytest.fixture
def df_with_mixed_types():
    """Create a DataFrame with mixed types in columns."""
    return pd.DataFrame({
        'value': [1, '2', 3.5, 'four', None],
        'flag': ['true', 'false', True, 0, 'yes']
    })
