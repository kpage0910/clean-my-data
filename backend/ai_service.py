import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from io import StringIO
import re


# ---------------------------
# Utilities
# ---------------------------

def _generate_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def _extract_csv_from_response(text: str) -> str:
    """Extract CSV content from LLM response, removing markdown blocks and extra text."""
    # Remove markdown code blocks
    text = re.sub(r'```csv\s*\n', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()
    return text


# ---------------------------
# Post-Processing Validation
# ---------------------------

def _validate_email(email):
    """Validate email format. Return cleaned email or None if invalid."""
    if pd.isna(email) or email == '':
        return None
    
    email = str(email).strip()
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, email):
        return email
    else:
        return None  # Invalid email


def _validate_age(age):
    """Validate age. Return valid age or None."""
    if pd.isna(age) or age == '':
        return None
    
    try:
        age_num = float(age)
        # Valid age range: 1-120 (exclude 0 and negative)
        if 1 <= age_num <= 120:
            return int(age_num)
        else:
            return None  # Out of range
    except (ValueError, TypeError):
        return None


def _validate_date(date_val):
    """Validate and standardize date to YYYY-MM-DD format."""
    if pd.isna(date_val) or date_val == '':
        return None
    
    try:
        # Try to parse the date
        parsed_date = pd.to_datetime(date_val, errors='coerce')
        if pd.isna(parsed_date):
            return None
        return parsed_date.strftime('%Y-%m-%d')
    except:
        return None


def _apply_post_processing_validation(df: pd.DataFrame) -> tuple:
    """
    Apply strict validation rules after AI cleaning.
    Returns (cleaned_df, validation_operations)
    """
    df_cleaned = df.copy()
    validation_ops = []
    
    # Detect column types by name (flexible detection)
    email_cols = [col for col in df.columns if 'email' in col.lower()]
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    date_cols = [col for col in df.columns if any(keyword in col.lower() 
                 for keyword in ['date', 'time', 'signup', 'created', 'updated'])]
    name_cols = [col for col in df.columns if any(keyword in col.lower() 
                 for keyword in ['name', 'first', 'last', 'full'])]
    
    # 1. Validate emails
    for col in email_cols:
        if col in df_cleaned.columns:
            invalid_count = 0
            for idx in df_cleaned.index:
                original = df_cleaned.at[idx, col]
                validated = _validate_email(original)
                if original != validated and pd.notna(original):
                    invalid_count += 1
                df_cleaned.at[idx, col] = validated
            
            if invalid_count > 0:
                validation_ops.append(f"Removed {invalid_count} invalid emails from '{col}'")
    
    # 2. Validate ages
    for col in age_cols:
        if col in df_cleaned.columns:
            invalid_count = 0
            for idx in df_cleaned.index:
                original = df_cleaned.at[idx, col]
                validated = _validate_age(original)
                if original != validated and pd.notna(original):
                    invalid_count += 1
                df_cleaned.at[idx, col] = validated
            
            if invalid_count > 0:
                validation_ops.append(f"Removed {invalid_count} invalid ages from '{col}' (must be 1-120)")
    
    # 3. Validate dates
    for col in date_cols:
        if col in df_cleaned.columns:
            invalid_count = 0
            for idx in df_cleaned.index:
                original = df_cleaned.at[idx, col]
                validated = _validate_date(original)
                if str(original) != str(validated) and pd.notna(original):
                    invalid_count += 1
                df_cleaned.at[idx, col] = validated
            
            if invalid_count > 0:
                validation_ops.append(f"Standardized {invalid_count} dates in '{col}' to YYYY-MM-DD")
    
    # 4. Standardize names (Title Case)
    for col in name_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: x.strip().title() if pd.notna(x) and x != '' else None
            )
    

    # 4.5. Standardize categorical columns (lowercase) and trim all strings
    status_cols = [col for col in df.columns if 'status' in col.lower() or 'type' in col.lower() or 'category' in col.lower()]
    for col in status_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) and x != '' else None
            )
            validation_ops.append(f"Standardized '{col}' values to lowercase")

    # 4.6. Standardize boolean-like columns (e.g., 'Active', 'Is', 'Enabled')
    bool_keywords = ['active', 'is', 'enabled', 'flag', 'valid', 'available', 'present', 'member', 'subscribed']
    bool_cols = [col for col in df.columns if any(kw in col.lower() for kw in bool_keywords)]
    def _standardize_bool(val):
        if pd.isna(val) or str(val).strip() == '' or str(val).strip() == '—':
            return False
        val_str = str(val).strip().lower()
        if val_str in ['true', '1', 'yes', 'y', 't', 'on', 'active', 'enabled', 'present', 'member', 'subscribed']:
            return True
        if val_str in ['false', '0', 'no', 'n', 'off', 'inactive', 'disabled', 'not present', 'not member', 'not subscribed']:
            return False
        # If ambiguous, treat as False
        return False
    for col in bool_cols:
        if col in df_cleaned.columns:
            orig_vals = df_cleaned[col].copy()
            df_cleaned[col] = df_cleaned[col].apply(_standardize_bool)
            if not orig_vals.equals(df_cleaned[col]):
                validation_ops.append(f"Standardized '{col}' to boolean True/False values")

    # Trim whitespace from all string columns
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
    
    # 5. Remove completely empty rows
    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(how='all')
    rows_after = len(df_cleaned)
    if rows_before != rows_after:
        validation_ops.append(f"Removed {rows_before - rows_after} completely empty rows")
    
    # 6. Drop rows with > 50% missing values
    threshold = len(df_cleaned.columns) * 0.5
    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(thresh=threshold)
    rows_after = len(df_cleaned)
    if rows_before != rows_after:
        validation_ops.append(f"Removed {rows_before - rows_after} rows with >50% missing data")
    
    # 7. Remove duplicate rows
    duplicates_before = df_cleaned.duplicated().sum()
    if duplicates_before > 0:
        df_cleaned = df_cleaned.drop_duplicates(keep='first')
        validation_ops.append(f"Removed {duplicates_before} duplicate rows")
    
    # 8. Reset index to prevent trailing empty rows
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned, validation_ops


# ---------------------------
# Pre-Check: Evaluate Data Quality
# ---------------------------

def evaluate_data_quality(df: pd.DataFrame) -> dict:
    """
    Ask the AI to evaluate if the dataset needs cleaning.
    Returns: {
        "needs_cleaning": bool,
        "confidence": str ("high", "medium", "low"),
        "assessment": str (explanation),
        "issues_found": list
    }
    """
    
    # Get a sample of the data (first 20 rows for analysis)
    sample_df = df.head(20)
    csv_sample = sample_df.to_csv(index=False)
    
    # Get basic stats
    summary = _generate_summary(df)
    
    prompt = f"""
You are a data quality analyst. Evaluate this dataset and determine if it needs cleaning.

DATASET OVERVIEW:
- Rows: {summary['rows']}
- Columns: {summary['columns']}
- Missing values: {summary['missing_values']}
- Duplicate rows: {summary['duplicates']}

SAMPLE DATA (first 20 rows):
{csv_sample}

ANALYZE AND RESPOND IN THIS EXACT JSON FORMAT:
{{
  "needs_cleaning": true/false,
  "confidence": "high/medium/low",
  "assessment": "Brief explanation of your assessment",
  "issues_found": ["list", "of", "specific", "issues"]
}}

EVALUATION CRITERIA:
- Look for invalid emails (missing @, malformed)
- Look for invalid dates (wrong format, impossible dates)
- Look for invalid ages (negative, text values, unrealistic)
- Look for inconsistent capitalization in names
- Look for placeholder values like "-", "N/A", "null"
- Look for obvious typos or formatting issues
- Look for duplicate rows

CONFIDENCE LEVELS:
- "high": Very clear whether data needs cleaning or not
- "medium": Some issues present but minor or ambiguous
- "low": Difficult to determine, need more context

If the data looks clean and well-formatted, set needs_cleaning to false.
If you're unsure, set confidence to "low" or "medium".

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: basic heuristic check
        return {
            "needs_cleaning": summary['missing_values'] > 0 or summary['duplicates'] > 0,
            "confidence": "low",
            "assessment": "Unable to perform AI evaluation. Basic check shows " + 
                         ("some" if summary['missing_values'] > 0 or summary['duplicates'] > 0 else "no") + 
                         " obvious issues.",
            "issues_found": []
        }
    
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )

        raw_output = response.choices[0].message.content.strip()
        
        # Extract JSON from response (in case there's extra text)
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            raw_output = json_match.group(0)
        
        # Parse JSON
        import json
        evaluation = json.loads(raw_output)
        
        # Validate the response structure
        required_keys = ["needs_cleaning", "confidence", "assessment", "issues_found"]
        if not all(key in evaluation for key in required_keys):
            raise ValueError("Invalid evaluation response structure")
        
        return evaluation

    except Exception as e:
        # Fallback to basic heuristic
        return {
            "needs_cleaning": summary['missing_values'] > 0 or summary['duplicates'] > 0,
            "confidence": "low",
            "assessment": f"AI evaluation failed. Basic check shows {summary['missing_values']} missing values and {summary['duplicates']} duplicates.",
            "issues_found": []
        }


# ---------------------------
# Data Cleaning
# ---------------------------

def clean_dataset(df: pd.DataFrame) -> tuple:
    """
    Clean the dataset using OpenAI GPT-4o Mini with post-processing validation.
    Returns (cleaned_df, operations_log, before_summary, after_summary, cleaning_instructions, warnings)
    """

    operations = []
    warnings = []
    df_original = df.copy()
    expected_columns = list(df.columns)

    # Convert DataFrame to CSV string
    csv_data = df.to_csv(index=False)

    prompt = f"""
You are a STRICT data cleaning engine. 
You ALWAYS transform the data.

Your job:
- Fix invalid emails (must have @ and valid domain)
- Fix invalid dates (force YYYY-MM-DD format)
- Fix invalid ages (convert text → numbers, must be 1-120)
- Replace "-" or blanks with empty values
- Standardize name capitalization (Title Case)
- Remove duplicate rows
- Fix obvious typos
- If you cannot infer a value, leave it blank
- Remove rows that are completely empty
- Output ONLY CLEAN CSV. No commentary. No markdown. No explanations.
- If you couldn't clean it thouroughly, let the user know in the warnings.


STRICT RULES:
- Ages must be between 1 and 120 (set invalid ages to blank)
- Emails MUST have @ and a valid domain (set invalid emails to blank)
- Dates MUST be in YYYY-MM-DD format



THE OUTPUT **MUST**:
- Keep the SAME columns: {expected_columns}
- Have the SAME number of columns
- Contain VALID CSV ONLY
- Start immediately with the header row

NOW CLEAN THIS CSV:

{csv_data}
"""

    # Create client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=0.1
        )

        raw_output = response.choices[0].message.content.strip()
        
        # Clean the output
        csv_output = _extract_csv_from_response(raw_output)

        # Attempt CSV parsing
        df_cleaned = pd.read_csv(StringIO(csv_output))

        # Schema validation
        missing_cols = set(expected_columns) - set(df_cleaned.columns)
        extra_cols = set(df_cleaned.columns) - set(expected_columns)
        
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        if extra_cols:
            warnings.append(f"Extra columns found and removed: {extra_cols}")
            df_cleaned = df_cleaned[expected_columns]

        # Reorder to original schema
        df_cleaned = df_cleaned[expected_columns]
        
        operations.append("AI Cleaning via GPT-4o Mini")

        # Apply post-processing validation (CRITICAL IMPROVEMENT!)
        df_cleaned, validation_ops = _apply_post_processing_validation(df_cleaned)
        operations.extend(validation_ops)

    except Exception as e:
        warnings.append(f"OpenAI cleaning failed: {str(e)}")
        df_cleaned = df_original.copy()
        operations.append("Cleaning failed - returned original data")

    before_summary = _generate_summary(df_original)
    after_summary = _generate_summary(df_cleaned)

    return df_cleaned, operations, before_summary, after_summary, {"ai_cleaning": "Performed"}, warnings



# ---------------------------
# Summary Generation
# ---------------------------

def generate_ai_summary(before: dict, after: dict, operations: list) -> str:
    prompt = f"""
You are a data cleaning assistant. Write a friendly 3-sentence summary explaining what changed.

BEFORE:
- {before['rows']} rows, {before['columns']} columns
- {before['missing_values']} missing values
- {before['duplicates']} duplicate rows

AFTER:
- {after['rows']} rows, {after['columns']} columns
- {after['missing_values']} missing values
- {after['duplicates']} duplicate rows

Operations performed:
{chr(10).join([f"- {op}" for op in operations])}
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _generate_fallback_summary(before, after, operations)

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _generate_fallback_summary(before, after, operations)


def _generate_fallback_summary(before: dict, after: dict, operations: list) -> str:
    summary = f"Found {before['missing_values']} missing values and {before['duplicates']} duplicates. "
    summary += f"Applied {len(operations)} cleaning operations. "
    summary += f"Your dataset now has {after['rows']} clean rows."
    return summary