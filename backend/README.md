# CSV Data Cleaner - Backend

FastAPI backend with AI-powered smart data cleaning capabilities.

## Features

- **Basic Analysis**: Get quick statistics about your CSV data
- **AI-Powered Smart Analysis**: Detect data quality issues and get intelligent cleaning recommendations
- **Automated Cleaning**: Apply AI-recommended transformations to clean your data

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Run the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Interactive docs: `http://localhost:8000/docs`

## Endpoints

### `POST /analyze`

Analyzes a CSV file and returns basic statistics.

**Request:** Multipart form data with a CSV file

**Response:**

```json
{
  "total_rows": 100,
  "total_columns": 5,
  "missing_values": 15,
  "duplicate_rows": 3
}
```

### `POST /smart-analyze` 🤖

AI-powered analysis that detects issues and generates cleaning recommendations.

**Request:** Multipart form data with a CSV file

**Response:**

```json
{
  "analysis": {
    "total_rows": 100,
    "total_columns": 5,
    "issues": [
      {
        "type": "missing_values",
        "column": "age",
        "severity": "high",
        "description": "15 missing values (15%)"
      }
    ],
    "issues_summary": {
      "total": 5,
      "high": 2,
      "medium": 2,
      "low": 1
    }
  },
  "recommendations": {
    "recommendations": [
      {
        "action": "fill_missing",
        "column": "age",
        "method": "mean",
        "reason": "Fill missing values with mean for numeric data",
        "priority": "high"
      }
    ],
    "summary": "Found 5 cleaning operations to improve data quality."
  }
}
```

### `POST /clean`

Cleans a CSV file using basic rule-based cleaning.

**Request:** Multipart form data with a CSV file

**Response:**

```json
{
  "csv_string": "cleaned,csv,data...",
  "rows_cleaned": 97,
  "preview": {
    "columns": ["col1", "col2"],
    "data": [
      [1, 2],
      [3, 4]
    ]
  }
}
```

### `POST /smart-clean` 🚀

Applies AI-recommended cleaning operations to the dataset.

**Request:** Multipart form data with a CSV file

**Response:**

```json
{
  "csv_string": "cleaned,csv,data...",
  "rows_cleaned": 97,
  "operations_applied": [
    "Removed 3 duplicate rows",
    "Filled 15 missing values in 'age' with mean",
    "Converted 'salary' to numeric type"
  ],
  "summary": "Found 5 cleaning operations to improve data quality.",
  "preview": {
    "columns": ["col1", "col2"],
    "data": [
      [1, 2],
      [3, 4]
    ]
  }
}
```

## AI-Powered Cleaning Operations

The AI service detects and fixes:

1. **Missing Values**: Intelligently fills with mean, median, or mode based on data type
2. **Duplicate Rows**: Identifies and removes exact duplicates
3. **Wrong Data Types**: Detects numeric data stored as text and converts it
4. **Inconsistent Formatting**: Standardizes text formatting (case, spacing)
5. **Outliers**: Can detect and handle outliers (future enhancement)

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI-powered analysis
