"""
Clean My Data - FastAPI Backend

This is the main application entry point. It defines all HTTP API routes that
the frontend (or any client) uses to upload, scan, preview, and clean CSV data.

ARCHITECTURE OVERVIEW:
─────────────────────
1. User uploads a CSV file → stored temporarily with a unique file_id
2. User requests a scan → backend analyzes data quality issues (no changes made)
3. User previews changes → see "before vs after" without committing
4. User applies changes → cleaned file saved with new file_id, available for download

TWO CLEANING MODES:
───────────────────
• Quick Apply (/autonomous-preview, /autonomous-apply):
  Automatic rule-based cleaning (capitalization, whitespace, number words, etc.)
  Good for: Fast cleanup of common issues

• Safe Review (/detect-issues, /apply-approved):
  User reviews and approves each suggested fix before applying
  Good for: Maximum control, sensitive data

FILE STORAGE:
─────────────
Files are stored in a temp directory and tracked in-memory via FILE_STORAGE dict.
This is intentionally simple—files are ephemeral and can be deleted at any time.
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# Load environment variables BEFORE importing other modules
# (some services like ai_summary.py read API keys from env)
load_dotenv()

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    UploadResponse,
    ScanRequest,
    ScanResponse,
    PreviewRequest,
    PreviewResponse,
    ApplyRequest,
    ApplyResponse,
    AutonomousCleanRequest,
    AutonomousCleanResponse,
    AutonomousPreviewRequest,
    ColumnTypeInference,
    GeneratedRuleInfo,
    DetectedIssuesReport,
    ApplyApprovedActionsRequest,
    ApplyApprovedActionsResponse,
    StrictModeConfig,
    AIQualitySummaryRequest,
    AIQualitySummaryResponse,
)
from app.services.scanner import scan_dataframe
from app.services.cleaner import apply_cleaning_rules, preview_cleaning
from app.services.autonomous import (
    autonomous_clean,
    autonomous_scan_and_suggest,
    autonomous_preview,
)
from app.services.suggestion_engine import (
    generate_suggestions,
    apply_approved_actions,
)
from app.services.ai_summary import (
    generate_data_quality_summary,
    get_raw_data_quality_analysis,
)

app = FastAPI(
    title="Clean My Data API",
    description="API for uploading, scanning, and cleaning CSV data",
    version="0.1.0",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "https://clean-my-data.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================
# In-Memory File Storage
# ============================================
# Maps file_id (UUID string) → absolute file path on disk.
# This is intentionally simple: no database, no persistence across restarts.
# For production, consider S3 or database-backed storage.
FILE_STORAGE: Dict[str, str] = {}

# Temporary directory for uploaded files
TEMP_DIR = Path(tempfile.gettempdir()) / "clean_my_data"
TEMP_DIR.mkdir(exist_ok=True)


# ============================================
# Helper Functions
# ============================================

def get_file_path(file_id: str) -> Path:
    """Get the file path for a given file_id, raising 404 if not found."""
    if file_id not in FILE_STORAGE:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    
    file_path = Path(FILE_STORAGE[file_id])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File no longer exists: {file_id}")
    
    return file_path


def load_csv(file_id: str) -> pd.DataFrame:
    """
    Load a CSV file by file_id and return as a pandas DataFrame.
    
    This is the standard way to retrieve uploaded data for processing.
    Raises HTTPException 404 if file not found, 400 if CSV is malformed.
    """
    file_path = get_file_path(file_id)
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")


# ============================================
# Core API Routes
# ============================================

@app.get("/")
async def root():
    """Health check endpoint. Returns 200 if the server is running."""
    return {"status": "ok", "message": "Clean My Data API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file for processing.
    
    This is typically the first step in any cleaning workflow:
    1. Client uploads CSV file
    2. Server validates it's a readable CSV
    3. Server returns a unique file_id for all subsequent operations
    
    Returns:
        file_id: UUID to reference this file in other endpoints
        filename: Original filename (for display purposes)
        message: Success confirmation
        
    Raises:
        400: File is not CSV, empty, or malformed
        500: Server error during upload
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Save file to temp directory
    file_path = TEMP_DIR / f"{file_id}.csv"
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Validate it's a readable CSV
        pd.read_csv(file_path, nrows=1)
        
        # Store file reference
        FILE_STORAGE[file_id] = str(file_path)
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            message="File uploaded successfully",
        )
    except pd.errors.EmptyDataError:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/scan", response_model=ScanResponse)
async def scan_file(request: ScanRequest):
    """
    Scan a CSV file to detect data quality issues.
    
    This is a READ-ONLY operation—no data is modified. The scan identifies:
    - Missing values (null, empty strings)
    - Duplicate rows
    - Mixed data types within columns
    - Format inconsistencies (emails, dates, etc.)
    - Whitespace issues
    
    The returned report helps users understand what cleaning may be needed
    before they decide which fixes to apply.
    """
    df = load_csv(request.file_id)
    
    try:
        # Call scanner service (to be implemented)
        report = scan_dataframe(df)
        return ScanResponse(
            file_id=request.file_id,
            report=report,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning file: {str(e)}")


@app.post("/preview", response_model=PreviewResponse)
async def preview_changes(request: PreviewRequest):
    """
    Preview cleaning changes without applying them.
    
    - Accepts file_id and cleaning rules
    - Applies rules to generate preview
    - Returns first N rows showing original vs cleaned data
    """
    df = load_csv(request.file_id)
    
    try:
        # Call cleaner service for preview (to be implemented)
        preview_data = preview_cleaning(
            df=df,
            rules=request.rules,
            n_rows=request.n_rows or 100,
        )
        return PreviewResponse(
            file_id=request.file_id,
            preview=preview_data,
            total_rows=len(df),
            preview_rows=min(request.n_rows or 100, len(df)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")


@app.post("/autonomous-preview", response_model=PreviewResponse)
async def autonomous_preview_changes(request: AutonomousPreviewRequest):
    """
    Preview autonomous cleaning changes without applying them.
    
    This endpoint uses the autonomous cleaning engine which:
    - Infers column types automatically
    - Applies smart cleaning (capitalization, number words, etc.)
    - Returns first N rows showing original vs cleaned data
    """
    df = load_csv(request.file_id)
    
    try:
        preview_data, result = autonomous_preview(
            df=df,
            n_rows=request.n_rows or 100,
        )
        return PreviewResponse(
            file_id=request.file_id,
            preview=preview_data,
            total_rows=len(df),
            preview_rows=min(request.n_rows or 100, len(df)),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating autonomous preview: {str(e)}")


@app.post("/autonomous-apply")
async def autonomous_apply_changes(request: AutonomousPreviewRequest):
    """
    Apply autonomous cleaning and save the cleaned CSV.
    
    This endpoint uses the autonomous cleaning engine which:
    - Infers column types automatically  
    - Applies smart cleaning (capitalization, number words, etc.)
    - Stores cleaned CSV with new file_id
    - Returns the new cleaned file_id
    """
    df = load_csv(request.file_id)
    
    try:
        cleaned_df, result = autonomous_clean(df)
        
        # Generate new file ID for cleaned data
        cleaned_file_id = str(uuid.uuid4())
        cleaned_file_path = TEMP_DIR / f"{cleaned_file_id}.csv"
        
        # Save cleaned DataFrame
        cleaned_df.to_csv(cleaned_file_path, index=False)
        
        # Store reference
        FILE_STORAGE[cleaned_file_id] = str(cleaned_file_path)
        
        return ApplyResponse(
            original_file_id=request.file_id,
            cleaned_file_id=cleaned_file_id,
            message="Autonomous cleaning applied successfully",
            rows_processed=len(cleaned_df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying autonomous cleaning: {str(e)}")


@app.post("/apply", response_model=ApplyResponse)
async def apply_changes(request: ApplyRequest):
    """
    Apply cleaning rules and save the cleaned CSV.
    
    - Accepts file_id and cleaning rules
    - Applies all specified cleaning rules
    - Stores cleaned CSV with new file_id
    - Returns the new cleaned file_id
    """
    df = load_csv(request.file_id)
    
    try:
        # Call cleaner service to apply rules (to be implemented)
        cleaned_df = apply_cleaning_rules(df=df, rules=request.rules)
        
        # Generate new file ID for cleaned data
        cleaned_file_id = str(uuid.uuid4())
        cleaned_file_path = TEMP_DIR / f"{cleaned_file_id}.csv"
        
        # Save cleaned DataFrame
        cleaned_df.to_csv(cleaned_file_path, index=False)
        
        # Store reference
        FILE_STORAGE[cleaned_file_id] = str(cleaned_file_path)
        
        return ApplyResponse(
            original_file_id=request.file_id,
            cleaned_file_id=cleaned_file_id,
            message="Cleaning rules applied successfully",
            rows_processed=len(cleaned_df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying rules: {str(e)}")


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download a CSV file by file_id.
    
    - Accepts file_id in URL path
    - Returns the CSV file for download
    """
    file_path = get_file_path(file_id)
    
    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=f"cleaned_{file_id}.csv",
    )


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file from storage.
    
    PRIVACY REQUIREMENT: Users can request deletion of their data at any time.
    This permanently removes the file from temporary storage.
    
    - Accepts file_id in URL path
    - Deletes the file from disk
    - Removes the file from the storage mapping
    - Returns confirmation of deletion
    """
    if file_id not in FILE_STORAGE:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    
    file_path = Path(FILE_STORAGE[file_id])
    
    try:
        # Delete the actual file if it exists
        if file_path.exists():
            file_path.unlink()
        
        # Remove from storage mapping
        del FILE_STORAGE[file_id]
        
        return {
            "status": "ok",
            "message": f"File {file_id} has been permanently deleted",
            "file_id": file_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.delete("/files")
async def delete_all_files():
    """
    Delete all files from storage.
    
    PRIVACY REQUIREMENT: Clear all user data from the system.
    This permanently removes all files from temporary storage.
    
    - Deletes all files from disk
    - Clears the storage mapping
    - Returns confirmation of deletion
    """
    deleted_count = 0
    errors = []
    
    for file_id, file_path_str in list(FILE_STORAGE.items()):
        try:
            file_path = Path(file_path_str)
            if file_path.exists():
                file_path.unlink()
            del FILE_STORAGE[file_id]
            deleted_count += 1
        except Exception as e:
            errors.append(f"Error deleting {file_id}: {str(e)}")
    
    return {
        "status": "ok",
        "message": f"Deleted {deleted_count} files",
        "deleted_count": deleted_count,
        "errors": errors if errors else None
    }


# ============================================
# SAFE DATA CLEANING WORKFLOW (USER APPROVAL REQUIRED)
# ============================================
# 
# This workflow follows strict principles:
# 1. NEVER invent, guess, or fabricate data
# 2. ONLY apply deterministic, meaning-preserving transformations
# 3. NEVER automatically fix or change anything without user approval
# 4. Keep all raw data intact until after user confirmation
# 5. Missing/invalid cells remain blank unless user selects a placeholder
# 6. Imputation is strictly opt-in and OFF by default
# 7. Dropping rows is strictly opt-in and OFF by default
#
# Workflow Steps:
#   Step 1: /detect-issues - Detect issues WITHOUT making changes
#   Step 2: (included in Step 1) - Generate suggested fixes
#   Step 3: /preview-actions - Preview what approved actions would do
#   Step 4: /apply-approved - Apply ONLY user-approved actions
# ============================================


@app.post("/detect-issues", response_model=DetectedIssuesReport)
async def detect_issues(file_id: str):
    """
    Step 1 & 2: Detect data quality issues and generate suggested fixes.
    
    SAFE MODE: This endpoint ONLY detects issues - NO changes are made.
    
    For each issue, the response includes:
    - Original value
    - Issue type and description
    - Available action options (leave as-is, replace with blank, etc.)
    - Recommended action (conservative default: leave as-is)
    - Preview of what each action would produce
    
    The user must then approve which actions to apply via /apply-approved.
    
    Detected issues include:
    - Missing values (no auto-fill - user chooses placeholder)
    - Invalid formats (email, date, etc.)
    - Type mismatches
    - Whitespace issues
    - Capitalization inconsistencies
    - Number words that could be converted
    - Duplicate rows
    - Empty rows
    """
    df = load_csv(file_id)
    
    try:
        report = generate_suggestions(df)
        report.file_id = file_id
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting issues: {str(e)}")


@app.post("/apply-approved", response_model=ApplyApprovedActionsResponse)
async def apply_approved_changes(request: ApplyApprovedActionsRequest):
    """
    Step 4: Apply ONLY user-approved actions.
    
    SAFE MODE: Only actions explicitly approved by the user are applied.
    Unapproved issues are left unchanged.
    
    FORBIDDEN ACTIONS (blocked even if requested):
    - Guessing names, emails, ages, or categories
    - Filling missing values with assumptions
    - Inventing replacement values
    - Generating random replacements
    - Changing the meaning of any cell
    
    ALLOWED ACTIONS:
    - leave_as_is: Keep original value (default)
    - replace_with_blank: Clear the cell
    - replace_with_placeholder: Use user-chosen placeholder (e.g., "Unknown")
    - apply_deterministic_fix: Apply reversible, meaning-preserving change
    - drop_row: Remove row (only if it qualifies - empty or corrupt)
    
    ROW DROPPING RULES:
    Rows can only be dropped if:
    1. The row is completely empty
    2. The row has fewer columns than header (corrupt)
    3. The row violates schema rules that cannot be repaired
    4. The user explicitly requested it AND it qualifies
    """
    df = load_csv(request.file_id)
    
    try:
        cleaned_df, summary = apply_approved_actions(
            df,
            request.approved_actions,
            request.strict_config
        )
        
        # Generate new file ID for cleaned data
        cleaned_file_id = str(uuid.uuid4())
        cleaned_file_path = TEMP_DIR / f"{cleaned_file_id}.csv"
        
        # Save cleaned DataFrame
        cleaned_df.to_csv(cleaned_file_path, index=False)
        
        # Store reference
        FILE_STORAGE[cleaned_file_id] = str(cleaned_file_path)
        
        return ApplyApprovedActionsResponse(
            original_file_id=request.file_id,
            cleaned_file_id=cleaned_file_id,
            actions_applied=summary["actions_applied"],
            actions_skipped=summary["actions_skipped"],
            rows_modified=summary["rows_modified"],
            rows_dropped=summary["rows_dropped"],
            applied_summary=summary["applied_summary"],
            skipped_summary=summary["skipped_summary"],
            warnings=summary["warnings"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying actions: {str(e)}")


@app.post("/autonomous-clean", response_model=AutonomousCleanResponse)
async def autonomous_clean_file(request: AutonomousCleanRequest):
    """
    Perform fully autonomous data cleaning.
    
    This endpoint:
    - Infers column types automatically
    - Generates cleaning rules based on best practices
    - Applies transformations automatically (unless preview_only=True)
    - Flags unsafe inferences
    - Returns cleaned dataset and comprehensive validation report
    
    The engine operates with full autonomy - no user configuration required.
    All cleaning decisions are made based on industry best practices.
    
    Best-practice cleaning rules applied:
    - NAME columns: Title Case, strip whitespace, remove invalid chars, placeholder for missing
    - NUMERIC columns: Convert number words to digits, fix typos, strip symbols
    - DATE columns: Standardize to ISO (YYYY-MM-DD), flag impossible dates
    - EMAIL columns: Lowercase, strip whitespace, validate format, mark invalid
    - CATEGORICAL columns: Normalize capitalization, merge duplicates
    - MISSING data: Infer only if safe, otherwise flag as UNABLE_TO_INFER
    """
    df = load_csv(request.file_id)
    
    try:
        if request.preview_only:
            # Just scan and suggest, don't apply
            result = autonomous_scan_and_suggest(df)
            
            return AutonomousCleanResponse(
                file_id=request.file_id,
                cleaned_file_id=None,
                summary=result.summary,
                column_inferences=[
                    ColumnTypeInference(
                        column=inf.column,
                        inferred_type=inf.inferred_type.value,
                        confidence=inf.confidence,
                        indicators=inf.indicators,
                        is_safe=inf.is_safe,
                        warning=inf.warning
                    )
                    for inf in result.column_inferences
                ],
                generated_rules=[
                    GeneratedRuleInfo(
                        rule_type=r.rule.rule_type,
                        column=r.column,
                        params=r.rule.params,
                        reason=r.reason,
                        priority=r.priority,
                        is_safe=r.is_safe
                    )
                    for r in result.generated_rules
                ],
                validation_report=result.validation_report,
                warnings=result.warnings,
                rows_processed=result.rows_processed,
                columns_processed=result.columns_processed
            )
        else:
            # Full autonomous cleaning
            cleaned_df, result = autonomous_clean(df)
            
            # Generate new file ID for cleaned data
            cleaned_file_id = str(uuid.uuid4())
            cleaned_file_path = TEMP_DIR / f"{cleaned_file_id}.csv"
            
            # Save cleaned DataFrame
            cleaned_df.to_csv(cleaned_file_path, index=False)
            
            # Store reference
            FILE_STORAGE[cleaned_file_id] = str(cleaned_file_path)
            
            return AutonomousCleanResponse(
                file_id=request.file_id,
                cleaned_file_id=cleaned_file_id,
                summary=result.summary,
                column_inferences=[
                    ColumnTypeInference(
                        column=inf.column,
                        inferred_type=inf.inferred_type.value,
                        confidence=inf.confidence,
                        indicators=inf.indicators,
                        is_safe=inf.is_safe,
                        warning=inf.warning
                    )
                    for inf in result.column_inferences
                ],
                generated_rules=[
                    GeneratedRuleInfo(
                        rule_type=r.rule.rule_type,
                        column=r.column,
                        params=r.rule.params,
                        reason=r.reason,
                        priority=r.priority,
                        is_safe=r.is_safe
                    )
                    for r in result.generated_rules
                ],
                validation_report=result.validation_report,
                warnings=result.warnings,
                rows_processed=result.rows_processed,
                columns_processed=result.columns_processed
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in autonomous cleaning: {str(e)}")


@app.post("/ai-quality-summary", response_model=AIQualitySummaryResponse)
async def ai_quality_summary(request: AIQualitySummaryRequest):
    """
    Generate an AI-powered natural-language summary of data quality issues.
    
    This endpoint analyzes a CSV file and uses OpenAI's GPT models to generate
    a comprehensive, readable summary of data quality issues including:
    
    - Missing value analysis and sparsity patterns
    - Duplicate record counts
    - Column-level issues (mixed types, inconsistent categories)
    - Outlier detection for numeric columns
    - Notable patterns and warnings
    
    NOTE: This endpoint does NOT clean the data - it only analyzes and reports.
    
    Requires OPENAI_API_KEY environment variable to be set.
    """
    df = load_csv(request.file_id)
    
    try:
        result = generate_data_quality_summary(
            df=df,
            model=request.model or "gpt-4.1",
            include_raw_analysis=request.include_raw_analysis or False
        )
        
        return AIQualitySummaryResponse(
            file_id=request.file_id,
            summary=result.get("summary", ""),
            success=result.get("success", False),
            ai_model=result.get("model_used"),
            analysis=result.get("analysis"),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI summary: {str(e)}")


@app.get("/ai-quality-analysis/{file_id}")
async def ai_quality_analysis(file_id: str):
    """
    Get raw data quality analysis statistics without AI summary.
    
    Useful for debugging or when you want to see the exact statistics
    that would be sent to the AI model without making an API call.
    
    Returns comprehensive statistics about:
    - Basic dataset info (rows, columns, memory usage)
    - Missing value statistics per column
    - Duplicate row analysis
    - Column type detection and mixed-type issues
    - Category anomalies (inconsistent values)
    - Outlier detection for numeric columns
    """
    df = load_csv(file_id)
    
    try:
        analysis = get_raw_data_quality_analysis(df)
        return {
            "file_id": file_id,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
