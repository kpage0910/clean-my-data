/**
 * API Client for Clean My Data Backend
 *
 * This module provides typed functions for all backend API calls.
 * It handles request formatting, error handling, and response parsing.
 *
 * USAGE PATTERN:
 * All functions are async and throw on error. Wrap calls in try/catch.
 *
 * WORKFLOW OVERVIEW:
 * 1. uploadFile() → Get a file_id
 * 2. scanFile() → See what issues exist
 * 3. Choose a cleaning approach:
 *    - Quick Apply: autonomousPreview() → autonomousApply()
 *    - Safe Review: detectIssues() → applyApprovedActions()
 * 4. downloadFile() → Get the cleaned CSV
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ============================================
// Type Definitions
// ============================================
// These match the backend's Pydantic schemas exactly.

export interface UploadResponse {
  file_id: string;
  filename: string;
  message: string;
}

export interface DataIssue {
  column: string;
  issue_type: string;
  severity: string;
  count: number;
  description: string;
  examples?: any[];
}

export interface ScanReport {
  total_rows: number;
  total_columns: number;
  issues: DataIssue[];
  column_stats: Record<string, any>;
  summary: Record<string, any>;
}

export interface ScanResponse {
  file_id: string;
  report: ScanReport;
}

export interface CleaningRule {
  rule_type: string;
  column?: string;
  params?: Record<string, any>;
}

export interface PreviewRow {
  row_index: number;
  original: Record<string, any>;
  cleaned: Record<string, any>;
  changes: string[];
}

export interface PreviewResponse {
  file_id: string;
  preview: PreviewRow[];
  total_rows: number;
  preview_rows: number;
}

// ============================================
// Core API Functions
// ============================================

/**
 * Upload a CSV file to the backend for processing.
 *
 * This is the first step in any cleaning workflow. The returned file_id
 * is used to reference this data in all subsequent operations.
 *
 * @param file - The CSV file to upload
 * @returns Promise with file_id for referencing this file
 * @throws Error if upload fails or file is invalid
 */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

/**
 * Scan an uploaded file for data quality issues.
 *
 * This is a READ-ONLY operation - no data is modified.
 * Use this to show users what problems exist before cleaning.
 *
 * @param fileId - The file_id from uploadFile()
 * @returns Promise with scan report containing detected issues
 */
export async function scanFile(fileId: string): Promise<ScanResponse> {
  const response = await fetch(`${API_BASE_URL}/scan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ file_id: fileId }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Scan failed");
  }

  return response.json();
}

// ============================================
// Quick Apply Workflow (Autonomous Cleaning)
// ============================================
// These functions power the "Quick Apply" mode where the system
// automatically applies safe transformations without per-cell approval.

/**
 * Preview what autonomous cleaning would do (without applying changes).
 *
 * Use this to show users a "before vs after" comparison before committing.
 * The preview shows how values would change for the first N rows.
 *
 * @param fileId - The file_id to preview
 * @param nRows - Number of rows to include in preview (default: 100)
 * @returns Promise with preview rows showing original vs cleaned values
 */
export async function autonomousPreview(
  fileId: string,
  nRows: number = 100
): Promise<PreviewResponse> {
  const response = await fetch(`${API_BASE_URL}/autonomous-preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      n_rows: nRows,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Preview failed");
  }

  return response.json();
}

/**
 * Apply autonomous cleaning and get a new cleaned file.
 *
 * This commits the changes previewed in autonomousPreview().
 * The original file is preserved; a new file_id is returned for the cleaned version.
 *
 * @param fileId - The file_id to clean
 * @returns Promise with the cleaned_file_id for downloading
 */
export async function autonomousApply(fileId: string): Promise<ApplyResponse> {
  const response = await fetch(`${API_BASE_URL}/autonomous-apply`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Apply failed");
  }

  return response.json();
}

export interface ApplyResponse {
  original_file_id: string;
  cleaned_file_id: string;
  message: string;
  rows_processed: number;
}

// ============================================
// Manual Rules Workflow (Legacy)
// ============================================
// These functions allow applying specific cleaning rules manually.
// Mostly used for advanced users who want fine-grained control.

/**
 * Preview what specific cleaning rules would do.
 * @deprecated Prefer autonomousPreview() for most use cases
 */
export async function previewClean(
  fileId: string,
  rules: CleaningRule[],
  nRows: number = 100
): Promise<PreviewResponse> {
  const response = await fetch(`${API_BASE_URL}/preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      rules,
      n_rows: nRows,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Preview failed");
  }

  return response.json();
}

/**
 * Apply specific cleaning rules and get a new cleaned file.
 * @deprecated Prefer autonomousApply() for most use cases
 */
export async function applyClean(
  fileId: string,
  rules: CleaningRule[]
): Promise<ApplyResponse> {
  const response = await fetch(`${API_BASE_URL}/apply`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      rules,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Apply failed");
  }

  return response.json();
}

// ============================================
// File Download
// ============================================

/**
 * Download a cleaned CSV file by triggering a browser download.
 *
 * This creates a temporary link element and clicks it to start the download.
 * Works in all modern browsers.
 *
 * @param fileId - The file_id to download
 * @param filename - Optional custom filename (default: cleaned_{fileId}.csv)
 */
export function downloadFile(fileId: string, filename?: string): void {
  const downloadUrl = `${API_BASE_URL}/download/${fileId}`;

  // Create a temporary anchor element to trigger download
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = filename || `cleaned_${fileId}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ============================================
// Autonomous Cleaning Types & Functions
// ============================================

export interface ColumnTypeInference {
  column: string;
  inferred_type: string;
  confidence: number;
  indicators: string[];
  is_safe: boolean;
  warning?: string;
}

export interface GeneratedRuleInfo {
  rule_type: string;
  column?: string;
  params: Record<string, any>;
  reason: string;
  priority: number;
  is_safe: boolean;
}

export interface AutonomousCleanResponse {
  file_id: string;
  cleaned_file_id?: string;
  summary: {
    total_rows: number;
    total_columns: number;
    issues_detected?: number;
    issues_fixed?: number;
    issues_unfixed?: number;
    rules_applied?: number;
    rules_to_apply?: number;
    column_types_inferred: Record<string, string>;
  };
  column_inferences: ColumnTypeInference[];
  generated_rules: GeneratedRuleInfo[];
  validation_report: {
    fixed?: Array<{
      column: string;
      issue_type: string;
      description: string;
      fix: string;
    }>;
    unfixed?: Array<{
      column: string;
      issue_type: string;
      description: string;
      reason: string;
      affected_rows: number;
    }>;
    warnings?: string[];
    preview_mode?: boolean;
  };
  warnings: string[];
  rows_processed: number;
  columns_processed: number;
}

/**
 * Perform fully autonomous data cleaning
 *
 * This function:
 * - Infers column types automatically
 * - Generates cleaning rules based on best practices
 * - Applies transformations automatically (unless preview_only=true)
 * - Flags unsafe inferences
 * - Returns cleaned dataset and comprehensive validation report
 */
export async function autonomousClean(
  fileId: string,
  previewOnly: boolean = false
): Promise<AutonomousCleanResponse> {
  const response = await fetch(`${API_BASE_URL}/autonomous-clean`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      preview_only: previewOnly,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Autonomous cleaning failed");
  }

  return response.json();
}

// ============================================
// Safe Review Workflow Types & Functions
// ============================================
// These power the "Safe Review" mode where users approve each change.

/**
 * Available actions for fixing issues.
 *
 * CONSERVATIVE BY DEFAULT:
 * - "leave_as_is": Keep the original value (safest)
 * - "replace_with_blank": Clear the cell
 * - "replace_with_placeholder": Use a placeholder like "Unknown"
 * - "apply_deterministic_fix": Apply a reversible, meaning-preserving fix
 * - "drop_row": Remove the row entirely (only for empty/corrupt rows)
 */
export type SuggestedAction =
  | "leave_as_is"
  | "replace_with_blank"
  | "replace_with_placeholder"
  | "apply_deterministic_fix"
  | "drop_row";

/**
 * Reasons a row can qualify for dropping (conservative rules)
 */
export type RowDropReason =
  | "completely_empty"
  | "structural_corruption"
  | "unrepairable_schema_violation"
  | "user_defined_rule";

/**
 * A suggested fix for a single cell issue
 */
export interface CellIssueSuggestion {
  row_index: number;
  column: string;
  original_value: any;
  issue_type: string;
  issue_description: string;
  available_actions: SuggestedAction[];
  recommended_action: SuggestedAction;
  action_previews: Record<string, any>;
  deterministic_fix_value?: any;
  deterministic_fix_explanation?: string;
}

/**
 * A suggested fix for a row-level issue
 */
export interface RowIssueSuggestion {
  row_index: number;
  row_data: Record<string, any>;
  issue_type: string;
  issue_description: string;
  drop_reason?: RowDropReason;
  drop_recommended: boolean;
  available_actions: SuggestedAction[];
}

/**
 * A suggested fix for a column-level issue (e.g., boolean standardization)
 */
export interface ColumnIssueSuggestion {
  column: string;
  issue_type: string;
  description: string;
  suggested_action: string;
  available_formats: string[];
  default_format?: string;
}

/**
 * Complete report of detected issues with suggestions
 */
export interface DetectedIssuesReport {
  file_id: string;
  total_rows: number;
  total_columns: number;
  total_issues: number;
  cell_issues_count: number;
  row_issues_count: number;
  cell_issues: CellIssueSuggestion[];
  row_issues: RowIssueSuggestion[];
  column_issues: ColumnIssueSuggestion[];
  column_inferences: ColumnTypeInference[];
  warnings: string[];
}

/**
 * A user-approved action to apply
 */
export interface UserApprovedAction {
  row_index?: number | null;
  column?: string;
  action: SuggestedAction;
  custom_placeholder?: string;
  target_format?: string;
}

/**
 * Response after applying approved actions
 */
export interface ApplyApprovedResponse {
  original_file_id: string;
  cleaned_file_id: string;
  actions_applied: number;
  actions_skipped: number;
  rows_modified: number;
  rows_dropped: number;
  applied_summary: Array<Record<string, any>>;
  skipped_summary: Array<Record<string, any>>;
  warnings: string[];
}

/**
 * Step 1 & 2: Detect issues and generate suggestions (NO CHANGES MADE)
 *
 * This function:
 * - Detects all data quality issues
 * - Generates suggested fixes with action options
 * - Returns report for user review
 *
 * NOTHING is changed until user explicitly approves via applyApprovedActions()
 */
export async function detectIssues(
  fileId: string
): Promise<DetectedIssuesReport> {
  const response = await fetch(
    `${API_BASE_URL}/detect-issues?file_id=${fileId}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Issue detection failed");
  }

  return response.json();
}

/**
 * Step 4: Apply ONLY user-approved actions
 *
 * SAFE MODE: Only actions explicitly approved by the user are applied.
 * Unapproved issues are left unchanged.
 *
 * FORBIDDEN ACTIONS (blocked even if requested):
 * - Guessing names, emails, ages, or categories
 * - Filling missing values with assumptions
 * - Inventing replacement values
 *
 * ALLOWED ACTIONS:
 * - leave_as_is: Keep original value (default)
 * - replace_with_blank: Clear the cell
 * - replace_with_placeholder: Use user-chosen placeholder
 * - apply_deterministic_fix: Apply reversible, meaning-preserving change
 * - drop_row: Remove row (only if it qualifies - empty or corrupt)
 */
export async function applyApprovedActions(
  fileId: string,
  approvedActions: UserApprovedAction[]
): Promise<ApplyApprovedResponse> {
  const response = await fetch(`${API_BASE_URL}/apply-approved`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      approved_actions: approvedActions,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Apply failed");
  }

  return response.json();
}

/**
 * Delete a file from storage
 *
 * PRIVACY: Permanently removes user data from the system.
 * Call this after exporting to ensure data is not retained.
 */
export async function deleteFile(
  fileId: string
): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/files/${fileId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Delete failed");
  }

  return response.json();
}

/**
 * Delete all files from storage
 *
 * PRIVACY: Clears all user data from the system.
 * Useful for bulk cleanup.
 */
export async function deleteAllFiles(): Promise<{
  status: string;
  message: string;
  deleted_count: number;
}> {
  const response = await fetch(`${API_BASE_URL}/files`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Delete all failed");
  }

  return response.json();
}

// ============================================
// AI Data Quality Summary
// ============================================

export interface AIQualitySummaryResponse {
  file_id: string;
  summary: string;
  success: boolean;
  ai_model?: string;
  analysis?: Record<string, any>;
  error?: string;
}

/**
 * Generate an AI-powered natural-language summary of data quality issues
 *
 * This uses OpenAI's GPT models to analyze the dataset and provide
 * a readable summary of issues, patterns, and recommendations.
 */
export async function getAIQualitySummary(
  fileId: string,
  includeRawAnalysis: boolean = false
): Promise<AIQualitySummaryResponse> {
  const response = await fetch(`${API_BASE_URL}/ai-quality-summary`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      file_id: fileId,
      include_raw_analysis: includeRawAnalysis,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "AI Summary generation failed");
  }

  return response.json();
}

/**
 * Get raw data quality analysis without AI summary
 *
 * Returns detailed statistics about data quality issues
 * without making an OpenAI API call.
 */
export async function getRawQualityAnalysis(
  fileId: string
): Promise<{ file_id: string; analysis: Record<string, any> }> {
  const response = await fetch(
    `${API_BASE_URL}/ai-quality-analysis/${fileId}`,
    {
      method: "GET",
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Analysis failed");
  }

  return response.json();
}
