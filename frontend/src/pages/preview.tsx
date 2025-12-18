/**
 * Preview Page - Quick Apply Workflow
 *
 * Shows users what autonomous cleaning will do BEFORE applying changes.
 * This is the "preview before commit" step for the fast workflow.
 *
 * USER FLOW:
 * 1. User arrives from home page with fileId in query
 * 2. Page fetches autonomous preview (before/after comparison)
 * 3. User reviews the changes in a table
 * 4. User clicks "Apply & Download" to commit changes
 * 5. Cleaned file downloads automatically
 *
 * KEY DESIGN DECISIONS:
 * - Preview is read-only (no changes made until user clicks Apply)
 * - Shows only first N rows to keep the page responsive
 * - Changes are highlighted so user can see what's different
 * - Can always go back to home page without applying
 */
import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import Link from "next/link";
import PreviewTable from "@/components/PreviewTable";
import {
  autonomousPreview,
  autonomousApply,
  downloadFile,
  PreviewRow,
} from "@/lib/api";

export default function PreviewPage() {
  const router = useRouter();
  const { fileId } = router.query;

  const [isLoading, setIsLoading] = useState(true);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewData, setPreviewData] = useState<PreviewRow[]>([]);
  const [totalRows, setTotalRows] = useState(0);
  const [previewRows, setPreviewRows] = useState(0);

  useEffect(() => {
    if (!fileId) return;

    const fetchPreview = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await autonomousPreview(fileId as string, 100);
        setPreviewData(response.preview);
        setTotalRows(response.total_rows);
        setPreviewRows(response.preview_rows);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Preview failed");
      } finally {
        setIsLoading(false);
      }
    };

    fetchPreview();
  }, [fileId]);

  const handleApply = async () => {
    if (!fileId) return;

    setIsApplying(true);
    setError(null);

    try {
      const response = await autonomousApply(fileId as string);
      downloadFile(response.cleaned_file_id, `cleaned_data.csv`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply changes");
    } finally {
      setIsApplying(false);
    }
  };

  return (
    <main className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="w-full px-6 py-4 border-b border-neutral-200 bg-white sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <Link
            href="/"
            className="text-sm font-medium text-neutral-900 hover:text-neutral-600"
          >
            Clean My Data
          </Link>
          <span className="text-sm text-neutral-500">Quick Apply Mode</span>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Back Button */}
        {fileId && (
          <Link
            href={{
              pathname: "/",
              query: { fileId },
            }}
            className="inline-flex items-center gap-1.5 text-sm text-neutral-500 hover:text-neutral-900 transition-colors mb-6"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
            Back to issues
          </Link>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm">
            {error}
          </div>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-16">
            <div className="inline-block w-8 h-8 border-2 border-neutral-300 border-t-neutral-900 rounded-full animate-spin"></div>
            <p className="mt-4 text-neutral-500">Generating preview...</p>
          </div>
        )}

        {/* Preview Content */}
        {!isLoading && !error && (
          <div className="space-y-6">
            {/* Info Banner */}
            <div className="bg-neutral-900 rounded-2xl p-6 text-white">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-white/10 rounded-xl flex items-center justify-center flex-shrink-0">
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <div>
                  <h3 className="font-medium mb-1">
                    Rule-Based Cleaning Applied
                  </h3>
                  <p className="text-neutral-400 text-sm">
                    Automatic fixes include: name capitalization, number word
                    conversion, whitespace trimming, date normalization, and
                    duplicate removal. These are deterministic rules, not AI.
                  </p>
                </div>
              </div>
            </div>

            {/* Preview Table */}
            <div className="bg-white rounded-2xl border border-neutral-200 p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-medium text-neutral-900">Data Preview</h3>
                <span className="text-sm text-neutral-500">
                  Showing {previewRows} of {totalRows.toLocaleString()} rows
                </span>
              </div>
              <PreviewTable
                previewData={previewData}
                totalRows={totalRows}
                previewRows={previewRows}
              />
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-3">
              <Link
                href="/"
                className="px-6 py-2.5 border border-neutral-200 rounded-full text-neutral-700 hover:bg-neutral-50 transition-colors text-sm font-medium"
              >
                Cancel
              </Link>
              <button
                onClick={handleApply}
                disabled={isApplying}
                className="px-6 py-2.5 bg-neutral-900 text-white rounded-full font-medium hover:bg-neutral-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm flex items-center gap-2"
              >
                {isApplying ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Applying...
                  </>
                ) : (
                  <>
                    Apply & Download
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                      />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* No Query Params */}
        {!isLoading && !fileId && (
          <div className="text-center py-16">
            <p className="text-neutral-500">
              No file selected.{" "}
              <Link
                href="/"
                className="text-neutral-900 hover:underline font-medium"
              >
                Upload a file
              </Link>{" "}
              to get started.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
