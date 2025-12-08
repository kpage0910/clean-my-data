import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import Link from "next/link";
import IssueApprovalList from "@/components/IssueApprovalList";
import {
  detectIssues,
  applyApprovedActions,
  downloadFile,
  DetectedIssuesReport,
  UserApprovedAction,
} from "@/lib/api";

export default function ReviewPage() {
  const router = useRouter();
  const { fileId } = router.query;

  const [isLoading, setIsLoading] = useState(true);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<DetectedIssuesReport | null>(null);
  const [approvedActions, setApprovedActions] = useState<UserApprovedAction[]>(
    []
  );
  const [applyResult, setApplyResult] = useState<{
    success: boolean;
    cleanedFileId?: string;
    applied: number;
    skipped: number;
    warnings: string[];
  } | null>(null);

  useEffect(() => {
    if (!fileId) return;

    const fetchIssues = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const issuesReport = await detectIssues(fileId as string);
        setReport(issuesReport);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to detect issues"
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchIssues();
  }, [fileId]);

  const handleApply = async () => {
    if (!fileId || approvedActions.length === 0) return;

    setIsApplying(true);
    setError(null);

    try {
      const result = await applyApprovedActions(
        fileId as string,
        approvedActions
      );
      setApplyResult({
        success: true,
        cleanedFileId: result.cleaned_file_id,
        applied: result.actions_applied,
        skipped: result.actions_skipped,
        warnings: result.warnings,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply changes");
    } finally {
      setIsApplying(false);
    }
  };

  const handleDownload = () => {
    if (applyResult?.cleanedFileId) {
      downloadFile(applyResult.cleanedFileId, "cleaned_data.csv");
    }
  };

  const actionCounts = approvedActions.reduce((acc, action) => {
    acc[action.action] = (acc[action.action] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <main className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="w-full px-6 py-4 border-b border-neutral-200 bg-white sticky top-0 z-10">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <Link
            href="/"
            className="text-sm font-medium text-neutral-900 hover:text-neutral-600"
          >
            Clean My Data
          </Link>
          <span className="text-sm text-neutral-500">Safe Review Mode</span>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Back Button */}
        {fileId && !applyResult?.success && (
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

        {/* Safety Notice */}
        <div className="mb-6 p-4 bg-emerald-50 border border-emerald-100 rounded-xl">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 bg-emerald-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <svg
                className="w-4 h-4 text-emerald-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                />
              </svg>
            </div>
            <div>
              <h3 className="text-sm font-medium text-emerald-900">
                Safe Mode Active
              </h3>
              <p className="text-sm text-emerald-700 mt-0.5">
                No changes are made until you explicitly approve each action.
              </p>
            </div>
          </div>
        </div>

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
            <p className="mt-4 text-neutral-500">Detecting issues...</p>
            <p className="text-sm text-neutral-400 mt-1">
              (No changes are being made)
            </p>
          </div>
        )}

        {/* Success State */}
        {applyResult?.success && (
          <div className="space-y-6">
            <div className="bg-white border border-neutral-200 rounded-2xl p-8 text-center">
              <div className="w-14 h-14 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-7 h-7 text-emerald-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-neutral-900">
                Changes Applied!
              </h3>
              <p className="text-neutral-500 mt-2">
                {applyResult.applied} actions applied, {applyResult.skipped}{" "}
                skipped
              </p>

              {applyResult.warnings.length > 0 && (
                <div className="mt-6 text-left bg-amber-50 border border-amber-100 rounded-xl p-4">
                  <h4 className="text-sm font-medium text-amber-800 mb-2">
                    Warnings
                  </h4>
                  <ul className="text-sm text-amber-700 space-y-1">
                    {applyResult.warnings.map((warning, idx) => (
                      <li key={idx}>• {warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              <button
                onClick={handleDownload}
                className="mt-6 px-6 py-3 bg-neutral-900 text-white rounded-full font-medium hover:bg-neutral-800 transition-colors inline-flex items-center gap-2"
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
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                Download Cleaned File
              </button>
            </div>

            <div className="text-center">
              <Link
                href="/"
                className="text-neutral-500 hover:text-neutral-700 text-sm"
              >
                ← Clean another file
              </Link>
            </div>
          </div>
        )}

        {/* Issue Review */}
        {!isLoading && !applyResult?.success && report && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="bg-white rounded-2xl border border-neutral-200 p-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-neutral-100 rounded-xl flex items-center justify-center">
                  <svg
                    className="w-6 h-6 text-neutral-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                    />
                  </svg>
                </div>
                <div>
                  <h2 className="font-medium text-neutral-900">File Summary</h2>
                  <p className="text-sm text-neutral-500 mt-0.5">
                    {report.total_rows.toLocaleString()} rows ·{" "}
                    {report.total_columns} columns · {report.total_issues}{" "}
                    issues
                  </p>
                </div>
              </div>
            </div>

            {/* Issues List with Approval Controls */}
            <div className="bg-white rounded-2xl border border-neutral-200 p-6">
              <h3 className="font-medium text-neutral-900 mb-4">
                Review & Select Actions
              </h3>
              <IssueApprovalList
                cellIssues={report.cell_issues}
                rowIssues={report.row_issues}
                columnIssues={report.column_issues || []}
                onApprovalChange={setApprovedActions}
                warnings={report.warnings}
              />
            </div>

            {/* Action Summary & Apply Button */}
            <div className="bg-white rounded-2xl border border-neutral-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-neutral-900">
                    Selected Actions
                  </h3>
                  <div className="flex gap-3 mt-2 text-xs text-neutral-500">
                    {Object.entries(actionCounts).length > 0 ? (
                      Object.entries(actionCounts).map(([action, count]) => (
                        <span
                          key={action}
                          className="px-2 py-1 bg-neutral-100 rounded-full"
                        >
                          {action.replace(/_/g, " ")}: {count}
                        </span>
                      ))
                    ) : (
                      <span className="text-neutral-400">
                        No actions selected
                      </span>
                    )}
                  </div>
                </div>
                <button
                  onClick={handleApply}
                  disabled={isApplying || approvedActions.length === 0}
                  className="px-6 py-2.5 bg-emerald-600 text-white rounded-full font-medium hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm"
                >
                  {isApplying ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      Applying...
                    </>
                  ) : (
                    <>
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
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      Apply Changes
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* No File Selected */}
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
