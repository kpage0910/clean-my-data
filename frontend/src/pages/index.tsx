import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import UploadForm from "@/components/UploadForm";
import IssuesList from "@/components/IssuesList";
import AISummary from "@/components/AISummary";
import { scanFile, DataIssue } from "@/lib/api";

interface ScanReport {
  total_rows: number;
  total_columns: number;
  issues: DataIssue[];
  column_stats: Record<string, any>;
  summary: Record<string, any>;
}

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 17) return "Good afternoon";
  if (hour >= 17 && hour < 21) return "Good evening";
  return "Good night";
}

export default function Home() {
  const router = useRouter();
  const [showUpload, setShowUpload] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [scanReport, setScanReport] = useState<ScanReport | null>(null);
  const [greeting, setGreeting] = useState("Good afternoon");

  useEffect(() => {
    setGreeting(getGreeting());
  }, []);

  // Handle returning from review/preview pages with fileId
  useEffect(() => {
    const queryFileId = router.query.fileId as string | undefined;
    if (queryFileId && !fileId && !scanReport) {
      setFileId(queryFileId);
      setShowUpload(true);
      setIsLoading(true);
      setError(null);

      scanFile(queryFileId)
        .then((response) => {
          setScanReport(response.report);
          setFilename("Uploaded file");
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to load file");
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [router.query.fileId, fileId, scanReport]);

  const handleUploadSuccess = async (
    uploadedFileId: string,
    uploadedFilename: string
  ) => {
    setFileId(uploadedFileId);
    setFilename(uploadedFilename);
    setError(null);
    setIsLoading(true);

    try {
      const response = await scanFile(uploadedFileId);
      setScanReport(response.report);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadError = (errorMessage: string) => {
    setError(errorMessage);
  };

  const handlePreview = () => {
    if (!fileId) return;
    router.push({
      pathname: "/preview",
      query: { fileId },
    });
  };

  const handleSafeReview = () => {
    if (!fileId) return;
    router.push({
      pathname: "/review",
      query: { fileId },
    });
  };

  const handleReset = () => {
    setFileId(null);
    setFilename(null);
    setScanReport(null);
    setError(null);
    setShowUpload(false);
  };

  // Landing page view
  if (!showUpload && !scanReport) {
    return (
      <main className="min-h-screen bg-neutral-50 flex flex-col">
        {/* Header */}
        <header className="w-full px-6 py-4">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <span className="text-sm font-medium text-neutral-900">
              Clean My Data
            </span>
            <span className="text-sm text-neutral-500">{greeting}</span>
          </div>
        </header>

        {/* Hero Section */}
        <div className="flex-1 flex flex-col items-center justify-center px-6 pb-24">
          <div className="text-center max-w-2xl mx-auto animate-fade-in">
            <h1 className="text-5xl md:text-6xl font-semibold text-neutral-900 tracking-tight leading-tight">
              Clean data. Simple.
            </h1>
            <p className="mt-6 text-lg text-neutral-500 max-w-md mx-auto leading-relaxed">
              Upload messy data. Get it back clean. No scripts, no hassle.
            </p>
            <button
              onClick={() => setShowUpload(true)}
              className="mt-10 inline-flex items-center gap-2 px-8 py-3 bg-neutral-900 text-white rounded-full text-sm font-medium hover:bg-neutral-800 transition-all hover:scale-105"
            >
              Get Started
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
                  d="M17 8l4 4m0 0l-4 4m4-4H3"
                />
              </svg>
            </button>
          </div>

          {/* Process Steps */}
          <div className="mt-20 flex flex-wrap items-center justify-center gap-4 animate-slide-up">
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-neutral-200 shadow-sm">
              <span className="text-lg">📁</span>
              <span className="text-sm text-neutral-600">Upload</span>
            </div>
            <span className="text-neutral-300">→</span>
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-neutral-200 shadow-sm">
              <span className="text-lg">🔧</span>
              <span className="text-sm text-neutral-600">Auto Clean</span>
            </div>
            <span className="text-neutral-300">→</span>
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-full border border-neutral-200 shadow-sm">
              <span className="text-lg">✅</span>
              <span className="text-sm text-neutral-600">Ready</span>
            </div>
          </div>

          {/* Feature Cards */}
          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto w-full animate-slide-up">
            <div className="p-6 bg-white rounded-2xl border border-neutral-200 hover:border-neutral-300 transition-colors">
              <div className="w-10 h-10 bg-neutral-100 rounded-xl flex items-center justify-center mb-4">
                <svg
                  className="w-5 h-5 text-neutral-600"
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
              <h3 className="font-medium text-neutral-900 mb-1">
                Rule-based cleaning.
              </h3>
              <p className="text-sm text-neutral-500">
                Smart algorithms, no scripts.
              </p>
            </div>
            <div className="p-6 bg-white rounded-2xl border border-neutral-200 hover:border-neutral-300 transition-colors">
              <div className="w-10 h-10 bg-violet-100 rounded-xl flex items-center justify-center mb-4">
                <svg
                  className="w-5 h-5 text-violet-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>
              <h3 className="font-medium text-neutral-900 mb-1">
                AI-powered insights.
              </h3>
              <p className="text-sm text-neutral-500">
                Get summaries & analysis.
              </p>
            </div>
            <div className="p-6 bg-white rounded-2xl border border-neutral-200 hover:border-neutral-300 transition-colors">
              <div className="w-10 h-10 bg-neutral-100 rounded-xl flex items-center justify-center mb-4">
                <svg
                  className="w-5 h-5 text-neutral-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="font-medium text-neutral-900 mb-1">
                Model-ready output.
              </h3>
              <p className="text-sm text-neutral-500">Download clean CSV.</p>
            </div>
          </div>
        </div>
      </main>
    );
  }

  // Upload view
  if (showUpload && !scanReport) {
    return (
      <main className="min-h-screen bg-neutral-50">
        {/* Header */}
        <header className="w-full px-6 py-4 border-b border-neutral-200 bg-white">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <button
              onClick={handleReset}
              className="text-sm font-medium text-neutral-900 hover:text-neutral-600 flex items-center gap-2"
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
              Clean My Data
            </button>
          </div>
        </header>

        <div className="max-w-2xl mx-auto px-6 py-16">
          <div className="text-center mb-10">
            <h2 className="text-2xl font-semibold text-neutral-900">
              Upload your file
            </h2>
            <p className="mt-2 text-neutral-500">CSV files up to 50MB</p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm">
              {error}
            </div>
          )}

          <UploadForm
            onUploadSuccess={handleUploadSuccess}
            onUploadError={handleUploadError}
            isLoading={isLoading}
          />

          {isLoading && (
            <div className="text-center mt-8">
              <div className="inline-block w-6 h-6 border-2 border-neutral-300 border-t-neutral-900 rounded-full animate-spin"></div>
              <p className="mt-3 text-sm text-neutral-500">
                Scanning your file...
              </p>
            </div>
          )}
        </div>
      </main>
    );
  }

  // Results view
  return (
    <main className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="w-full px-6 py-4 border-b border-neutral-200 bg-white sticky top-0 z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <button
            onClick={handleReset}
            className="text-sm font-medium text-neutral-900 hover:text-neutral-600 flex items-center gap-2"
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
            Clean My Data
          </button>
          <button
            onClick={handleReset}
            className="text-sm text-neutral-500 hover:text-neutral-700"
          >
            New upload
          </button>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        {error && (
          <div className="p-4 bg-red-50 border border-red-100 rounded-xl text-red-600 text-sm">
            {error}
          </div>
        )}

        {/* File Info Card */}
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
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <div className="flex-1">
              <h2 className="font-medium text-neutral-900">{filename}</h2>
              <p className="text-sm text-neutral-500 mt-0.5">
                {scanReport?.total_rows.toLocaleString()} rows ·{" "}
                {scanReport?.total_columns} columns ·{" "}
                {scanReport?.issues.length}{" "}
                {scanReport?.issues.length === 1 ? "issue" : "issues"} found
              </p>
            </div>
          </div>
        </div>

        {/* Issues Card */}
        {scanReport && scanReport.issues.length > 0 && (
          <div className="bg-white rounded-2xl border border-neutral-200 p-6">
            <h3 className="font-medium text-neutral-900 mb-4">
              Issues detected
            </h3>
            <IssuesList issues={scanReport.issues} />
          </div>
        )}

        {/* No Issues */}
        {scanReport && scanReport.issues.length === 0 && (
          <div className="bg-emerald-50 rounded-2xl border border-emerald-100 p-8 text-center">
            <div className="w-12 h-12 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg
                className="w-6 h-6 text-emerald-600"
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
            <h3 className="font-medium text-emerald-900">
              Your data looks great!
            </h3>
            <p className="text-sm text-emerald-600 mt-1">
              No issues detected in your file.
            </p>
          </div>
        )}

        {/* AI Summary */}
        {fileId && <AISummary fileId={fileId} />}

        {/* Action Buttons */}
        <div className="bg-white rounded-2xl border border-neutral-200 p-6 space-y-4">
          <div
            onClick={handleSafeReview}
            className="p-5 border-2 border-emerald-200 bg-emerald-50 rounded-xl cursor-pointer hover:border-emerald-300 transition-colors group"
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full">
                    Recommended
                  </span>
                </div>
                <h4 className="font-medium text-neutral-900 mt-2">
                  Safe Review Mode
                </h4>
                <p className="text-sm text-neutral-500 mt-1">
                  Review each fix before applying. Nothing changes without
                  approval.
                </p>
              </div>
              <div className="w-10 h-10 bg-emerald-600 rounded-full flex items-center justify-center group-hover:bg-emerald-700 transition-colors">
                <svg
                  className="w-5 h-5 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4"
                  />
                </svg>
              </div>
            </div>
          </div>

          <div
            onClick={handlePreview}
            className="p-5 border border-neutral-200 rounded-xl cursor-pointer hover:border-neutral-300 transition-colors group"
          >
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-medium text-neutral-900">Quick Apply</h4>
                <p className="text-sm text-neutral-500 mt-1">
                  Apply all suggested fixes automatically.
                </p>
              </div>
              <div className="w-10 h-10 bg-neutral-900 rounded-full flex items-center justify-center group-hover:bg-neutral-800 transition-colors">
                <svg
                  className="w-5 h-5 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
