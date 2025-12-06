import React, { useState } from "react";
import { getAIQualitySummary, AIQualitySummaryResponse } from "@/lib/api";

interface AISummaryProps {
  fileId: string;
}

export default function AISummary({ fileId }: AISummaryProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [summary, setSummary] = useState<AIQualitySummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateSummary = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await getAIQualitySummary(fileId, false);
      setSummary(result);

      if (!result.success && result.error) {
        setError(result.error);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to generate summary"
      );
    } finally {
      setIsLoading(false);
    }
  };

  const formatSummary = (text: string) => {
    const lines = text.split("\n");

    return lines.map((line, index) => {
      if (line.startsWith("**") && line.endsWith("**")) {
        const headerText = line.replace(/\*\*/g, "");
        return (
          <h4
            key={index}
            className="font-medium text-neutral-900 mt-4 mb-2 first:mt-0"
          >
            {headerText}
          </h4>
        );
      }

      if (line.includes("**")) {
        const parts = line.split(/(\*\*.*?\*\*)/);
        return (
          <p key={index} className="text-neutral-600 mb-1">
            {parts.map((part, partIndex) => {
              if (part.startsWith("**") && part.endsWith("**")) {
                return (
                  <strong key={partIndex} className="text-neutral-900">
                    {part.replace(/\*\*/g, "")}
                  </strong>
                );
              }
              return part;
            })}
          </p>
        );
      }

      if (line.trim().startsWith("- ") || line.trim().startsWith("• ")) {
        return (
          <li key={index} className="text-neutral-600 ml-4 mb-1 list-disc">
            {line.trim().substring(2)}
          </li>
        );
      }

      if (line.trim() === "") {
        return <div key={index} className="h-2" />;
      }

      return (
        <p key={index} className="text-neutral-600 mb-1">
          {line}
        </p>
      );
    });
  };

  return (
    <div className="bg-white rounded-2xl border border-neutral-200 p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-violet-100 rounded-xl flex items-center justify-center">
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
        <div>
          <h3 className="font-medium text-neutral-900">AI Data Insights</h3>
          <p className="text-sm text-neutral-500">
            Get an AI-generated analysis of your data quality
          </p>
        </div>
      </div>

      {!summary && !isLoading && (
        <div>
          <button
            onClick={handleGenerateSummary}
            disabled={isLoading}
            className="w-full py-3 px-4 rounded-xl font-medium transition-all flex items-center justify-center gap-2 bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50"
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
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
            Generate AI Insights
          </button>
          <p className="text-xs text-neutral-400 text-center mt-2">
            AI analyzes your data quality — cleaning is done by rule-based
            algorithms
          </p>
        </div>
      )}

      {isLoading && (
        <div className="text-center py-8">
          <div className="inline-block w-6 h-6 border-2 border-violet-300 border-t-violet-600 rounded-full animate-spin"></div>
          <p className="mt-3 text-sm text-neutral-500">Analyzing with AI...</p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-100 rounded-xl">
          <p className="text-red-600 text-sm">{error}</p>
          <button
            onClick={handleGenerateSummary}
            className="mt-2 text-sm text-red-600 hover:text-red-700 underline"
          >
            Try again
          </button>
        </div>
      )}

      {summary && summary.success && (
        <div className="mt-4">
          <div className="bg-neutral-50 rounded-xl p-4 border border-neutral-100">
            <div className="text-sm">{formatSummary(summary.summary)}</div>
          </div>

          <div className="flex items-center justify-between mt-4 pt-4 border-t border-neutral-100">
            <p className="text-xs text-neutral-400">
              Generated by {summary.ai_model || "AI"}
            </p>
            <button
              onClick={handleGenerateSummary}
              className="text-sm text-violet-600 hover:text-violet-700 flex items-center gap-1"
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
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
              Regenerate
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
