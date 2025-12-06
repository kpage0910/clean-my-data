import React from "react";

interface DataIssue {
  column: string;
  issue_type: string;
  severity: string;
  count: number;
  description: string;
  examples?: any[];
}

interface IssuesListProps {
  issues: DataIssue[];
}

export default function IssuesList({ issues }: IssuesListProps) {
  const issuesByColumn: Record<string, DataIssue[]> = {};
  issues.forEach((issue) => {
    if (!issuesByColumn[issue.column]) {
      issuesByColumn[issue.column] = [];
    }
    issuesByColumn[issue.column].push(issue);
  });

  const getSeverityStyles = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "high":
        return "bg-red-50 text-red-600 border-red-100";
      case "medium":
        return "bg-amber-50 text-amber-600 border-amber-100";
      case "low":
        return "bg-blue-50 text-blue-600 border-blue-100";
      default:
        return "bg-neutral-50 text-neutral-600 border-neutral-100";
    }
  };

  if (issues.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="w-12 h-12 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-3">
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
        <p className="text-neutral-600">No issues detected in your data!</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {Object.entries(issuesByColumn).map(([column, columnIssues]) => (
        <div
          key={column}
          className="border border-neutral-200 rounded-xl overflow-hidden"
        >
          <div className="bg-neutral-50 px-4 py-3 border-b border-neutral-200">
            <h4 className="text-sm font-medium text-neutral-900">
              Column: {column}
            </h4>
          </div>

          <div className="divide-y divide-neutral-100">
            {columnIssues.map((issue, idx) => (
              <div key={idx} className="p-4 bg-white">
                <div className="flex items-start gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-medium text-neutral-900">
                        {issue.issue_type.replace(/_/g, " ")}
                      </span>
                      <span
                        className={`px-2 py-0.5 text-xs rounded-full border ${getSeverityStyles(
                          issue.severity
                        )}`}
                      >
                        {issue.severity}
                      </span>
                      <span className="text-xs text-neutral-400">
                        {issue.count} rows
                      </span>
                    </div>

                    <p className="text-sm text-neutral-500 mb-2">
                      {issue.description}
                    </p>

                    {issue.examples && issue.examples.length > 0 && (
                      <div className="flex items-center gap-1 text-xs text-neutral-400">
                        <span>Examples:</span>
                        {issue.examples.slice(0, 3).map((ex, i) => (
                          <code
                            key={i}
                            className="bg-neutral-100 px-1.5 py-0.5 rounded text-neutral-600"
                          >
                            {ex === null ? "null" : String(ex)}
                          </code>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
