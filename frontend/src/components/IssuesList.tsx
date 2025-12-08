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

const getIssueIcon = (issueType: string) => {
  switch (issueType) {
    case "missing_values":
      return (
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
            d="M20 12H4"
          />
        </svg>
      );
    case "boolean_inconsistency":
      return (
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
            d="M8 9l4-4 4 4m0 6l-4 4-4-4"
          />
        </svg>
      );
    case "mixed_types":
      return (
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
            d="M4 6h16M4 12h16m-7 6h7"
          />
        </svg>
      );
    case "invalid_email":
    case "invalid_date":
      return (
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
            d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      );
    case "whitespace":
      return (
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
            d="M4 6h16M4 12h8m-8 6h16"
          />
        </svg>
      );
    case "duplicate_rows":
      return (
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
            d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
      );
    default:
      return (
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
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      );
  }
};

const getIssueLabel = (issueType: string) => {
  switch (issueType) {
    case "boolean_inconsistency":
      return "Boolean Inconsistency";
    case "missing_values":
      return "Missing Values";
    case "mixed_types":
      return "Mixed Types";
    case "invalid_email":
      return "Invalid Email";
    case "invalid_date":
      return "Invalid Date";
    case "whitespace":
      return "Whitespace Issues";
    case "number_formatting":
      return "Number Formatting";
    case "number_words":
      return "Number Words";
    case "duplicate_rows":
      return "Duplicate Rows";
    default:
      return issueType.replace(/_/g, " ");
  }
};

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
                  <div
                    className={`mt-0.5 ${
                      issue.issue_type === "boolean_inconsistency"
                        ? "text-amber-500"
                        : "text-neutral-400"
                    }`}
                  >
                    {getIssueIcon(issue.issue_type)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-medium text-neutral-900">
                        {getIssueLabel(issue.issue_type)}
                      </span>
                      <span
                        className={`px-2 py-0.5 text-xs rounded-full border ${getSeverityStyles(
                          issue.severity
                        )}`}
                      >
                        {issue.severity}
                      </span>
                      <span className="text-xs text-neutral-400">
                        {issue.count}{" "}
                        {issue.issue_type === "duplicate_rows"
                          ? "rows"
                          : "values"}
                      </span>
                    </div>

                    <p className="text-sm text-neutral-500 mb-2">
                      {issue.description}
                    </p>

                    {issue.examples && issue.examples.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1 text-xs text-neutral-400">
                        <span>
                          {issue.issue_type === "boolean_inconsistency"
                            ? "Formats found:"
                            : "Examples:"}
                        </span>
                        {issue.examples.slice(0, 5).map((ex, i) => (
                          <code
                            key={i}
                            className={`px-1.5 py-0.5 rounded ${
                              issue.issue_type === "boolean_inconsistency"
                                ? "bg-amber-50 text-amber-700 border border-amber-200"
                                : "bg-neutral-100 text-neutral-600"
                            }`}
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
