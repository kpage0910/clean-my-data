import React, { useState, useEffect } from "react";
import {
  CellIssueSuggestion,
  RowIssueSuggestion,
  ColumnIssueSuggestion,
  SuggestedAction,
  UserApprovedAction,
} from "@/lib/api";

interface IssueApprovalListProps {
  cellIssues: CellIssueSuggestion[];
  rowIssues: RowIssueSuggestion[];
  columnIssues?: ColumnIssueSuggestion[];
  onApprovalChange: (approvedActions: UserApprovedAction[]) => void;
  warnings: string[];
}

const ACTION_LABELS: Record<SuggestedAction, string> = {
  leave_as_is: "Keep original",
  replace_with_blank: "Make blank",
  replace_with_placeholder: "Use placeholder",
  apply_deterministic_fix: "Apply fix",
  drop_row: "Remove row",
};

const BOOLEAN_FORMAT_OPTIONS = ["True/False", "Yes/No", "1/0"];

const ISSUE_ICONS: Record<string, string> = {
  missing_value: "❓",
  whitespace: "⎵",
  capitalization: "Aa",
  number_word: "🔢",
  invalid_format: "⚠️",
  currency_format: "💰",
  percentage_format: "%",
  invalid_date: "📅",
  empty_row: "🗑️",
  duplicate_row: "👥",
  boolean_inconsistency: "⚡",
};

export default function IssueApprovalList({
  cellIssues,
  rowIssues,
  columnIssues = [],
  onApprovalChange,
  warnings,
}: IssueApprovalListProps) {
  const [cellActions, setCellActions] = useState<
    Record<string, { action: SuggestedAction; placeholder?: string }>
  >(() => {
    const initial: Record<string, { action: SuggestedAction }> = {};
    cellIssues.forEach((issue) => {
      const key = `${issue.row_index}-${issue.column}`;
      initial[key] = { action: issue.recommended_action };
    });
    return initial;
  });

  const [rowActions, setRowActions] = useState<
    Record<number, { action: SuggestedAction }>
  >(() => {
    const initial: Record<number, { action: SuggestedAction }> = {};
    rowIssues.forEach((issue) => {
      initial[issue.row_index] = {
        action: issue.drop_recommended ? "drop_row" : "leave_as_is",
      };
    });
    return initial;
  });

  const [columnActions, setColumnActions] = useState<
    Record<string, { action: SuggestedAction; targetFormat?: string }>
  >(() => {
    const initial: Record<
      string,
      { action: SuggestedAction; targetFormat?: string }
    > = {};
    columnIssues.forEach((issue) => {
      initial[issue.column] = {
        action: "apply_deterministic_fix",
        targetFormat: issue.default_format || "True/False",
      };
    });
    return initial;
  });

  const [expandedIssues, setExpandedIssues] = useState<Set<string>>(new Set());
  const [customPlaceholder, setCustomPlaceholder] = useState("Unknown");

  const buildApprovedActions = (): UserApprovedAction[] => {
    const actions: UserApprovedAction[] = [];

    // Column-level actions first
    columnIssues.forEach((issue) => {
      const selected = columnActions[issue.column];
      if (selected && selected.action !== "leave_as_is") {
        actions.push({
          row_index: null,
          column: issue.column,
          action: selected.action,
          target_format: selected.targetFormat,
        });
      }
    });

    cellIssues.forEach((issue) => {
      const key = `${issue.row_index}-${issue.column}`;
      const selected = cellActions[key];
      if (selected) {
        actions.push({
          row_index: issue.row_index,
          column: issue.column,
          action: selected.action,
          custom_placeholder:
            selected.action === "replace_with_placeholder"
              ? customPlaceholder
              : undefined,
        });
      }
    });

    rowIssues.forEach((issue) => {
      const selected = rowActions[issue.row_index];
      if (selected) {
        actions.push({
          row_index: issue.row_index,
          action: selected.action,
        });
      }
    });

    return actions;
  };

  const handleCellActionChange = (
    rowIndex: number,
    column: string,
    action: SuggestedAction
  ) => {
    const key = `${rowIndex}-${column}`;
    setCellActions((prev) => ({ ...prev, [key]: { action } }));
  };

  const handleRowActionChange = (rowIndex: number, action: SuggestedAction) => {
    setRowActions((prev) => ({ ...prev, [rowIndex]: { action } }));
  };

  const handleColumnActionChange = (
    column: string,
    action: SuggestedAction,
    targetFormat?: string
  ) => {
    setColumnActions((prev) => ({
      ...prev,
      [column]: {
        action,
        targetFormat:
          targetFormat || prev[column]?.targetFormat || "True/False",
      },
    }));
  };

  const handleColumnFormatChange = (column: string, targetFormat: string) => {
    setColumnActions((prev) => ({
      ...prev,
      [column]: { ...prev[column], targetFormat },
    }));
  };

  const toggleExpanded = (key: string) => {
    setExpandedIssues((prev) => {
      const updated = new Set(prev);
      if (updated.has(key)) {
        updated.delete(key);
      } else {
        updated.add(key);
      }
      return updated;
    });
  };

  const applyAllRecommended = () => {
    const newCellActions: typeof cellActions = {};
    cellIssues.forEach((issue) => {
      const key = `${issue.row_index}-${issue.column}`;
      newCellActions[key] = { action: issue.recommended_action };
    });
    setCellActions(newCellActions);

    const newRowActions: typeof rowActions = {};
    rowIssues.forEach((issue) => {
      newRowActions[issue.row_index] = {
        action: issue.drop_recommended ? "drop_row" : "leave_as_is",
      };
    });
    setRowActions(newRowActions);

    const newColumnActions: typeof columnActions = {};
    columnIssues.forEach((issue) => {
      newColumnActions[issue.column] = {
        action: "apply_deterministic_fix",
        targetFormat: issue.default_format || "True/False",
      };
    });
    setColumnActions(newColumnActions);
  };

  const leaveAllAsIs = () => {
    const newCellActions: typeof cellActions = {};
    cellIssues.forEach((issue) => {
      const key = `${issue.row_index}-${issue.column}`;
      newCellActions[key] = { action: "leave_as_is" };
    });
    setCellActions(newCellActions);

    const newRowActions: typeof rowActions = {};
    rowIssues.forEach((issue) => {
      newRowActions[issue.row_index] = { action: "leave_as_is" };
    });
    setRowActions(newRowActions);

    const newColumnActions: typeof columnActions = {};
    columnIssues.forEach((issue) => {
      newColumnActions[issue.column] = { action: "leave_as_is" };
    });
    setColumnActions(newColumnActions);
  };

  // Initialize columnActions when columnIssues prop changes (async from API)
  useEffect(() => {
    if (columnIssues.length > 0) {
      const newColumnActions: Record<
        string,
        { action: SuggestedAction; targetFormat?: string }
      > = {};
      columnIssues.forEach((issue) => {
        newColumnActions[issue.column] = {
          action: "apply_deterministic_fix",
          targetFormat: issue.default_format || "True/False",
        };
      });
      setColumnActions(newColumnActions);
    }
  }, [columnIssues]);

  useEffect(() => {
    onApprovalChange(buildApprovedActions());
  }, [cellActions, rowActions, columnActions, customPlaceholder]);

  const totalIssues =
    cellIssues.length + rowIssues.length + columnIssues.length;

  if (totalIssues === 0) {
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
    <div className="space-y-5">
      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-amber-50 border border-amber-100 rounded-xl p-4">
          <h4 className="text-sm font-medium text-amber-900 mb-2">
            Important Notes
          </h4>
          <ul className="text-sm text-amber-700 space-y-1">
            {warnings.map((warning, idx) => (
              <li key={idx}>• {warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Bulk Actions */}
      <div className="flex items-center justify-between bg-neutral-50 rounded-xl p-4">
        <div className="text-sm text-neutral-600">
          <span className="font-medium text-neutral-900">{totalIssues}</span>{" "}
          issues found
          <span className="text-neutral-400 ml-2">
            ({cellIssues.length} cells, {rowIssues.length} rows)
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={leaveAllAsIs}
            className="px-3 py-1.5 text-xs border border-neutral-200 rounded-full hover:bg-white transition-colors text-neutral-600"
          >
            Keep All
          </button>
          <button
            onClick={applyAllRecommended}
            className="px-3 py-1.5 text-xs bg-neutral-900 text-white rounded-full hover:bg-neutral-800 transition-colors"
          >
            Apply Recommended
          </button>
        </div>
      </div>

      {/* Custom Placeholder Input */}
      <div className="bg-neutral-50 rounded-xl p-4">
        <label className="block text-sm font-medium text-neutral-700 mb-2">
          Custom Placeholder
        </label>
        <input
          type="text"
          value={customPlaceholder}
          onChange={(e) => setCustomPlaceholder(e.target.value)}
          className="w-full max-w-xs px-3 py-2 border border-neutral-200 rounded-lg text-sm focus:ring-2 focus:ring-neutral-900 focus:border-neutral-900 outline-none transition-all"
          placeholder="Unknown"
        />
      </div>

      {/* Column Issues (e.g., Boolean Standardization) */}
      {columnIssues.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-neutral-900">
            Column Issues
          </h4>
          {columnIssues.map((issue) => {
            const key = `col-${issue.column}`;
            const isExpanded = expandedIssues.has(key);
            const selectedAction =
              columnActions[issue.column]?.action || "leave_as_is";
            const selectedFormat =
              columnActions[issue.column]?.targetFormat || "True/False";

            return (
              <div
                key={key}
                className="border border-amber-200 rounded-xl overflow-hidden bg-amber-50"
              >
                <div
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-amber-100 transition-colors"
                  onClick={() => toggleExpanded(key)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">
                      {ISSUE_ICONS[issue.issue_type] || "⚡"}
                    </span>
                    <div>
                      <div className="text-sm font-medium text-neutral-900">
                        Column "{issue.column}"
                      </div>
                      <div className="text-xs text-neutral-600">
                        {issue.description}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2.5 py-1 rounded-full ${
                        selectedAction === "apply_deterministic_fix"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-neutral-100 text-neutral-600"
                      }`}
                    >
                      {selectedAction === "apply_deterministic_fix"
                        ? `Standardize to ${selectedFormat}`
                        : ACTION_LABELS[selectedAction]}
                    </span>
                    <svg
                      className={`w-4 h-4 text-neutral-400 transition-transform ${
                        isExpanded ? "rotate-180" : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-amber-200 p-4 bg-white">
                    <div className="text-xs text-neutral-600 mb-4 flex items-start gap-2">
                      <span>💡</span>
                      <span>{issue.suggested_action}</span>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <div className="text-xs font-medium text-neutral-700 mb-2">
                          Choose Action
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleColumnActionChange(
                                issue.column,
                                "leave_as_is"
                              );
                            }}
                            className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                              selectedAction === "leave_as_is"
                                ? "bg-neutral-900 text-white border-neutral-900"
                                : "bg-white text-neutral-700 border-neutral-200 hover:border-neutral-400"
                            }`}
                          >
                            Keep as is
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleColumnActionChange(
                                issue.column,
                                "apply_deterministic_fix"
                              );
                            }}
                            className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                              selectedAction === "apply_deterministic_fix"
                                ? "bg-neutral-900 text-white border-neutral-900"
                                : "bg-white text-neutral-700 border-neutral-200 hover:border-neutral-400"
                            }`}
                          >
                            Standardize All ★
                          </button>
                        </div>
                      </div>

                      {selectedAction === "apply_deterministic_fix" && (
                        <div>
                          <div className="text-xs font-medium text-neutral-700 mb-2">
                            Target Format
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {(
                              issue.available_formats || BOOLEAN_FORMAT_OPTIONS
                            ).map((format) => (
                              <button
                                key={format}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleColumnFormatChange(
                                    issue.column,
                                    format
                                  );
                                }}
                                className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                                  selectedFormat === format
                                    ? "bg-emerald-600 text-white border-emerald-600"
                                    : "bg-white text-neutral-700 border-neutral-200 hover:border-neutral-400"
                                }`}
                              >
                                {format}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Cell Issues */}
      {cellIssues.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-neutral-900">Cell Issues</h4>
          {cellIssues.map((issue) => {
            const key = `${issue.row_index}-${issue.column}`;
            const isExpanded = expandedIssues.has(key);
            const selectedAction = cellActions[key]?.action || "leave_as_is";

            return (
              <div
                key={key}
                className="border border-neutral-200 rounded-xl overflow-hidden bg-white"
              >
                <div
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-neutral-50 transition-colors"
                  onClick={() => toggleExpanded(key)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">
                      {ISSUE_ICONS[issue.issue_type] || "❔"}
                    </span>
                    <div>
                      <div className="text-sm font-medium text-neutral-900">
                        Row {issue.row_index}, Column "{issue.column}"
                      </div>
                      <div className="text-xs text-neutral-500">
                        {issue.issue_description}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-2.5 py-1 rounded-full ${
                        selectedAction === "leave_as_is"
                          ? "bg-neutral-100 text-neutral-600"
                          : selectedAction === "apply_deterministic_fix"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-violet-100 text-violet-700"
                      }`}
                    >
                      {ACTION_LABELS[selectedAction]}
                    </span>
                    <svg
                      className={`w-4 h-4 text-neutral-400 transition-transform ${
                        isExpanded ? "rotate-180" : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-neutral-100 p-4 bg-neutral-50">
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <div className="text-xs text-neutral-500 mb-1">
                          Original
                        </div>
                        <div className="text-sm font-mono bg-red-50 text-red-600 px-3 py-2 rounded-lg">
                          {issue.original_value === null
                            ? "(null)"
                            : issue.original_value === ""
                            ? "(empty)"
                            : String(issue.original_value)}
                        </div>
                      </div>
                      {issue.deterministic_fix_value !== undefined && (
                        <div>
                          <div className="text-xs text-neutral-500 mb-1">
                            Fixed
                          </div>
                          <div className="text-sm font-mono bg-emerald-50 text-emerald-600 px-3 py-2 rounded-lg">
                            {String(issue.deterministic_fix_value)}
                          </div>
                        </div>
                      )}
                    </div>

                    {issue.deterministic_fix_explanation && (
                      <div className="text-xs text-neutral-600 mb-4 flex items-start gap-2">
                        <span>💡</span>
                        <span>{issue.deterministic_fix_explanation}</span>
                      </div>
                    )}

                    <div className="space-y-2">
                      <div className="text-xs font-medium text-neutral-700">
                        Choose Action
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {issue.available_actions.map((action) => (
                          <button
                            key={action}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCellActionChange(
                                issue.row_index,
                                issue.column,
                                action
                              );
                            }}
                            className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                              selectedAction === action
                                ? "bg-neutral-900 text-white border-neutral-900"
                                : "bg-white text-neutral-700 border-neutral-200 hover:border-neutral-400"
                            }`}
                          >
                            {ACTION_LABELS[action]}
                            {action === issue.recommended_action && (
                              <span className="ml-1">★</span>
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Row Issues */}
      {rowIssues.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-neutral-900">Row Issues</h4>
          {rowIssues.map((issue) => {
            const key = `row-${issue.row_index}`;
            const isExpanded = expandedIssues.has(key);
            const selectedAction =
              rowActions[issue.row_index]?.action || "leave_as_is";

            return (
              <div
                key={key}
                className="border border-neutral-200 rounded-xl overflow-hidden bg-white"
              >
                <div
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-neutral-50 transition-colors"
                  onClick={() => toggleExpanded(key)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">
                      {ISSUE_ICONS[issue.issue_type] || "❔"}
                    </span>
                    <div>
                      <div className="text-sm font-medium text-neutral-900">
                        Row {issue.row_index}
                      </div>
                      <div className="text-xs text-neutral-500">
                        {issue.issue_description}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {issue.drop_reason && (
                      <span className="text-xs px-2.5 py-1 bg-amber-100 text-amber-700 rounded-full">
                        Can drop
                      </span>
                    )}
                    <span
                      className={`text-xs px-2.5 py-1 rounded-full ${
                        selectedAction === "drop_row"
                          ? "bg-red-100 text-red-700"
                          : "bg-neutral-100 text-neutral-600"
                      }`}
                    >
                      {ACTION_LABELS[selectedAction]}
                    </span>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-neutral-100 p-4 bg-neutral-50">
                    <div className="mb-4">
                      <div className="text-xs text-neutral-500 mb-2">
                        Row Data
                      </div>
                      <div className="text-xs font-mono bg-neutral-100 p-3 rounded-lg overflow-x-auto">
                        {Object.entries(issue.row_data).map(([col, val]) => (
                          <span key={col} className="mr-4">
                            <span className="text-neutral-400">{col}:</span>{" "}
                            <span
                              className={
                                val === null
                                  ? "text-neutral-400 italic"
                                  : "text-neutral-800"
                              }
                            >
                              {val === null ? "null" : String(val)}
                            </span>
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-xs font-medium text-neutral-700">
                        Choose Action
                      </div>
                      <div className="flex gap-2">
                        {issue.available_actions.map((action) => (
                          <button
                            key={action}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleRowActionChange(issue.row_index, action);
                            }}
                            className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                              selectedAction === action
                                ? action === "drop_row"
                                  ? "bg-red-600 text-white border-red-600"
                                  : "bg-neutral-900 text-white border-neutral-900"
                                : "bg-white text-neutral-700 border-neutral-200 hover:border-neutral-400"
                            }`}
                          >
                            {ACTION_LABELS[action]}
                            {action === "drop_row" &&
                              issue.drop_recommended && (
                                <span className="ml-1">★</span>
                              )}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
