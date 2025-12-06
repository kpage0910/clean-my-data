import React from "react";

interface PreviewRow {
  row_index: number;
  original: Record<string, any>;
  cleaned: Record<string, any>;
  changes: string[];
}

interface PreviewTableProps {
  previewData: PreviewRow[];
  totalRows: number;
  previewRows: number;
}

export default function PreviewTable({
  previewData,
  totalRows,
  previewRows,
}: PreviewTableProps) {
  if (!previewData || previewData.length === 0) {
    return (
      <div className="text-center py-8 text-neutral-500">
        No preview data available
      </div>
    );
  }

  const columns = Object.keys(previewData[0].original);

  const hasChanged = (row: PreviewRow, column: string): boolean => {
    return row.original[column] !== row.cleaned[column];
  };

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return "—";
    if (value === "") return "(empty)";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  return (
    <div className="w-full">
      <div className="overflow-x-auto rounded-xl border border-neutral-200">
        <table className="min-w-full divide-y divide-neutral-200">
          <thead>
            <tr className="bg-neutral-50">
              <th className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider sticky left-0 bg-neutral-50">
                Row
              </th>
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-4 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider"
                  colSpan={2}
                >
                  {column}
                </th>
              ))}
            </tr>
            <tr className="bg-neutral-100/50">
              <th className="px-4 py-2 text-left text-xs text-neutral-400 sticky left-0 bg-neutral-100/50">
                #
              </th>
              {columns.map((column) => (
                <React.Fragment key={column}>
                  <th className="px-4 py-2 text-left text-xs text-neutral-400">
                    Original
                  </th>
                  <th className="px-4 py-2 text-left text-xs text-neutral-400">
                    Cleaned
                  </th>
                </React.Fragment>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-neutral-100">
            {previewData.map((row) => (
              <tr
                key={row.row_index}
                className="hover:bg-neutral-50 transition-colors"
              >
                <td className="px-4 py-3 text-sm text-neutral-400 sticky left-0 bg-white">
                  {row.row_index}
                </td>
                {columns.map((column) => {
                  const changed = hasChanged(row, column);
                  return (
                    <React.Fragment key={column}>
                      <td
                        className={`px-4 py-3 text-sm ${
                          changed
                            ? "bg-red-50 text-red-600"
                            : "text-neutral-600"
                        }`}
                      >
                        {formatValue(row.original[column])}
                      </td>
                      <td
                        className={`px-4 py-3 text-sm ${
                          changed
                            ? "bg-emerald-50 text-emerald-600 font-medium"
                            : "text-neutral-600"
                        }`}
                      >
                        {formatValue(row.cleaned[column])}
                      </td>
                    </React.Fragment>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Changes Summary */}
      {previewData.filter((row) => row.changes.length > 0).length > 0 && (
        <div className="mt-4 p-4 bg-neutral-50 rounded-xl border border-neutral-100">
          <h4 className="text-sm font-medium text-neutral-900 mb-3">
            Changes Summary
          </h4>
          <ul className="text-sm text-neutral-600 space-y-1.5">
            {previewData
              .filter((row) => row.changes.length > 0)
              .slice(0, 5)
              .map((row) => (
                <li key={row.row_index} className="flex items-start gap-2">
                  <span className="text-xs font-medium text-neutral-400 bg-neutral-200 px-1.5 py-0.5 rounded">
                    Row {row.row_index}
                  </span>
                  <span>{row.changes.join(", ")}</span>
                </li>
              ))}
            {previewData.filter((row) => row.changes.length > 0).length > 5 && (
              <li className="text-neutral-400 text-xs mt-2">
                +{" "}
                {previewData.filter((row) => row.changes.length > 0).length - 5}{" "}
                more rows with changes
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
