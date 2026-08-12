import { useMemo, useState } from "react";

import type { ObservationRow } from "./types";
import { SourceBadge } from "./SourceBadge";

type SortKey = "clientId" | "lidMean" | "actualNoise" | "actualEmd";

const SORT_KEYS: readonly { key: SortKey; label: string }[] = [
  { key: "clientId", label: "Client ID" },
  { key: "lidMean", label: "LID Mean" },
  { key: "actualNoise", label: "Actual Noise" },
  { key: "actualEmd", label: "Actual EMD" },
];

const valueAt = (row: ObservationRow, key: SortKey): number => row[key];

export function ClientTable({ rows }: { rows: ObservationRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("clientId");
  const [ascending, setAscending] = useState(true);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((left, right) => {
      const diff = valueAt(left, sortKey) - valueAt(right, sortKey);
      return ascending ? diff : -diff;
    });
    return copy;
  }, [rows, sortKey, ascending]);

  const toggle = (key: SortKey): void => {
    if (key === sortKey) {
      setAscending((current) => !current);
    } else {
      setSortKey(key);
      setAscending(true);
    }
  };

  return (
    <div className="client-table-wrap">
      <table className="client-table">
        <thead>
          <tr>
            {SORT_KEYS.map(({ key, label }) => (
              <th
                key={key}
                aria-sort={
                  sortKey === key
                    ? ascending
                      ? "ascending"
                      : "descending"
                    : "none"
                }
              >
                <button type="button" onClick={() => toggle(key)}>
                  {label}
                </button>
              </th>
            ))}
            <th>Sample</th>
            <th>LID k</th>
            <th>Test Loss</th>
            <th>Test Acc</th>
            <th>Target</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={`${row.runId}-${row.clientId}`}>
              <td>{row.clientId}</td>
              <td>{row.lidMean.toFixed(4)}</td>
              <td>{row.actualNoise.toFixed(3)}</td>
              <td>{row.actualEmd.toFixed(3)}</td>
              <td>{row.sampleCount}</td>
              <td>{row.lidK}</td>
              <td>{row.testLoss.toFixed(4)}</td>
              <td>{row.testAccuracy.toFixed(4)}</td>
              <td>
                {row.targetNoise.provenance === "inferred" ? "推导" : row.targetNoise.value.toFixed(3)}
              </td>
              <td><SourceBadge source={row.source} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      {sorted.length === 0 ? <p className="empty-table">暂无观测</p> : null}
    </div>
  );
}
