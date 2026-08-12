import type { Source } from "./types";

const LABELS: Record<Source, string> = {
  fixture: "Fixture",
  fresh: "Fresh",
  legacy: "Legacy",
};

export function SourceBadge({ source }: { source: Source }) {
  return (
    <span
      className={`source-badge source-${source}`}
      title={source === "fixture" ? "仅用于测试，不得作为研究结论" : undefined}
    >
      {LABELS[source]}
    </span>
  );
}
