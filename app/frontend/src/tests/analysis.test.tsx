import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { FactorScatter } from "../features/analysis/FactorScatter";
import {
  defaultFilters,
  filterObservations,
  normalizeLegacy,
} from "../features/analysis/normalize";
import type { ObservationRow } from "../features/analysis/types";

const observation = (
  source: ObservationRow["source"],
  id: number,
): ObservationRow => ({
  source,
  studyId: `study-${id}`,
  runId: `run-${id}`,
  clientId: id,
  seed: 1,
  cycle: 0,
  step: 0,
  role: "probe",
  targetNoise: { value: 0.2, provenance: "configured" },
  actualNoise: 0.2,
  targetEmd: { value: 0.3, provenance: "configured" },
  actualEmd: 0.3,
  sampleCount: 200,
  lidK: 20,
  lidMean: 2 + id,
  lidStd: 0.1,
  trainLoss: 1.1,
  testLoss: 1.2,
  testAccuracy: 0.8,
});

describe("analysis normalization", () => {
  it("hides fixture observations by default", () => {
    const rows = [
      observation("fixture", 1),
      observation("fresh", 2),
      observation("legacy", 3),
    ];
    expect(
      filterObservations(rows, defaultFilters()).map((row) => row.source),
    ).toEqual(["fresh", "legacy"]);
  });

  it("never treats inferred legacy target as measured", () => {
    const row = normalizeLegacy({
      actual_noise: 0.3,
      categorical_emd_01: 0.4,
      inferred_target_noise: 0.2,
      inferred_target_emd: 0.35,
      lid: 2.5,
      sample_count: 200,
      client_id: 100,
      block_index: 1,
    });
    expect(row.actualNoise).toBe(0.3);
    expect(row.targetNoise).toEqual({
      value: 0.2,
      provenance: "inferred",
    });
    expect(row.targetEmd).toEqual({
      value: 0.35,
      provenance: "inferred",
    });
  });
});

describe("FactorScatter", () => {
  it("renders accessible point labels and source legend", () => {
    const html = renderToStaticMarkup(
      <FactorScatter
        rows={[
          observation("fixture", 1),
          observation("fresh", 2),
          observation("legacy", 3),
        ]}
        x="actualNoise"
        y="lidMean"
      />,
    );
    expect(html).toContain("aria-label=\"因素散点图");
    expect(html).toContain("Fresh");
    expect(html).toContain("Legacy");
    expect(html).toContain("Fixture");
  });
});
