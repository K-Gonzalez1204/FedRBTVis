import type { ClientMetric, Observation } from "../../api/client";
import type {
  AnalysisFilters,
  ObservationRow,
} from "./types";

export const defaultFilters = (): AnalysisFilters => ({
  includeFixture: false,
  sources: [],
  roles: [],
});

export function filterObservations(
  rows: ObservationRow[],
  filters: AnalysisFilters,
): ObservationRow[] {
  return rows.filter((row) => {
    if (row.source === "fixture" && !filters.includeFixture) return false;
    if (filters.sources.length > 0 && !filters.sources.includes(row.source)) {
      return false;
    }
    if (filters.roles.length > 0 && !filters.roles.includes(row.role)) {
      return false;
    }
    return true;
  });
}

export interface LegacyRecord {
  client_id: number;
  block_index: number;
  actual_noise: number;
  categorical_emd_01: number;
  inferred_target_noise: number;
  inferred_target_emd?: number;
  lid: number;
  sample_count: number;
}

export function normalizeLegacy(record: LegacyRecord): ObservationRow {
  return {
    source: "legacy",
    studyId: `legacy-${record.block_index}`,
    runId: `legacy-${record.block_index}`,
    clientId: record.client_id,
    seed: 0,
    cycle: 0,
    step: 0,
    role: "probe",
    targetNoise: {
      value: record.inferred_target_noise,
      provenance: "inferred",
    },
    actualNoise: record.actual_noise,
    targetEmd: {
      value: record.inferred_target_emd ?? 0,
      provenance: "inferred",
    },
    actualEmd: record.categorical_emd_01,
    sampleCount: record.sample_count,
    lidK: 0,
    lidMean: record.lid,
    lidStd: 0,
    trainLoss: 0,
    testLoss: 0,
    testAccuracy: 0,
  };
}

export function normalizeApiObservation(
  observation: Observation,
): ObservationRow {
  return {
    source: observation.source,
    studyId: observation.study_id,
    runId: observation.run_id,
    clientId: observation.client_id,
    seed: observation.seed,
    cycle: observation.cycle,
    step: observation.step,
    role: observation.role,
    targetNoise: {
      value: observation.target_noise,
      provenance: "configured",
    },
    actualNoise: observation.actual_noise,
    targetEmd: {
      value: observation.target_emd,
      provenance: "configured",
    },
    actualEmd: observation.actual_emd,
    sampleCount: observation.sample_count,
    lidK: observation.lid_k,
    lidMean: observation.lid_mean,
    lidStd: observation.lid_std,
    trainLoss: observation.train_loss,
    testLoss: observation.test_loss,
    testAccuracy: observation.test_accuracy,
  };
}

export function normalizeClientMetric(
  metric: ClientMetric,
  runId: string,
): ObservationRow {
  return {
    source: "fresh",
    studyId: runId,
    runId,
    clientId: metric.client_id,
    seed: 0,
    cycle: metric.cycle,
    step: metric.step,
    role: metric.role,
    targetNoise: {
      value: metric.target_noise,
      provenance: "configured",
    },
    actualNoise: metric.actual_noise,
    targetEmd: {
      value: metric.target_emd,
      provenance: "configured",
    },
    actualEmd: metric.actual_emd,
    sampleCount: metric.sample_count,
    lidK: metric.lid_k,
    lidMean: metric.lid_mean,
    lidStd: metric.lid_std,
    trainLoss: metric.train_loss,
    testLoss: metric.test_loss,
    testAccuracy: metric.test_accuracy,
  };
}
