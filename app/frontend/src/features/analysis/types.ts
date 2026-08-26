import type { DataSource } from "../../api/types";

export type Source = DataSource;
export type Provenance = "measured" | "configured" | "inferred";
export type XFactor = "actualNoise" | "actualEmd" | "sampleCount" | "lidK";
export type YFactor = "lidMean" | "testLoss" | "testAccuracy";

export interface ScalarWithProvenance {
  value: number;
  provenance: Provenance;
}

export interface ObservationRow {
  source: Source;
  studyId: string;
  runId: string;
  clientId: number;
  seed: number;
  cycle: number;
  step: number;
  role: string;
  targetNoise: ScalarWithProvenance;
  actualNoise: number;
  targetEmd: ScalarWithProvenance;
  actualEmd: number;
  sampleCount: number;
  lidK: number;
  lidMean: number;
  lidStd: number;
  trainLoss: number;
  testLoss: number;
  testAccuracy: number;
}

export interface AnalysisFilters {
  includeFixture: boolean;
  sources: Source[];
  roles: string[];
}
