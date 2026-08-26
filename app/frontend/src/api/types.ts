export type RunStatus = "queued" | "running" | "stopping" | "stopped" | "completed" | "failed";
export type DataSource = "fixture" | "fresh" | "legacy";
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export type TrainingEventType =
  | "run.started"
  | "client.started"
  | "client.completed"
  | "aggregation.completed"
  | "run.stop_requested"
  | "run.stopped"
  | "run.completed"
  | "run.failed";

export interface TrainingEvent {
  schema_version: 1;
  event_id: string;
  run_id: string;
  sequence: number;
  type: TrainingEventType;
  created_at: string;
  payload: Record<string, JsonValue>;
}

export interface ArtifactFile {
  path: string;
  bytes: number;
  sha256: string;
}

export interface RunSummary {
  schema_version: 1;
  run_id: string;
  status: RunStatus;
  preset: "test-fixture" | "research-lite" | "historical-compatible";
  source: DataSource;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_code: string | null;
  error_message: string | null;
  files: ArtifactFile[];
}
