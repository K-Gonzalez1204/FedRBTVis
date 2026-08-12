import type {
  ArtifactFile,
  DataSource,
  JsonValue,
  RunStatus,
  RunSummary,
  TrainingEvent,
  TrainingEventType,
} from "./types";

export type PresetName = "test-fixture" | "research-lite" | "historical-compatible";
export type StudyStatus = "queued" | "running" | "completed" | "failed" | "stopped";
export type StudyFactor = "target_noise" | "target_emd" | "sample_count" | "lid_k";

export interface ProbePreset {
  client_id: number;
  target_noise: number;
  target_emd: number;
  sample_count: number;
  lid_k: number;
}

export interface Preset {
  schema_version: 1;
  preset: PresetName;
  source: DataSource;
  dataset: "synthetic-cifar" | "cifar10";
  model: "tiny-cnn" | "cifar-resnet18";
  seed: number;
  num_classes: number;
  background_clients: number;
  background_samples: number;
  background_lid_k: number;
  background_noise_fraction: number;
  background_noise_min: number;
  background_noise_max: number;
  probes: ProbePreset[];
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  cycles: number;
  clients_per_step: number;
  checkpoint_policy: "none" | "server-only" | "probe-clients";
}

export interface CreatedRun {
  run_id: string;
  status: RunStatus;
}

export interface ClientMetric {
  cycle: number;
  step: number;
  client_id: number;
  role: string;
  sample_count: number;
  target_noise: number;
  actual_noise: number;
  target_emd: number;
  actual_emd: number;
  lid_k: number;
  train_loss: number;
  test_loss: number;
  test_accuracy: number;
  test_correct: number;
  test_samples: number;
  lid_mean: number;
  lid_std: number;
  state_sha256: string;
}

export interface AggregationMetric {
  cycle: number;
  step: number;
  client_ids: number[];
  test_loss: number;
  test_accuracy: number;
}

export interface StudySpec {
  preset: "research-lite" | "historical-compatible";
  factors: Partial<Record<StudyFactor, Array<number>>>;
  seeds: number[];
}

export interface CreatedStudy {
  study_id: string;
  status: StudyStatus;
}

export interface StudySummary {
  schema_version: 1;
  study_id: string;
  status: StudyStatus;
  preset: "research-lite" | "historical-compatible";
  total_runs: number;
  run_ids: string[];
  active_run_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_code: string | null;
  error_message: string | null;
}

export interface Observation {
  source: DataSource;
  study_id: string;
  run_id: string;
  client_id: number;
  seed: number;
  cycle: number;
  step: number;
  role: string;
  target_noise: number;
  actual_noise: number;
  target_emd: number;
  actual_emd: number;
  sample_count: number;
  lid_k: number;
  lid_mean: number;
  lid_std: number;
  train_loss: number;
  test_loss: number;
  test_accuracy: number;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

type Decoder<T> = (value: unknown, path: string) => T;

const invalid = (path: string, expectation: string): never => {
  throw new Error(`${path} must be ${expectation}`);
};

const record = (value: unknown, path: string): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return invalid(path, "an object");
  }
  return value as Record<string, unknown>;
};

const field = (value: Record<string, unknown>, name: string, path: string): unknown => {
  if (!Object.hasOwn(value, name)) return invalid(`${path}.${name}`, "present");
  return value[name];
};

const stringValue = (value: unknown, path: string): string =>
  typeof value === "string" && value.length > 0 ? value : invalid(path, "a non-empty string");

const nullableString = (value: unknown, path: string): string | null =>
  value === null ? null : stringValue(value, path);

const numberValue = (value: unknown, path: string): number =>
  typeof value === "number" && Number.isFinite(value) ? value : invalid(path, "a finite number");

const integerValue = (value: unknown, path: string): number => {
  const decoded = numberValue(value, path);
  return Number.isInteger(decoded) ? decoded : invalid(path, "an integer");
};

const oneOf = <T extends string>(choices: readonly T[]): Decoder<T> => (value, path) =>
  typeof value === "string" && choices.includes(value as T)
    ? value as T
    : invalid(path, `one of ${choices.join(", ")}`);

const arrayOf = <T>(decoder: Decoder<T>): Decoder<T[]> => (value, path) => {
  if (!Array.isArray(value)) return invalid(path, "an array");
  return value.map((item, index) => decoder(item, `${path}[${index}]`));
};

const jsonValue = (value: unknown, path: string): JsonValue => {
  if (value === null || typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.map((item, index) => jsonValue(item, `${path}[${index}]`));
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, jsonValue(item, `${path}.${key}`)]),
    );
  }
  return invalid(path, "valid JSON");
};

const dataSource = oneOf<DataSource>(["fixture", "fresh", "legacy"]);
const presetName = oneOf<PresetName>(["test-fixture", "research-lite", "historical-compatible"]);
const runStatus = oneOf<RunStatus>(["queued", "running", "stopping", "stopped", "completed", "failed"]);
const eventType = oneOf<TrainingEventType>([
  "run.started",
  "client.started",
  "client.completed",
  "aggregation.completed",
  "run.stop_requested",
  "run.stopped",
  "run.completed",
  "run.failed",
]);
const studyStatus = oneOf<StudyStatus>(["queued", "running", "completed", "failed", "stopped"]);

const literalOne = (value: unknown, path: string): 1 => value === 1 ? 1 : invalid(path, "1");

const artifactFile: Decoder<ArtifactFile> = (value, path) => {
  const item = record(value, path);
  return {
    path: stringValue(field(item, "path", path), `${path}.path`),
    bytes: integerValue(field(item, "bytes", path), `${path}.bytes`),
    sha256: stringValue(field(item, "sha256", path), `${path}.sha256`),
  };
};

const runSummary: Decoder<RunSummary> = (value, path) => {
  const item = record(value, path);
  return {
    schema_version: literalOne(field(item, "schema_version", path), `${path}.schema_version`),
    run_id: stringValue(field(item, "run_id", path), `${path}.run_id`),
    status: runStatus(field(item, "status", path), `${path}.status`),
    preset: presetName(field(item, "preset", path), `${path}.preset`),
    source: dataSource(field(item, "source", path), `${path}.source`),
    created_at: stringValue(field(item, "created_at", path), `${path}.created_at`),
    started_at: nullableString(field(item, "started_at", path), `${path}.started_at`),
    finished_at: nullableString(field(item, "finished_at", path), `${path}.finished_at`),
    error_code: nullableString(field(item, "error_code", path), `${path}.error_code`),
    error_message: nullableString(field(item, "error_message", path), `${path}.error_message`),
    files: arrayOf(artifactFile)(field(item, "files", path), `${path}.files`),
  };
};

const trainingEvent: Decoder<TrainingEvent> = (value, path) => {
  const item = record(value, path);
  const payload = record(field(item, "payload", path), `${path}.payload`);
  return {
    schema_version: literalOne(field(item, "schema_version", path), `${path}.schema_version`),
    event_id: stringValue(field(item, "event_id", path), `${path}.event_id`),
    run_id: stringValue(field(item, "run_id", path), `${path}.run_id`),
    sequence: integerValue(field(item, "sequence", path), `${path}.sequence`),
    type: eventType(field(item, "type", path), `${path}.type`),
    created_at: stringValue(field(item, "created_at", path), `${path}.created_at`),
    payload: Object.fromEntries(
      Object.entries(payload).map(([key, itemValue]) => [key, jsonValue(itemValue, `${path}.payload.${key}`)]),
    ),
  };
};

export const validateTrainingEvent = (value: unknown): TrainingEvent =>
  trainingEvent(value, "event");

const envelope = <T>(decoder: Decoder<T>): Decoder<T[]> => (value, path) => {
  const item = record(value, path);
  return arrayOf(decoder)(field(item, "items", path), `${path}.items`);
};

const probePreset: Decoder<ProbePreset> = (value, path) => {
  const item = record(value, path);
  return {
    client_id: integerValue(field(item, "client_id", path), `${path}.client_id`),
    target_noise: numberValue(field(item, "target_noise", path), `${path}.target_noise`),
    target_emd: numberValue(field(item, "target_emd", path), `${path}.target_emd`),
    sample_count: integerValue(field(item, "sample_count", path), `${path}.sample_count`),
    lid_k: integerValue(field(item, "lid_k", path), `${path}.lid_k`),
  };
};

const preset: Decoder<Preset> = (value, path) => {
  const item = record(value, path);
  return {
    schema_version: literalOne(field(item, "schema_version", path), `${path}.schema_version`),
    preset: presetName(field(item, "preset", path), `${path}.preset`),
    source: dataSource(field(item, "source", path), `${path}.source`),
    dataset: oneOf(["synthetic-cifar", "cifar10"] as const)(field(item, "dataset", path), `${path}.dataset`),
    model: oneOf(["tiny-cnn", "cifar-resnet18"] as const)(field(item, "model", path), `${path}.model`),
    seed: integerValue(field(item, "seed", path), `${path}.seed`),
    num_classes: integerValue(field(item, "num_classes", path), `${path}.num_classes`),
    background_clients: integerValue(field(item, "background_clients", path), `${path}.background_clients`),
    background_samples: integerValue(field(item, "background_samples", path), `${path}.background_samples`),
    background_lid_k: integerValue(field(item, "background_lid_k", path), `${path}.background_lid_k`),
    background_noise_fraction: numberValue(field(item, "background_noise_fraction", path), `${path}.background_noise_fraction`),
    background_noise_min: numberValue(field(item, "background_noise_min", path), `${path}.background_noise_min`),
    background_noise_max: numberValue(field(item, "background_noise_max", path), `${path}.background_noise_max`),
    probes: arrayOf(probePreset)(field(item, "probes", path), `${path}.probes`),
    local_epochs: integerValue(field(item, "local_epochs", path), `${path}.local_epochs`),
    batch_size: integerValue(field(item, "batch_size", path), `${path}.batch_size`),
    learning_rate: numberValue(field(item, "learning_rate", path), `${path}.learning_rate`),
    cycles: integerValue(field(item, "cycles", path), `${path}.cycles`),
    clients_per_step: integerValue(field(item, "clients_per_step", path), `${path}.clients_per_step`),
    checkpoint_policy: oneOf(["none", "server-only", "probe-clients"] as const)(field(item, "checkpoint_policy", path), `${path}.checkpoint_policy`),
  };
};

const createdRun: Decoder<CreatedRun> = (value, path) => {
  const item = record(value, path);
  return {
    run_id: stringValue(field(item, "run_id", path), `${path}.run_id`),
    status: runStatus(field(item, "status", path), `${path}.status`),
  };
};

const clientMetric: Decoder<ClientMetric> = (value, path) => {
  const item = record(value, path);
  const integers = (name: keyof ClientMetric) => integerValue(field(item, name, path), `${path}.${name}`);
  const numbers = (name: keyof ClientMetric) => numberValue(field(item, name, path), `${path}.${name}`);
  return {
    cycle: integers("cycle"),
    step: integers("step"),
    client_id: integers("client_id"),
    role: stringValue(field(item, "role", path), `${path}.role`),
    sample_count: integers("sample_count"),
    target_noise: numbers("target_noise"),
    actual_noise: numbers("actual_noise"),
    target_emd: numbers("target_emd"),
    actual_emd: numbers("actual_emd"),
    lid_k: integers("lid_k"),
    train_loss: numbers("train_loss"),
    test_loss: numbers("test_loss"),
    test_accuracy: numbers("test_accuracy"),
    test_correct: integers("test_correct"),
    test_samples: integers("test_samples"),
    lid_mean: numbers("lid_mean"),
    lid_std: numbers("lid_std"),
    state_sha256: stringValue(field(item, "state_sha256", path), `${path}.state_sha256`),
  };
};

const aggregationMetric: Decoder<AggregationMetric> = (value, path) => {
  const item = record(value, path);
  return {
    cycle: integerValue(field(item, "cycle", path), `${path}.cycle`),
    step: integerValue(field(item, "step", path), `${path}.step`),
    client_ids: arrayOf(integerValue)(field(item, "client_ids", path), `${path}.client_ids`),
    test_loss: numberValue(field(item, "test_loss", path), `${path}.test_loss`),
    test_accuracy: numberValue(field(item, "test_accuracy", path), `${path}.test_accuracy`),
  };
};

const createdStudy: Decoder<CreatedStudy> = (value, path) => {
  const item = record(value, path);
  return {
    study_id: stringValue(field(item, "study_id", path), `${path}.study_id`),
    status: studyStatus(field(item, "status", path), `${path}.status`),
  };
};

const studySummary: Decoder<StudySummary> = (value, path) => {
  const item = record(value, path);
  return {
    schema_version: literalOne(field(item, "schema_version", path), `${path}.schema_version`),
    study_id: stringValue(field(item, "study_id", path), `${path}.study_id`),
    status: studyStatus(field(item, "status", path), `${path}.status`),
    preset: oneOf(["research-lite", "historical-compatible"] as const)(field(item, "preset", path), `${path}.preset`),
    total_runs: integerValue(field(item, "total_runs", path), `${path}.total_runs`),
    run_ids: arrayOf(stringValue)(field(item, "run_ids", path), `${path}.run_ids`),
    active_run_id: nullableString(field(item, "active_run_id", path), `${path}.active_run_id`),
    created_at: stringValue(field(item, "created_at", path), `${path}.created_at`),
    started_at: nullableString(field(item, "started_at", path), `${path}.started_at`),
    finished_at: nullableString(field(item, "finished_at", path), `${path}.finished_at`),
    error_code: nullableString(field(item, "error_code", path), `${path}.error_code`),
    error_message: nullableString(field(item, "error_message", path), `${path}.error_message`),
  };
};

const observation: Decoder<Observation> = (value, path) => {
  const item = record(value, path);
  const integers = (name: keyof Observation) => integerValue(field(item, name, path), `${path}.${name}`);
  const numbers = (name: keyof Observation) => numberValue(field(item, name, path), `${path}.${name}`);
  return {
    source: dataSource(field(item, "source", path), `${path}.source`),
    study_id: stringValue(field(item, "study_id", path), `${path}.study_id`),
    run_id: stringValue(field(item, "run_id", path), `${path}.run_id`),
    client_id: integers("client_id"),
    seed: integers("seed"),
    cycle: integers("cycle"),
    step: integers("step"),
    role: stringValue(field(item, "role", path), `${path}.role`),
    target_noise: numbers("target_noise"),
    actual_noise: numbers("actual_noise"),
    target_emd: numbers("target_emd"),
    actual_emd: numbers("actual_emd"),
    sample_count: integers("sample_count"),
    lid_k: integers("lid_k"),
    lid_mean: numbers("lid_mean"),
    lid_std: numbers("lid_std"),
    train_loss: numbers("train_loss"),
    test_loss: numbers("test_loss"),
    test_accuracy: numbers("test_accuracy"),
  };
};

export class ApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl = "", fetchImpl: typeof fetch = fetch) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.fetchImpl = fetchImpl;
  }

  presets(): Promise<Preset[]> {
    return this.request("/api/presets", undefined, envelope(preset));
  }

  createRun(presetValue: PresetName, overrides: Record<string, JsonValue> = {}): Promise<CreatedRun> {
    return this.request("/api/runs", this.jsonPost({ preset: presetValue, overrides }), createdRun);
  }

  listRuns(): Promise<RunSummary[]> {
    return this.request("/api/runs", undefined, envelope(runSummary));
  }

  getRun(runId: string): Promise<RunSummary> {
    return this.request(`/api/runs/${encodeURIComponent(runId)}`, undefined, runSummary);
  }

  stopRun(runId: string): Promise<RunSummary> {
    return this.request(`/api/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" }, runSummary);
  }

  events(runId: string, afterSequence = 0): Promise<TrainingEvent[]> {
    return this.request(
      `/api/runs/${encodeURIComponent(runId)}/events?after_sequence=${encodeURIComponent(String(afterSequence))}`,
      undefined,
      envelope(trainingEvent),
    );
  }

  clientMetrics(runId: string): Promise<ClientMetric[]> {
    return this.request(`/api/runs/${encodeURIComponent(runId)}/metrics/clients`, undefined, envelope(clientMetric));
  }

  aggregationMetrics(runId: string): Promise<AggregationMetric[]> {
    return this.request(`/api/runs/${encodeURIComponent(runId)}/metrics/aggregations`, undefined, envelope(aggregationMetric));
  }

  createStudy(spec: StudySpec): Promise<CreatedStudy> {
    return this.request("/api/studies", this.jsonPost(spec), createdStudy);
  }

  getStudy(studyId: string): Promise<StudySummary> {
    return this.request(`/api/studies/${encodeURIComponent(studyId)}`, undefined, studySummary);
  }

  studyObservations(studyId: string): Promise<Observation[]> {
    return this.request(`/api/studies/${encodeURIComponent(studyId)}/observations`, undefined, envelope(observation));
  }

  legacyObservations(): Promise<Observation[]> {
    return this.request("/api/observations/legacy", undefined, envelope(observation));
  }

  private jsonPost(value: unknown): RequestInit {
    return {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(value),
    };
  }

  private async request<T>(path: string, init: RequestInit | undefined, decoder: Decoder<T>): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, init);
    const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";

    if (!response.ok) {
      if (contentType.includes("application/json")) {
        try {
          const body: unknown = await response.json();
          const outer = record(body, "response");
          const error = record(field(outer, "error", "response"), "response.error");
          const code = stringValue(field(error, "code", "response.error"), "response.error.code");
          const message = stringValue(field(error, "message", "response.error"), "response.error.message");
          throw new ApiError(response.status, code, message);
        } catch (error) {
          if (error instanceof ApiError) throw error;
        }
      }
      throw new ApiError(response.status, `HTTP_${response.status}`, `request failed with HTTP ${response.status}`);
    }

    if (!contentType.includes("application/json")) {
      throw new ApiError(response.status, "INVALID_RESPONSE", "successful response is not JSON");
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new ApiError(response.status, "INVALID_RESPONSE", "successful response contains invalid JSON");
    }
    try {
      return decoder(body, "response");
    } catch (error) {
      const message = error instanceof Error ? error.message : "successful response has an invalid structure";
      throw new ApiError(response.status, "INVALID_RESPONSE", message);
    }
  }
}
