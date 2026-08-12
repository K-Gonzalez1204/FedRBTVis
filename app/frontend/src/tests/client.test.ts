import { describe, expect, it } from "vitest";
import { ApiClient, ApiError } from "../api/client";

const jsonResponse = (value: unknown, status = 200): Response =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });

const run = {
  schema_version: 1,
  run_id: "run-1",
  status: "running",
  preset: "test-fixture",
  source: "fixture",
  created_at: "2026-08-01T00:00:00+00:00",
  started_at: "2026-08-01T00:00:01+00:00",
  finished_at: null,
  error_code: null,
  error_message: null,
  files: [],
} as const;

const trainingEvent = {
  schema_version: 1,
  event_id: "event-1",
  run_id: "run-1",
  sequence: 1,
  type: "run.started",
  created_at: "2026-08-01T00:00:01+00:00",
  payload: {},
} as const;

const clientMetric = {
  cycle: 1,
  step: 2,
  client_id: 3,
  role: "probe",
  sample_count: 100,
  target_noise: 0.1,
  actual_noise: 0.11,
  target_emd: 0.2,
  actual_emd: 0.21,
  lid_k: 5,
  train_loss: 0.8,
  test_loss: 0.7,
  test_accuracy: 0.75,
  test_correct: 75,
  test_samples: 100,
  lid_mean: 2.5,
  lid_std: 0.4,
  state_sha256: "a".repeat(64),
};

const aggregationMetric = {
  cycle: 1,
  step: 2,
  client_ids: [1, 3],
  test_loss: 0.6,
  test_accuracy: 0.8,
};

const observation = {
  source: "fresh",
  study_id: "study-1",
  run_id: "run-1",
  client_id: 3,
  seed: 7,
  cycle: 1,
  step: 2,
  role: "probe",
  target_noise: 0.1,
  actual_noise: 0.11,
  target_emd: 0.2,
  actual_emd: 0.21,
  sample_count: 100,
  lid_k: 5,
  lid_mean: 2.5,
  lid_std: 0.4,
  train_loss: 0.8,
  test_loss: 0.7,
  test_accuracy: 0.75,
};

describe("ApiClient", () => {
  it("surfaces the stable backend error code", async () => {
    const fetchImpl: typeof fetch = async () => jsonResponse(
      { error: { code: "RUN_ALREADY_ACTIVE", message: "one run at a time" } },
      409,
    );
    const client = new ApiClient("", fetchImpl);

    await expect(client.createRun("test-fixture")).rejects.toMatchObject({
      code: "RUN_ALREADY_ACTIVE",
      message: "one run at a time",
      status: 409,
    });
  });

  it("rejects a successful response whose required fields are missing", async () => {
    const fetchImpl: typeof fetch = async () => jsonResponse({
      ...run,
      started_at: undefined,
    });
    const client = new ApiClient("", fetchImpl);

    await expect(client.getRun("run-1")).rejects.toMatchObject({
      code: "INVALID_RESPONSE",
      status: 200,
    });
  });

  it("rejects a successful non-JSON response", async () => {
    const fetchImpl: typeof fetch = async () => new Response("ok", {
      status: 200,
      headers: { "content-type": "text/plain" },
    });
    const client = new ApiClient("", fetchImpl);

    await expect(client.listRuns()).rejects.toBeInstanceOf(ApiError);
    await expect(client.listRuns()).rejects.toMatchObject({
      code: "INVALID_RESPONSE",
      status: 200,
    });
  });

  it("validates and unwraps every backend endpoint", async () => {
    const preset = {
      schema_version: 1,
      preset: "test-fixture",
      source: "fixture",
      dataset: "synthetic-cifar",
      model: "tiny-cnn",
      seed: 7,
      num_classes: 10,
      background_clients: 2,
      background_samples: 100,
      background_lid_k: 5,
      background_noise_fraction: 0.2,
      background_noise_min: 0,
      background_noise_max: 0.1,
      probes: [{ client_id: 2, target_noise: 0.1, target_emd: 0.2, sample_count: 100, lid_k: 5 }],
      local_epochs: 1,
      batch_size: 32,
      learning_rate: 0.01,
      cycles: 2,
      clients_per_step: 2,
      checkpoint_policy: "none",
    };
    const study = {
      schema_version: 1,
      study_id: "study-1",
      status: "running",
      preset: "research-lite",
      total_runs: 1,
      run_ids: ["run-1"],
      active_run_id: "run-1",
      created_at: "2026-08-01T00:00:00+00:00",
      started_at: "2026-08-01T00:00:01+00:00",
      finished_at: null,
      error_code: null,
      error_message: null,
    };
    const requests: Array<{ url: string; method: string; body: unknown }> = [];
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      requests.push({
        url,
        method,
        body: typeof init?.body === "string" ? JSON.parse(init.body) : null,
      });
      if (url === "/root/api/presets") return jsonResponse({ items: [preset] });
      if (url === "/root/api/runs" && method === "POST") return jsonResponse({ run_id: "run-1", status: "queued" }, 202);
      if (url === "/root/api/runs") return jsonResponse({ items: [run] });
      if (url === "/root/api/runs/run-1/stop") return jsonResponse({ ...run, status: "stopping" }, 202);
      if (url === "/root/api/runs/run-1/events?after_sequence=4") return jsonResponse({ items: [trainingEvent] });
      if (url === "/root/api/runs/run-1/metrics/clients") return jsonResponse({ items: [clientMetric] });
      if (url === "/root/api/runs/run-1/metrics/aggregations") return jsonResponse({ items: [aggregationMetric] });
      if (url === "/root/api/runs/run-1") return jsonResponse(run);
      if (url === "/root/api/studies" && method === "POST") return jsonResponse({ study_id: "study-1", status: "queued" }, 202);
      if (url === "/root/api/studies/study-1/observations") return jsonResponse({ items: [observation] });
      if (url === "/root/api/studies/study-1") return jsonResponse(study);
      if (url === "/root/api/observations/legacy") return jsonResponse({ items: [{ ...observation, source: "legacy" }] });
      return jsonResponse({ error: { code: "NOT_FOUND", message: "unexpected request" } }, 404);
    };
    const client = new ApiClient("/root/", fetchImpl);

    expect(await client.presets()).toEqual([preset]);
    expect(await client.createRun("test-fixture", { cycles: 2 })).toEqual({ run_id: "run-1", status: "queued" });
    expect(await client.listRuns()).toEqual([run]);
    expect(await client.getRun("run-1")).toEqual(run);
    expect(await client.stopRun("run-1")).toMatchObject({ status: "stopping" });
    expect(await client.events("run-1", 4)).toEqual([trainingEvent]);
    expect(await client.clientMetrics("run-1")).toEqual([clientMetric]);
    expect(await client.aggregationMetrics("run-1")).toEqual([aggregationMetric]);
    expect(await client.createStudy({ preset: "research-lite", factors: { target_noise: [0.1] }, seeds: [7] })).toEqual({ study_id: "study-1", status: "queued" });
    expect(await client.getStudy("study-1")).toEqual(study);
    expect(await client.studyObservations("study-1")).toEqual([observation]);
    expect(await client.legacyObservations()).toEqual([{ ...observation, source: "legacy" }]);

    expect(requests).toContainEqual({
      url: "/root/api/runs",
      method: "POST",
      body: { preset: "test-fixture", overrides: { cycles: 2 } },
    });
    expect(requests).toContainEqual({
      url: "/root/api/studies",
      method: "POST",
      body: { preset: "research-lite", factors: { target_noise: [0.1] }, seeds: [7] },
    });
  });

  it("keeps a missing legacy endpoint as a stable 404 error", async () => {
    const fetchImpl: typeof fetch = async () => jsonResponse(
      { error: { code: "NOT_FOUND", message: "not found" } },
      404,
    );
    const client = new ApiClient("", fetchImpl);

    await expect(client.legacyObservations()).rejects.toMatchObject({
      code: "NOT_FOUND",
      status: 404,
    });
  });
});
