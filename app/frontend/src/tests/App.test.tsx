import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { initialRunStream } from "../api/events";
import type { RunStreamState } from "../api/events";
import type { RunStatus, RunSummary } from "../api/types";
import { App, type AppServices } from "../App";
import { isLaunchDisabled } from "../features/runs/RunLauncher";
import { RunMonitor } from "../features/runs/RunMonitor";

const notUsed = (): never => {
  throw new Error("services must not be called during static render");
};

const fixtureServices: AppServices = {
  presets: notUsed,
  createRun: notUsed,
  getRun: notUsed,
  stopRun: notUsed,
  clientMetrics: notUsed,
  aggregationMetrics: notUsed,
  studyObservations: notUsed,
  legacyObservations: notUsed,
  createStream: notUsed,
};

const runSummary = (overrides: Partial<RunSummary>): RunSummary => ({
  schema_version: 1,
  run_id: "run-1",
  status: "completed",
  preset: "test-fixture",
  source: "fixture",
  created_at: "2026-08-01T00:00:00+00:00",
  started_at: null,
  finished_at: null,
  error_code: null,
  error_message: null,
  files: [],
  ...overrides,
});

const emptyStream = (): RunStreamState => initialRunStream("run-1");

describe("App", () => {
  it("labels fixture data as non-research", () => {
    const html = renderToStaticMarkup(
      <App
        services={fixtureServices}
        initialRun={runSummary({})}
      />,
    );
    expect(html).toContain("Fixture");
    expect(html).toContain("仅用于测试，不得作为研究结论");
  });
});

describe("RunMonitor", () => {
  it("shows a failed run message instead of completion", () => {
    const html = renderToStaticMarkup(
      <RunMonitor
        run={runSummary({
          status: "failed",
          error_code: "ARTIFACT_CORRUPT",
          error_message: "manifest hash mismatch",
        })}
        stream={emptyStream()}
      />,
    );
    expect(html).toContain("运行失败");
    expect(html).toContain("ARTIFACT_CORRUPT");
    expect(html).not.toContain("运行完成");
  });
});

describe("RunLauncher", () => {
  it("disables launch only while a run is active", () => {
    const activeStatuses: RunStatus[] = ["queued", "running", "stopping"];
    const terminalStatuses: RunStatus[] = ["completed", "failed", "stopped"];

    for (const status of activeStatuses) {
      expect(isLaunchDisabled(runSummary({ status }), false)).toBe(true);
    }
    for (const status of terminalStatuses) {
      expect(isLaunchDisabled(runSummary({ status }), false)).toBe(false);
    }
  });
});
