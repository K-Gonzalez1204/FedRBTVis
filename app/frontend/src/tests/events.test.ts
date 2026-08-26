import { describe, expect, it } from "vitest";
import { initialRunStream, reduceRunEvent } from "../api/events";
import type { TrainingEvent } from "../api/types";

const event = (sequence: number, type: TrainingEvent["type"]): TrainingEvent => ({
  schema_version: 1,
  event_id: `event-${sequence}`,
  run_id: "run-1",
  sequence,
  type,
  created_at: "2026-08-01T00:00:00+00:00",
  payload: {},
});

describe("reduceRunEvent", () => {
  it("accepts ordered events and derives terminal state", () => {
    const started = reduceRunEvent(initialRunStream("run-1"), event(1, "run.started"));
    const completed = reduceRunEvent(started, event(2, "run.completed"));
    expect(completed.lastSequence).toBe(2);
    expect(completed.status).toBe("completed");
    expect(completed.events).toHaveLength(2);
  });

  it("deduplicates replayed event ids", () => {
    const once = reduceRunEvent(initialRunStream("run-1"), event(1, "run.started"));
    const twice = reduceRunEvent(once, event(1, "run.started"));
    expect(twice.events).toHaveLength(1);
  });

  it("marks a sequence gap without inventing events", () => {
    const gap = reduceRunEvent(initialRunStream("run-1"), event(2, "client.started"));
    expect(gap.connection).toBe("gap");
    expect(gap.events).toHaveLength(0);
    expect(gap.expectedSequence).toBe(1);
  });

  it("rejects events from another run", () => {
    expect(() => reduceRunEvent(initialRunStream("run-1"), {
      ...event(1, "run.started"),
      run_id: "run-2",
    })).toThrow("run_id");
  });

  it("rejects unsupported schema versions", () => {
    expect(() => reduceRunEvent(initialRunStream("run-1"), {
      ...event(1, "run.started"),
      schema_version: 2,
    } as unknown as TrainingEvent)).toThrow("schema_version");
  });

  it("rejects unseen events that arrive behind the expected sequence", () => {
    const started = reduceRunEvent(initialRunStream("run-1"), event(1, "run.started"));
    expect(() => reduceRunEvent(started, event(0, "client.started"))).toThrow("sequence");
  });

  it("keeps client completion from creating a terminal state", () => {
    const started = reduceRunEvent(initialRunStream("run-1"), event(1, "run.started"));
    const clientCompleted = reduceRunEvent(started, event(2, "client.completed"));
    expect(clientCompleted.status).toBe("running");
  });
});
