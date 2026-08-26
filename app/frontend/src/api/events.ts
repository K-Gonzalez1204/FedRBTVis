import type { RunStatus, TrainingEvent } from "./types";

export interface RunStreamState {
  runId: string;
  status: RunStatus | "unknown";
  connection: "idle" | "connecting" | "live" | "reconnecting" | "closed" | "gap";
  expectedSequence: number;
  lastSequence: number;
  events: TrainingEvent[];
  eventIds: ReadonlySet<string>;
}

export const initialRunStream = (runId: string): RunStreamState => ({
  runId,
  status: "unknown",
  connection: "idle",
  expectedSequence: 1,
  lastSequence: 0,
  events: [],
  eventIds: new Set<string>(),
});

const statusForEvent = (event: TrainingEvent): RunStatus | "unknown" => {
  switch (event.type) {
    case "run.started":
      return "running";
    case "run.stop_requested":
      return "stopping";
    case "run.stopped":
      return "stopped";
    case "run.completed":
      return "completed";
    case "run.failed":
      return "failed";
    default:
      return "unknown";
  }
};

const isTerminalEvent = (event: TrainingEvent): boolean =>
  event.type === "run.completed" || event.type === "run.failed" || event.type === "run.stopped";

export const reduceRunEvent = (state: RunStreamState, event: TrainingEvent): RunStreamState => {
  if (event.run_id !== state.runId) {
    throw new Error("event run_id does not match stream runId");
  }
  if (event.schema_version !== 1) {
    throw new Error("event schema_version is unsupported");
  }
  if (state.eventIds.has(event.event_id)) {
    return state;
  }
  if (event.sequence < state.expectedSequence) {
    throw new Error("event sequence is behind the expected sequence");
  }
  if (event.sequence > state.expectedSequence) {
    return { ...state, connection: "gap" };
  }

  const status = statusForEvent(event);
  return {
    ...state,
    status: status === "unknown" ? state.status : status,
    connection: isTerminalEvent(event) ? "closed" : "live",
    expectedSequence: event.sequence + 1,
    lastSequence: event.sequence,
    events: [...state.events, event],
    eventIds: new Set(state.eventIds).add(event.event_id),
  };
};
