import "./styles.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ApiClient } from "./api/client";
import {
  RunStream,
  type TimerApi,
  type WebSocketFactory,
} from "./api/runStream";
import { App, type AppServices } from "./App";

const client = new ApiClient("");

const webSocketFactory: WebSocketFactory = (url) => {
  const socket = new WebSocket(url);
  const adapter = {
    onopen: null as (() => void) | null,
    onmessage: null as ((event: { data: unknown }) => void) | null,
    onclose: null as ((event: { code: number }) => void) | null,
    close(code?: number, reason?: string): void {
      socket.close(code, reason);
    },
  };
  socket.onopen = () => adapter.onopen?.();
  socket.onmessage = (message) => adapter.onmessage?.({ data: message.data });
  socket.onclose = (event) => adapter.onclose?.({ code: event.code });
  return adapter;
};

const browserTimers: TimerApi = {
  setTimeout(callback, delay) {
    return window.setTimeout(callback, delay);
  },
  clearTimeout(handle) {
    window.clearTimeout(handle as number);
  },
};

const services: AppServices = {
  presets: () => client.presets(),
  createRun: (preset, overrides) => client.createRun(preset, overrides),
  getRun: (runId) => client.getRun(runId),
  stopRun: (runId) => client.stopRun(runId),
  clientMetrics: (runId) => client.clientMetrics(runId),
  aggregationMetrics: (runId) => client.aggregationMetrics(runId),
  studyObservations: (studyId) => client.studyObservations(studyId),
  legacyObservations: () => client.legacyObservations(),
  createStream: (runId) => new RunStream(runId, webSocketFactory, browserTimers, client),
};

const root = document.getElementById("root");
if (root === null) {
  throw new Error("missing #root mount element");
}
createRoot(root).render(
  <StrictMode>
    <App services={services} />
  </StrictMode>,
);
