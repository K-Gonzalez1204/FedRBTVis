import { useEffect, useRef, useState } from "react";

import { initialRunStream, type RunStreamState } from "../../api/events";
import type { RunStream } from "../../api/runStream";
import type { PresetName } from "../../api/client";
import type { JsonValue, RunSummary } from "../../api/types";
import type { AppServices } from "../../App";

export interface RunController {
  run: RunSummary | null;
  stream: RunStreamState;
  error: Error | null;
  loading: boolean;
  launch(preset: PresetName, overrides: Record<string, JsonValue>): Promise<void>;
  stop(): Promise<void>;
}

const TERMINAL_STATES = new Set(["completed", "failed", "stopped"]);

const asError = (value: unknown): Error =>
  value instanceof Error ? value : new Error(String(value));

export function useRunController(
  services: AppServices,
  initialRun: RunSummary | null,
): RunController {
  const [run, setRun] = useState<RunSummary | null>(initialRun);
  const [stream, setStream] = useState<RunStreamState>(() =>
    initialRun ? initialRunStream(initialRun.run_id) : initialRunStream(""),
  );
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const streamRef = useRef<RunStream | null>(null);
  const runId = run?.run_id ?? null;

  useEffect(() => {
    if (runId === null) return;
    const stream = services.createStream(runId);
    setStream(initialRunStream(runId));
    streamRef.current = stream;
    const unsubscribeState = stream.subscribe((next) => {
      setStream(next);
      if (TERMINAL_STATES.has(next.status)) {
        void services.getRun(runId)
          .then(setRun)
          .catch((reason: unknown) => setError(asError(reason)));
      }
    });
    const unsubscribeError = stream.onError(setError);
    stream.start();
    return () => {
      unsubscribeState();
      unsubscribeError();
      stream.stop();
      streamRef.current = null;
    };
  }, [services, runId]);

  const launch = async (
    preset: PresetName,
    overrides: Record<string, JsonValue>,
  ): Promise<void> => {
    if (loading) return;
    setLoading(true);
    setError(null);
    try {
      const created = await services.createRun(preset, overrides);
      setRun(await services.getRun(created.run_id));
    } catch (reason) {
      setError(asError(reason));
    } finally {
      setLoading(false);
    }
  };

  const stop = async (): Promise<void> => {
    if (run === null || loading) return;
    setLoading(true);
    setError(null);
    try {
      setRun(await services.stopRun(run.run_id));
    } catch (reason) {
      setError(asError(reason));
    } finally {
      setLoading(false);
    }
  };

  return { run, stream, error, loading, launch, stop };
}
