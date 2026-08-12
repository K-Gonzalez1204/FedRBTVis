import { useEffect, useState } from "react";

import type {
  AggregationMetric,
  ClientMetric,
  CreatedRun,
  Observation,
  Preset,
  PresetName,
} from "./api/client";
import type {
  JsonValue,
  RunSummary,
} from "./api/types";
import { ApiError } from "./api/client";
import { AnalysisPanel } from "./features/analysis/AnalysisPanel";
import {
  normalizeApiObservation,
  normalizeClientMetric,
} from "./features/analysis/normalize";
import type { ObservationRow } from "./features/analysis/types";
import type { RunStream } from "./api/runStream";
import { RunLauncher } from "./features/runs/RunLauncher";
import { RunMonitor } from "./features/runs/RunMonitor";
import { useRunController } from "./features/runs/useRunController";

export interface AppServices {
  presets(): Promise<Preset[]>;
  createRun(
    preset: PresetName,
    overrides: Record<string, JsonValue>,
  ): Promise<CreatedRun>;
  getRun(runId: string): Promise<RunSummary>;
  stopRun(runId: string): Promise<RunSummary>;
  clientMetrics(runId: string): Promise<ClientMetric[]>;
  aggregationMetrics(runId: string): Promise<AggregationMetric[]>;
  studyObservations(studyId: string): Promise<Observation[]>;
  legacyObservations(): Promise<Observation[]>;
  createStream(runId: string): RunStream;
}

export interface AppProps {
  services: AppServices;
  initialRun?: RunSummary | null;
}

export function App({ services, initialRun = null }: AppProps) {
  const { run, stream, error, loading, launch, stop } =
    useRunController(services, initialRun);
  const [freshRows, setFreshRows] = useState<ObservationRow[]>([]);
  const [legacyRows, setLegacyRows] = useState<ObservationRow[]>([]);
  const [freshError, setFreshError] = useState<string | null>(null);
  const [legacyError, setLegacyError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setFreshError(null);
    if (run === null) {
      setFreshRows([]);
      return;
    }
    services.clientMetrics(run.run_id)
      .then((items) => {
        if (!cancelled) {
          setFreshRows(
            items.map((item) => normalizeClientMetric(item, run.run_id)),
          );
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setFreshRows([]);
          setFreshError(reason instanceof Error ? reason.message : "观测加载失败");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [services, run]);

  useEffect(() => {
    let cancelled = false;
    services.legacyObservations()
      .then((items) => {
        if (!cancelled) setLegacyRows(items.map(normalizeApiObservation));
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setLegacyRows([]);
        if (reason instanceof ApiError && reason.status === 404) {
          setLegacyError("历史数据尚未导入");
        } else {
          setLegacyError(reason instanceof Error ? reason.message : "历史数据加载失败");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [services]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>FedRBTVis</h1>
        <p>联邦数据异质性实验与可视分析工作台</p>
      </header>
      <div className="workbench-grid">
        <RunLauncher
          services={services}
          activeRun={run}
          loading={loading}
          onLaunch={launch}
          onStop={stop}
        />
        <RunMonitor run={run} stream={stream} />
      </div>
      <AnalysisPanel
        freshRows={freshRows}
        legacyRows={legacyRows}
        freshError={freshError}
        legacyError={legacyError}
      />
      {error ? <p className="app-error">{error.message}</p> : null}
    </main>
  );
}
