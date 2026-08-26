import { useEffect, useState } from "react";

import type { Preset, PresetName } from "../../api/client";
import type { JsonValue, RunStatus, RunSummary } from "../../api/types";
import type { AppServices } from "../../App";

export interface RunLauncherProps {
  services: Pick<AppServices, "presets">;
  activeRun: RunSummary | null;
  loading: boolean;
  onLaunch(preset: PresetName, overrides: Record<string, JsonValue>): Promise<void>;
  onStop(): Promise<void>;
}

type CheckpointPolicy = "none" | "server-only" | "probe-clients";

const PRESET_ORDER: readonly PresetName[] = [
  "test-fixture",
  "research-lite",
  "historical-compatible",
];

const CHECKPOINT_OPTIONS: readonly CheckpointPolicy[] = [
  "none",
  "server-only",
  "probe-clients",
];

const ACTIVE_RUN_STATES: ReadonlySet<RunStatus> = new Set([
  "queued",
  "running",
  "stopping",
]);

export function isLaunchDisabled(
  activeRun: RunSummary | null,
  loading: boolean,
): boolean {
  return loading || (activeRun !== null && ACTIVE_RUN_STATES.has(activeRun.status));
}

const presetTitle = (preset: PresetName): string => {
  switch (preset) {
    case "test-fixture":
      return "Test Fixture";
    case "research-lite":
      return "Research Lite";
    case "historical-compatible":
      return "Historical Compatible";
    default:
      return preset;
  }
};

export function RunLauncher({
  services,
  activeRun,
  loading,
  onLaunch,
  onStop,
}: RunLauncherProps) {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selected, setSelected] = useState<PresetName>("test-fixture");
  const [seed, setSeed] = useState(0);
  const [cycles, setCycles] = useState(1);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [clientsPerStep, setClientsPerStep] = useState(2);
  const [checkpointPolicy, setCheckpointPolicy] =
    useState<CheckpointPolicy>("none");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    services.presets()
      .then((items) => {
        if (cancelled) return;
        setPresets(items);
        const first = items.find((item) => item.preset === selected) ?? items[0];
        if (first) setSelected(first.preset);
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setError(reason instanceof Error ? reason.message : "预设加载失败");
      });
    return () => {
      cancelled = true;
    };
  }, [services]);

  const disabled = isLaunchDisabled(activeRun, loading);
  const selectedPreset = presets.find((item) => item.preset === selected) ?? null;
  const overrides: Record<string, JsonValue> = {
    seed,
    cycles,
    local_epochs: localEpochs,
    clients_per_step: clientsPerStep,
    checkpoint_policy: checkpointPolicy,
  };

  const launch = async (): Promise<void> => {
    if (disabled) return;
    setError(null);
    try {
      await onLaunch(selected, overrides);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建运行失败");
    }
  };

  const stop = async (): Promise<void> => {
    setError(null);
    try {
      await onStop();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "停止运行失败");
    }
  };

  const canStop = activeRun?.status === "running" || activeRun?.status === "stopping";

  return (
    <section className="run-launcher" aria-label="运行启动器">
      <header className="panel-header">
        <h2>Run Launcher</h2>
      </header>
      <div className="preset-grid">
        {PRESET_ORDER.map((preset) => {
          const known = presets.find((item) => item.preset === preset);
          return (
            <button
              key={preset}
              className="preset-card"
              type="button"
              aria-pressed={selected === preset}
              disabled={disabled}
              onClick={() => setSelected(preset)}
            >
              <strong>{presetTitle(preset)}</strong>
              <span>{known?.dataset ?? "加载中"}</span>
            </button>
          );
        })}
      </div>
      {selected === "historical-compatible" ? (
        <p className="preset-warning">长时、不会自动执行</p>
      ) : null}
      <fieldset className="run-fields" disabled={disabled}>
        <legend>运行参数</legend>
        <label>
          Seed
          <input
            type="number"
            value={seed}
            min={0}
            onChange={(event) => setSeed(Number(event.target.value))}
          />
        </label>
        <label>
          Cycles
          <input
            type="number"
            value={cycles}
            min={1}
            onChange={(event) => setCycles(Number(event.target.value))}
          />
        </label>
        <label>
          Local epochs
          <input
            type="number"
            value={localEpochs}
            min={1}
            onChange={(event) => setLocalEpochs(Number(event.target.value))}
          />
        </label>
        <label>
          Clients per step
          <input
            type="number"
            value={clientsPerStep}
            min={1}
            onChange={(event) => setClientsPerStep(Number(event.target.value))}
          />
        </label>
        <label>
          Checkpoint policy
          <select
            value={checkpointPolicy}
            onChange={(event) => {
              setCheckpointPolicy(event.target.value as CheckpointPolicy);
            }}
          >
            {CHECKPOINT_OPTIONS.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </label>
      </fieldset>
      {selectedPreset ? (
        <pre className="config-summary">
          {JSON.stringify(
            {
              preset: selected,
              ...overrides,
            },
            null,
            2,
          )}
        </pre>
      ) : null}
      {error ? <p className="form-error">{error}</p> : null}
      <div className="launcher-actions">
        <button type="button" className="primary-action" disabled={disabled} onClick={() => void launch()}>
          启动运行
        </button>
        {canStop ? (
          <button type="button" className="danger-action" disabled={loading} onClick={() => void stop()}>
            停止运行
          </button>
        ) : null}
      </div>
    </section>
  );
}
