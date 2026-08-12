import { useMemo, useState } from "react";

import {
  defaultFilters,
  filterObservations,
} from "./normalize";
import type {
  AnalysisFilters,
  ObservationRow,
  Source,
  XFactor,
  YFactor,
} from "./types";
import { ClientTable } from "./ClientTable";
import { FactorScatter } from "./FactorScatter";

export interface AnalysisPanelProps {
  freshRows: ObservationRow[];
  legacyRows: ObservationRow[];
  freshError: string | null;
  legacyError: string | null;
}

const X_OPTIONS: readonly { value: XFactor; label: string }[] = [
  { value: "actualNoise", label: "实际噪声" },
  { value: "actualEmd", label: "实际类别距离" },
  { value: "sampleCount", label: "样本量" },
  { value: "lidK", label: "LID k" },
];

const Y_OPTIONS: readonly { value: YFactor; label: string }[] = [
  { value: "lidMean", label: "LID 均值" },
  { value: "testLoss", label: "测试损失" },
  { value: "testAccuracy", label: "测试准确率" },
];

const SOURCES: readonly Source[] = ["fixture", "fresh", "legacy"];
const SOURCE_LABELS: Record<Source, string> = {
  fixture: "Fixture",
  fresh: "Fresh",
  legacy: "Legacy",
};

export function AnalysisPanel({
  freshRows,
  legacyRows,
  freshError,
  legacyError,
}: AnalysisPanelProps) {
  const [filters, setFilters] = useState<AnalysisFilters>(defaultFilters());
  const [x, setX] = useState<XFactor>("actualNoise");
  const [y, setY] = useState<YFactor>("lidMean");
  const allRoles = useMemo(
    () =>
      Array.from(
        new Set([...freshRows, ...legacyRows].map((row) => row.role)),
      ).sort(),
    [freshRows, legacyRows],
  );
  const rows = useMemo(
    () => filterObservations([...freshRows, ...legacyRows], filters),
    [freshRows, legacyRows, filters],
  );

  const toggleSource = (source: Source): void => {
    setFilters((current) => ({
      ...current,
      sources: current.sources.includes(source)
        ? current.sources.filter((item) => item !== source)
        : [...current.sources, source],
    }));
  };

  const toggleRole = (role: string): void => {
    setFilters((current) => ({
      ...current,
      roles: current.roles.includes(role)
        ? current.roles.filter((item) => item !== role)
        : [...current.roles, role],
    }));
  };

  return (
    <section className="analysis-panel" aria-label="因素分析">
      <header className="panel-header">
        <h2>Factor Analysis</h2>
      </header>
      {legacyError ? <p className="legacy-missing">历史数据尚未导入</p> : null}
      {freshError ? <p className="data-error">{freshError}</p> : null}
      <div className="analysis-controls">
        <label>
          X
          <select value={x} onChange={(event) => setX(event.target.value as XFactor)}>
            {X_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <label>
          Y
          <select value={y} onChange={(event) => setY(event.target.value as YFactor)}>
            {Y_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={filters.includeFixture}
            onChange={(event) =>
              setFilters((current) => ({
                ...current,
                includeFixture: event.target.checked,
              }))
            }
          />
          显示 Fixture
        </label>
        <fieldset className="source-filter">
          <legend>来源</legend>
          {SOURCES.map((source) => (
            <label key={source}>
              <input
                type="checkbox"
                checked={filters.sources.includes(source)}
                onChange={() => toggleSource(source)}
              />
              {SOURCE_LABELS[source]}
            </label>
          ))}
        </fieldset>
        {allRoles.length > 0 ? (
          <fieldset className="role-filter">
            <legend>角色</legend>
            {allRoles.map((role) => (
              <label key={role}>
                <input
                  type="checkbox"
                  checked={filters.roles.includes(role)}
                  onChange={() => toggleRole(role)}
                />
                {role}
              </label>
            ))}
          </fieldset>
        ) : null}
      </div>
      <FactorScatter rows={rows} x={x} y={y} />
      <ClientTable rows={rows} />
    </section>
  );
}
