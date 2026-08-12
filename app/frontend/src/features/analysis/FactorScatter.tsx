import type {
  ObservationRow,
  XFactor,
  YFactor,
} from "./types";

export interface FactorScatterProps {
  rows: ObservationRow[];
  x: XFactor;
  y: YFactor;
}

const WIDTH = 720;
const HEIGHT = 420;
const MARGIN = { top: 56, right: 24, bottom: 44, left: 64 };
const INNER_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const INNER_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

const extent = (values: number[]): [number, number] => {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return [0, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (min === max) {
    min -= 0.5;
    max += 0.5;
  }
  return [min, max];
};

const scale = (
  value: number,
  from: [number, number],
  to: [number, number],
): number => to[0] + ((value - from[0]) / (from[1] - from[0])) * (to[1] - to[0]);

export function FactorScatter({ rows, x, y }: FactorScatterProps) {
  const valid = rows.filter(
    (row) => Number.isFinite(row[x]) && Number.isFinite(row[y]),
  );
  const invalidCount = rows.length - valid.length;
  const xExtent = extent(valid.map((row) => row[x]));
  const yExtent = extent(valid.map((row) => row[y]));
  const points = valid.map((row) => ({
    row,
    cx: scale(row[x], xExtent, [MARGIN.left, WIDTH - MARGIN.right]),
    cy: scale(row[y], yExtent, [HEIGHT - MARGIN.bottom, MARGIN.top]),
  }));

  return (
    <figure
      className="factor-scatter"
      role="img"
      aria-label="因素散点图"
    >
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="presentation"
        focusable="false"
      >
        <line
          className="axis axis-x"
          x1={MARGIN.left}
          y1={HEIGHT - MARGIN.bottom}
          x2={WIDTH - MARGIN.right}
          y2={HEIGHT - MARGIN.bottom}
        />
        <line
          className="axis axis-y"
          x1={MARGIN.left}
          y1={MARGIN.top}
          x2={MARGIN.left}
          y2={HEIGHT - MARGIN.bottom}
        />
        <text className="axis-label" x={WIDTH / 2} y={HEIGHT - 10} textAnchor="middle">
          {x}
        </text>
        <text
          className="axis-label"
          transform={`rotate(-90 ${16} ${HEIGHT / 2})`}
          x={16}
          y={HEIGHT / 2}
          textAnchor="middle"
        >
          {y}
        </text>
        {points.map(({ row, cx, cy }) => {
          const label = `${row.source} client ${row.clientId}`;
          const title = (
            <title>{`${label}: ${x}=${row[x]}, ${y}=${row[y]}`}</title>
          );
          if (row.source === "fresh") {
            return (
              <circle
                key={`${row.runId}-${row.clientId}`}
                className="point point-fresh"
                cx={cx}
                cy={cy}
                r={5}
              >
                {title}
              </circle>
            );
          }
          if (row.source === "legacy") {
            return (
              <rect
                key={`${row.runId}-${row.clientId}`}
                className="point point-legacy"
                x={cx - 5}
                y={cy - 5}
                width={10}
                height={10}
              >
                {title}
              </rect>
            );
          }
          return (
            <polygon
              key={`${row.runId}-${row.clientId}`}
              className="point point-fixture"
              points={`${cx},${cy - 6} ${cx - 6},${cy + 5} ${cx + 6},${cy + 5}`}
            >
              {title}
            </polygon>
          );
        })}
        <g className="legend">
          <circle className="legend-point" cx={MARGIN.left + 8} cy={MARGIN.top + 12} r={5} />
          <text className="legend-label" x={MARGIN.left + 18} y={MARGIN.top + 16}>Fresh</text>
          <rect
            className="legend-point"
            x={MARGIN.left + 72}
            y={MARGIN.top + 7}
            width={10}
            height={10}
          />
          <text className="legend-label" x={MARGIN.left + 88} y={MARGIN.top + 16}>Legacy</text>
          <polygon
            className="legend-point"
            points={`${MARGIN.left + 146},${MARGIN.top + 6} ${MARGIN.left + 140},${MARGIN.top + 17} ${MARGIN.left + 152},${MARGIN.top + 17}`}
          />
          <text className="legend-label" x={MARGIN.left + 158} y={MARGIN.top + 16}>Fixture</text>
        </g>
      </svg>
      {invalidCount > 0 ? (
        <p className="chart-error">数据错误数：{invalidCount}</p>
      ) : null}
    </figure>
  );
}
