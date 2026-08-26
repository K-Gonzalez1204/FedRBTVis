import type { RunStreamState } from "../../api/events";
import type { RunSummary } from "../../api/types";

export interface RunMonitorProps {
  run: RunSummary | null;
  stream: RunStreamState;
}

const statusLabel = (status: RunSummary["status"] | "unknown"): string => {
  switch (status) {
    case "queued":
      return "排队中";
    case "running":
      return "运行中";
    case "stopping":
      return "停止中";
    case "stopped":
      return "已停止";
    case "completed":
      return "运行完成";
    case "failed":
      return "运行失败";
    default:
      return "未知";
  }
};

const connectionLabel = (connection: RunStreamState["connection"]): string => {
  switch (connection) {
    case "idle":
      return "未连接";
    case "connecting":
      return "连接中";
    case "live":
      return "实时";
    case "reconnecting":
      return "重连中";
    case "closed":
      return "已关闭";
    case "gap":
      return "事件缺口";
    default:
      return "未知";
  }
};

export function RunMonitor({ run, stream }: RunMonitorProps) {
  const failed = run?.status === "failed";
  const fixture = run?.source === "fixture";
  return (
    <section className="run-monitor" aria-label="运行监控">
      <header className="run-monitor-header">
        <h2>Run Monitor</h2>
        <span className="status-badge">{run ? statusLabel(run.status) : "无运行"}</span>
        {run ? (
          <span className={`source-badge source-${run.source}`}>{run.source}</span>
        ) : null}
      </header>
      {fixture ? (
        <p className="source-warning">仅用于测试，不得作为研究结论</p>
      ) : null}
      {failed ? (
        <p className="run-error">
          运行失败：{run?.error_code ?? "UNKNOWN"}
          {run?.error_message ? ` ${run.error_message}` : ""}
        </p>
      ) : run ? (
        <dl className="run-summary">
          <div>
            <dt>Run ID</dt>
            <dd>{run.run_id}</dd>
          </div>
          <div>
            <dt>预设</dt>
            <dd>{run.preset}</dd>
          </div>
          <div>
            <dt>连接</dt>
            <dd>{connectionLabel(stream.connection)}</dd>
          </div>
          <div>
            <dt>事件序号</dt>
            <dd>{stream.lastSequence}</dd>
          </div>
        </dl>
      ) : (
        <p>尚未创建运行</p>
      )}
    </section>
  );
}
