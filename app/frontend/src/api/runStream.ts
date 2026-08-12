import { ApiClient, validateTrainingEvent } from "./client";
import { initialRunStream, reduceRunEvent, type RunStreamState } from "./events";
import type { TrainingEvent } from "./types";

export interface WebSocketLike {
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: ((event: { code: number }) => void) | null;
  close(code?: number, reason?: string): void;
}

export type WebSocketFactory = (url: string) => WebSocketLike;

export interface TimerApi {
  setTimeout(callback: () => void, delay: number): unknown;
  clearTimeout(handle: unknown): void;
}

export interface RunEventSource {
  events(runId: string, afterSequence?: number): Promise<TrainingEvent[]>;
}

type StateListener = (state: RunStreamState) => void;
type ErrorListener = (error: Error) => void;

const BACKOFF_MS = [250, 500, 1000, 2000, 5000] as const;
const TERMINAL_STATUSES = new Set(["completed", "failed", "stopped"]);

const socketPath = (runId: string, afterSequence: number): string =>
  `/ws/runs/${encodeURIComponent(runId)}?after_sequence=${encodeURIComponent(String(afterSequence))}`;

const socketUrl = (runId: string, afterSequence: number): string => {
  const path = socketPath(runId, afterSequence);
  if (typeof location === "undefined" || !location.host) return path;
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}`;
};

const asError = (value: unknown): Error =>
  value instanceof Error ? value : new Error(String(value));

export class RunStream {
  private state: RunStreamState;
  private socket: WebSocketLike | null = null;
  private retryTimer: unknown | null = null;
  private retryIndex = 0;
  private stopped = false;
  private recoveringGap = false;
  private recoveryGeneration = 0;
  private readonly stateListeners = new Set<StateListener>();
  private readonly errorListeners = new Set<ErrorListener>();
  private readonly eventSource: RunEventSource;

  constructor(
    private readonly runId: string,
    private readonly createSocket: WebSocketFactory,
    private readonly timers: TimerApi,
    eventSource: RunEventSource = new ApiClient(""),
  ) {
    this.state = initialRunStream(runId);
    this.eventSource = eventSource;
  }

  start(): void {
    const lifecycleActive = this.state.connection !== "idle" && this.state.connection !== "closed";
    if (
      this.isTerminal()
      || this.recoveringGap
      || lifecycleActive
      || this.socket !== null
      || this.retryTimer !== null
    ) {
      return;
    }
    this.stopped = false;
    this.retryIndex = 0;
    this.updateState({ ...this.state, connection: "connecting" });
    this.openSocket();
  }

  stop(): void {
    this.stopped = true;
    this.recoveringGap = false;
    this.recoveryGeneration += 1;
    if (this.retryTimer !== null) {
      this.timers.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    this.updateState({ ...this.state, connection: "closed" });
    socket?.close(1000, "stream stopped");
  }

  snapshot(): RunStreamState {
    return {
      ...this.state,
      events: [...this.state.events],
      eventIds: new Set(this.state.eventIds),
    };
  }

  subscribe(listener: StateListener): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  onError(listener: ErrorListener): () => void {
    this.errorListeners.add(listener);
    return () => this.errorListeners.delete(listener);
  }

  private openSocket(): void {
    if (this.stopped || this.isTerminal()) return;
    let socket: WebSocketLike;
    try {
      socket = this.createSocket(socketUrl(this.runId, this.state.lastSequence));
    } catch (error) {
      this.notifyError(asError(error));
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      if (this.socket !== socket || this.stopped) return;
      this.updateState({ ...this.state, connection: "live" });
    };
    socket.onmessage = (message) => {
      if (this.socket !== socket || this.stopped) return;
      this.receiveMessage(message.data, socket);
    };
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      if (this.stopped || this.isTerminal()) {
        this.updateState({ ...this.state, connection: "closed" });
        return;
      }
      this.scheduleReconnect();
    };
  }

  private receiveMessage(data: unknown, socket: WebSocketLike): void {
    let event: TrainingEvent;
    try {
      if (typeof data !== "string") throw new Error("WebSocket message must be JSON text");
      event = validateTrainingEvent(JSON.parse(data));
      if (event.run_id !== this.runId) throw new Error("event run_id does not match stream runId");
    } catch (error) {
      this.closeForInvalidMessage(asError(error), socket);
      return;
    }

    if (event.sequence > this.state.expectedSequence) {
      this.beginGapRecovery(socket);
      return;
    }

    try {
      this.acceptEvent(event);
    } catch (error) {
      this.closeForInvalidMessage(asError(error), socket);
    }
  }

  private acceptEvent(event: TrainingEvent): void {
    const next = reduceRunEvent(this.state, event);
    if (next.connection === "gap") throw new Error("event sequence contains a gap");
    if (next !== this.state) this.retryIndex = 0;
    this.updateState(next);
  }

  private beginGapRecovery(socket: WebSocketLike): void {
    if (this.recoveringGap) return;
    this.recoveringGap = true;
    this.updateState({ ...this.state, connection: "gap" });
    if (this.socket === socket) this.socket = null;
    socket.close(1000, "recovering sequence gap");
    const generation = ++this.recoveryGeneration;
    void this.recoverGap(generation);
  }

  private async recoverGap(generation: number): Promise<void> {
    const afterSequence = this.state.lastSequence;
    try {
      const replay = await this.eventSource.events(this.runId, afterSequence);
      if (generation !== this.recoveryGeneration || this.stopped) return;
      for (const value of replay) {
        const event = validateTrainingEvent(value);
        if (event.run_id !== this.runId) throw new Error("replayed event run_id does not match stream runId");
        this.acceptEvent(event);
      }
      this.recoveringGap = false;
      if (this.stopped || this.isTerminal()) return;
      this.updateState({ ...this.state, connection: "reconnecting" });
      this.openSocket();
    } catch (error) {
      if (generation !== this.recoveryGeneration || this.stopped) return;
      this.recoveringGap = false;
      this.notifyError(asError(error));
      if (!this.stopped && !this.isTerminal()) this.scheduleGapRecovery(generation);
    }
  }

  private scheduleReconnect(): void {
    this.scheduleBackoff("reconnecting", () => this.openSocket());
  }

  private scheduleGapRecovery(generation: number): void {
    this.scheduleBackoff("gap", () => {
      this.recoveringGap = true;
      void this.recoverGap(generation);
    });
  }

  private scheduleBackoff(
    connection: "reconnecting" | "gap",
    callback: () => void,
  ): void {
    if (this.stopped || this.isTerminal() || this.retryTimer !== null) return;
    this.updateState({ ...this.state, connection });
    const index = Math.min(this.retryIndex, BACKOFF_MS.length - 1);
    const delay = BACKOFF_MS[index] ?? BACKOFF_MS[4];
    this.retryIndex += 1;
    let firedSynchronously = false;
    const handle = this.timers.setTimeout(() => {
      firedSynchronously = true;
      this.retryTimer = null;
      callback();
    }, delay);
    if (!firedSynchronously) this.retryTimer = handle;
  }

  private closeForInvalidMessage(error: Error, socket: WebSocketLike): void {
    this.stopped = true;
    this.notifyError(error);
    if (this.socket === socket) this.socket = null;
    this.updateState({ ...this.state, connection: "closed" });
    socket.close(1003, "invalid event");
  }

  private isTerminal(): boolean {
    return TERMINAL_STATUSES.has(this.state.status);
  }

  private updateState(state: RunStreamState): void {
    this.state = state;
    const snapshot = this.snapshot();
    for (const listener of this.stateListeners) {
      try {
        listener(snapshot);
      } catch {
        // Observers must not control the stream lifecycle.
      }
    }
  }

  private notifyError(error: Error): void {
    for (const listener of this.errorListeners) {
      try {
        listener(error);
      } catch {
        // One observer must not prevent the stream from closing or notifying others.
      }
    }
  }
}
