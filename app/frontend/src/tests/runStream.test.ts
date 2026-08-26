import { describe, expect, it, vi } from "vitest";
import { RunStream, type TimerApi, type WebSocketLike } from "../api/runStream";
import type { TrainingEvent } from "../api/types";

const event = (sequence: number, type: TrainingEvent["type"] = "client.started"): TrainingEvent => ({
  schema_version: 1,
  event_id: `event-${sequence}`,
  run_id: "run-1",
  sequence,
  type,
  created_at: "2026-08-01T00:00:00+00:00",
  payload: {},
});

const eventJson = (sequence: number, type?: TrainingEvent["type"]): string =>
  JSON.stringify(event(sequence, type));

class FakeSocket implements WebSocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((message: { data: unknown }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  closeCalls = 0;
  closeObserved: (() => void) | null = null;

  open(): void {
    this.onopen?.();
  }

  message(data: unknown): void {
    this.onmessage?.({ data });
  }

  closeUnexpectedly(): void {
    this.onclose?.({ code: 1006 });
  }

  close(): void {
    this.closeCalls += 1;
    this.closeObserved?.();
    this.onclose?.({ code: 1000 });
  }
}

class FakeSocketFactory {
  readonly urls: string[] = [];
  readonly sockets: FakeSocket[] = [];
  readonly create = (url: string): WebSocketLike => {
    this.urls.push(url);
    const socket = new FakeSocket();
    this.sockets.push(socket);
    return socket;
  };

  get latest(): FakeSocket {
    const socket = this.sockets.at(-1);
    if (socket === undefined) throw new Error("no socket was created");
    return socket;
  }
}

const immediateTimer: TimerApi = {
  setTimeout(callback) {
    callback();
    return 1;
  },
  clearTimeout() {},
};

class QueuedTimers implements TimerApi {
  readonly delays: number[] = [];
  readonly callbacks: Array<() => void> = [];
  readonly cleared: unknown[] = [];

  setTimeout(callback: () => void, delay: number): unknown {
    this.delays.push(delay);
    this.callbacks.push(callback);
    return this.callbacks.length;
  }

  clearTimeout(handle: unknown): void {
    this.cleared.push(handle);
  }

  runNext(): void {
    const callback = this.callbacks.shift();
    if (callback === undefined) throw new Error("no queued timer");
    callback();
  }
}

describe("RunStream", () => {
  it("reconnects from the last accepted sequence", () => {
    const sockets = new FakeSocketFactory();
    const stream = new RunStream("run-1", sockets.create, immediateTimer);

    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(1, "run.started"));
    sockets.latest.closeUnexpectedly();

    expect(sockets.urls).toEqual([
      "/ws/runs/run-1?after_sequence=0",
      "/ws/runs/run-1?after_sequence=1",
    ]);
  });

  it("does not turn a disconnect into completion", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    const stream = new RunStream("run-1", sockets.create, timers);
    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(1, "run.started"));

    sockets.latest.closeUnexpectedly();

    expect(stream.snapshot().status).toBe("running");
    expect(stream.snapshot().connection).toBe("reconnecting");
  });

  it("uses the bounded reconnect backoff sequence", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    const stream = new RunStream("run-1", sockets.create, timers);
    stream.start();

    for (let attempt = 0; attempt < 7; attempt += 1) {
      sockets.latest.open();
      sockets.latest.closeUnexpectedly();
      timers.runNext();
    }

    expect(timers.delays).toEqual([250, 500, 1000, 2000, 5000, 5000, 5000]);
  });

  it("backs off when the initial socket factory call throws", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    const errors: string[] = [];
    let attempts = 0;
    const stream = new RunStream("run-1", (url) => {
      attempts += 1;
      if (attempts === 1) throw new Error("constructor failed");
      return sockets.create(url);
    }, timers);
    stream.onError((error) => errors.push(error.message));

    expect(() => stream.start()).not.toThrow();
    expect(stream.snapshot().connection).toBe("reconnecting");
    expect(timers.delays).toEqual([250]);
    timers.runNext();

    expect(errors).toEqual(["constructor failed"]);
    expect(sockets.urls).toEqual(["/ws/runs/run-1?after_sequence=0"]);
  });

  it("continues backoff when a reconnect socket factory call throws", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    let attempts = 0;
    const stream = new RunStream("run-1", (url) => {
      attempts += 1;
      if (attempts === 2) throw new Error("reconnect failed");
      return sockets.create(url);
    }, timers);
    stream.start();
    sockets.latest.open();
    sockets.latest.closeUnexpectedly();

    expect(() => timers.runNext()).not.toThrow();

    expect(stream.snapshot().connection).toBe("reconnecting");
    expect(timers.delays).toEqual([250, 500]);
    timers.runNext();
    expect(sockets.urls).toHaveLength(2);
  });

  it("stops reconnecting after an explicit stop and cancels its timer", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    const stream = new RunStream("run-1", sockets.create, timers);
    stream.start();
    sockets.latest.open();
    sockets.latest.closeUnexpectedly();

    stream.stop();

    expect(timers.cleared).toEqual([1]);
    expect(stream.snapshot().connection).toBe("closed");
  });

  it("does not reconnect after a terminal event", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    const stream = new RunStream("run-1", sockets.create, timers);
    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(1, "run.completed"));

    sockets.latest.closeUnexpectedly();

    expect(stream.snapshot().status).toBe("completed");
    expect(stream.snapshot().connection).toBe("closed");
    expect(timers.delays).toEqual([]);
  });

  it("keeps start as a no-op after a terminal event", () => {
    const sockets = new FakeSocketFactory();
    const stream = new RunStream("run-1", sockets.create, immediateTimer);
    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(1, "run.completed"));
    sockets.latest.closeUnexpectedly();

    stream.start();

    expect(stream.snapshot().status).toBe("completed");
    expect(stream.snapshot().connection).toBe("closed");
    expect(sockets.urls).toHaveLength(1);
  });

  it("reports an invalid message before closing without reconnecting", () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    const stream = new RunStream("run-1", sockets.create, timers);
    const observations: string[] = [];
    stream.onError((error) => observations.push(`error:${error.message}`));
    stream.subscribe((state) => {
      if (state.connection === "closed") observations.push("closed");
    });
    stream.start();
    sockets.latest.open();
    sockets.latest.closeObserved = () => observations.push("socket-closed");

    sockets.latest.message(JSON.stringify({ ...event(1), event_id: undefined }));

    expect(observations[0]).toMatch(/^error:/);
    expect(observations[1]).toBe("closed");
    expect(observations[2]).toBe("socket-closed");
    expect(sockets.latest.closeCalls).toBe(1);
    expect(timers.delays).toEqual([]);
  });

  it("still closes an invalid stream when an error listener throws", () => {
    const sockets = new FakeSocketFactory();
    const stream = new RunStream("run-1", sockets.create, immediateTimer);
    stream.onError(() => {
      throw new Error("observer failed");
    });
    stream.start();
    sockets.latest.open();

    expect(() => sockets.latest.message("not-json")).not.toThrow();

    expect(sockets.latest.closeCalls).toBe(1);
    expect(stream.snapshot().connection).toBe("closed");
  });

  it("does not treat a state listener failure as an invalid event", () => {
    const sockets = new FakeSocketFactory();
    const stream = new RunStream("run-1", sockets.create, immediateTimer);
    stream.subscribe(() => {
      throw new Error("observer failed");
    });

    expect(() => stream.start()).not.toThrow();
    sockets.latest.open();
    sockets.latest.message(eventJson(1, "run.started"));

    expect(stream.snapshot().status).toBe("running");
    expect(sockets.latest.closeCalls).toBe(0);
  });

  it("fills a sequence gap over HTTP before reopening the socket", async () => {
    const sockets = new FakeSocketFactory();
    const calls: Array<{ runId: string; afterSequence: number }> = [];
    const eventSource = {
      async events(runId: string, afterSequence: number): Promise<TrainingEvent[]> {
        calls.push({ runId, afterSequence });
        expect(sockets.urls).toHaveLength(1);
        return [event(1, "run.started"), event(2, "client.completed")];
      },
    };
    const stream = new RunStream("run-1", sockets.create, immediateTimer, eventSource);
    stream.start();
    sockets.latest.open();

    sockets.latest.message(eventJson(2, "client.completed"));
    await vi.waitFor(() => expect(sockets.urls).toHaveLength(2));

    expect(calls).toEqual([{ runId: "run-1", afterSequence: 0 }]);
    expect(stream.snapshot().lastSequence).toBe(2);
    expect(sockets.urls[1]).toBe("/ws/runs/run-1?after_sequence=2");
  });

  it("keeps start as a no-op while HTTP gap recovery is in flight", async () => {
    const sockets = new FakeSocketFactory();
    let resolveReplay: ((events: TrainingEvent[]) => void) | undefined;
    const replay = new Promise<TrainingEvent[]>((resolve) => {
      resolveReplay = resolve;
    });
    const stream = new RunStream("run-1", sockets.create, immediateTimer, {
      events: async () => replay,
    });
    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(2));

    stream.start();

    expect(stream.snapshot().connection).toBe("gap");
    expect(sockets.urls).toHaveLength(1);
    resolveReplay?.([event(1, "run.started"), event(2)]);
    await vi.waitFor(() => expect(sockets.urls).toHaveLength(2));
    expect(sockets.urls[1]).toBe("/ws/runs/run-1?after_sequence=2");
  });

  it("retries failed HTTP gap recovery before reopening the socket", async () => {
    const sockets = new FakeSocketFactory();
    const timers = new QueuedTimers();
    let calls = 0;
    const stream = new RunStream("run-1", sockets.create, timers, {
      async events(): Promise<TrainingEvent[]> {
        calls += 1;
        if (calls === 1) throw new Error("HTTP unavailable");
        return [event(1, "run.started"), event(2)];
      },
    });
    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(2));
    await vi.waitFor(() => expect(timers.delays).toEqual([250]));

    expect(sockets.urls).toHaveLength(1);
    timers.runNext();
    await vi.waitFor(() => expect(sockets.urls).toHaveLength(2));

    expect(calls).toBe(2);
    expect(sockets.urls[1]).toBe("/ws/runs/run-1?after_sequence=2");
  });

  it("ignores an in-flight gap replay after stop", async () => {
    const sockets = new FakeSocketFactory();
    let resolveReplay: ((events: TrainingEvent[]) => void) | undefined;
    const replay = new Promise<TrainingEvent[]>((resolve) => {
      resolveReplay = resolve;
    });
    const stream = new RunStream("run-1", sockets.create, immediateTimer, {
      events: async () => replay,
    });
    stream.start();
    sockets.latest.open();
    sockets.latest.message(eventJson(2));

    stream.stop();
    resolveReplay?.([event(1, "run.started"), event(2)]);
    for (let turn = 0; turn < 4; turn += 1) await Promise.resolve();

    expect(stream.snapshot().connection).toBe("closed");
    expect(stream.snapshot().events).toEqual([]);
    expect(sockets.urls).toHaveLength(1);
  });

  it("notifies subscribers and supports unsubscription", () => {
    const sockets = new FakeSocketFactory();
    const stream = new RunStream("run-1", sockets.create, immediateTimer);
    const states: string[] = [];
    const unsubscribe = stream.subscribe((state) => states.push(state.connection));
    stream.start();
    sockets.latest.open();
    unsubscribe();
    sockets.latest.message(eventJson(1, "run.started"));

    expect(states).toEqual(["connecting", "live"]);
  });

  it("uses a secure absolute socket URL on an HTTPS page", () => {
    vi.stubGlobal("location", { protocol: "https:", host: "example.test" });
    const sockets = new FakeSocketFactory();
    const stream = new RunStream("run-1", sockets.create, immediateTimer);

    stream.start();

    expect(sockets.urls).toEqual(["wss://example.test/ws/runs/run-1?after_sequence=0"]);
    vi.unstubAllGlobals();
  });
});
