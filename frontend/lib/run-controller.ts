// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
  cancelRun,
  getRun,
  type ConversationTurn,
} from '../api/conversations.ts';
import {createSSEParser, parseData} from './sse.ts';
import {
  answerEventCursorStore,
  type AnswerEventCursorStore,
} from '../stores/answer-event-cursor-store.ts';

const DEFAULT_MAX_RECONNECT_ATTEMPTS = 5;
const DEFAULT_RECONNECT_DELAY_MS = 500;

export type AnswerRunEvent =
  | {kind: 'token'; text: string}
  | {kind: 'reset'}
  | {kind: 'progress'; payload: unknown}
  | {
      kind: 'tool';
      eventType: 'tool_start' | 'tool_progress' | 'tool_end';
      payload: unknown;
    }
  | {kind: 'memory'; payload: unknown}
  | {kind: 'error'; payload: unknown}
  | {kind: 'done'; payload: unknown};

type PendingAnswerRunEvent =
  | Exclude<AnswerRunEvent, {kind: 'token'}>
  | {kind: 'token'; parts: string[]};

interface FrameTicket {
  handle: unknown;
}

interface AnswerRunBatch {
  readonly onBatch: (events: readonly AnswerRunEvent[]) => void;
  pending: PendingAnswerRunEvent[];
  frame: FrameTicket | null;
}

export type RunFrameScheduler = (callback: () => void) => unknown;
export type RunFrameCanceller = (handle: unknown) => void;

interface DefaultFrameHandle {
  kind: 'animation' | 'timeout';
  handle: number | ReturnType<typeof setTimeout>;
}

function scheduleDefaultFrame(callback: () => void): DefaultFrameHandle {
  if (typeof globalThis.requestAnimationFrame === 'function') {
    return {kind: 'animation', handle: globalThis.requestAnimationFrame(callback)};
  }
  return {kind: 'timeout', handle: setTimeout(callback, 16)};
}

function cancelDefaultFrame(value: unknown): void {
  const frame = value as DefaultFrameHandle;
  if (frame.kind === 'animation' && typeof globalThis.cancelAnimationFrame === 'function') {
    globalThis.cancelAnimationFrame(frame.handle as number);
  } else if (frame.kind === 'timeout') {
    clearTimeout(frame.handle as ReturnType<typeof setTimeout>);
  }
}

export type FollowResult =
  | {kind: 'aborted'}
  | {kind: 'terminal'; stored?: ConversationTurn}
  | {kind: 'retryable'; stored: ConversationTurn}
  | {kind: 'error'; message: string};

export interface RunControllerOptions {
  fetch?: typeof fetch;
  getRun?: typeof getRun;
  cancelRun?: typeof cancelRun;
  maxReconnectAttempts?: number;
  reconnectDelayMs?: number;
  onStateChange?: () => void;
  scheduleFrame?: RunFrameScheduler;
  cancelFrame?: RunFrameCanceller;
  cursorStore?: AnswerEventCursorStore;
}

/** Owns the transport and lifecycle resources for this tab's one followed run. */
export class RunController {
  readonly #fetch: typeof fetch;
  readonly #getRun: typeof getRun;
  readonly #cancelRun: typeof cancelRun;
  readonly #maxReconnectAttempts: number;
  readonly #reconnectDelayMs: number;
  readonly #onStateChange: () => void;
  readonly #scheduleFrame: RunFrameScheduler;
  readonly #cancelFrame: RunFrameCanceller;
  readonly #cursors: AnswerEventCursorStore;

  #lifecycleController: AbortController | null = null;
  readonly #cancelControllers = new Map<string, AbortController>();
  readonly #readers = new Set<ReadableStreamDefaultReader<Uint8Array>>();
  #batch: AnswerRunBatch | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #active = false;
  #stopping = false;
  #runId: string | null = null;

  constructor(options: RunControllerOptions = {}) {
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.#getRun = options.getRun ?? getRun;
    this.#cancelRun = options.cancelRun ?? cancelRun;
    this.#maxReconnectAttempts = options.maxReconnectAttempts ?? DEFAULT_MAX_RECONNECT_ATTEMPTS;
    this.#reconnectDelayMs = options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS;
    this.#onStateChange = options.onStateChange ?? (() => {});
    if ((options.scheduleFrame === undefined) !== (options.cancelFrame === undefined)) {
      throw new Error('scheduleFrame and cancelFrame must be provided together');
    }
    this.#scheduleFrame = options.scheduleFrame ?? scheduleDefaultFrame;
    this.#cancelFrame = options.cancelFrame ?? cancelDefaultFrame;
    this.#cursors = options.cursorStore ?? answerEventCursorStore;
  }

  get active(): boolean {
    return this.#active;
  }

  get stopping(): boolean {
    return this.#stopping;
  }

  get runId(): string | null {
    return this.#runId;
  }

  /** Return the current run lifecycle signal only while that run remains attached. */
  signalFor(runId: string): AbortSignal | null {
    const signal = this.#lifecycleController?.signal;
    return this.#runId === runId && signal && !signal.aborted ? signal : null;
  }

  /** Begin following a durable run discovered from conversation history. */
  beginFollow(runId: string, cancelRequested: boolean): AbortSignal | null {
    if (this.#active) return null;
    this.#startLifecycle();
    this.#active = true;
    this.#runId = runId;
    this.#stopping = cancelRequested || this.#cancelControllers.has(runId);
    this.#notify();
    return this.#lifecycleController?.signal ?? null;
  }

  /** Stop this tab's reader and timers without asking the server to cancel. */
  detach(): void {
    this.#releaseBatch();
    this.#clearTimer();
    this.#lifecycleController?.abort();
    this.#lifecycleController = null;
    this.#cancelReaders();
    this.#resetState();
  }

  /** Release a completed or failed local lifecycle if it is still current. */
  finish(runId?: string): void {
    if (runId !== undefined && this.#runId !== runId) return;
    this.#releaseBatch();
    this.#clearTimer();
    this.#lifecycleController?.abort();
    this.#lifecycleController = null;
    this.#cancelReaders();
    this.#resetState();
  }

  /** Ask the server to stop the durable run; disconnecting alone never does this. */
  async cancel(): Promise<void> {
    const runId = this.#runId;
    if (!runId || this.#stopping || this.#cancelControllers.has(runId)) return;
    this.#stopping = true;
    this.#notify();
    const controller = new AbortController();
    this.#cancelControllers.set(runId, controller);
    try {
      await this.#cancelRun(runId, controller.signal);
    } catch {
      if (!controller.signal.aborted && this.#runId === runId) {
        this.#stopping = false;
        this.#notify();
      }
    } finally {
      if (this.#cancelControllers.get(runId) === controller) {
        this.#cancelControllers.delete(runId);
      }
    }
  }

  /** Abort every owned request when the Feature itself leaves the document. */
  disconnect(): void {
    this.detach();
    for (const controller of this.#cancelControllers.values()) controller.abort();
    this.#cancelControllers.clear();
  }

  /** Follow and replay one durable event stream from its last consumed sequence. */
  async follow(
    conversationId: string,
    runId: string,
    onBatch: (events: readonly AnswerRunEvent[]) => void,
  ): Promise<FollowResult> {
    const signal = this.#lifecycleController?.signal;
    if (!signal || signal.aborted || this.#runId !== runId) return {kind: 'aborted'};
    const batch: AnswerRunBatch = {onBatch, pending: [], frame: null};
    this.#batch = batch;
    let barrenAttempts = 0;

    const mayRetry = (before: number): boolean => {
      barrenAttempts = this.#cursors.lastSequence(conversationId, runId) > before
        ? 0
        : barrenAttempts + 1;
      return barrenAttempts <= this.#maxReconnectAttempts;
    };

    try {
      while (!signal.aborted) {
        const after = this.#cursors.lastSequence(conversationId, runId);
        let response: Response;
        try {
          response = await this.#fetch(`/web/api/runs/${encodeURIComponent(runId)}/events`, {
            signal,
            headers: after > 0 ? {'Last-Event-ID': String(after)} : undefined,
          });
        } catch {
          this.#flushBatch(batch);
          if (signal.aborted) return {kind: 'aborted'};
          if (!mayRetry(after)) break;
          if (!await this.#delay(signal)) return {kind: 'aborted'};
          continue;
        }
        if (response.status === 404 || response.status === 410) break;
        if (!response.ok) return {kind: 'error', message: 'Service error. Please try again.'};

        let terminal = false;
        try {
          terminal = await this.#readEvents(
            response,
            conversationId,
            runId,
            signal,
            batch,
          );
        } catch {
          this.#flushBatch(batch);
          if (signal.aborted) return {kind: 'aborted'};
          if (!mayRetry(after)) break;
          if (!await this.#delay(signal)) return {kind: 'aborted'};
          continue;
        }
        this.#flushBatch(batch);
        if (terminal) return {kind: 'terminal'};
        if (!mayRetry(after)) break;
        if (!await this.#delay(signal)) return {kind: 'aborted'};
      }

      if (signal.aborted) return {kind: 'aborted'};
      try {
        const stored = await this.#getRun(runId, signal);
        if (stored.status === 'queued' || stored.status === 'running') {
          return {kind: 'retryable', stored};
        }
        return {kind: 'terminal', stored};
      } catch {
        if (signal.aborted) return {kind: 'aborted'};
        return {kind: 'error', message: 'Service error. Please try again.'};
      }
    } finally {
      this.#flushBatch(batch);
      if (this.#batch === batch) this.#batch = null;
    }
  }

  #startLifecycle(): void {
    this.#releaseBatch();
    this.#clearTimer();
    this.#lifecycleController?.abort();
    this.#cancelReaders();
    this.#lifecycleController = new AbortController();
    this.#runId = null;
    this.#stopping = false;
  }

  #resetState(): void {
    const changed = this.#active || this.#stopping || this.#runId !== null;
    this.#active = false;
    this.#stopping = false;
    this.#runId = null;
    if (changed) this.#notify();
  }

  #notify(): void {
    this.#onStateChange();
  }

  #releaseBatch(): void {
    const batch = this.#batch;
    if (!batch) return;
    this.#flushBatch(batch);
    if (this.#batch === batch) this.#batch = null;
  }

  #scheduleBatch(batch: AnswerRunBatch): void {
    if (batch.frame || batch.pending.length === 0) return;
    const ticket: FrameTicket = {handle: undefined};
    batch.frame = ticket;
    ticket.handle = this.#scheduleFrame(() => {
      if (batch.frame !== ticket) return;
      batch.frame = null;
      this.#flushBatch(batch);
    });
  }

  #cancelBatchFrame(batch: AnswerRunBatch): void {
    const ticket = batch.frame;
    if (!ticket) return;
    batch.frame = null;
    this.#cancelFrame(ticket.handle);
  }

  #flushBatch(batch: AnswerRunBatch): void {
    this.#cancelBatchFrame(batch);
    if (batch.pending.length === 0) return;
    const pending = batch.pending;
    batch.pending = [];
    const events: AnswerRunEvent[] = pending.map((event) => event.kind === 'token'
      ? {kind: 'token', text: event.parts.join('')}
      : event);
    batch.onBatch(events);
  }

  #queueEvent(batch: AnswerRunBatch, event: AnswerRunEvent): void {
    if (event.kind === 'reset' || event.kind === 'error' || event.kind === 'done') {
      this.#flushBatch(batch);
      batch.pending.push(event);
      this.#flushBatch(batch);
      return;
    }
    if (event.kind === 'token') {
      const previous = batch.pending.at(-1);
      if (previous?.kind === 'token') previous.parts.push(event.text);
      else batch.pending.push({kind: 'token', parts: [event.text]});
    } else {
      batch.pending.push(event);
    }
    this.#scheduleBatch(batch);
  }

  #interpretEvent(eventType: string, data: string): AnswerRunEvent | null {
    if (eventType === 'token') {
      const parsed = parseData(data);
      return {kind: 'token', text: typeof parsed === 'string' ? parsed : String(parsed)};
    }
    if (eventType === 'reset') return {kind: 'reset'};
    if (eventType === 'progress') return {kind: 'progress', payload: parseData(data)};
    if (eventType === 'tool_start' || eventType === 'tool_progress' || eventType === 'tool_end') {
      return {kind: 'tool', eventType, payload: parseData(data)};
    }
    if (eventType === 'memory_operation_settled') {
      return {kind: 'memory', payload: parseData(data)};
    }
    if (eventType === 'error') return {kind: 'error', payload: parseData(data)};
    if (eventType === 'done') return {kind: 'done', payload: parseData(data)};
    return null;
  }

  #cancelReaders(): void {
    for (const reader of this.#readers) void reader.cancel().catch(() => {});
    this.#readers.clear();
  }

  #clearTimer(): void {
    if (this.#timer === null) return;
    clearTimeout(this.#timer);
    this.#timer = null;
  }

  #delay(signal: AbortSignal): Promise<boolean> {
    if (signal.aborted) return Promise.resolve(false);
    return new Promise((resolve) => {
      const onAbort = (): void => {
        this.#clearTimer();
        resolve(false);
      };
      signal.addEventListener('abort', onAbort, {once: true});
      this.#timer = setTimeout(() => {
        this.#timer = null;
        signal.removeEventListener('abort', onAbort);
        resolve(!signal.aborted);
      }, this.#reconnectDelayMs);
    });
  }

  async #readEvents(
    response: Response,
    conversationId: string,
    runId: string,
    signal: AbortSignal,
    batch: AnswerRunBatch,
  ): Promise<boolean> {
    if (!response.body) throw new Error('Response body is not streamable');
    let terminal = false;
    const parser = createSSEParser((eventType, data, id) => {
      if (signal.aborted || this.#batch !== batch) return;
      const sequence = Number(id);
      if (Number.isFinite(sequence) && sequence > 0) {
        if (sequence <= this.#cursors.lastSequence(conversationId, runId)) return;
        this.#cursors.recordSequence(conversationId, runId, sequence);
      }
      const event = this.#interpretEvent(eventType, data);
      if (!event) return;
      this.#queueEvent(batch, event);
      if (event.kind === 'done' || event.kind === 'error') terminal = true;
    });
    const reader = response.body.getReader();
    this.#readers.add(reader);
    const decoder = new TextDecoder();
    try {
      while (!signal.aborted) {
        const result = await reader.read();
        if (result.done) break;
        parser.push(decoder.decode(result.value, {stream: true}));
      }
      if (!signal.aborted) {
        parser.push(decoder.decode());
        parser.flush();
      }
    } finally {
      this.#readers.delete(reader);
      void reader.cancel().catch(() => {});
      this.#flushBatch(batch);
    }
    return terminal;
  }
}
