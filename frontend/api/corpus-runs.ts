// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

export const corpusRunReceipt = v.pipe(
  v.object({
    run_id: v.string(),
    run_kind: v.literal('corpus_mutation'),
    lane: v.literal('corpus_mutation'),
    status: v.picklist(['queued', 'running', 'succeeded', 'failed', 'cancelled']),
    status_url: v.string(),
    events_url: v.string(),
    cancel_url: v.string(),
    resume_url: v.string(),
    workspace: v.string(),
    file_count: v.optional(v.nullable(v.number())),
  }),
  v.transform((wire) => ({
    runId: wire.run_id,
    runKind: wire.run_kind,
    lane: wire.lane,
    status: wire.status,
    statusUrl: wire.status_url,
    eventsUrl: wire.events_url,
    cancelUrl: wire.cancel_url,
    resumeUrl: wire.resume_url,
    workspace: wire.workspace,
    fileCount: wire.file_count ?? null,
  })),
);
export type WebCorpusRunReceipt = v.InferOutput<typeof corpusRunReceipt>;

const corpusRunStatus = v.pipe(
  v.object({
    run_id: v.string(),
    run_kind: v.literal('corpus_mutation'),
    lane: v.literal('corpus_mutation'),
    status: v.picklist(['queued', 'running', 'succeeded', 'failed', 'cancelled']),
    status_url: v.string(),
    events_url: v.string(),
    cancel_url: v.string(),
    resume_url: v.string(),
    result: v.optional(v.nullable(v.record(v.string(), v.unknown()))),
    phase: v.optional(v.nullable(v.string())),
    error_kind: v.optional(v.nullable(v.string())),
    error_message: v.optional(v.nullable(v.string())),
    repair_reason: v.optional(v.nullable(v.string())),
    repair_remedy: v.optional(v.nullable(v.string())),
  }),
  v.transform((wire) => ({
    runId: wire.run_id,
    runKind: wire.run_kind,
    lane: wire.lane,
    status: wire.status,
    statusUrl: wire.status_url,
    eventsUrl: wire.events_url,
    cancelUrl: wire.cancel_url,
    resumeUrl: wire.resume_url,
    result: wire.result ?? null,
    phase: wire.phase ?? null,
    errorKind: wire.error_kind ?? null,
    errorMessage: wire.error_message ?? null,
    repairReason: wire.repair_reason ?? null,
    repairRemedy: wire.repair_remedy ?? null,
  })),
);
export type WebCorpusRunStatus = v.InferOutput<typeof corpusRunStatus>;

export function corpusRunActive<T extends {status: string}>(
  run: T | null | undefined,
): run is T & {status: 'queued' | 'running'} {
  return run?.status === 'queued' || run?.status === 'running';
}

export async function getCorpusRunStatus(
  statusUrl: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunStatus> {
  const response = await fetch(statusUrl, {signal});
  return parseWire(response, corpusRunStatus, makeError, 'Failed to read Corpus update status');
}

export async function resumeCorpusRun(
  resumeUrl: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunStatus> {
  const response = await fetch(resumeUrl, {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  return parseWire(response, corpusRunStatus, makeError, 'Corpus repair resume failed');
}

function makeError(status: number, message: string): Error {
  return new CorpusRunApiError(status, message);
}

export class CorpusRunApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'CorpusRunApiError';
    this.status = status;
  }
}
