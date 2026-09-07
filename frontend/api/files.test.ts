// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {getFailedFiles, getFilePanel, startFailedFileRetry} from './files.ts';

const originalDocument = globalThis.document;
const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;

test.beforeEach(() => {
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {location: {origin: 'http://localhost'}},
  });
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: {cookie: 'dlightrag_web_csrf=test-token'},
  });
});

test.afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: originalDocument,
  });
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: originalWindow,
  });
});

test('file pages encode the opaque cursor, pass abort, and normalize older payloads', async () => {
  const seen: Array<{url: string; signal: AbortSignal | null | undefined}> = [];
  globalThis.fetch = async (input, init) => {
    seen.push({url: String(input), signal: init?.signal});
    return new Response(JSON.stringify({
      workspace: 'finance',
      files: [],
    }), {headers: {'Content-Type': 'application/json'}});
  };
  const controller = new AbortController();

  const first = await getFilePanel('finance', null, controller.signal);
  const older = await getFilePanel('finance', 'opaque/cursor +', controller.signal);

  assert.equal(first.nextCursor, null);
  assert.equal(older.nextCursor, null);
  assert.deepEqual(seen, [
    {
      url: '/web/api/files?workspace=finance',
      signal: controller.signal,
    },
    {
      url: '/web/api/files?workspace=finance&cursor=opaque%2Fcursor+%2B',
      signal: controller.signal,
    },
  ]);
});

test('failed-file recovery uses bounded pages and accepts a durable Run', async () => {
  const seen: Array<{url: string; method: string; headers?: HeadersInit}> = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    seen.push({url, method, headers: init?.headers});
    if (method === 'POST') {
      return new Response(JSON.stringify({
        run_id: 'run-retry-1',
        run_kind: 'corpus_mutation',
        lane: 'corpus_mutation',
        workspace: 'finance',
        status: 'queued',
        status_url: '/web/api/corpus-runs/run-retry-1',
        events_url: '/web/api/corpus-runs/run-retry-1/events',
        cancel_url: '/web/api/corpus-runs/run-retry-1',
        resume_url: '/web/api/corpus-runs/run-retry-1/resume',
      }), {headers: {'Content-Type': 'application/json'}});
    }
    return new Response(JSON.stringify({
      workspace: 'finance',
      failed: [],
      next_cursor: null,
    }), {headers: {'Content-Type': 'application/json'}});
  };

  await getFailedFiles('finance', 'opaque cursor');
  const receipt = await startFailedFileRetry('finance');
  assert.equal(receipt.runId, 'run-retry-1');

  assert.deepEqual(seen, [
    {
      url: '/web/api/files/failed?workspace=finance&cursor=opaque+cursor',
      method: 'GET',
      headers: undefined,
    },
    {
      url: '/web/api/files/retry?workspace=finance',
      method: 'POST',
      headers: {'X-CSRF-Token': 'test-token'},
    },
  ]);
});
