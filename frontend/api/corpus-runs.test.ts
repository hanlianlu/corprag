// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import {corpusRunActive, getCorpusRunStatus, resumeCorpusRun} from './corpus-runs.ts';

const originalDocument = globalThis.document;
const originalFetch = globalThis.fetch;

test.beforeEach(() => {
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
});

test('Corpus mutation polling follows the accepted canonical status URL', async () => {
  let requested = '';
  globalThis.fetch = async (input) => {
    requested = String(input);
    return new Response(JSON.stringify({
      run_id: 'run-1',
      run_kind: 'corpus_mutation',
      lane: 'corpus_mutation',
      status: 'succeeded',
      status_url: '/web/api/corpus-runs/run-1',
      events_url: '/web/api/corpus-runs/run-1/events',
      cancel_url: '/web/api/corpus-runs/run-1',
      resume_url: '/web/api/corpus-runs/run-1/resume',
      result: {action: 'retry', document_count: 1},
    }), {headers: {'Content-Type': 'application/json'}});
  };

  const run = await getCorpusRunStatus('/web/api/corpus-runs/run-1');

  assert.equal(requested, '/web/api/corpus-runs/run-1');
  assert.equal(run.status, 'succeeded');
  assert.equal(run.result?.document_count, 1);
  assert.equal(corpusRunActive(run), false);
  assert.equal(corpusRunActive({...run, status: 'running'}), true);
});

test('Corpus repair resume uses the advertised same-origin endpoint', async () => {
  let requested: {url: string; method: string} | null = null;
  globalThis.fetch = async (input, init) => {
    requested = {url: String(input), method: init?.method ?? 'GET'};
    return new Response(JSON.stringify({
      run_id: 'run-1',
      run_kind: 'corpus_mutation',
      lane: 'corpus_mutation',
      status: 'queued',
      status_url: '/web/api/corpus-runs/run-1',
      events_url: '/web/api/corpus-runs/run-1/events',
      cancel_url: '/web/api/corpus-runs/run-1',
      resume_url: '/web/api/corpus-runs/run-1/resume',
    }), {status: 202, headers: {'Content-Type': 'application/json'}});
  };

  const run = await resumeCorpusRun('/web/api/corpus-runs/run-1/resume');

  assert.deepEqual(requested, {
    url: '/web/api/corpus-runs/run-1/resume',
    method: 'POST',
  });
  assert.equal(run.status, 'queued');
});
