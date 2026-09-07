// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './failed-file-recovery.ts';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';

const originalFetch = window.fetch;
const originalSetTimeout = window.setTimeout;

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => originalSetTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function failedPage(workspace = 'personel', failed = true) {
  return {
    workspace,
    failed: failed ? [{
      document_id: 'doc-1',
      file_name: '货币、权力与人.pdf',
      error: 'technical embedding failure',
      updated_at: '2026-08-31T21:36:15',
    }] : [],
    next_cursor: null,
  };
}

function receipt() {
  return {
    run_id: 'run-retry-1',
    run_kind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status: 'queued',
    status_url: '/web/api/corpus-runs/run-retry-1',
    events_url: '/web/api/corpus-runs/run-retry-1/events',
    cancel_url: '/web/api/corpus-runs/run-retry-1',
    resume_url: '/web/api/corpus-runs/run-retry-1/resume',
    workspace: 'personel',
  };
}

function terminalRun() {
  return {
    ...receipt(),
    status: 'succeeded',
    result: {action: 'retry', documents: [{document_id: 'doc-1', status: 'succeeded'}]},
  };
}

function mount(): DlFailedFileRecovery {
  const recovery = document.createElement('dl-failed-file-recovery') as DlFailedFileRecovery;
  recovery.workspace = 'personel';
  recovery.active = true;
  document.body.appendChild(recovery);
  return recovery;
}

afterEach(() => {
  window.fetch = originalFetch;
  window.setTimeout = originalSetTimeout;
  document.body.replaceChildren();
});

it('keeps Retry all available while failed-document details are collapsed', async () => {
  window.fetch = async () => new Response(JSON.stringify(failedPage()), {
    headers: {'Content-Type': 'application/json'},
  });

  const recovery = mount();
  await waitFor(() => recovery.loading === false && recovery.page !== null);

  const disclosure = recovery.querySelector<HTMLDetailsElement>('.failed-file-recovery')!;
  const retry = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(disclosure.open).to.equal(false);
  expect(disclosure.textContent).to.contain('1 document needs attention');
  expect(retry.textContent?.trim()).to.equal('Retry all');
  expect(retry.disabled).to.equal(false);
  expect(recovery.textContent).to.contain('technical embedding failure');
});

it('accepts one durable retry Run and polls its canonical status URL', async () => {
  const requests: Array<{url: string; method: string}> = [];
  let failedLists = 0;
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({url, method});
    if (method === 'POST') {
      return new Response(JSON.stringify(receipt()), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
    }
    if (url === '/web/api/corpus-runs/run-retry-1') {
      return new Response(JSON.stringify(terminalRun()), {
        headers: {'Content-Type': 'application/json'},
      });
    }
    failedLists += 1;
    return new Response(JSON.stringify(failedPage('personel', failedLists === 1)), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  let completed = false;
  recovery.addEventListener('dl-failed-file-recovery-complete', () => { completed = true; });
  await waitFor(() => recovery.page?.failed.length === 1);
  recovery.querySelector<HTMLButtonElement>('.failed-file-retry')?.click();
  const dialog = recovery.querySelector<HTMLDialogElement>('#retry-failed-files-dialog')!;
  await waitFor(() => dialog.open);
  dialog.returnValue = 'retry';
  dialog.close();

  await waitFor(() => completed);
  expect(recovery.recovery?.runId).to.equal('run-retry-1');
  expect(recovery.recovery?.status).to.equal('succeeded');
  expect(requests).to.deep.include({url: '/web/api/files/retry?workspace=personel', method: 'POST'});
  expect(requests).to.deep.include({url: '/web/api/corpus-runs/run-retry-1', method: 'GET'});
  expect(requests.some(({url}) => url.includes('/files/retry/run-'))).to.equal(false);
});

it('offers explicit same-Run resume while waiting for operator repair', async () => {
  const requests: Array<{url: string; method: string}> = [];
  window.fetch = async (input, init) => {
    const url = String(input);
    const method = init?.method ?? 'GET';
    requests.push({url, method});
    if (method === 'POST') {
      return new Response(JSON.stringify({...receipt(), status: 'queued'}), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response(JSON.stringify(failedPage()), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.page?.failed.length === 1);
  recovery.recovery = {
    runId: 'run-retry-1',
    runKind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status: 'running',
    statusUrl: '/web/api/corpus-runs/run-retry-1',
    eventsUrl: '/web/api/corpus-runs/run-retry-1/events',
    cancelUrl: '/web/api/corpus-runs/run-retry-1',
    resumeUrl: '/web/api/corpus-runs/run-retry-1/resume',
    workspace: 'personel',
    fileCount: null,
    result: null,
    phase: 'waiting_for_repair',
    errorKind: null,
    errorMessage: null,
    repairReason: 'Inspect upstream state.',
    repairRemedy: 'Repair it, then resume.',
  };
  await recovery.updateComplete;

  const resume = recovery.querySelector<HTMLButtonElement>('.failed-file-retry')!;
  expect(resume.textContent?.trim()).to.equal('Resume after repair');
  expect(recovery.textContent).to.contain('Inspect upstream state.');
  resume.click();

  await waitFor(() => requests.some(({url, method}) => (
    url === '/web/api/corpus-runs/run-retry-1/resume' && method === 'POST'
  )));
  expect(recovery.recovery?.runId).to.equal('run-retry-1');
});

it('clears workspace-scoped state before loading the next workspace', async () => {
  let resolveOther!: (response: Response) => void;
  const other = new Promise<Response>((resolve) => { resolveOther = resolve; });
  window.fetch = async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.searchParams.get('workspace') === 'other') return other;
    return new Response(JSON.stringify(failedPage()), {
      headers: {'Content-Type': 'application/json'},
    });
  };

  const recovery = mount();
  await waitFor(() => recovery.page?.failed.length === 1);
  recovery.workspace = 'other';
  await recovery.updateComplete;
  expect(recovery.page).to.equal(null);
  expect(recovery.textContent).not.to.contain('货币、权力与人.pdf');

  resolveOther(new Response(JSON.stringify(failedPage('other', false)), {
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => recovery.loading === false);
});
