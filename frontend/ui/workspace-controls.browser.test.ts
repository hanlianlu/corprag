// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {ingestStore} from '../stores/ingest-store.ts';
import {workspaceStore} from '../stores/workspace-store.ts';
import type {DlWorkspaceScope} from './workspace-scope.ts';
import './workspace-scope.ts';
import type {DlIngestTarget} from './ingest-target.ts';
import './ingest-target.ts';
import type {ToastRequestDetail} from './toast.ts';

const originalFetch = window.fetch;

function buttonNamed(root: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(root.querySelectorAll<HTMLButtonElement>('button'))
    .find((button) => (button.getAttribute('aria-label') || button.textContent?.trim()) === name)
    ?? null;
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function corpusReceipt(runId: string, workspace: string) {
  return {
    run_id: runId,
    run_kind: 'corpus_mutation',
    lane: 'corpus_mutation',
    status: 'queued',
    status_url: `/web/api/corpus-runs/${runId}`,
    events_url: `/web/api/corpus-runs/${runId}/events`,
    cancel_url: `/web/api/corpus-runs/${runId}`,
    resume_url: `/web/api/corpus-runs/${runId}/resume`,
    workspace,
  };
}

function mountScope(): DlWorkspaceScope {
  const scope = document.createElement('dl-workspace-scope') as DlWorkspaceScope;
  document.body.appendChild(scope);
  return scope;
}

beforeEach(() => {
  workspaceStore.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
  ], ['default'], 'default');
  ingestStore.resetToPrimary();
});

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('owns a native expanded trigger and closes after typed creation intent', async () => {
  window.fetch = async () => new Response(JSON.stringify({
    workspace: 'research', display_name: 'Research',
  }), {status: 200, headers: {'Content-Type': 'application/json'}});
  const scope = mountScope();
  await scope.updateComplete;

  expect(customElements.get('workspace-scope')).to.equal(undefined);
  expect(customElements.get('workspace-create')).to.equal(undefined);
  expect(customElements.get('ingest-target')).to.equal(undefined);
  expect(scope.hasAttribute('role')).to.equal(false);
  expect(scope.hasAttribute('tabindex')).to.equal(false);
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  expect(trigger.getAttribute('aria-haspopup')).to.equal('dialog');
  expect(trigger.getAttribute('aria-expanded')).to.equal('false');
  expect(trigger.getAttribute('aria-controls')).to.equal('workspace-popover');

  trigger.click();
  await scope.updateComplete;
  const popover = scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')!;
  expect(popover.hidden).to.equal(false);
  expect(popover.querySelector('[role="option"]')).to.equal(null);
  expect(popover.querySelector('dl-workspace-create')).not.to.equal(null);
  expect(trigger.getAttribute('aria-expanded')).to.equal('true');
  const createButton = scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')!;
  expect(createButton.textContent?.trim()).to.equal('');
  expect(createButton.querySelector('svg.dl-popover-create-icon')?.getAttribute('viewBox'))
    .to.equal('0 0 24 24');

  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();
  await waitFor(() => workspaceStore.primary === 'research'
    && trigger.getAttribute('aria-expanded') === 'false');
  await scope.updateComplete;

  expect(scope.textContent).to.contain('Research');
  expect(popover.hidden).to.equal(true);
  expect(trigger.getAttribute('aria-expanded')).to.equal('false');
  expect(document.activeElement).to.equal(trigger);
});

it('restores typed-name deletion intent after a failed workspace request', async () => {
  let calls = 0;
  window.fetch = async () => {
    calls += 1;
    return new Response(JSON.stringify({error: 'Deletion denied'}), {
      status: 500,
      headers: {'Content-Type': 'application/json'},
    });
  };
  const scope = mountScope();
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Reset Corpus Default"]')?.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLDialogElement>('dialog')?.open));
  const input = scope.querySelector<HTMLInputElement>('[aria-label="Type Default to confirm"]')!;
  input.value = 'Default';
  input.dispatchEvent(new Event('input'));
  await scope.updateComplete;
  const submit = buttonNamed(scope, 'Reset Corpus')!;
  expect(submit.disabled).to.equal(false);

  submit.click();
  await waitFor(() => calls === 1 && submit.disabled === false);

  expect(input.value).to.equal('Default');
  expect(scope.querySelector<HTMLDialogElement>('dialog')?.open).to.equal(true);
  expect(document.activeElement).to.equal(input);
});

it('keeps submitted creation connected while its popover is dismissed', async () => {
  let resolveCreate!: (response: Response) => void;
  let requestSignal: AbortSignal | null | undefined;
  window.fetch = async (_input, init) => {
    requestSignal = init?.signal;
    return await new Promise<Response>((resolve) => { resolveCreate = resolve; });
  };
  const scope = mountScope();
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();

  scope.close();
  await scope.updateComplete;
  const composer = document.createElement('textarea');
  document.body.appendChild(composer);
  composer.focus();
  expect(scope.querySelector('dl-workspace-create')).not.to.equal(null);
  expect(scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')?.hidden)
    .to.equal(true);
  resolveCreate(new Response(JSON.stringify({
    workspace: 'research', display_name: 'Research',
  }), {status: 200, headers: {'Content-Type': 'application/json'}}));
  await waitFor(() => workspaceStore.primary === 'research');

  expect(requestSignal?.aborted).to.equal(false);
  expect(workspaceStore.records.some((record) => record.workspace === 'research')).to.equal(true);
  expect(document.activeElement).to.equal(composer);
});

it('aborts a submitted creation only when its Feature disconnects', async () => {
  let requestSignal: AbortSignal | null | undefined;
  window.fetch = async (_input, init) => await new Promise<Response>((_resolve, reject) => {
    requestSignal = init?.signal;
    requestSignal?.addEventListener('abort', () => {
      reject(new DOMException('Aborted', 'AbortError'));
    }, {once: true});
  });
  const scope = mountScope();
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();
  await waitFor(() => requestSignal !== undefined);

  scope.remove();
  await waitFor(() => requestSignal?.aborted === true);
  expect(workspaceStore.records.some((record) => record.workspace === 'research')).to.equal(false);

  document.body.appendChild(scope);
  await scope.updateComplete;
  expect(scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')?.disabled)
    .to.equal(false);
});

it('reports a submitted creation failure after its popover is dismissed', async () => {
  let rejectCreate!: (reason: unknown) => void;
  window.fetch = async () => await new Promise<Response>((_resolve, reject) => {
    rejectCreate = reject;
  });
  let receipt: ToastRequestDetail | null = null;
  const scope = mountScope();
  scope.addEventListener('dl-toast-request', (event) => { receipt = event.detail; });
  await scope.updateComplete;
  buttonNamed(scope, 'Choose search workspaces')?.click();
  await scope.updateComplete;
  const input = scope.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  input.value = 'Research';
  scope.querySelector<HTMLButtonElement>('[aria-label="Create workspace"]')?.click();

  scope.close();
  await scope.updateComplete;
  rejectCreate(new TypeError('network unavailable'));
  await waitFor(() => receipt !== null);

  expect(receipt).to.deep.equal({message: 'Failed to create workspace', duration: 3000});
  expect(scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')?.hidden)
    .to.equal(true);
});

it('keeps a pending reset modal and isolates the next reset operation', async () => {
  workspaceStore.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
    {workspace: 'research', displayName: 'Research', embeddingModel: 'embed'},
  ], ['default'], 'default');
  const requests: Array<(response: Response) => void> = [];
  window.fetch = async () => await new Promise<Response>((resolve) => { requests.push(resolve); });
  const scope = mountScope();
  await scope.updateComplete;
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  trigger.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Reset Corpus Default"]')?.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLDialogElement>('dialog')?.open));
  let input = scope.querySelector<HTMLInputElement>('[aria-label="Type Default to confirm"]')!;
  input.value = 'Default';
  input.dispatchEvent(new Event('input'));
  await scope.updateComplete;
  buttonNamed(scope, 'Reset Corpus')?.click();
  await waitFor(() => requests.length === 1
    && buttonNamed(scope, 'Accepting reset…')?.disabled === true);

  let dialog = scope.querySelector<HTMLDialogElement>('dialog')!;
  const cancel = buttonNamed(scope, 'Cancel')!;
  expect(cancel.disabled).to.equal(true);
  cancel.click();
  const cancelEvent = new Event('cancel', {cancelable: true});
  dialog.dispatchEvent(cancelEvent);
  expect(cancelEvent.defaultPrevented).to.equal(true);
  expect(dialog.open).to.equal(true);

  requests[0]!(new Response(JSON.stringify({error: 'Deletion denied'}), {
    status: 500,
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => buttonNamed(scope, 'Cancel')?.disabled === false
    && input.readOnly === false);
  buttonNamed(scope, 'Cancel')?.click();
  await waitFor(() => !scope.querySelector<HTMLDialogElement>('dialog')?.open
    && document.activeElement === buttonNamed(scope, 'Choose search workspaces'));
  await scope.updateComplete;

  const nextTrigger = buttonNamed(scope, 'Choose search workspaces')!;
  nextTrigger.click();
  await scope.updateComplete;
  expect(nextTrigger.getAttribute('aria-expanded')).to.equal('true');
  const deleteResearch = scope.querySelector<HTMLButtonElement>(
    '[aria-label="Reset Corpus Research"]',
  )!;
  expect(deleteResearch).not.to.equal(null);
  deleteResearch.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLInputElement>(
    '[aria-label="Type Research to confirm"]',
  )) && Boolean(scope.querySelector<HTMLDialogElement>('dialog[open]')));
  dialog = scope.querySelector<HTMLDialogElement>('dialog[open]')!;
  input = scope.querySelector<HTMLInputElement>('[aria-label="Type Research to confirm"]')!;
  input.value = 'Research';
  input.dispatchEvent(new Event('input'));
  await scope.updateComplete;
  const submit = buttonNamed(scope, 'Reset Corpus')!;
  expect(submit.disabled).to.equal(false);
  submit.click();
  await waitFor(() => requests.length === 2);
  expect(scope.querySelector<HTMLInputElement>('[aria-label="Type Research to confirm"]')?.readOnly)
    .to.equal(true);
  requests[1]!(new Response(JSON.stringify(corpusReceipt('run-reset', 'research')), {
    status: 202,
    headers: {'Content-Type': 'application/json'},
  }));
  await waitFor(() => !dialog.open);

  expect(workspaceStore.records.some((record) => record.workspace === 'research')).to.equal(true);
  expect(dialog.open).to.equal(false);
});

it('uses native dialog popover controls and restores ingest-trigger focus', async () => {
  workspaceStore.init([
    {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
    {workspace: 'research', displayName: 'Research', embeddingModel: 'embed'},
  ], ['default'], 'default');
  const ingest = document.createElement('dl-ingest-target') as DlIngestTarget;
  ingest.active = true;
  document.body.appendChild(ingest);
  await ingest.updateComplete;
  const trigger = ingest.querySelector<HTMLButtonElement>('#ingest-target-trigger')!;
  expect(trigger.tagName).to.equal('BUTTON');
  expect(ingest.querySelector('.ingest-target-label')).to.equal(null);
  expect(trigger.textContent?.trim()).to.equal('Default');
  expect(trigger.getAttribute('aria-haspopup')).to.equal('dialog');
  expect(trigger.getAttribute('aria-controls')).to.equal('ingest-target-popover');

  trigger.click();
  await ingest.updateComplete;
  const popover = ingest.querySelector<HTMLElement>(
    '[role="dialog"][aria-label="Select ingest workspace"]',
  )!;
  expect(popover.hidden).to.equal(false);
  expect(popover.querySelector('[role="option"]')).to.equal(null);
  expect(document.activeElement).to.equal(
    popover.querySelector('[data-ingest-workspace-choice][aria-pressed="true"]'),
  );

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => popover.hidden && document.activeElement === trigger);

  trigger.click();
  await ingest.updateComplete;
  const createInput = ingest.querySelector<HTMLInputElement>('[aria-label="New workspace name"]')!;
  createInput.focus();
  ingest.querySelector('dl-workspace-create')?.dispatchEvent(new CustomEvent(
    'dl-workspace-created',
    {detail: {workspace: 'new'}, bubbles: true, composed: true},
  ));
  await waitFor(() => popover.hidden && document.activeElement === trigger);
});

it('resets workspace popover and deletion state across disconnect and reconnect', async () => {
  const scope = mountScope();
  const modalStates: boolean[] = [];
  scope.addEventListener('dl-modal-state-change', (event) => {
    modalStates.push(event.detail.open);
  });
  await scope.updateComplete;
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  trigger.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Reset Corpus Default"]')?.click();
  const dialog = scope.querySelector<HTMLDialogElement>('dialog')!;
  await waitFor(() => dialog.open);
  expect(modalStates.at(-1)).to.equal(true);

  scope.remove();
  document.body.appendChild(scope);
  await scope.updateComplete;

  const reconnectedTrigger = buttonNamed(scope, 'Choose search workspaces')!;
  const popover = scope.querySelector<HTMLElement>('[role="dialog"][aria-label="Workspaces"]')!;
  expect(reconnectedTrigger.getAttribute('aria-expanded')).to.equal('false');
  expect(popover.hidden).to.equal(true);
  expect(dialog.open).to.equal(false);
  await waitFor(() => modalStates.at(-1) === false);

  reconnectedTrigger.click();
  await scope.updateComplete;
  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => popover.hidden && document.activeElement === reconnectedTrigger);
});

it('resets ingest popover state and document listeners across reconnect', async () => {
  const ingest = document.createElement('dl-ingest-target') as DlIngestTarget;
  ingest.active = true;
  document.body.appendChild(ingest);
  await ingest.updateComplete;
  const trigger = ingest.querySelector<HTMLButtonElement>(
    '[aria-label="Files in Default; choose file workspace"]',
  )!;
  trigger.click();
  await ingest.updateComplete;
  expect(trigger.getAttribute('aria-expanded')).to.equal('true');

  ingest.remove();
  document.body.appendChild(ingest);
  await ingest.updateComplete;
  const reconnectedTrigger = ingest.querySelector<HTMLButtonElement>(
    '[aria-label="Files in Default; choose file workspace"]',
  )!;
  const popover = ingest.querySelector<HTMLElement>(
    '[role="dialog"][aria-label="Select ingest workspace"]',
  )!;
  expect(reconnectedTrigger.getAttribute('aria-expanded')).to.equal('false');
  expect(popover.hidden).to.equal(true);

  reconnectedTrigger.click();
  await ingest.updateComplete;
  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await waitFor(() => popover.hidden && document.activeElement === reconnectedTrigger);
});

it('restores workspace-selector trigger focus when deletion is cancelled', async () => {
  const scope = mountScope();
  await scope.updateComplete;
  const trigger = buttonNamed(scope, 'Choose search workspaces')!;
  trigger.click();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('[aria-label="Reset Corpus Default"]')?.click();
  await waitFor(() => Boolean(scope.querySelector<HTMLDialogElement>('dialog')?.open));

  buttonNamed(scope, 'Cancel')?.click();
  await waitFor(() => document.activeElement === trigger);

  expect(scope.querySelector<HTMLDialogElement>('dialog')?.open).to.equal(false);
  expect(document.activeElement).to.equal(trigger);
});

it('loads more workspaces with coalescing, dedup, retry, and exhaustion', async () => {
  let olderRequests = 0;
  const loader = async (cursor: string | null) => {
    if (cursor === null) throw new Error('unexpected first-page fetch');
    olderRequests += 1;
    if (olderRequests === 1) {
      throw new Error('transient failure');
    }
    return {
      workspaces: [
        {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
        {workspace: 'finance', displayName: 'Finance', embeddingModel: 'embed'},
        {workspace: 'research', displayName: 'Research', embeddingModel: 'embed'},
      ],
      nextCursor: olderRequests === 2 ? 'cursor-2' : null,
    };
  };
  workspaceStore.init(
    [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed'}],
    ['default'],
    'default',
    loader,
    'cursor-1',
  );
  expect(workspaceStore.hasMoreWorkspaces).to.equal(true);

  const flight = workspaceStore.loadMoreWorkspaces();
  expect(workspaceStore.loadMoreWorkspaces()).to.equal(flight);
  await flight;
  expect(workspaceStore.workspaceLoadMoreState).to.equal('error');
  expect(workspaceStore.records.map((record) => record.workspace)).to.deep.equal(['default']);

  await workspaceStore.loadMoreWorkspaces();
  expect(workspaceStore.workspaceLoadMoreState).to.equal('idle');
  expect(workspaceStore.records.map((record) => record.workspace)).to.deep.equal([
    'default', 'finance', 'research',
  ]);
  expect(workspaceStore.hasMoreWorkspaces).to.equal(true);

  await workspaceStore.loadMoreWorkspaces();
  expect(workspaceStore.hasMoreWorkspaces).to.equal(false);
  expect(workspaceStore.records).to.have.length(3);
  expect(olderRequests).to.equal(3);
});

it('rejects stale load-more pages after a fresh init invalidates the flight', async () => {
  let resolve!: (page: {workspaces: {workspace: string; displayName: string; embeddingModel: string}[]; nextCursor: string | null}) => void;
  const pending = new Promise<{workspaces: {workspace: string; displayName: string; embeddingModel: string}[]; nextCursor: string | null}>((done) => {
    resolve = done;
  });
  const loader = async () => await pending;
  workspaceStore.init(
    [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed'}],
    ['default'],
    'default',
    loader,
    'cursor-1',
  );

  const flight = workspaceStore.loadMoreWorkspaces();
  workspaceStore.init(
    [{workspace: 'fresh', displayName: 'Fresh', embeddingModel: 'embed'}],
    ['fresh'],
    'fresh',
    loader,
    'cursor-fresh',
  );
  resolve({workspaces: [{workspace: 'stale', displayName: 'Stale', embeddingModel: 'e'}],
    nextCursor: null});
  await flight;

  expect(workspaceStore.records.map((record) => record.workspace)).to.deep.equal(['fresh']);
  expect(workspaceStore.hasMoreWorkspaces).to.equal(true);
  expect(workspaceStore.workspaceLoadMoreState).to.equal('idle');
});

it('renders an accessible load-more workspaces control in the picker', async () => {
  const loader = async () => ({
    workspaces: [{workspace: 'finance', displayName: 'Finance', embeddingModel: 'embed'}],
    nextCursor: null,
  });
  workspaceStore.init(
    [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed'}],
    ['default'],
    'default',
    loader,
    'cursor-1',
  );
  const scope = mountScope();
  await scope.updateComplete;
  scope.querySelector<HTMLButtonElement>('#workspace-trigger')!.click();
  await scope.updateComplete;

  const control = scope.querySelector<HTMLButtonElement>('[data-load-more-workspaces]');
  expect(control).not.to.equal(null);
  expect(control!.type).to.equal('button');
  expect(control!.textContent?.trim()).to.equal('Load more workspaces');

  control!.click();
  await waitFor(() => scope.querySelector('[data-load-more-workspaces]') === null);
  await waitFor(() => scope.querySelector('[data-workspaces-status]')?.textContent
    ?.includes('Loaded more workspaces.') ?? false);

  expect(scope.querySelector('[data-load-more-workspaces]')).to.equal(null);
  expect(scope.querySelector('[data-workspaces-status]')?.textContent).to.contain(
    'Loaded more workspaces.',
  );
  expect([...scope.querySelectorAll('[data-workspace-choice]:not([data-workspace-all])')]
    .map((item) => item.textContent?.trim())).to.contain('Finance');
});

it('preserves server-validated active and primary beyond the first display page', async () => {
  const loader = async () => ({
    workspaces: [{workspace: 'archive', displayName: 'Archive', embeddingModel: 'embed'}],
    nextCursor: null,
  });
  // biome-ignore lint/suspicious/noDocumentCookie: clearing the preference channel under test
  document.cookie = 'dlightrag_workspace_ids=;path=/;SameSite=Lax;Max-Age=0';
  // biome-ignore lint/suspicious/noDocumentCookie: clearing the preference channel under test
  document.cookie = 'dlightrag_workspace=;path=/;SameSite=Lax;Max-Age=0';
  workspaceStore.init(
    [{workspace: 'default', displayName: 'Default', embeddingModel: 'embed'}],
    ['default', 'finance', 'research'],
    'finance',
    loader,
    'cursor-1',
    ['default', 'finance', 'research'],
  );

  expect(workspaceStore.active).to.deep.equal(['default', 'finance', 'research']);
  expect(workspaceStore.primary).to.equal('finance');
  expect(document.cookie).to.contain('dlightrag_workspace_ids=default%2Cfinance%2Cresearch');
  expect(document.cookie).to.contain('dlightrag_workspace=finance');

  // Select-all stays complete over the full known set, not the loaded page.
  workspaceStore.selectAll();
  expect(workspaceStore.active).to.deep.equal(['default', 'finance', 'research']);

  const scope = mountScope();
  await scope.updateComplete;
  expect(scope.querySelector('#workspace-label')?.textContent).to.equal('All workspaces (3)');
});
