// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerPresentation} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {ingestStore} from '../stores/ingest-store.ts';
import {workspaceStore} from '../stores/workspace-store.ts';
import './inspector.ts';

defineDesignSystemElements();
import type {DlInspector, InspectorStateDetail} from './inspector.ts';

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;

function media(compact: boolean): (query: string) => MediaQueryList {
  return (query: string) => ({
    matches: compact ? query === '(width < 1200px)' : query === '(min-width: 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  });
}

const presentation: AnswerPresentation = {
  answerText: 'See sources.',
  parts: [],
  sources: [
    {
      id: '1', title: 'First source', sourceUrl: null, downloadUrl: null,
      chunks: [{
        chunkIdx: 1, contentHtml: '<p>First evidence</p>', pageNumber: 2,
        imageUrl: null, thumbnailUrl: null,
      }],
    },
    {
      id: '2', title: 'Second source', sourceUrl: null, downloadUrl: null,
      chunks: [],
    },
  ],
  evidenceImages: [],
  artifacts: [],
  artifactOutcome: {status: 'complete', issues: []},
};

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function buttonNamed(root: ParentNode, name: string): HTMLElement | null {
  return Array.from(root.querySelectorAll<HTMLElement>('button, dl-icon-button'))
    .find((button) => button.getAttribute('aria-label') === name || button.textContent?.trim() === name)
    ?? null;
}

beforeEach(() => {
  workspaceStore.init(
    [
      {workspace: 'default', displayName: 'Default', embeddingModel: 'embed'},
      {workspace: 'secondary', displayName: 'Secondary', embeddingModel: 'embed'},
    ],
    ['default'],
    'default',
  );
  ingestStore.resetToPrimary();
});

afterEach(() => {
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  document.body.replaceChildren();
});

it('owns Sources state, selection, commands, and focus restoration through its public seam', async () => {
  window.matchMedia = media(false);
  const returnFocus = document.createElement('button');
  returnFocus.textContent = 'Citation 1';
  const inspector = document.createElement('dl-inspector') as DlInspector;
  document.body.append(returnFocus, inspector);
  returnFocus.focus();

  await inspector.openSources(presentation, '1', '1', returnFocus);
  await inspector.updateComplete;
  const panel = inspector.querySelector<HTMLElement>('aside[aria-labelledby="panel-title"]')!;
  expect(inspector.open).to.equal(true);
  expect(inspector.kind).to.equal('sources');
  expect(inspector.querySelector('#panel-title')?.textContent).to.equal('Sources');
  expect(panel.hasAttribute('aria-label')).to.equal(false);
  expect(panel.hasAttribute('aria-modal')).to.equal(false);
  expect(panel.querySelector('[aria-expanded="true"]')).not.to.equal(null);

  buttonNamed(inspector, 'Show all')?.click();
  await waitFor(() => buttonNamed(inspector, 'Collapse all') !== null);
  expect(inspector.sourcesExpanded).to.equal(true);

  buttonNamed(inspector, 'Close panel')?.click();
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(inspector.open).to.equal(false);
  expect(document.activeElement).to.equal(returnFocus);
  expect(customElements.get('source-panel-view')).to.equal(undefined);
});

async function scrollableSources(): Promise<{
  inspector: DlInspector;
  content: HTMLElement;
  answer: AnswerPresentation;
}> {
  window.matchMedia = media(false);
  const inspector = document.createElement('dl-inspector');
  const style = document.createElement('style');
  style.textContent = `
    #panel-content { height: 160px; overflow-y: auto; overflow-anchor: none; }
    dl-inspector-sources { display: block; }
    [data-source-header] { height: 48px; }
    [data-chunk] { height: 240px; }
  `;
  document.body.append(style, inspector);
  const answer: AnswerPresentation = {
    ...presentation,
    sources: Array.from({length: 12}, (_, index) => ({
      ...presentation.sources[0], id: String(index + 1),
    })),
  };
  await inspector.openSources(answer);
  await inspector.querySelector('dl-inspector-sources')!.updateComplete;
  const content = inspector.querySelector<HTMLElement>('#panel-content')!;
  expect(content.scrollHeight).to.be.greaterThan(content.clientHeight);
  return {inspector, content, answer};
}

for (const chunkId of [undefined, '1']) {
  it(`reveals ${chunkId ? 'a chunk citation' : 'a Reference'} after rendering, including repeated selection`, async () => {
    const {inspector, content, answer} = await scrollableSources();
    const sources = inspector.querySelector('dl-inspector-sources')!;
    const selector = chunkId
      ? '[data-ref="12"][data-chunk="1"]'
      : '[data-ref="12"] > [data-source-header]';

    for (let attempt = 0; attempt < 2; attempt += 1) {
      content.scrollTop = 0;
      await inspector.openSources(answer, '12', chunkId);
      await sources.updateComplete;

      expect(sources.querySelector('[data-ref="12"][data-expanded]')).not.to.equal(null);
      if (chunkId) expect(sources.activeChunk).to.equal(chunkId);
      const target = sources.querySelector<HTMLElement>(selector)!;
      const viewport = content.getBoundingClientRect();
      const bounds = target.getBoundingClientRect();
      expect(content.scrollTop).to.be.greaterThan(0);
      expect(bounds.top).to.be.at.least(viewport.top - 1);
      expect(bounds.top).to.be.lessThan(viewport.bottom);
    }

    content.scrollTop = 0;
    sources.requestUpdate();
    await sources.updateComplete;
    expect(content.scrollTop).to.equal(0);
  });
}

it('reveals only the latest selection and ignores a missing reference', async () => {
  const {inspector, content, answer} = await scrollableSources();
  const sources = inspector.querySelector('dl-inspector-sources')!;
  await Promise.all([
    inspector.openSources(answer, '12', '1'),
    inspector.openSources(answer, '1'),
  ]);
  await sources.updateComplete;
  expect(sources.expandedRef).to.equal('1');
  expect(content.scrollTop).to.equal(0);

  await inspector.openSources(answer, 'missing');
  await sources.updateComplete;
  expect(content.scrollTop).to.equal(0);
});

it('does not restore stale focus when close is immediately followed by reopen', async () => {
  window.matchMedia = media(true);
  const firstTrigger = document.createElement('button');
  const secondTrigger = document.createElement('button');
  const inspector = document.createElement('dl-inspector') as DlInspector;
  document.body.append(firstTrigger, secondTrigger, inspector);

  await inspector.openSources(presentation, undefined, undefined, firstTrigger);
  inspector.close();
  const reopened = inspector.openSources(presentation, undefined, undefined, secondTrigger);
  await reopened;
  await new Promise((resolve) => requestAnimationFrame(resolve));

  expect(inspector.open).to.equal(true);
  expect(document.activeElement).not.to.equal(firstTrigger);
  expect(document.activeElement).to.equal(buttonNamed(inspector, 'Close panel'));
});

it('owns compact dialog semantics, entry focus, Escape, and typed state', async () => {
  window.matchMedia = media(true);
  const inspector = document.createElement('dl-inspector') as DlInspector;
  const states: InspectorStateDetail[] = [];
  inspector.addEventListener('dl-inspector-state-change', (event) => {
    states.push((event as CustomEvent<InspectorStateDetail>).detail);
  });
  document.body.appendChild(inspector);

  await inspector.openSources(presentation);
  const panel = inspector.querySelector<HTMLElement>('aside[aria-labelledby="panel-title"]')!;
  expect(panel.getAttribute('role')).to.equal('dialog');
  expect(panel.getAttribute('aria-modal')).to.equal('true');
  expect(document.activeElement).to.equal(buttonNamed(inspector, 'Close panel'));

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await inspector.updateComplete;
  expect(inspector.open).to.equal(false);
  expect(states.some((state) => state.open && state.compact)).to.equal(true);
  expect(states.at(-1)?.open).to.equal(false);
});

it('activates and pauses typed Files content without a legacy element alias', async () => {
  window.matchMedia = media(false);
  window.fetch = async () => new Response(JSON.stringify({
    workspace: 'default',
    files: [],
    ingest: {
      busy: false, message: '', progress_percent: null, current_batch: null,
      total_batches: null, documents: null, pending_enqueues: 0,
    },
  }), {status: 200, headers: {'Content-Type': 'application/json'}});
  const inspector = document.createElement('dl-inspector') as DlInspector;
  document.body.appendChild(inspector);

  await inspector.openFiles();
  const files = inspector.querySelector('dl-inspector-files')!;
  await waitFor(() => files.loading === false);
  const panel = inspector.querySelector('aside')!;
  const title = inspector.querySelector<HTMLElement>('#panel-title')!;
  expect(title.textContent).to.equal('Files');
  expect(title.hidden).to.equal(false);
  expect(panel.getAttribute('aria-labelledby')).to.equal(title.id);
  expect(panel.hasAttribute('aria-label')).to.equal(false);
  expect(files.active).to.equal(true);
  expect(buttonNamed(inspector, 'Choose files')).not.to.equal(null);

  ingestStore.set('secondary');
  await inspector.openFiles();
  expect(ingestStore.workspace).to.equal('default');

  inspector.close(false);
  await inspector.updateComplete;
  await files.updateComplete;
  expect(files.active).to.equal(false);
  expect(customElements.get('file-panel')).to.equal(undefined);
});
