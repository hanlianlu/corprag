// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {ConversationSummary} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {conversationRoute, newChatRoute} from '../lib/router.ts';
import {conversationStore} from '../stores/conversation-store.ts';
import type {
  ConversationIntentDetail,
  ConversationRenameDetail,
  DlConversationList,
} from './conversation-list.ts';
import './conversation-list.ts';
import type {
  ConversationChat,
  ConversationSidebarStateDetail,
  DlConversationSidebar,
} from './conversation-sidebar.ts';
import './conversation-sidebar.ts';
import {webRouter} from './router.ts';

defineDesignSystemElements();

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;

const first: ConversationSummary = {
  conversationId: 'conversation-1',
  title: 'Research notes',
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-02T00:00:00Z',
  forkedFromConversationId: null,
  forkedFromTitle: null,
};

function media(desktop: boolean): (query: string) => MediaQueryList {
  return (query) => ({
    matches: desktop ? query === '(min-width: 1200px)' : query === '(width < 1200px)',
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  });
}

const firstWire = {
  conversation_id: 'conversation-1',
  title: 'Research notes',
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-02T00:00:00Z',
};

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {'Content-Type': 'application/json'},
  });
}

function conversationPage(items: ConversationSummary[], nextCursor: string | null = null): Response {
  return response({
    items: items.map((item) => ({
      conversation_id: item.conversationId,
      title: item.title,
      created_at: item.createdAt,
      updated_at: item.updatedAt,
      forked_from_conversation_id: item.forkedFromConversationId,
      forked_from_title: item.forkedFromTitle,
    })),
    next_cursor: nextCursor,
  });
}

function button(root: ParentNode, name: string): HTMLButtonElement {
  const match = [...root.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => {
    return candidate.getAttribute('aria-label') === name || candidate.textContent?.trim() === name;
  });
  if (!match) throw new Error(`button not found: ${name}`);
  return match;
}

function conversationsOpenButton(): HTMLButtonElement {
  const match = document.querySelector<HTMLButtonElement>('#conversation-sidebar-open');
  if (!match) throw new Error('open conversations button not found');
  return match;
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

beforeEach(() => {
  document.body.replaceChildren();
  document.body.className = '';
  const trigger = document.createElement('button');
  trigger.id = 'conversation-sidebar-open';
  trigger.type = 'button';
  trigger.setAttribute('aria-label', 'Open conversations');
  document.body.append(trigger);
  window.localStorage.removeItem('dlightrag.conversation_sidebar_collapsed');
  conversationStore.openNew();
});

afterEach(async () => {
  document.body.replaceChildren();
  document.body.className = '';
  conversationStore.dispose();
  conversationStore.openNew();
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  window.localStorage.removeItem('dlightrag.conversation_sidebar_collapsed');
  await webRouter.navigate(newChatRoute(), {
    replace: true,
    notify: false,
    bypassGuard: true,
  });
});

it('publishes list item intent and owns menu keyboard behavior through ARIA', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  let selected: ConversationIntentDetail | null = null;
  list.addEventListener('dl-conversation-select', (event) => { selected = event.detail; });
  button(list, 'Research notes').click();
  expect(selected).to.deep.equal({conversationId: first.conversationId});

  const actions = button(list, 'Conversation actions');
  actions.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'ArrowDown', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;
  const menu = list.querySelector<HTMLElement>('[role="menu"][aria-label="Conversation actions"]');
  expect(menu).not.to.equal(null);
  expect(document.activeElement?.textContent?.trim()).to.equal('Rename');

  menu?.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;
  expect(list.querySelector('[role="menu"]')).to.equal(null);
  expect(document.activeElement).to.equal(actions);
  expect(customElements.get('conversation-list')).to.equal(undefined);
});

it('renders JavaScript-sensitive conversation titles as inert exact text', async () => {
  const title = 'line\u2028separator\u2029</script><script>window.__conversationTitleRan=true</script>';
  const sensitive = {...first, title};
  const browser = window as Window & {__conversationTitleRan?: boolean};
  delete browser.__conversationTitleRan;
  window.fetch = async () => conversationPage([sensitive]);

  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  expect(list.querySelector<HTMLButtonElement>('.conversation-select')?.textContent).to.equal(title);
  expect(list.querySelector('script')).to.equal(null);
  expect(browser.__conversationTitleRan).to.equal(undefined);
});

it('keeps accessible Load older and retry controls outside list ownership', async () => {
  const older = {...first, conversation_id: 'conversation-older', title: 'Older notes'};
  const urls: string[] = [];
  let olderAttempts = 0;
  window.fetch = async (input) => {
    const url = String(input);
    urls.push(url);
    if (!url.includes('?cursor=')) return conversationPage([first], 'opaque-next');
    olderAttempts += 1;
    return olderAttempts === 1
      ? response({detail: 'temporarily unavailable'}, 503)
      : conversationPage([older]);
  };
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  const loadOlder = button(list, 'Load older conversations');
  const ownedList = list.querySelector<HTMLElement>('[role="list"]')!;
  expect(list.getAttribute('role')).to.equal(null);
  expect([...ownedList.children].every(
    (child) => child.getAttribute('role') === 'listitem',
  )).to.equal(true);
  expect(loadOlder.closest('[role="list"]')).to.equal(null);
  expect(urls).to.deep.equal(['/web/api/conversations']);
  loadOlder.click();
  await waitFor(() => list.textContent?.includes('Could not load older conversations.') ?? false);
  const retry = button(list, 'Retry loading older conversations');
  expect(retry.closest('[role="list"]')).to.equal(null);
  expect([...ownedList.children].every(
    (child) => child.getAttribute('role') === 'listitem',
  )).to.equal(true);
  retry.click();
  await waitFor(() => list.textContent?.includes('Older notes') ?? false);

  expect(urls).to.deep.equal([
    '/web/api/conversations',
    '/web/api/conversations?cursor=opaque-next',
    '/web/api/conversations?cursor=opaque-next',
  ]);
  expect(list.querySelector('[aria-label="Load older conversations"]')).to.equal(null);
});

it('starts rename from a double-click on the conversation title', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  list.querySelector('[data-conversation-id]')!.dispatchEvent(new MouseEvent('dblclick', {
    bubbles: true, cancelable: true,
  }));
  await list.updateComplete;
  const input = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]');
  expect(input).not.to.equal(null);
  expect(document.activeElement).to.equal(input);
});

it('restores row focus after keyboard rename completion and cancellation', async () => {
  window.fetch = async () => conversationPage([first]);
  await conversationStore.loadList();
  const list = document.createElement('dl-conversation-list') as DlConversationList;
  document.body.appendChild(list);
  await list.updateComplete;

  const renamed: ConversationRenameDetail[] = [];
  list.addEventListener('dl-conversation-rename', (event) => { renamed.push(event.detail); });

  button(list, 'Conversation actions').click();
  await list.updateComplete;
  button(list, 'Rename').click();
  await list.updateComplete;
  const input = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  input.value = 'Updated title';
  input.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Enter', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;

  expect(renamed).to.deep.equal([{
    conversationId: first.conversationId,
    title: 'Updated title',
  }]);
  expect(document.activeElement).to.equal(button(list, 'Conversation actions'));

  button(list, 'Conversation actions').click();
  await list.updateComplete;
  button(list, 'Rename').click();
  await list.updateComplete;
  const cancelled = list.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  cancelled.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await list.updateComplete;

  expect(renamed).to.have.length(1);
  expect(document.activeElement).to.equal(button(list, 'Conversation actions'));
});

it('owns desktop collapse state, focus, and typed Shell state', async () => {
  window.matchMedia = media(true);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let state: ConversationSidebarStateDetail | null = null;
  sidebar.addEventListener('dl-conversation-sidebar-state-change', (event) => {
    state = event.detail;
  });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const navigation = sidebar.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  expect(navigation.inert).to.equal(false);
  expect(navigation.hasAttribute('aria-hidden')).to.equal(false);
  expect(state).to.deep.equal({expanded: true, compact: false});
  expect(document.body.classList.contains('conversation-sidebar-open')).to.equal(false);

  button(sidebar, 'Collapse conversations').click();
  await sidebar.updateComplete;
  expect(navigation.inert).to.equal(true);
  expect(navigation.getAttribute('aria-hidden')).to.equal('true');
  expect(state).to.deep.equal({expanded: false, compact: false});
  expect(document.activeElement).to.equal(conversationsOpenButton());
  expect(window.localStorage.getItem('dlightrag.conversation_sidebar_collapsed')).to.equal('true');

  await sidebar.open(conversationsOpenButton());
  expect(navigation.inert).to.equal(false);
  expect(state).to.deep.equal({expanded: true, compact: false});
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));
  sidebar.shellInert = true;
  await sidebar.updateComplete;
  expect(navigation.inert).to.equal(true);
});

it('owns compact modality, a cancelable open command, focus wrapping, and Escape restore', async () => {
  window.matchMedia = media(false);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let state: ConversationSidebarStateDetail | null = null;
  sidebar.addEventListener('dl-conversation-sidebar-state-change', (event) => {
    state = event.detail;
  });
  const trigger = document.createElement('button');
  trigger.textContent = 'External trigger';
  document.body.append(trigger, sidebar);
  await sidebar.updateComplete;

  sidebar.addEventListener('dl-conversation-sidebar-opening', (event) => {
    event.preventDefault();
  }, {once: true});
  expect(await sidebar.open(trigger)).to.equal(false);
  expect(state).to.deep.equal({expanded: false, compact: true});

  expect(await sidebar.open(trigger)).to.equal(true);
  const navigation = sidebar.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  expect(navigation.getAttribute('role')).to.equal('dialog');
  expect(navigation.getAttribute('aria-modal')).to.equal('true');
  expect(state).to.deep.equal({expanded: true, compact: true});
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  const settings = button(sidebar, 'Settings');
  settings.focus();
  settings.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Tab', bubbles: true, cancelable: true,
  }));
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  document.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Escape', bubbles: true, cancelable: true,
  }));
  await sidebar.updateComplete;
  expect(navigation.getAttribute('aria-hidden')).to.equal('true');
  expect(state).to.deep.equal({expanded: false, compact: true});
  expect(document.activeElement).to.equal(trigger);
});

it('normalizes drawer state and focus across compact and desktop breakpoints', async () => {
  let desktop = false;
  window.matchMedia = (query) => media(desktop)(query);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.enabled = true;
  let state: ConversationSidebarStateDetail | null = null;
  sidebar.addEventListener('dl-conversation-sidebar-state-change', (event) => {
    state = event.detail;
  });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  expect(await sidebar.open(conversationsOpenButton())).to.equal(true);
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  desktop = true;
  window.dispatchEvent(new Event('resize'));
  await sidebar.updateComplete;
  const navigation = sidebar.querySelector<HTMLElement>('nav[aria-label="Conversations"]')!;
  expect(state).to.deep.equal({expanded: true, compact: false});
  expect(navigation.inert).to.equal(false);
  expect(navigation.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(button(sidebar, 'New chat'));

  desktop = false;
  window.dispatchEvent(new Event('resize'));
  await sidebar.updateComplete;
  expect(state).to.deep.equal({expanded: false, compact: true});
  expect(navigation.inert).to.equal(true);
  expect(navigation.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(conversationsOpenButton());
});

it('owns list loading and route selection while exposing only typed Shell intent', async () => {
  window.matchMedia = media(true);
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/conversations') return conversationPage([first]);
    if (url.endsWith(`/${first.conversationId}/history`)) {
      return response({conversation: firstWire, turns: []});
    }
    return response({detail: 'not found'}, 404);
  };
  let detached = 0;
  const chat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: false,
    submissionPending: false,
    detachRun: () => { detached += 1; },
    focusComposer: () => undefined,
    clearDraft: () => undefined,
  };
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = chat;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await waitFor(() => [...sidebar.querySelectorAll('button')].some(
    (candidate) => candidate.textContent?.trim() === 'Research notes',
  ));

  let routeChanges = 0;
  sidebar.addEventListener('dl-conversation-route-change', (event) => {
    if (event.detail.nextConversationId === first.conversationId) routeChanges += 1;
  });
  button(sidebar, 'Research notes').click();
  await waitFor(() => chat.view.kind === 'ready');
  await waitFor(() => sidebar.querySelector(
    '[role="listitem"][aria-current="page"]',
  ) !== null);

  expect(chat.view.kind).to.equal('ready');
  expect(sidebar.querySelector('[role="listitem"][aria-current="page"]')).not.to.equal(null);
  expect(detached).to.equal(1);
  expect(routeChanges).to.equal(1);

  expect(await webRouter.navigate({kind: 'not-found'})).to.equal(true);
  await waitFor(() => chat.view.kind === 'unavailable');
  expect(chat.view).to.deep.equal({kind: 'unavailable', hasRecent: false});
});

it('keeps route navigation available while an independent rename settles', async () => {
  window.matchMedia = media(true);
  let resolveRename: ((response: Response) => void) | undefined;
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/conversations') return conversationPage([first]);
    if (url.endsWith(`/${first.conversationId}/history`)) {
      return response({conversation: firstWire, turns: []});
    }
    if (url.endsWith(`/${first.conversationId}`) && init?.method === 'PATCH') {
      return await new Promise<Response>((resolve) => { resolveRename = resolve; });
    }
    return response({detail: 'not found'}, 404);
  };
  const chat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: false,
    submissionPending: false,
    detachRun: () => undefined,
    focusComposer: () => undefined,
    clearDraft: () => undefined,
  };
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = chat;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await waitFor(() => sidebar.querySelector('[role="listitem"]') !== null);

  button(sidebar, 'Research notes').click();
  await waitFor(() => chat.view.kind === 'ready');
  button(sidebar, 'Conversation actions').click();
  await sidebar.updateComplete;
  button(sidebar, 'Rename').click();
  await sidebar.updateComplete;
  const input = sidebar.querySelector<HTMLInputElement>('[aria-label="Conversation title"]')!;
  input.value = 'Renaming';
  input.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Enter', bubbles: true, cancelable: true,
  }));
  await waitFor(() => resolveRename !== undefined);

  expect(await webRouter.navigate(newChatRoute())).to.equal(true);
  await waitFor(() => chat.view.kind === 'new');
  const finishRename = resolveRename;
  if (!finishRename) throw new Error('rename request was not started');
  finishRename(response({...first, title: 'Renaming'}));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(chat.view.kind).to.equal('new');
});

it('publishes a route reset after delete-all succeeds on the new route', async () => {
  window.matchMedia = media(true);
  window.fetch = async (input, init) => {
    if (String(input) === '/web/api/conversations' && init?.method === 'DELETE') {
      return new Response(null, {status: 204});
    }
    if (String(input) === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  const chat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: false,
    submissionPending: false,
    detachRun: () => undefined,
    focusComposer: () => undefined,
    clearDraft: () => undefined,
  };
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = chat;
  sidebar.enabled = true;
  let routeReset = 0;
  sidebar.addEventListener('dl-conversation-route-change', (event) => {
    if (
      event.detail.previousConversationId === null
      && event.detail.nextConversationId === null
    ) routeReset += 1;
  });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  button(sidebar, 'Delete all').click();

  expect(await deletion).to.equal(true);
  expect(routeReset).to.equal(1);
});

it('aborts a confirmed mutation and ignores its result after disconnect', async () => {
  window.matchMedia = media(true);
  let mutationStarted = false;
  let mutationAborted = false;
  window.fetch = async (input, init) => {
    if (String(input) === '/web/api/conversations' && init?.method === 'DELETE') {
      mutationStarted = true;
      return await new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          mutationAborted = true;
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    }
    if (String(input) === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  let draftClears = 0;
  const chat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: false,
    submissionPending: false,
    detachRun: () => undefined,
    focusComposer: () => undefined,
    clearDraft: () => { draftClears += 1; },
  };
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = chat;
  sidebar.enabled = true;
  let routeResets = 0;
  sidebar.addEventListener('dl-conversation-route-change', () => { routeResets += 1; });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  button(sidebar, 'Delete all').click();
  await waitFor(() => mutationStarted);
  sidebar.remove();

  expect(await deletion).to.equal(false);
  expect(mutationAborted).to.equal(true);
  expect(conversationStore.mutationPending).to.equal(false);
  expect(draftClears).to.equal(0);
  expect(routeResets).to.equal(0);
});

it('aborts optional memory clearing and ignores its result after disconnect', async () => {
  window.matchMedia = media(true);
  let memoryStarted = false;
  let memoryAborted = false;
  window.fetch = async (input, init) => {
    const url = String(input);
    if (url === '/web/api/conversations' && init?.method === 'DELETE') {
      return new Response(null, {status: 204});
    }
    if (url === '/web/api/memory/clear' && init?.method === 'POST') {
      memoryStarted = true;
      return await new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          memoryAborted = true;
          reject(new DOMException('Aborted', 'AbortError'));
        }, {once: true});
      });
    }
    if (url === '/web/api/conversations') return conversationPage([]);
    return response({detail: 'not found'}, 404);
  };
  let draftClears = 0;
  const chat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: false,
    submissionPending: false,
    detachRun: () => undefined,
    focusComposer: () => undefined,
    clearDraft: () => { draftClears += 1; },
  };
  const toast = document.createElement('div');
  toast.id = 'toast';
  document.body.appendChild(toast);
  const focusSentinel = document.createElement('button');
  document.body.appendChild(focusSentinel);
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = chat;
  sidebar.enabled = true;
  let routeResets = 0;
  sidebar.addEventListener('dl-conversation-route-change', () => { routeResets += 1; });
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const deletion = sidebar.deleteAll();
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#delete-all-conversations-dialog',
  )?.open));
  const clearProfileMemories = sidebar.querySelector<HTMLInputElement>(
    '#delete-all-also-clear-memory',
  );
  if (!clearProfileMemories) throw new Error('memory clearing checkbox was not rendered');
  clearProfileMemories.checked = true;
  button(sidebar, 'Delete all').click();
  await waitFor(() => memoryStarted);
  sidebar.remove();
  focusSentinel.focus();

  expect(await deletion).to.equal(false);
  expect(memoryAborted).to.equal(true);
  expect(conversationStore.mutationPending).to.equal(false);
  expect(draftClears).to.equal(1);
  expect(routeResets).to.equal(0);
  expect(toast.classList.contains('visible')).to.equal(false);
  expect(toast.textContent).to.equal('');
  expect(document.activeElement).to.equal(focusSentinel);
});

it('protects unresolved failed submissions from tab unload without blocking in-app navigation', async () => {
  const chat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: false,
    hasUnresolvedSubmission: true,
    submissionPending: false,
    detachRun: () => undefined,
    focusComposer: () => undefined,
    clearDraft: () => undefined,
  };
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = chat;
  document.body.appendChild(sidebar);
  await sidebar.updateComplete;

  const event = new Event('beforeunload', {cancelable: true}) as BeforeUnloadEvent;
  window.dispatchEvent(event);
  expect(event.defaultPrevented).to.equal(true);
});

it('cancels a pending draft dialog on disconnect without wedging later navigation', async () => {
  window.matchMedia = media(true);
  window.fetch = async (input) => {
    const url = String(input);
    if (url === '/web/api/conversations') return conversationPage([first]);
    if (url.endsWith(`/${first.conversationId}/history`)) {
      return response({conversation: firstWire, turns: []});
    }
    return response({detail: 'not found'}, 404);
  };
  const draftChat: ConversationChat = {
    view: {kind: 'new'},
    hasDraft: true,
    submissionPending: false,
    detachRun: () => undefined,
    focusComposer: () => undefined,
    clearDraft: () => undefined,
  };
  const sidebar = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  sidebar.chatFeature = draftChat;
  sidebar.enabled = true;
  document.body.appendChild(sidebar);
  await waitFor(() => sidebar.querySelector('[role="listitem"]') !== null);

  const blockedNavigation = webRouter.navigate(conversationRoute(first.conversationId));
  await waitFor(() => Boolean(sidebar.querySelector<HTMLDialogElement>(
    '#discard-draft-dialog',
  )?.open));
  sidebar.remove();
  expect(await blockedNavigation).to.equal(false);

  const replacementChat: ConversationChat = {
    ...draftChat,
    hasDraft: false,
  };
  const replacement = document.createElement('dl-conversation-sidebar') as DlConversationSidebar;
  replacement.chatFeature = replacementChat;
  replacement.enabled = true;
  document.body.appendChild(replacement);
  await waitFor(() => replacement.querySelector('[role="listitem"]') !== null);

  expect(await webRouter.navigate(conversationRoute(first.conversationId))).to.equal(true);
  await waitFor(() => replacementChat.view.kind === 'ready');
  expect(replacementChat.view.kind).to.equal('ready');
});
