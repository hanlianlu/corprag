// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges } from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  corpusRunActive,
  getCorpusRunStatus,
  type WebCorpusRunReceipt,
} from '../api/corpus-runs.ts';
import {resetWorkspaceRequest, WorkspaceApiError } from '../api/workspaces.ts';
import {icon} from '../design-system/index.ts';
import {rovingArrowKeydown} from '../lib/listbox.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import type {WorkspaceRecord} from '../stores/workspace-store.ts';
import workspaceStyles from '../styles/workspaces.module.css';
import {publishModalState, showOwnedModal} from './modal.ts';
import {requestToast} from './toast-request.ts';
import './workspace-create.ts';

/** Search-scope selection, Corpus reset, popover, and Dialog lifecycle. */
export class DlWorkspaceScope extends LightElement {
  static properties = {
    handles: {attribute: false},
    open: {state: true},
    resetWorkspace: {state: true},
    resetPending: {state: true},
    resetConfirmed: {state: true},
  };

  declare handles: AppHandles;
  declare open: boolean;
  declare resetWorkspace: string | null;
  declare resetPending: boolean;
  declare resetConfirmed: boolean;

  #resetOperation: AbortController | null = null;
  #resetPoll: AbortController | null = null;
  #resetReturnFocus: HTMLElement | null = null;
  #restoreLoadMoreFocus = false;
  #settledFocusRestore = false;
  #loadMoreAnnouncement = '';
  #lastLoadMoreState: 'idle' | 'loading' | 'error' = 'idle';
  readonly #dismiss = createAutoDismiss({
    getAnchor: () => this,
    isOpen: () => this.open,
    onDismiss: (reason) => { this.#dismissPopover(reason === 'escape'); },
  });

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.handles = productionHandles();
    this.open = false;
    this.resetWorkspace = null;
    this.resetPending = false;
    this.resetConfirmed = false;
    /** Store reads: records, active, primary. */
    new StoreController(this, this.handles.workspaces);
  }

  override disconnectedCallback(): void {
    const dialog = this.querySelector<HTMLDialogElement>('#reset-workspace-dialog');
    if (dialog?.open) dialog.close();
    this.#resetOperation?.abort();
    this.#resetOperation = null;
    this.#resetPoll?.abort();
    this.#resetPoll = null;
    this.#resetReturnFocus = null;
    this.open = false;
    this.resetWorkspace = null;
    this.resetPending = false;
    this.resetConfirmed = false;
    this.#dismiss.deactivate();
    publishModalState(this);
    super.disconnectedCallback();
  }

  close(): void {
    this.open = false;
  }

  protected override willUpdate(_changed: PropertyValues<this>): void {
    const state = this.handles.workspaces.workspaceLoadMoreState;
    const previous = this.#lastLoadMoreState;
    if (state === previous) return;
    this.#lastLoadMoreState = state;
    if (state === 'loading') {
      this.#loadMoreAnnouncement = msg('Loading workspaces…', {id: 'workspaceScope.loadingMore'});
    } else if (state === 'error') {
      this.#loadMoreAnnouncement = msg('Workspaces could not be loaded.', {id: 'workspaceScope.loadMoreFailed'});
    } else if (previous === 'loading') {
      this.#loadMoreAnnouncement = msg('Loaded more workspaces.', {id: 'workspaceScope.loadedMore'});
      this.#settledFocusRestore = true;
    }
  }

  protected override updated(): void {
    this.classList.toggle('open', this.open);
    if (this.open) this.#dismiss.activate();
    else this.#dismiss.deactivate();
    if (this.#settledFocusRestore) {
      this.#settledFocusRestore = false;
      if (this.#restoreLoadMoreFocus) {
        this.#restoreLoadMoreFocus = false;
        const control = this.querySelector<HTMLButtonElement>('[data-load-more-workspaces]');
        if (control) control.focus({preventScroll: true});
      }
    }
  }

  protected override render(): TemplateResult {
    const multi = this.handles.workspaces.active.length > 1 || this.#allSelected;
    return html`
      <button class="workspace-selector-trigger" id="workspace-trigger" type="button"
              aria-label=${msg('Choose search workspaces', {id: 'workspaceScope.chooseSearchWorkspaces'})}
              aria-haspopup="dialog"
              aria-expanded=${this.open ? 'true' : 'false'} aria-controls="workspace-popover"
              @click=${this.#togglePopover}>
        <span class="workspace-dot${multi ? ' multi' : ''}" id="workspace-dot"></span>
        <span class="workspace-label" id="workspace-label">${this.#label}</span>
        ${icon('chevron-down', {size: 'xs', className: 'workspace-caret'})}
      </button>
      ${this.#popover()}
      ${this.#resetDialog()}
    `;
  }

  #trigger(): HTMLButtonElement | null {
    return this.querySelector<HTMLButtonElement>('#workspace-trigger');
  }

  #togglePopover = (): void => {
    if (this.open) {
      this.open = false;
      return;
    }
    this.open = true;
    void this.updateComplete.then(() => {
      const selected = this.querySelector<HTMLButtonElement>(
        '[data-workspace-choice][aria-pressed="true"]',
      );
      (selected ?? this.querySelector<HTMLButtonElement>('[data-workspace-choice]'))?.focus();
    });
  };

  #dismissPopover(restoreFocus: boolean): void {
    this.open = false;
    if (restoreFocus) {
      void this.updateComplete.then(() => { this.#trigger()?.focus(); });
    }
  }

  get #allSelected(): boolean {
    const known = this.handles.workspaces.knownWorkspaces;
    const active = this.handles.workspaces.active;
    return known.length > 0 && known.every((workspace) => active.includes(workspace));
  }

  get #label(): string {
    const total = this.handles.workspaces.knownWorkspaces.length;
    const active = this.handles.workspaces.active;
    if (active.length === 0 || this.#allSelected) {
      return total > 0
        ? msg(str`All workspaces (${total})`, {id: 'workspaceScope.allWithCount'})
        : msg('All workspaces', {id: 'workspaceScope.all'});
    }
    const anchor = active.includes(this.handles.workspaces.primary) ? this.handles.workspaces.primary : active[0];
    const name = this.handles.workspaces.records.find((record) => record.workspace === anchor)?.displayName
      ?? anchor;
    return active.length === 1 ? name : `${name} + ${active.length - 1}`;
  }

  #check(selected: boolean): TemplateResult {
    return html`<div class="${workspaceStyles.workspacePopoverCheck}${selected
      ? ` ${workspaceStyles.on}` : ''}">${selected ? icon('check', {size: 'xs'}) : nothing}</div>`;
  }

  #popover(): TemplateResult {
    const sorted = [...this.handles.workspaces.records]
      .sort((left, right) => left.displayName.localeCompare(right.displayName));
    return html`
      <div class="dl-popover dl-popover--workspace" id="workspace-popover"
           role="dialog" aria-label=${msg('Workspaces', {id: 'workspaceScope.workspacesAria'})}
           ?hidden=${!this.open}
           @keydown=${(event: KeyboardEvent) => {
             rovingArrowKeydown(event, '[data-workspace-choice]');
           }}
           @dl-workspace-created=${this.#workspaceCreated}>
        ${this.#allOption()}
        ${repeat(sorted, (record) => record.workspace, (record) => this.#option(record))}
        ${this.#loadMoreControl()}
        <span class="sr-only" data-workspaces-status role="status" aria-live="polite">
          ${this.#loadMoreAnnouncement}
        </span>
        <dl-workspace-create .handles=${this.handles}></dl-workspace-create>
      </div>
    `;
  }

  #loadMoreControl(): TemplateResult | typeof nothing {
    if (!this.handles.workspaces.hasMoreWorkspaces) return nothing;
    const state = this.handles.workspaces.workspaceLoadMoreState;
    return html`
      <div class="workspace-load-more">
        <button type="button" data-load-more-workspaces class="dl-popover-item"
                aria-busy=${state === 'loading' ? 'true' : 'false'}
                ?disabled=${state === 'loading'} @click=${this.#loadMore}>
          ${state === 'error'
            ? msg('Retry loading workspaces', {id: 'workspaceScope.retryLoadMore'})
            : msg('Load more workspaces', {id: 'workspaceScope.loadMore'})}
        </button>
      </div>
    `;
  }

  #loadMore = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    this.#restoreLoadMoreFocus = document.activeElement === button;
    void this.handles.workspaces.loadMoreWorkspaces();
  };

  #allOption(): TemplateResult {
    const selected = this.#allSelected;
    const selectAll = (event: Event): void => {
      event.stopPropagation();
      this.handles.workspaces.selectAll();
    };
    return html`
      <button class="dl-popover-item ${workspaceStyles.workspacePopoverAll}" type="button"
              data-workspace-choice data-workspace-all="true"
              aria-pressed=${selected ? 'true' : 'false'} @click=${selectAll}>
        ${this.#check(selected)}${msg('All workspaces', {id: 'workspaceScope.all'})}
      </button>
    `;
  }

  #option(record: WorkspaceRecord): TemplateResult {
    const selected = this.handles.workspaces.active.includes(record.workspace);
    const toggle = (event: Event): void => {
      event.stopPropagation();
      this.handles.workspaces.toggle(record.workspace);
    };
    return html`
      <div class=${workspaceStyles.workspacePopoverItem}>
        <button class="dl-popover-item ${workspaceStyles.workspacePopoverOption}" type="button"
                data-workspace-choice aria-pressed=${selected ? 'true' : 'false'} @click=${toggle}>
          ${this.#check(selected)}
          <span class=${workspaceStyles.workspacePopoverName}>${record.displayName}</span>
        </button>
        <button type="button" class=${workspaceStyles.workspacePopoverDelete}
                title=${msg('Reset Corpus', {id: 'workspaceScope.resetTitle'})}
                aria-label=${msg(str`Reset Corpus ${record.displayName}`, {id: 'workspaceScope.resetWorkspaceAria'})}
                @click=${(event: MouseEvent) => {
                  event.stopPropagation();
                  void this.#requestReset(record.workspace, event.currentTarget as HTMLElement);
                }}>${icon('close', {size: 'xs'})}</button>
      </div>
    `;
  }

  #workspaceCreated = (): void => {
    const active = document.activeElement;
    const restoreFocus = active === document.body || this.contains(active);
    this.open = false;
    if (restoreFocus) {
      void this.updateComplete.then(() => { this.#trigger()?.focus(); });
    }
  };

  async #requestReset(workspace: string, trigger: HTMLElement): Promise<void> {
    this.#resetReturnFocus = trigger;
    this.open = false;
    this.resetWorkspace = workspace;
    this.resetConfirmed = false;
    await this.updateComplete;
    const dialog = this.querySelector<HTMLDialogElement>('#reset-workspace-dialog');
    if (!dialog) return;
    dialog.returnValue = '';
    showOwnedModal(this, dialog);
    window.requestAnimationFrame(() => {
      this.querySelector<HTMLInputElement>('#reset-workspace-confirm-input')?.focus();
    });
  }

  #resetDialog(): TemplateResult {
    const workspace = this.resetWorkspace ?? '';
    const displayName = this.handles.workspaces.records.find((record) => record.workspace === workspace)
      ?.displayName ?? workspace;
    return html`
      <dialog id="reset-workspace-dialog" class="workspace-dialog"
              aria-labelledby="reset-workspace-title" @cancel=${this.#resetCancelled}
              @close=${this.#resetClosed}>
        <form @submit=${this.#submitReset}>
          <h3 class="workspace-dialog-title" id="reset-workspace-title">${msg('Reset Corpus', {id: 'workspaceScope.resetTitle'})}</h3>
          <p class="workspace-dialog-text">${msg('This will permanently remove all Corpus data while preserving workspace', {id: 'workspaceScope.resetWarning'})} <strong>${displayName}</strong>.</p>
          <p class="workspace-dialog-text">${msg('Type the workspace name to confirm', {id: 'workspaceScope.typeToConfirm'})}</p>
          <input type="text" id="reset-workspace-confirm-input" class="dl-dialog-input"
                 autocomplete="off"
                 placeholder=${msg('Type workspace name...', {id: 'workspaceScope.confirmPlaceholder'})}
                 aria-label=${msg(str`Type ${displayName} to confirm`, {id: 'workspaceScope.typeNameToConfirmAria'})}
                 .readOnly=${this.resetPending}
                 @input=${this.#resetInput}>
          <div class="dl-dialog-actions">
            <button type="button" ?disabled=${this.resetPending}
                    @click=${() => this.querySelector<HTMLDialogElement>(
                      '#reset-workspace-dialog',
                    )?.close()}>${msg('Cancel', {id: 'workspaceScope.cancel'})}</button>
            <button type="submit" class="dl-dialog-danger"
                    ?disabled=${this.resetPending || !this.resetConfirmed}>
              ${this.resetPending
                ? msg('Accepting reset…', {id: 'workspaceScope.resetting'})
                : msg('Reset Corpus', {id: 'workspaceScope.reset'})}
            </button>
          </div>
        </form>
      </dialog>
    `;
  }

  #resetInput = (event: Event): void => {
    const input = event.currentTarget as HTMLInputElement;
    const workspace = this.resetWorkspace ?? '';
    const displayName = this.handles.workspaces.records.find((record) => record.workspace === workspace)
      ?.displayName ?? workspace;
    this.resetConfirmed = input.value.trim() === displayName || input.value.trim() === workspace;
  };

  #submitReset = async (event: SubmitEvent): Promise<void> => {
    event.preventDefault();
    const workspace = this.resetWorkspace;
    if (!workspace || this.resetPending || !this.resetConfirmed) return;
    const operation = new AbortController();
    this.#resetOperation = operation;
    this.resetPending = true;
    try {
      const receipt = await resetWorkspaceRequest(workspace, operation.signal);
      if (
        operation.signal.aborted || this.#resetOperation !== operation
        || this.resetWorkspace !== workspace
      ) return;
      this.querySelector<HTMLDialogElement>('#reset-workspace-dialog')?.close();
      requestToast(this, {
        message: msg(str`Corpus reset accepted for ${workspace}.`, {id: 'workspaceScope.resetAccepted'}),
      });
      void this.#watchReset(receipt);
    } catch (error) {
      if (!operation.signal.aborted && this.#resetOperation === operation) {
        requestToast(this, {
          message: error instanceof WorkspaceApiError
            ? error.message
            : msg('Could not accept Corpus reset.', {id: 'workspaceScope.resetFailed'}),
          duration: 3000,
        });
      }
    } finally {
      if (this.#resetOperation === operation) {
        this.#resetOperation = null;
        this.resetPending = false;
        await this.updateComplete;
        if (this.querySelector<HTMLDialogElement>('#reset-workspace-dialog')?.open) {
          this.querySelector<HTMLInputElement>('#reset-workspace-confirm-input')?.focus();
        }
      }
    }
  };

  async #watchReset(receipt: WebCorpusRunReceipt): Promise<void> {
    this.#resetPoll?.abort();
    const controller = new AbortController();
    this.#resetPoll = controller;
    let current: {status: string} = receipt;
    try {
      while (corpusRunActive(current) && !controller.signal.aborted) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        if (controller.signal.aborted) return;
        current = await getCorpusRunStatus(receipt.statusUrl, controller.signal);
      }
      if (controller.signal.aborted) return;
      requestToast(this, {
        message: current.status === 'succeeded'
          ? msg(str`Corpus reset finished for ${receipt.workspace}.`, {id: 'workspaceScope.resetFinished'})
          : msg(str`Corpus reset did not finish for ${receipt.workspace}.`, {id: 'workspaceScope.resetDidNotFinish'}),
        duration: 4000,
      });
    } catch {
      if (!controller.signal.aborted) {
        requestToast(this, {
          message: msg('Corpus reset status is temporarily unavailable.', {id: 'workspaceScope.resetStatusUnavailable'}),
          duration: 3000,
        });
      }
    } finally {
      if (this.#resetPoll === controller) this.#resetPoll = null;
    }
  }

  #resetCancelled = (event: Event): void => {
    if (this.resetPending) event.preventDefault();
  };

  #resetClosed = (): void => {
    publishModalState(this);
    this.#resetOperation?.abort();
    this.#resetOperation = null;
    this.resetWorkspace = null;
    this.resetPending = false;
    this.resetConfirmed = false;
    const returnFocus = this.#resetReturnFocus;
    this.#resetReturnFocus = null;
    const target = returnFocus?.isConnected && !returnFocus.inert
      && !returnFocus.closest('[hidden]')
      ? returnFocus
      : this.#trigger();
    if (target?.isConnected && !target.inert) target.focus();
  };

}

customElements.define('dl-workspace-scope', DlWorkspaceScope);

declare global {
  interface HTMLElementTagNameMap {
    'dl-workspace-scope': DlWorkspaceScope;
  }
}
