// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges } from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import {
  corpusRunActive,
  getCorpusRunStatus,
  resumeCorpusRun,
  type WebCorpusRunReceipt,
  type WebCorpusRunStatus,
} from '../api/corpus-runs.ts';
import {
  deleteFileRequest,
  FilesApiError,
  getFilePanel,
  uploadFileBatch,
  type WebFilePanelSnapshot,
} from '../api/files.ts';
import {icon} from '../design-system/index.ts';
import {isAbortError} from '../lib/errors.ts';
import {LightElement, StoreController} from '../lib/lit-host.ts';
import {type AppHandles, productionHandles } from '../stores/app-handles.ts';
import {withRelativePath} from './folder-upload.ts';
import {modalResult} from './modal.ts';
import {requestToast} from './toast-request.ts';
import './failed-file-recovery.ts';
import fileStyles from '../styles/inspector-files.module.css';
import type {DlFailedFileRecovery} from './failed-file-recovery.ts';
import {InspectorFilesSession} from './inspector-files-session.ts';

type MutationRun = WebCorpusRunReceipt | (
  WebCorpusRunStatus & Pick<WebCorpusRunReceipt, 'workspace' | 'fileCount'>
);

function waitingForRepair(
  run: MutationRun | null | undefined,
): run is WebCorpusRunStatus & Pick<WebCorpusRunReceipt, 'workspace' | 'fileCount'> {
  return run !== null
    && run !== undefined
    && 'phase' in run
    && run.phase === 'waiting_for_repair';
}

function uploadLabel(files: readonly File[], label?: string | null): string {
  if (label) return label;
  return files.length === 1
    ? files[0].name
    : msg(str`${files.length} files`, {id: 'inspectorFiles.nFiles'});
}

/** File-management content, async work, and upload intent owned by the Inspector. */
export class DlInspectorFiles extends LightElement {
  static properties = {
    handles: {attribute: false},
    active: {attribute: false},
    snapshot: {state: true},
    loading: {state: true},
    error: {state: true},
    uploading: {state: true},
    acceptedFiles: {state: true},
    mutationRun: {state: true},
    filesLoadMoreState: {state: true},
  };

  declare handles: AppHandles;
  declare active: boolean;
  declare snapshot: WebFilePanelSnapshot | null;
  declare loading: boolean;
  declare error: string | null;
  declare uploading: boolean;
  declare acceptedFiles: number;
  declare mutationRun: MutationRun | null;
  declare filesLoadMoreState: 'idle' | 'loading' | 'error';

  #workspace = '';
  #requestGeneration = 0;
  readonly #session = new InspectorFilesSession();
  #olderFilesFlight: Promise<void> | null = null;
  #olderFilesAnnouncement = '';
  #restoreOlderFocus = false;
  #deleteTrigger: HTMLElement | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.handles = productionHandles();
    this.active = false;
    this.snapshot = null;
    this.loading = true;
    this.error = null;
    this.uploading = false;
    this.acceptedFiles = 0;
    this.mutationRun = null;
    this.filesLoadMoreState = 'idle';
    this.#workspace = this.handles.ingest.workspace;
    /** Store reads: this.handles.ingest.workspace. */
    new StoreController(this, this.handles.ingest);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    if (this.active) queueMicrotask(() => { void this.reload(); });
  }

  override disconnectedCallback(): void {
    this.pause();
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('active')) {
      if (this.active) void this.reload();
      else this.pause();
    }
    const workspace = this.handles.ingest.workspace;
    if (this.active && workspace !== this.#workspace && this.isConnected) {
      void this.reload();
    }
    this.querySelectorAll<HTMLElement>('[data-pct]').forEach((fill) => {
      const value = Number(fill.dataset.pct);
      fill.style.width = `${Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0))}%`;
    });
  }

  get hasActiveMutation(): boolean {
    return this.#session.mutating;
  }

  async reload(showLoading = true): Promise<void> {
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    if (workspace !== this.#workspace) {
      // Hide the old Workspace before any new-Workspace I/O. A failed load must
      // never leave actionable rows from the previously selected Workspace.
      this.snapshot = null;
      this.acceptedFiles = 0;
      this.mutationRun = null;
      this.#stopPolling();
    }
    this.#workspace = workspace;
    this.uploading = false;
    const {controller, generation} = this.#startRequest();
    if (!corpusRunActive(this.mutationRun)) this.#stopPolling();
    if (showLoading) this.loading = true;
    this.error = null;
    try {
      const snapshot = await getFilePanel(workspace, null, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      if (snapshot.workspace !== workspace) {
        throw new Error('file panel response changed workspace identity');
      }
      this.snapshot = snapshot;
      if (!corpusRunActive(this.mutationRun)) this.acceptedFiles = 0;
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      // Keep the workspace transition fail closed even when a transport ignores
      // AbortSignal and resolves an invalidated request later.
      if (this.snapshot?.workspace !== workspace) this.snapshot = null;
      this.error = error instanceof FilesApiError
        ? error.message
        : msg('Failed to load files.', {id: 'inspectorFiles.loadFailed'});
    } finally {
      if (this.#session.finishRequest(controller)) this.loading = false;
    }
  }

  loadOlderFiles(): Promise<void> {
    if (this.#olderFilesFlight !== null) return this.#olderFilesFlight;
    const workspace = this.handles.ingest.workspace;
    const cursor = this.snapshot?.workspace === workspace
      ? this.snapshot.nextCursor
      : null;
    if (!cursor || this.loading || this.#session.requestBusy || !this.active) {
      return Promise.resolve();
    }
    const flight = this.#loadOlderFilesPage(
      workspace,
      cursor,
      this.#session.olderGeneration,
    );
    this.#olderFilesFlight = flight;
    void flight.finally(() => {
      if (this.#olderFilesFlight === flight) this.#olderFilesFlight = null;
    });
    return flight;
  }

  async #loadOlderFilesPage(
    workspace: string,
    cursor: string,
    generation: number,
  ): Promise<void> {
    const controller = this.#session.startOlder();
    this.filesLoadMoreState = 'loading';
    this.#olderFilesAnnouncement = msg('Loading older files…', {id: 'inspectorFiles.loadingOlder'});
    try {
      const older = await getFilePanel(workspace, cursor, controller.signal);
      const current = this.snapshot;
      if (
        !this.#session.isOlderCurrent(controller, generation)
        || workspace !== this.handles.ingest.workspace
        || current?.workspace !== workspace
        || current.nextCursor !== cursor
      ) {
        if (this.#session.isOlderCurrent(controller, generation)) {
          this.filesLoadMoreState = 'idle';
          this.#olderFilesAnnouncement = '';
        }
        return;
      }
      if (older.workspace !== workspace) {
        throw new Error('older file page changed workspace identity');
      }
      const paths = new Set(current.files.map((file) => file.filePath));
      const appended = older.files.filter((file) => {
        if (paths.has(file.filePath)) return false;
        paths.add(file.filePath);
        return true;
      });
      this.snapshot = {
        ...current,
        files: [...current.files, ...appended],
        nextCursor: older.nextCursor,
      };
      this.filesLoadMoreState = 'idle';
      this.#olderFilesAnnouncement = appended.length === 1
        ? msg('Loaded 1 older file.', {id: 'inspectorFiles.loadedOneOlder'})
        : msg(str`Loaded ${appended.length} older files.`, {id: 'inspectorFiles.loadedOlder'});
      if (older.nextCursor === null && this.#restoreOlderFocus) {
        this.#restoreOlderFocus = false;
        await this.updateComplete;
        this.querySelector<HTMLElement>('#file-list')?.focus({preventScroll: true});
      }
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#session.isOlderCurrent(controller, generation)
      ) return;
      this.filesLoadMoreState = 'error';
      this.#olderFilesAnnouncement = msg('Older files could not be loaded.', {id: 'inspectorFiles.olderFilesFailed'});
    } finally {
      this.#session.finishOlder(controller);
    }
  }

  async upload(files: readonly File[], label?: string | null): Promise<void> {
    if (files.length === 0) return;
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    this.#workspace = workspace;
    const {controller, generation} = this.#startRequest();
    this.#stopPolling();
    this.#beginMutation();
    this.uploading = true;
    this.error = null;
    const name = uploadLabel(files, label);
    requestToast(this, {message: msg(str`Uploading ${name}...`, {id: 'inspectorFiles.uploadingToast'})});
    try {
      const receipt = await uploadFileBatch(workspace, files, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      this.mutationRun = receipt;
      this.acceptedFiles = receipt.fileCount ?? files.length;
      requestToast(this, {
        message: msg('Files received — Corpus update accepted', {id: 'inspectorFiles.filesReceived'}),
        duration: 3000,
      });
      void this.#poll(workspace);
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      const message = error instanceof FilesApiError
        ? error.message
        : msg('Upload failed.', {id: 'inspectorFiles.uploadFailed'});
      this.error = message;
      requestToast(this, {message, duration: 3000});
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) {
        this.uploading = false;
        this.loading = false;
      }
    }
  }

  pause(): void {
    this.#invalidateOlderFiles();
    this.#session.pause();
    this.uploading = false;
  }

  async #deleteFile(filePath: string): Promise<void> {
    if (!filePath) return;
    const filename = filePath.split('/').pop() || filePath;
    const dialog = this.querySelector<HTMLDialogElement>('#delete-file-dialog');
    const message = this.querySelector<HTMLElement>('#delete-file-message');
    if (!dialog || !message) return;
    message.textContent = msg(
      str`${filename} will be permanently removed from this workspace.`,
      {id: 'inspectorFiles.deleteNotice'},
    );
    if (await modalResult(this, dialog, () => this.#restoreDeleteTrigger()) !== 'confirm') return;
    const workspace = this.handles.ingest.workspace;
    this.#invalidateOlderFiles();
    this.#stopPolling();
    const {controller, generation} = this.#startRequest();
    this.#beginMutation();
    this.error = null;
    try {
      const receipt = await deleteFileRequest(workspace, filePath, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      this.mutationRun = receipt;
      requestToast(this, {
        message: msg('File deletion accepted.', {id: 'inspectorFiles.fileDeleted'}),
        duration: 3000,
      });
      void this.#poll(workspace);
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      const message = error instanceof FilesApiError
        ? error.message
        : msg('Deletion failed.', {id: 'inspectorFiles.deletionFailed'});
      this.error = message;
      requestToast(this, {message, duration: 3000});
    } finally {
      this.#finishMutation();
      if (this.#session.finishRequest(controller)) this.loading = false;
    }
  }

  async #poll(workspace: string): Promise<void> {
    const receipt = this.mutationRun;
    if (!receipt) return;
    const controller = this.#session.startPollRequest();
    try {
      const status = await getCorpusRunStatus(receipt.statusUrl, controller.signal);
      if (workspace !== this.handles.ingest.workspace || !this.active || !this.isConnected) return;
      if (this.mutationRun?.runId !== receipt.runId) return;
      this.mutationRun = {
        ...status,
        workspace: receipt.workspace,
        fileCount: receipt.fileCount,
      };
      if (waitingForRepair(this.mutationRun)) return;
      if (corpusRunActive(status)) {
        this.#schedulePoll(workspace);
        return;
      }
      this.acceptedFiles = 0;
      requestToast(this, {
        message: status.status === 'succeeded'
          ? msg('Corpus update finished.', {id: 'inspectorFiles.corpusUpdateFinished'})
          : msg('Corpus update did not finish.', {id: 'inspectorFiles.corpusUpdateFailed'}),
        duration: 4000,
      });
      await this.reload(false);
      const recovery = this.querySelector<DlFailedFileRecovery>('dl-failed-file-recovery');
      await recovery?.refresh(false);
    } catch (error) {
      if (isAbortError(error)) return;
      if (workspace === this.handles.ingest.workspace && this.active && this.isConnected) {
        this.#schedulePoll(workspace);
      }
    } finally {
      this.#session.finishPollRequest(controller);
    }
  }

  async #resumeMutation(): Promise<void> {
    const waiting = this.mutationRun;
    const workspace = this.handles.ingest.workspace;
    if (!waitingForRepair(waiting) || this.#session.mutating) return;
    const {controller, generation} = this.#startRequest();
    this.#beginMutation();
    try {
      const status = await resumeCorpusRun(waiting.resumeUrl, controller.signal);
      if (!this.#isCurrent(controller, workspace, generation)) return;
      this.mutationRun = {
        ...status,
        workspace: waiting.workspace,
        fileCount: waiting.fileCount,
      };
      requestToast(this, {
        message: msg('Corpus repair resume accepted.', {
          id: 'inspectorFiles.corpusResumeAccepted',
        }),
        duration: 3000,
      });
      if (corpusRunActive(status) && !waitingForRepair(this.mutationRun)) {
        this.#schedulePoll(workspace);
      }
    } catch (error) {
      if (
        isAbortError(error)
        || !this.#isCurrent(controller, workspace, generation)
      ) return;
      const message = msg('Corpus repair resume failed.', {
        id: 'inspectorFiles.corpusResumeFailed',
      });
      this.error = message;
      requestToast(this, {message, duration: 4000});
    } finally {
      this.#finishMutation();
      this.#session.finishRequest(controller);
    }
  }

  #schedulePoll(workspace: string): void {
    this.#session.schedulePoll(workspace, (next) => { void this.#poll(next); });
  }

  #invalidateOlderFiles(): void {
    this.#session.invalidateOlder();
    this.#olderFilesFlight = null;
    this.filesLoadMoreState = 'idle';
    this.#olderFilesAnnouncement = '';
    this.#restoreOlderFocus = false;
  }

  #stopPolling(): void {
    this.#session.stopPolling();
  }

  #startRequest(): {controller: AbortController; generation: number} {
    this.#requestGeneration += 1;
    return {
      controller: this.#session.startRequest(),
      generation: this.#requestGeneration,
    };
  }

  #isCurrent(
    controller: AbortController,
    workspace: string,
    generation: number,
  ): boolean {
    return generation === this.#requestGeneration
      && this.#session.isCurrent(controller, workspace, this.handles.ingest.workspace);
  }

  #beginMutation(): void {
    this.#session.beginMutation();
  }

  #finishMutation(): void {
    this.#session.finishMutation();
  }

  #chooseFiles(): void {
    this.querySelector<HTMLInputElement>('#file-input')?.click();
  }

  #chooseFolder(): void {
    this.querySelector<HTMLInputElement>('#folder-input')?.click();
  }

  #fileInputChanged(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = '';
    if (files.length > 0) void this.upload(files);
  }

  #folderInputChanged(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const rawFiles = Array.from(input.files ?? []);
    input.value = '';
    if (rawFiles.length === 0) return;
    let folderName: string | null = null;
    const files = rawFiles.map((file) => {
      const path = file.webkitRelativePath || file.name;
      if (!folderName && file.webkitRelativePath) folderName = path.split('/')[0];
      return withRelativePath(file, path);
    });
    void this.upload(files, folderName);
  }

  #loadOlderFiles = (event: Event): void => {
    const button = event.currentTarget as HTMLButtonElement;
    this.#restoreOlderFocus = document.activeElement === button;
    void this.loadOlderFiles();
  };


  #restoreDeleteTrigger(): void {
    const trigger = this.#deleteTrigger;
    this.#deleteTrigger = null;
    if (trigger?.isConnected) trigger.focus();
  }

  #deleteDialog(): TemplateResult {
    return html`
      <dialog id="delete-file-dialog" class="confirm-dialog"
              aria-labelledby="delete-file-title" aria-describedby="delete-file-message">
        <form method="dialog">
          <h2 id="delete-file-title">${msg('Delete file', {id: 'inspectorFiles.deleteTitle'})}</h2>
          <p id="delete-file-message"></p>
          <div class="dl-dialog-actions">
            <button type="submit" value="cancel">${msg('Cancel', {id: 'inspectorFiles.cancel'})}</button>
            <button type="submit" value="confirm" class="dl-dialog-danger">${msg('Delete', {id: 'inspectorFiles.delete'})}</button>
          </div>
        </form>
      </dialog>
    `;
  }

  #progress(run: MutationRun | null): TemplateResult | typeof nothing {
    if (waitingForRepair(run)) {
      return html`
        <div id="ingest-progress" role="status">
          <div class=${fileStyles['file-status']}>
            <span>
              ${run.repairReason ?? msg('The corpus outcome needs operator repair.', {
                id: 'inspectorFiles.corpusRepairRequired',
              })}
              ${run.repairRemedy ?? msg('Repair it, then resume this same Run.', {
                id: 'inspectorFiles.corpusRepairRemedy',
              })}
            </span>
            <button type="button" ?disabled=${this.hasActiveMutation}
                    @click=${() => { void this.#resumeMutation(); }}>
              ${msg('Resume after repair', {id: 'inspectorFiles.corpusResumeRepair'})}
            </button>
          </div>
        </div>
      `;
    }
    if (!corpusRunActive(run)) return nothing;
    return html`
      <div id="ingest-progress">
        <div class=${fileStyles['file-status']}>
          <div class=${fileStyles.spinner}></div>
          <span>${msg('Corpus update in progress…', {id: 'inspectorFiles.corpusUpdateRunning'})}</span>
        </div>
      </div>
    `;
  }

  protected override render(): TemplateResult {
    const snapshot = this.snapshot;
    const files = snapshot?.files ?? [];
    return html`
      ${this.#progress(this.mutationRun)}
      ${this.error ? html`<div class="file-error" role="alert">${this.error}</div>` : nothing}
      <div class=${`${fileStyles['upload-zone']}${this.uploading ? ` ${fileStyles['is-uploading']}` : ''}`} id="upload-zone">
        <button type="button" class=${fileStyles['upload-zone-file-action']}
                data-upload-file-action
                aria-label=${msg('Choose files', {id: 'inspectorFiles.chooseFilesAria'})}
                @click=${() => { this.#chooseFiles(); }}>
          <span class=${fileStyles['upload-text']}>${msg('Drop files or folders, or click to choose files', {id: 'inspectorFiles.dropHint'})}</span>
        </button>
        <button type="button" class=${fileStyles['upload-folder-action']}
                @click=${() => { this.#chooseFolder(); }}>${msg('Choose folder', {id: 'inspectorFiles.chooseFolder'})}</button>
        <input class="hidden" type="file" id="file-input" name="files" multiple
               @change=${(event: Event) => { this.#fileInputChanged(event); }}>
        <input class="hidden" type="file" id="folder-input" webkitdirectory directory multiple
               @change=${(event: Event) => { this.#folderInputChanged(event); }}>
        <div id="upload-spinner" class=${fileStyles['file-status']}>${msg('Uploading...', {id: 'inspectorFiles.uploadingStatus'})}</div>
      </div>
      <dl-failed-file-recovery
        .workspace=${this.handles.ingest.workspace}
        .active=${this.active}
        @dl-failed-file-recovery-complete=${() => { void this.reload(false); }}
      ></dl-failed-file-recovery>
      ${this.loading ? html`
        <div class=${fileStyles['file-status']}><div class=${fileStyles.spinner}></div><span>${msg('Loading files...', {id: 'inspectorFiles.loadingFiles'})}</span></div>
      ` : nothing}
      ${!this.loading ? html`
        <div id="file-list" role="list" aria-label=${msg('Processed files', {id: 'inspectorFiles.processedFilesAria'})} tabindex="-1">
          ${repeat(
            files,
            (file) => file.filePath,
            (file) => html`
              <div class=${fileStyles['file-item']} role="listitem" data-file-item>
                <span class=${fileStyles['file-name']} title=${file.filePath}>${file.fileName}</span>
                <button class=${fileStyles['file-delete']} type="button" data-file-delete
                        aria-label=${msg(str`Delete ${file.fileName}`, {id: 'inspectorFiles.deleteFileAria'})}
                        @click=${(event: Event) => {
                          this.#deleteTrigger = event.currentTarget as HTMLElement;
                          void this.#deleteFile(file.filePath);
                        }}>
                  ${icon('close', {size: 'sm', className: fileStyles['file-delete-icon']})}
                </button>
              </div>
            `,
          )}
        </div>
        ${snapshot?.nextCursor ? html`
          <div class=${fileStyles['file-page-control']}>
            <button type="button" data-load-older-files
                    aria-busy=${this.filesLoadMoreState === 'loading' ? 'true' : 'false'}
                    ?disabled=${this.filesLoadMoreState === 'loading'}
                    @click=${this.#loadOlderFiles}>
              ${this.filesLoadMoreState === 'error'
                ? msg('Retry loading older files', {id: 'inspectorFiles.retryLoadOlder'})
                : msg('Load older files', {id: 'inspectorFiles.loadOlder'})}
            </button>
          </div>
        ` : nothing}
        <span class="sr-only" data-older-files-status role="status" aria-live="polite">
          ${this.#olderFilesAnnouncement}
        </span>
      ` : nothing}
      ${!this.loading && !this.error && files.length === 0 && !corpusRunActive(this.mutationRun) ? html`
        <div class="empty-state">${msg(str`No files ingested in workspace “${this.#workspace}”.`, {id: 'inspectorFiles.emptyState'})}</div>
      ` : nothing}
      ${this.acceptedFiles > 0 && corpusRunActive(this.mutationRun) ? html`
        <div class=${`${fileStyles['ingest-queue-notice']} ${fileStyles['ingest-queue-notice--inline']}`}>
          ${msg(str`${this.acceptedFiles} new file(s) accepted for ingest`, {id: 'inspectorFiles.acceptedForIngest'})}
        </div>
      ` : nothing}
      ${this.#deleteDialog()}
    `;
  }
}

customElements.define('dl-inspector-files', DlInspectorFiles);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-files': DlInspectorFiles;
  }
}
