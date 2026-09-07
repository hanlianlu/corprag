// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Abort, generation, and poll bookkeeping for failed-document recovery.

 *  The Feature still owns page/recovery rendering and toasts. This session is
 *  the single in-flight list/mutation/poll loop.
 */

const RECOVERY_POLL_INTERVAL_MS = 2000;

export class FailedFileRecoverySession {
  #list: AbortController | null = null;
  #loadMore: AbortController | null = null;
  #mutation: AbortController | null = null;
  #modal: AbortController | null = null;
  #poll: AbortController | null = null;
  #pollTimer: number | null = null;
  #contextGeneration = 0;
  #listGeneration = 0;

  get contextGeneration(): number {
    return this.#contextGeneration;
  }

  get listGeneration(): number {
    return this.#listGeneration;
  }

  startList(): {controller: AbortController; generation: number} {
    this.#list?.abort();
    this.#loadMore?.abort();
    this.stopPolling();
    const generation = ++this.#listGeneration;
    const controller = new AbortController();
    this.#list = controller;
    return {controller, generation};
  }

  isListCurrent(
    controller: AbortController,
    workspace: string,
    currentWorkspace: string,
    generation: number,
    active: boolean,
  ): boolean {
    return this.#list === controller
      && workspace === currentWorkspace
      && generation === this.#listGeneration
      && active;
  }

  finishList(controller: AbortController): boolean {
    if (this.#list !== controller) return false;
    this.#list = null;
    return true;
  }

  startLoadMore(): AbortController {
    this.#loadMore?.abort();
    const controller = new AbortController();
    this.#loadMore = controller;
    return controller;
  }

  isLoadMoreCurrent(controller: AbortController, generation: number): boolean {
    return this.#loadMore === controller && generation === this.#contextGeneration;
  }

  finishLoadMore(controller: AbortController): boolean {
    if (this.#loadMore !== controller) return false;
    this.#loadMore = null;
    return true;
  }

  startMutation(): AbortController {
    this.#mutation?.abort();
    const controller = new AbortController();
    this.#mutation = controller;
    return controller;
  }

  isMutationCurrent(
    controller: AbortController,
    workspace: string,
    currentWorkspace: string,
    generation: number,
    active: boolean,
  ): boolean {
    return this.#mutation === controller
      && workspace === currentWorkspace
      && generation === this.#contextGeneration
      && active;
  }

  finishMutation(controller: AbortController): boolean {
    if (this.#mutation !== controller) return false;
    this.#mutation = null;
    return true;
  }

  startModal(): AbortController {
    this.#modal?.abort();
    const controller = new AbortController();
    this.#modal = controller;
    return controller;
  }

  finishModal(controller: AbortController): boolean {
    if (this.#modal !== controller) return false;
    this.#modal = null;
    return true;
  }

  schedulePoll(
    workspace: string,
    statusUrl: string,
    tick: (workspace: string, statusUrl: string) => void,
  ): void {
    this.stopPolling();
    this.#pollTimer = window.setTimeout(() => {
      this.#pollTimer = null;
      tick(workspace, statusUrl);
    }, RECOVERY_POLL_INTERVAL_MS);
  }

  startPollRequest(): AbortController {
    const controller = new AbortController();
    this.#poll = controller;
    return controller;
  }

  isPollCurrent(controller: AbortController): boolean {
    return this.#poll === controller;
  }

  finishPollRequest(controller: AbortController): void {
    if (this.#poll === controller) this.#poll = null;
  }

  stopPolling(): void {
    if (this.#pollTimer !== null) window.clearTimeout(this.#pollTimer);
    this.#pollTimer = null;
    this.#poll?.abort();
    this.#poll = null;
  }

  cancelContext(): void {
    this.#contextGeneration += 1;
    this.#listGeneration += 1;
    this.#list?.abort();
    this.#list = null;
    this.#loadMore?.abort();
    this.#loadMore = null;
    this.#mutation?.abort();
    this.#mutation = null;
    this.#modal?.abort();
    this.#modal = null;
    this.stopPolling();
  }
}
