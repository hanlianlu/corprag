// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges } from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {PresentationSource} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeExternalHttpHref, safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import sourceStyles from '../styles/inspector-sources.module.css';
import type {ImageOpenDetail} from './image-lightbox.ts';
import {mountRichHtml, typesetRichContent} from './rich-rendering.ts';

export interface InspectorSourcesStateDetail {
  hasSources: boolean;
  fullyExpanded: boolean;
}

/** Source list content owned by the Inspector. */
export class DlInspectorSources extends LightElement {
  static properties = {
    sources: {attribute: false},
    expandedRef: {state: true},
    onlyChunk: {state: true},
    activeRef: {state: true},
    activeChunk: {state: true},
    showAll: {state: true},
  };

  declare sources: PresentationSource[];
  declare expandedRef: string | null;
  declare onlyChunk: string | null;
  declare activeRef: string | null;
  declare activeChunk: string | null;
  declare showAll: boolean;

  #pendingReveal: string | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.sources = [];
    this.expandedRef = null;
    this.onlyChunk = null;
    this.activeRef = null;
    this.activeChunk = null;
    this.showAll = false;
  }

  setSelection(ref?: string, chunk?: string): void {
    this.showAll = false;
    this.expandedRef = ref || null;
    this.onlyChunk = chunk || null;
    this.activeRef = ref && chunk ? ref : null;
    this.activeChunk = ref && chunk ? chunk : null;
    this.#pendingReveal = ref
      ? `[data-ref="${CSS.escape(ref)}"]${chunk
        ? `[data-chunk="${CSS.escape(chunk)}"]`
        : ' > [data-source-header]'}`
      : null;
    // Navigation is an intent, even when the selected values have not changed.
    this.requestUpdate();
  }

  expandAll(): void {
    this.#pendingReveal = null;
    this.showAll = true;
    this.expandedRef = null;
    this.onlyChunk = null;
  }

  collapseAll(): void {
    this.#pendingReveal = null;
    this.showAll = false;
    this.expandedRef = null;
    this.onlyChunk = null;
  }

  get fullyExpanded(): boolean {
    return this.sources.length > 0 && this.showAll;
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('sources')) {
      this.sources.forEach((source) => {
        source.chunks.forEach((chunk, index) => {
          const key = this.#chunkKey(chunk.chunkIdx, index);
          const host = this.querySelector<HTMLElement>(
            `[data-source-content="${CSS.escape(source.id)}:${CSS.escape(key)}"]`,
          );
          if (host) mountRichHtml(host, chunk.contentHtml);
        });
      });
    }
    if (changed.has('sources') || changed.has('expandedRef') || changed.has('showAll')) {
      this.querySelectorAll('[data-source-chunks]').forEach((element) => {
        if (element.closest('[data-expanded]')) typesetRichContent(element);
      });
    }
    const reveal = this.#pendingReveal;
    this.#pendingReveal = null;
    if (reveal) {
      const target = this.querySelector<HTMLElement>(reveal);
      if (target?.getClientRects().length) {
        target.scrollIntoView({behavior: 'auto', block: 'nearest', inline: 'nearest'});
      }
    }
    this.dispatchEvent(new CustomEvent<InspectorSourcesStateDetail>(
      'dl-inspector-sources-state-change',
      {
        bubbles: true,
        composed: true,
        detail: {
          hasSources: this.sources.length > 0,
          fullyExpanded: this.fullyExpanded,
        },
      },
    ));
  }

  #chunkKey(chunkIndex: number | null, fallback: number): string {
    return String(chunkIndex ?? fallback + 1);
  }

  #toggle(sourceId: string): void {
    this.#pendingReveal = null;
    if (!this.showAll && this.expandedRef === sourceId) {
      this.expandedRef = null;
      this.onlyChunk = null;
      return;
    }
    this.showAll = false;
    this.expandedRef = sourceId;
    this.onlyChunk = null;
  }

  #source(source: PresentationSource): TemplateResult {
    const expanded = this.showAll || this.expandedRef === source.id;
    const download = safeSameOriginHref(source.downloadUrl);
    const external = safeExternalHttpHref(source.sourceUrl);
    const s = sourceStyles;
    return html`
      <div class=${`${s['source-doc']}${expanded ? ` ${s.expanded}` : ''}`}
           data-ref=${source.id} ?data-expanded=${expanded}>
        <div class=${s['source-doc-header']} data-source-header>
          <button class=${s['source-doc-toggle']} type="button" data-source-toggle aria-expanded=${String(expanded)}
                  @click=${() => { this.#toggle(source.id); }}>
            <span class=${s['collapse-icon']}>${icon('disclosure', {size: 'xs'})}</span>
            <span class=${s['source-doc-title']}>${source.title}</span>
            <span class=${s['source-doc-badge']}>${source.id}</span>
            <span class=${s['source-doc-count']}>${source.chunks.length}</span>
          </button>
          ${download ? html`
            <a href=${download} class=${s['source-action-icon']}
               title=${msg('Download source', {id: 'inspectorSources.downloadSource'})}
               aria-label=${msg('Download source', {id: 'inspectorSources.downloadSource'})} download>
              ${icon('download', {size: 'sm', className: s['source-action-icon-svg']})}
            </a>
          ` : nothing}
          ${external ? html`
            <a href=${external} class=${s['source-action-icon']}
               title=${msg('Open source', {id: 'inspectorSources.openSource'})}
               aria-label=${msg('Open source', {id: 'inspectorSources.openSource'})}
               target="_blank" rel="noopener noreferrer">
              ${icon('open-external', {size: 'sm', className: s['source-action-icon-svg']})}
            </a>
          ` : nothing}
        </div>
        <div class=${s['source-doc-chunks']} data-source-chunks ?hidden=${!expanded}>
          ${repeat(
            source.chunks,
            (chunk, index) => this.#chunkKey(chunk.chunkIdx, index),
            (chunk, index) => {
              const key = this.#chunkKey(chunk.chunkIdx, index);
              const hidden = expanded && !this.showAll && this.onlyChunk !== null
                && key !== this.onlyChunk;
              const active = this.activeRef === source.id && this.activeChunk === key;
              const image = safeImageSrc(chunk.imageUrl || chunk.thumbnailUrl);
              const thumbnail = safeImageSrc(chunk.thumbnailUrl || chunk.imageUrl);
              return html`
                <div class=${`${s['source-chunk']}${active ? ` ${s.active}` : ''}`} data-ref=${source.id}
                     data-chunk=${key} ?hidden=${hidden}>
                  <div class=${s['source-chunk-header']}>
                    <span class=${s['source-chunk-page']}>
                      ${chunk.pageNumber === null ? `#${key}` : `p.${chunk.pageNumber}`}
                    </span>
                  </div>
                  ${image && thumbnail ? html`
                    <div class=${s['source-chunk-image']}>
                      <img src=${thumbnail}
                           alt=${msg(str`Page ${chunk.pageNumber ?? ''}`, {id: 'inspectorSources.pageAlt'})}
                           loading="lazy" role="button" tabindex="0"
                           aria-label=${msg('Open page image', {id: 'inspectorSources.openPageImage'})}
                           @click=${(event: Event) => this.#openImage(
                             image,
                             event.currentTarget as HTMLElement,
                           )}
                           @keydown=${(event: KeyboardEvent) => {
                             if (event.key !== 'Enter' && event.key !== ' ') return;
                             event.preventDefault();
                             this.#openImage(image, event.currentTarget as HTMLElement);
                           }}>
                    </div>
                  ` : nothing}
                  ${chunk.contentHtml ? html`
                    <div class=${s['source-chunk-content']}
                         data-source-content=${`${source.id}:${key}`}></div>
                  ` : nothing}
                </div>
              `;
            },
          )}
        </div>
      </div>
    `;
  }

  protected override render(): TemplateResult {
    return html`${repeat(this.sources, (source) => source.id, (source) => this.#source(source))}`;
  }

  #openImage(src: string, returnFocus: HTMLElement): void {
    const gallery = this.sources.flatMap((source) => source.chunks)
      .map((chunk) => safeImageSrc(chunk.imageUrl || chunk.thumbnailUrl))
      .filter(Boolean);
    this.dispatchEvent(new CustomEvent<ImageOpenDetail>('dl-image-open', {
      bubbles: true,
      composed: true,
      detail: {src, gallery: [...new Set(gallery)], returnFocus},
    }));
  }

}

customElements.define('dl-inspector-sources', DlInspectorSources);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-sources': DlInspectorSources;
  }

  interface HTMLElementEventMap {
    'dl-inspector-sources-state-change': CustomEvent<InspectorSourcesStateDetail>;
  }
}
