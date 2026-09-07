// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {corpusRunReceipt, type WebCorpusRunReceipt} from './corpus-runs.ts';
import {csrfHeaders} from './csrf.ts';

const webFileItem = v.pipe(
  v.object({file_name: v.string(), file_path: v.string()}),
  v.transform((w) => ({fileName: w.file_name, filePath: w.file_path})),
);
export type WebFileItem = v.InferOutput<typeof webFileItem>;

const webFilePanelSnapshot = v.pipe(
  v.object({
    workspace: v.string(),
    files: v.array(webFileItem),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    workspace: w.workspace,
    files: w.files,
    nextCursor: w.next_cursor ?? null,
  })),
);
export type WebFilePanelSnapshot = v.InferOutput<typeof webFilePanelSnapshot>;

const webFailedFileItem = v.pipe(
  v.object({
    document_id: v.string(),
    file_name: v.string(),
    error: v.string(),
    updated_at: v.string(),
  }),
  v.transform((w) => ({
    documentId: w.document_id,
    fileName: w.file_name,
    error: w.error,
    updatedAt: w.updated_at,
  })),
);
export type WebFailedFileItem = v.InferOutput<typeof webFailedFileItem>;

const webFailedFilesPage = v.pipe(
  v.object({
    workspace: v.string(),
    failed: v.array(webFailedFileItem),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    workspace: w.workspace,
    failed: w.failed,
    nextCursor: w.next_cursor ?? null,
  })),
);
export type WebFailedFilesPage = v.InferOutput<typeof webFailedFilesPage>;

export class FilesApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'FilesApiError';
    this.status = status;
  }
}

function url(path: string, workspace: string): string {
  const target = new URL(path, window.location.origin);
  if (workspace) target.searchParams.set('workspace', workspace);
  return target.pathname + target.search;
}

async function json<Input, Output>(
  response: Response,
  schema: v.GenericSchema<Input, Output>,
  fallback: string,
): Promise<Output> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as {
      detail?: unknown;
      error?: unknown;
    } | null;
    const detail = typeof payload?.detail === 'string'
      ? payload.detail
      : typeof payload?.error === 'string'
        ? payload.error
        : fallback;
    throw new FilesApiError(response.status, detail);
  }
  try {
    return v.parse(schema, await response.json());
  } catch {
    throw new FilesApiError(response.status, fallback);
  }
}

export async function getFilePanel(
  workspace: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<WebFilePanelSnapshot> {
  const target = new URL(url('/web/api/files', workspace), window.location.origin);
  if (cursor !== null) target.searchParams.set('cursor', cursor);
  const response = await fetch(target.pathname + target.search, {signal});
  return json(response, webFilePanelSnapshot, 'Failed to load files');
}

export async function getFailedFiles(
  workspace: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<WebFailedFilesPage> {
  const target = new URL(url('/web/api/files/failed', workspace), window.location.origin);
  if (cursor !== null) target.searchParams.set('cursor', cursor);
  const response = await fetch(target.pathname + target.search, {signal});
  return json(response, webFailedFilesPage, 'Failed to load documents needing attention');
}

export async function startFailedFileRetry(
  workspace: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunReceipt> {
  const response = await fetch(url('/web/api/files/retry', workspace), {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  return json(response, corpusRunReceipt, 'Document recovery could not be started');
}

export async function uploadFileBatch(
  workspace: string,
  files: readonly File[],
  signal?: AbortSignal,
): Promise<WebCorpusRunReceipt> {
  const body = new FormData();
  body.append('workspace', workspace);
  for (const file of files) {
    const relative = file as File & {_relativePath?: string; webkitRelativePath?: string};
    body.append('files', file, relative._relativePath || relative.webkitRelativePath || file.name);
  }
  const response = await fetch('/web/api/files/upload', {
    method: 'POST',
    headers: csrfHeaders(),
    body,
    signal,
  });
  return json(response, corpusRunReceipt, 'Upload failed');
}

export async function deleteFileRequest(
  workspace: string,
  filePath: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunReceipt> {
  const target = new URL('/web/api/files', window.location.origin);
  target.searchParams.set('workspace', workspace);
  target.searchParams.set('file_path', filePath);
  const response = await fetch(target.pathname + target.search, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  return json(response, corpusRunReceipt, 'Deletion failed');
}
