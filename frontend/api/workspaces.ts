// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {corpusRunReceipt, type WebCorpusRunReceipt} from './corpus-runs.ts';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

const createdWorkspace = v.pipe(
  v.object({workspace: v.string(), display_name: v.string()}),
  v.transform((w) => ({workspace: w.workspace, displayName: w.display_name})),
);
export type CreatedWorkspace = v.InferOutput<typeof createdWorkspace>;

export const workspacePageItem = v.pipe(
  v.object({workspace: v.string(), display_name: v.string(), embedding_model: v.string()}),
  v.transform((w) => ({
    workspace: w.workspace,
    displayName: w.display_name,
    embeddingModel: w.embedding_model,
  })),
);
export type WorkspacePageItem = v.InferOutput<typeof workspacePageItem>;

const workspacePage = v.pipe(
  v.object({
    workspaces: v.optional(v.array(workspacePageItem)),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({workspaces: w.workspaces ?? [], nextCursor: w.next_cursor ?? null})),
);
export type WorkspacePage = v.InferOutput<typeof workspacePage>;

export class WorkspaceApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'WorkspaceApiError';
    this.status = status;
  }
}

export async function getWorkspacesPage(
  cursor: string | null,
  signal?: AbortSignal,
): Promise<WorkspacePage> {
  const query = cursor === null ? '' : `?cursor=${encodeURIComponent(cursor)}`;
  const response = await fetch(`/web/api/workspaces${query}`, {signal});
  return parseWire(response, workspacePage, makeError, 'Failed to load workspaces');
}

async function post<Input, Output>(
  path: string,
  body: Record<string, string>,
  schema: v.GenericSchema<Input, Output>,
  fallback: string,
  signal?: AbortSignal,
): Promise<Output> {
  const response = await fetch(path, {
    method: 'POST',
    headers: csrfHeaders('application/x-www-form-urlencoded'),
    body: new URLSearchParams(body).toString(),
    signal,
  });
  if (!response.ok) {
    // The route answers with {"error": ...}; fall back when it cannot.
    const detail = await response.json().catch(() => null) as {error?: unknown} | null;
    const message = typeof detail?.error === 'string' ? detail.error : fallback;
    throw new WorkspaceApiError(response.status, message);
  }
  return v.parse(schema, await response.json());
}

function makeError(status: number, message: string): Error {
  return new WorkspaceApiError(status, message);
}

export function createWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<CreatedWorkspace> {
  return post(
    '/web/api/workspaces/create',
    {workspace_name: name},
    createdWorkspace,
    'Failed to create workspace',
    signal,
  );
}

export function resetWorkspaceRequest(
  name: string,
  signal?: AbortSignal,
): Promise<WebCorpusRunReceipt> {
  return post(
    '/web/api/workspaces/reset',
    {workspace_name: name, confirm_name: name},
    corpusRunReceipt,
    'Could not accept Corpus reset.',
    signal,
  );
}
