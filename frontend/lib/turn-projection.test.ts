// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import type {ChatTurnView} from './chat-views.ts';
import {ANSWER_PHASE_LABELS, answerPhaseLabel, applyAnswerEvent} from './turn-projection.ts';

function turn(overrides: Partial<ChatTurnView> = {}): ChatTurnView {
  return {
    id: 't1',
    userText: 'q',
    userAttachments: [],
    runId: 'r1',
    state: 'pending',
    streamText: '',
    presentation: null,
    usage: {},
    evidence: {},
    error: '',
    progress: '',
    liveStatus: '',
    sawChildren: false,
    cancelRequested: false,
    steeringMessages: [],
    toolRows: [],
    toolTotal: 0,
    toolExpanded: false,
    ...overrides,
  };
}

test('phase labels map every server phase and reject unknowns', () => {
  assert.equal(answerPhaseLabel('searching'), 'Searching knowledge base...');
  assert.equal(answerPhaseLabel('bogus'), null);
  assert.ok(Object.keys(ANSWER_PHASE_LABELS).length >= 5);
});

test('tokens accumulate within a batch and settle the stream state', () => {
  let view = applyAnswerEvent(turn(), {kind: 'token', text: 'He'});
  view = applyAnswerEvent(view, {kind: 'token', text: 'llo'});
  assert.equal(view.streamText, 'Hello');
  assert.equal(view.state, 'streaming');
  assert.equal(view.error, '');
});

test('reset clears the stream back to pending', () => {
  const streamed = turn({state: 'streaming', streamText: 'abc', progress: 'x', error: ''});
  const view = applyAnswerEvent(streamed, {kind: 'reset'});
  assert.deepEqual(
    {state: view.state, streamText: view.streamText, progress: view.progress},
    {state: 'pending', streamText: '', progress: ''},
  );
});

test('progress applies known phases and ignores unknown ones', () => {
  const base = turn();
  const known = applyAnswerEvent(base, {kind: 'progress', payload: {phase: 'searching'}});
  assert.ok(known.progress.includes('Searching'));
  const ignored = applyAnswerEvent(known, {kind: 'progress', payload: {phase: 'bogus'}});
  assert.equal(ignored, known);
});

test('memory events never change the view', () => {
  const base = turn();
  assert.equal(applyAnswerEvent(base, {kind: 'memory', payload: {operation: 'remember'}}), base);
});

test('tool events drive the trace and sawChildren', () => {
  let view = applyAnswerEvent(turn(), {
    kind: 'tool',
    eventType: 'tool_start',
    payload: {tool_name: 'spawn_agent', call_id: 'c1'},
  });
  assert.equal(view.toolTotal, 1);
  assert.equal(view.sawChildren, true);
  view = applyAnswerEvent(view, {
    kind: 'tool',
    eventType: 'tool_progress',
    payload: {tool_name: 'spawn_agent', call_id: 'c1', object_label: 'review'},
  });
  assert.ok(view.progress.includes('review'));
});

test('errors fail the turn', () => {
  const view = applyAnswerEvent(turn(), {
    kind: 'error',
    payload: {kind: 'run_abandoned', message: 'Run could not be recovered.'},
  });
  assert.equal(view.state, 'failed');
  assert.equal(view.error, 'Run could not be recovered.');
});

test('done settles succeeded, cancelled, and malformed payloads', () => {
  const succeeded = applyAnswerEvent(turn(), {
    kind: 'done',
    payload: {
      status: 'succeeded',
      presentation: {answer_text: 'answer', sources: []},
      usage: {tokens: 1},
      evidence: {sources: 2},
    },
  });
  assert.equal(succeeded.state, 'succeeded');
  assert.equal(succeeded.streamText, 'answer');
  assert.equal(succeeded.evidence.sources, 2);

  const cancelled = applyAnswerEvent(turn(), {kind: 'done', payload: {status: 'cancelled'}});
  assert.equal(cancelled.state, 'cancelled');

  const malformed = applyAnswerEvent(turn(), {kind: 'done', payload: {status: 'running'}});
  assert.equal(malformed.state, 'failed');
});
