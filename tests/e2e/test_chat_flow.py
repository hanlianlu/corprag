# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for chat flow: durable run submission, streaming, and reload."""

from typing import Any
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, Route, expect


@pytest.mark.e2e
def test_conversation_list_endpoint_returns_two_exact_keyset_pages(
    page: Page,
    e2e_conversation_service: Any,
) -> None:
    expected_ids = e2e_conversation_service.seed_conversations(count=10)
    page.goto("/web/")

    result = page.evaluate(
        """
        async () => {
          const firstResponse = await fetch('/web/api/conversations?limit=7');
          const first = await firstResponse.json();
          const secondResponse = await fetch(
            `/web/api/conversations?limit=7&cursor=${encodeURIComponent(first.next_cursor)}`
          );
          return {
            firstStatus: firstResponse.status,
            secondStatus: secondResponse.status,
            first,
            second: await secondResponse.json(),
          };
        }
        """
    )

    assert result["firstStatus"] == 200
    assert result["secondStatus"] == 200
    assert [item["conversation_id"] for item in result["first"]["items"]] == expected_ids[:7]
    assert isinstance(result["first"]["next_cursor"], str)
    assert [item["conversation_id"] for item in result["second"]["items"]] == expected_ids[7:]
    assert result["second"]["next_cursor"] is None


@pytest.mark.e2e
def test_chat_submit_streams_answer(page):
    """Submit a query via the composer and verify the AI response appears in the DOM.

    The mocked backend accepts a durable run, then replays its progress, token,
    and terminal events; the frontend renders them into .ai-message-content.
    """
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    composer = page.locator(".composer-input")
    composer.fill("What is DlightRAG?")
    page.click(".composer-send")

    # After acceptance the composer clears and the unpersisted root route adopts
    # the server-created conversation without replacing the live viewport.
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    page.wait_for_url("**/web/conversations/*")
    assert page.locator("[aria-current='page']").count() == 1

    # AI message container should appear with progressive content
    page.wait_for_selector(".app.has-messages", timeout=10000)
    ai_messages = page.locator('[class*="aiMessageContent"]')
    assert ai_messages.count() >= 1


@pytest.mark.e2e
def test_chat_answer_shows_text(page):
    """Verify the answer text is rendered and visible after stream completion."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    page.locator(".composer-input").fill("test")
    page.click(".composer-send")

    # Wait for text content to appear in any AI message
    page.wait_for_function(
        """
        () => {
          const msgs = document.querySelectorAll('[class*="aiMessageContent"]');
          return Array.from(msgs).some(m => m.textContent.trim().length > 0);
        }
        """,
        timeout=15000,
    )

    ai_block = page.locator('[class*="aiMessageContent"]').first
    assert "DlightRAG" in ai_block.text_content()


@pytest.mark.e2e
def test_terminal_answer_exposes_minimal_agent_branch_controls(page):
    page.goto("/web/")
    page.locator(".composer-input").fill("Show controls")
    page.click(".composer-send")
    page.get_by_role("button", name="Follow up").last.wait_for(timeout=10000)

    actions = page.locator('[class*="runActions"]').last
    assert actions.get_by_role("button", name="Follow up").is_visible()
    assert actions.get_by_role("button", name="Fork").is_visible()
    assert actions.get_by_role("button", name="Child agents").count() == 0


@pytest.mark.e2e
def test_chat_history_appends_turns(page):
    """Verify that submitting a second query adds another user-message to the DOM."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    # First query
    page.locator(".composer-input").fill("First query")
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    page.wait_for_selector(".app.has-messages", timeout=10000)
    # One answer at a time: wait for terminal answer controls before submitting
    # the next query.
    page.get_by_role("button", name="Follow up").last.wait_for(timeout=10000)

    initial_user_messages = page.locator('[class*="userMessageWrapper"]').count()

    # Second query
    page.locator(".composer-input").fill("Second query")
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")

    # Should have at least one more user message wrapper
    final_user_messages = page.locator('[class*="userMessageWrapper"]').count()
    assert final_user_messages > initial_user_messages


@pytest.mark.e2e
def test_chat_submit_keeps_open_panel_visible(page):
    """Submitting a query should not dismiss a panel the user already opened."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)

    page.click("#files-btn")
    page.wait_for_function("document.querySelector('#panel').classList.contains('open')")

    page.locator(".composer-input").fill("Keep the panel open")
    page.click(".composer-send")

    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    assert page.locator("#panel").evaluate("el => el.classList.contains('open')")
    assert page.locator("body").evaluate("el => el.classList.contains('panel-open')")


@pytest.mark.e2e
def test_reloading_recovers_the_answer_from_the_run_not_the_response(page):
    """The 202 response is not the answer; the reloaded page reads the run."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    page.locator(".composer-input").fill("What is DlightRAG?")
    page.click(".composer-send")
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal'))
        """,
        timeout=15000,
    )

    page.reload()

    # Nothing from the original response survives a reload, so this text can only
    # have come from the run the conversation still links to.
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal'))
        """,
        timeout=15000,
    )
    assert page.locator('[class*="userMessageWrapper"]').count() == 1


@pytest.mark.e2e
def test_a_replayed_submission_never_creates_a_second_turn(page, e2e_base_url):
    """Resending one submission id returns the run it already created."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    page.locator(".composer-input").fill("What is DlightRAG?")
    page.click(".composer-send")
    page.get_by_role("button", name="Follow up").last.wait_for(timeout=15000)

    replay = page.evaluate(
        """
        async () => {
          const history = await (await fetch('/web/api/conversations')).json();
          const conversation = history.items[0].conversation_id;
          const turns = await (
            await fetch(`/web/api/conversations/${conversation}/history`)
          ).json();
          const turn = turns.turns[0];
          const response = await fetch('/web/api/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              query: 'What is DlightRAG?',
              workspaces: ['default'],
              conversation_id: conversation,
              submission_id: turn.submission_id,
            }),
          });
          const descriptor = await response.json();
          const after = await (
            await fetch(`/web/api/conversations/${conversation}/history`)
          ).json();
          return {
            status: response.status,
            runId: descriptor.turn?.answer_run_id,
            originalRunId: turn.answer_run_id,
            turnCount: after.turns.length,
          };
        }
        """
    )

    assert replay["status"] == 202
    assert replay["runId"] == replay["originalRunId"]
    assert replay["turnCount"] == 1


def _frame(sequence: int, event_type: str, data: str) -> str:
    return f"id: {sequence}\nevent: {event_type}\ndata: {data}\n\n"


_PROGRESS = _frame(1, "progress", '{"phase": "planning"}')
_TOKENS = [_frame(2, "token", '"DlightRAG is a "'), _frame(3, "token", '"multimodal RAG system."')]
_DONE = _frame(
    4,
    "done",
    '{"status":"succeeded","presentation":{'
    '"answer_text":"DlightRAG is a multimodal RAG system.",'
    '"parts":[{"type":"markdown","text":"DlightRAG is a multimodal RAG system.",'
    '"html":"<p>DlightRAG is a multimodal RAG system.</p>","artifact":null,'
    '"evidence_image":null,"inline":false}],"sources":[],"evidence_images":[],'
    '"artifacts":[],"artifact_outcome":{"status":"complete","issues":[]}}}',
)
_DONE_RENDERED = _DONE


def _install_event_transport(
    page: Page,
    slices: list[list[str]],
    service: Any = None,
) -> dict[str, Any]:
    """Serve one run's durable event log as a scripted sequence of dropped connections.

    Each entry is the frames one connection delivers before the transport ends
    without a terminal event. Entries beyond the script repeat the last one, so a
    test can express either separated drops that make progress or a barren stall.
    Serving a terminal frame also records the run's terminal state, because a
    ``done`` event only ever reaches a browser after the run committed it.
    """
    state: dict[str, Any] = {"attempts": 0, "cursors": [], "cancelled": 0}

    def handle(route: Route) -> None:
        index = state["attempts"]
        state["attempts"] = index + 1
        state["cursors"].append(route.request.headers.get("last-event-id"))
        frames = slices[index] if index < len(slices) else slices[-1]
        if service is not None and any("event: done" in frame for frame in frames):
            service.finish_run(urlparse(route.request.url).path.split("/")[4])
        route.fulfill(
            status=200,
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
            body="".join(frames),
        )

    def count_cancel(route: Route) -> None:
        if route.request.method == "DELETE":
            state["cancelled"] += 1
        route.continue_()

    page.route("**/web/api/runs/*/events", handle)
    page.route("**/web/api/runs/*", count_cancel)
    return state


def _submit(page: Page, query: str) -> None:
    page.locator(".composer-input").fill(query)
    page.click(".composer-send")
    page.wait_for_function("document.querySelector('.composer-input').value === ''")


@pytest.mark.e2e
def test_navigating_away_from_a_pending_run_detaches_instead_of_blocking(page: Page) -> None:
    """Following a durable run is this tab's business, not the conversation's."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    # Every attempt delivers a new sequence, so the run stays pending and
    # followed for the whole test instead of exhausting the reconnect budget.
    transport = _install_event_transport(
        page,
        [[_frame(index + 1, "progress", '{"phase": "planning"}')] for index in range(200)],
    )

    _submit(page, "What is DlightRAG?")
    page.wait_for_selector(".composer-send.is-stop", timeout=10000)
    conversation_id = page.locator("[aria-current='page']").get_attribute("data-conversation-id")
    assert conversation_id

    # Following must leave the conversation shell usable. New Chat is a route,
    # not an eager database row, and leaving only detaches this tab's reader.
    assert page.locator("#new-conversation-btn").is_enabled()
    page.locator("#new-conversation-btn").click()
    page.wait_for_url("**/web/")
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=5000)
    assert page.locator("[aria-current='page']").count() == 0
    assert page.locator('[class*="userMessageWrapper"]').count() == 0
    assert transport["cancelled"] == 0
    # The run the conversation owns is untouched and still pending.
    status = page.evaluate(
        "id => fetch(`/web/api/conversations/${id}/history`)"
        ".then(r => r.json()).then(h => h.turns[0].status)",
        conversation_id,
    )
    assert status in ("queued", "running")


@pytest.mark.e2e
def test_steering_clears_only_text_and_keeps_pending_attachments(page: Page) -> None:
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    _install_event_transport(
        page,
        [[_frame(index + 1, "progress", '{"phase":"planning"}')] for index in range(200)],
    )
    page.route("**/web/api/answer/*/steer", lambda route: route.fulfill(status=202, json={}))

    _submit(page, "Start research")
    page.wait_for_selector(".composer-send.is-stop", timeout=10000)
    page.locator("#attachment-input").set_input_files(
        {"name": "notes.md", "mimeType": "text/markdown", "buffer": b"notes"}
    )
    page.get_by_role("textbox", name="Message").fill("Use these notes")
    page.get_by_role("button", name="Steer").click()

    page.wait_for_function("document.querySelector('.composer-input').value === ''")
    assert page.locator("#thumbnail-strip").get_by_text("notes.md").is_visible()
    assert page.get_by_text("Use these notes", exact=True).is_visible()


@pytest.mark.e2e
def test_separated_drops_that_make_progress_never_exhaust_the_reconnect_budget(
    page: Page,
    e2e_conversation_service: Any,
) -> None:
    """The budget bounds consecutive barren attempts, not total reconnects."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    script = [[_PROGRESS], *[[token] for token in _TOKENS], [], [], [], [], [], [_DONE]]
    transport = _install_event_transport(page, script, e2e_conversation_service)

    _submit(page, "What is DlightRAG?")

    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal RAG system.'))
        """,
        timeout=20000,
    )
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)
    answer = page.evaluate(
        """() => Array.from(document.querySelectorAll('[data-run-id]'))
            .map(node => node.textContent).join('')"""
    )
    assert answer.count("DlightRAG is a multimodal RAG system.") == 1
    assert transport["attempts"] == len(script)
    # Every reconnect resumed after the last durable sequence it consumed.
    assert transport["cursors"][:4] == [None, "1", "2", "3"]


@pytest.mark.e2e
def test_a_run_still_pending_after_the_budget_offers_an_explicit_reconnect(page: Page) -> None:
    """Exhausting the budget is a recoverable connection error, not a dead spinner."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    transport = _install_event_transport(page, [[]])

    _submit(page, "What is DlightRAG?")

    page.get_by_role("button", name="Reconnect").wait_for(timeout=20000)
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=10000)
    assert transport["cancelled"] == 0


@pytest.mark.e2e
def test_a_second_failed_reconnect_replaces_the_offer_instead_of_stacking_it(
    page: Page,
    e2e_conversation_service: Any,
) -> None:
    """One offer at a time, and a reconnect that succeeds leaves none behind."""
    page.goto("/web/")
    page.wait_for_selector(".composer-input", timeout=10000)
    # Six barren attempts exhaust the budget, so the first twelve strand the run
    # twice; the thirteenth delivers the run's terminal event.
    transport = _install_event_transport(
        page, [*([[]] * 12), [_PROGRESS, _DONE_RENDERED]], e2e_conversation_service
    )

    _submit(page, "What is DlightRAG?")

    reconnect = page.get_by_role("button", name="Reconnect")
    reconnect.wait_for(timeout=20000)
    reconnect.click()
    page.wait_for_selector(".composer-send.is-stop", timeout=10000)
    page.wait_for_selector(".composer-send:not(.is-stop)", timeout=20000)
    assert transport["attempts"] == 12
    expect(reconnect).to_have_count(1)
    expect(
        page.get_by_text("Connection lost while this answer is running.", exact=True)
    ).to_have_count(1)

    reconnect.click()
    page.wait_for_function(
        """
        () => Array.from(document.querySelectorAll('[class*="aiMessageContent"]'))
            .some(node => node.textContent.includes('DlightRAG is a multimodal RAG system.'))
        """,
        timeout=20000,
    )
    assert transport["attempts"] == 13
    expect(page.get_by_role("button", name="Reconnect")).to_have_count(0)
    expect(
        page.get_by_text("Connection lost while this answer is running.", exact=True)
    ).to_have_count(0)
