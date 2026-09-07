# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""E2E tests for workspace creation, selection, and identity-preserving reset."""

import pytest
from playwright.sync_api import expect


@pytest.mark.e2e
def test_workspace_popover_opens(page):
    """Click the workspace selector → popover appears with workspace items."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    selector = page.locator("#workspace-selector")
    selector.click()

    # Popover should be visible with at least one workspace item
    page.wait_for_selector(".dl-popover--workspace", timeout=5000)
    popover = page.locator(".dl-popover--workspace")
    assert popover.is_visible()

    # "Default" workspace should be listed
    items = popover.locator(".dl-popover-item")
    assert items.count() >= 1


@pytest.mark.e2e
def test_workspace_popover_closes(page):
    """Click outside the popover or press Escape → popover closes."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.wait_for_selector(".dl-popover--workspace", timeout=5000)
    assert page.locator(".dl-popover--workspace").is_visible()

    # Click outside — the popover should go away
    page.locator(".app").click(position={"x": 10, "y": 10})
    expect(page.locator(".dl-popover--workspace")).to_be_hidden()


@pytest.mark.e2e
def test_scope_toggle_keeps_a_half_typed_workspace_name(page):
    """Changing the search scope must not discard the name being typed."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    popover = page.get_by_role("dialog", name="Workspaces")
    popover.wait_for()
    create_input = popover.get_by_label("New workspace name")
    create_input.fill("half typed name")

    page.locator('[data-workspace-all="true"]').click()
    page.wait_for_timeout(300)

    assert create_input.input_value() == "half typed name"


@pytest.mark.e2e
def test_workspace_popover_is_arrow_navigable(page):
    """Arrow and Home/End keys must move focus between scope options."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.wait_for_selector(".dl-popover--workspace", timeout=5000)

    focused_index = """() => {
        const choices = [...document.querySelectorAll('[data-workspace-choice]')];
        return choices.indexOf(document.activeElement);
    }"""
    choices = page.locator("[data-workspace-choice]")
    last = choices.count() - 1
    choices.first.focus()

    page.keyboard.press("ArrowDown")
    assert page.evaluate(focused_index) == 1

    page.keyboard.press("End")
    assert page.evaluate(focused_index) == last

    page.keyboard.press("Home")
    assert page.evaluate(focused_index) == 0


@pytest.mark.e2e
def test_workspace_controls_inherit_application_typography(page):
    """Native workspace triggers and choices inherit the application font."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)
    app_font = page.locator("body").evaluate("element => getComputedStyle(element).fontFamily")

    scope_trigger = page.get_by_role("button", name="Choose search workspaces")
    assert scope_trigger.evaluate("element => getComputedStyle(element).fontFamily") == app_font
    scope_trigger.click()
    workspace_dialog = page.get_by_role("dialog", name="Workspaces")
    workspace_choice = workspace_dialog.get_by_role("button", name="All workspaces")
    assert workspace_choice.evaluate("element => getComputedStyle(element).fontFamily") == app_font
    page.keyboard.press("Escape")
    expect(workspace_dialog).to_be_hidden()

    page.get_by_role("button", name="Files", exact=True).click()
    page.wait_for_selector("#upload-zone", timeout=10000)
    ingest_trigger = page.get_by_role("button", name="Files in Default; choose file workspace")
    assert ingest_trigger.evaluate("element => getComputedStyle(element).fontFamily") == app_font
    ingest_trigger.click()
    ingest_choice = page.get_by_role("dialog", name="Select ingest workspace").get_by_role(
        "button", name="Default", exact=True
    )
    assert ingest_choice.evaluate("element => getComputedStyle(element).fontFamily") == app_font


@pytest.mark.e2e
def test_workspace_create_input_visible(page):
    """Verify the new-workspace input row exists inside the popover."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.wait_for_selector(".dl-popover--workspace", timeout=5000)

    # The create-row with input and button should be present
    popover = page.get_by_role("dialog", name="Workspaces")
    create_row = popover.locator("dl-workspace-create")
    assert create_row.is_visible()
    assert create_row.get_by_label("New workspace name").is_visible()
    assert create_row.get_by_label("Create workspace").is_visible()


@pytest.mark.e2e
def test_workspace_selector_labels_all_when_scope_covers_every_workspace(page):
    """Selecting every workspace individually promotes the topbar scope to All."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Research").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Research").click()

    assert page.locator("#workspace-label").text_content() == "All workspaces (2)"
    assert (
        page.locator(
            ".dl-popover--workspace .dl-popover-item", has_text="All workspaces"
        ).get_attribute("aria-pressed")
        == "true"
    )


@pytest.mark.e2e
def test_workspace_selector_all_sets_default_primary(page):
    """Selecting All is explicit and resets single-workspace surfaces to default."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Research").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Research").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="All workspaces").click()
    page.locator("#files-btn").click()

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    assert page.locator("#workspace-label").text_content() == "All workspaces (2)"
    assert page.locator("[data-ingest-name]").text_content() == "Default"


@pytest.mark.e2e
def test_workspace_selector_auto_all_keeps_last_explicit_primary(page):
    """An all-workspace query scope still keeps the last explicit workspace primary."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Research").click()
    page.locator(".dl-popover--workspace .dl-popover-item", has_text="Research").click()
    page.locator("#files-btn").click()

    page.wait_for_selector("#panel-content #upload-zone", timeout=10000)
    assert page.locator("#workspace-label").text_content() == "All workspaces (2)"
    assert page.locator("[data-ingest-name]").text_content() == "Research"


@pytest.mark.e2e
def test_workspace_corpus_reset_server_round_trip(page):
    """Reset corpus through the real Web route without deleting Workspace identity."""
    page.goto("/web/")
    page.wait_for_selector("#workspace-selector", timeout=10000)

    page.locator("#workspace-selector").click()
    page.get_by_label("Reset Corpus Research").click()
    dialog = page.get_by_role("dialog", name="Reset Corpus")
    dialog.get_by_label("Type Research to confirm").fill("Research")
    with page.expect_response("**/web/api/workspaces/reset") as reset_response:
        dialog.get_by_role("button", name="Reset Corpus", exact=True).click()
    assert reset_response.value.ok

    expect(dialog).to_be_hidden()
    page.locator("#workspace-selector").click()
    expect(page.get_by_label("Reset Corpus Research")).to_have_count(1)
