"""Focused tests for deterministic Gradio handler behavior."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command, Interrupt

from src import ui


class FakeGraph:
    def __init__(self, snapshot, result=None, error=None):
        self.snapshot = snapshot
        self.result = result or {}
        self.error = error
        self.get_state_calls = []
        self.invoke_calls = []

    def get_state(self, config):
        self.get_state_calls.append(config)
        return self.snapshot

    def invoke(self, payload, config):
        self.invoke_calls.append((payload, config))
        if self.error:
            raise self.error
        return self.result


class TestExtractReply:
    def test_extracts_plain_string_content(self):
        result = {"messages": [AIMessage(content="Hello there")]}
        assert ui._extract_reply(result) == "Hello there"

    def test_extracts_standard_text_content_blocks(self):
        result = {
            "messages": [
                AIMessage(content=[{"type": "text", "text": "Block response"}])
            ]
        }
        assert ui._extract_reply(result) == "Block response"

    def test_extracts_output_text_content_blocks(self):
        result = {
            "messages": [
                AIMessage(content=[{"type": "output_text", "text": "API response"}])
            ]
        }
        assert ui._extract_reply(result) == "API response"

    def test_skips_empty_assistant_messages(self):
        result = {"messages": [AIMessage(content="Earlier"), AIMessage(content="  ")]}
        assert ui._extract_reply(result) == "Earlier"

    def test_returns_fallback_without_assistant_text(self):
        assert "wasn't able to generate" in ui._extract_reply({"messages": []})


def test_add_user_message_trims_input_without_mutating_history():
    history = [{"role": "assistant", "content": "Welcome"}]
    textbox, updated = ui._add_user_message("  hello  ", history)

    assert textbox == ""
    assert updated[-1] == {"role": "user", "content": "hello"}
    assert history == [{"role": "assistant", "content": "Welcome"}]


def test_add_user_message_rejects_oversized_input_without_invoking_graph():
    history = [{"role": "assistant", "content": "Welcome"}]

    _, updated = ui._add_user_message("x" * (ui.MAX_USER_MESSAGE_LENGTH + 1), history)

    assert updated[-1]["role"] == "assistant"
    assert "too long" in updated[-1]["content"]


def test_pending_node_without_interrupt_starts_a_new_turn(monkeypatch):
    snapshot = SimpleNamespace(next=("recoverable_node",), interrupts=())
    graph = FakeGraph(snapshot, {"messages": [AIMessage(content="Done")]})
    times = iter([10.0, 12.5])
    monkeypatch.setattr(ui.time, "perf_counter", lambda: next(times))
    history = [{"role": "user", "content": "hello"}]

    updates = list(ui._respond(graph, history, "thread-1"))

    payload, config = graph.invoke_calls[0]
    assert isinstance(payload, dict)
    assert payload == {"user_input": "hello"}
    assert config == {"configurable": {"thread_id": "thread-1"}}
    assert "Processing" in updates[0][2]
    assert "Done in 2.5s" in updates[-1][2]
    assert updates[-1][0][-1] == {"role": "assistant", "content": "Done"}


def test_actual_interrupt_resumes_with_command():
    snapshot = SimpleNamespace(next=(), interrupts=(Interrupt("Verify"),))
    graph = FakeGraph(snapshot, {"messages": [AIMessage(content="Verified")]})
    history = [{"role": "user", "content": "5"}]

    list(ui._respond(graph, history, "thread-2"))

    payload, _ = graph.invoke_calls[0]
    assert isinstance(payload, Command)
    assert payload.resume == "5"


def test_new_interrupt_is_shown_as_waiting():
    snapshot = SimpleNamespace(next=(), interrupts=())
    graph = FakeGraph(snapshot, {"__interrupt__": [Interrupt("Customer ID?")]})
    history = [{"role": "user", "content": "show invoices"}]

    updates = list(ui._respond(graph, history, "thread-3"))

    assert updates[-1][0][-1]["content"] == "Customer ID?"
    assert "Waiting for your input" in updates[-1][2]


def test_graph_error_returns_safe_message_and_status():
    snapshot = SimpleNamespace(next=(), interrupts=())
    graph = FakeGraph(snapshot, error=RuntimeError("sensitive detail"))
    history = [{"role": "user", "content": "hello"}]

    updates = list(ui._respond(graph, history, "thread-4"))

    assert "something went wrong" in updates[-1][0][-1]["content"]
    assert "sensitive detail" not in updates[-1][0][-1]["content"]
    assert "Error: RuntimeError" in updates[-1][2]


def test_new_conversation_clears_textbox_and_rotates_thread(monkeypatch):
    monkeypatch.setattr(ui.uuid, "uuid4", lambda: "new-thread")

    textbox, history, thread_id, status = ui._new_conversation()

    assert textbox == ""
    assert history == [{"role": "assistant", "content": ui.WELCOME}]
    assert thread_id == "new-thread"
    assert "Ready" in status


def test_new_conversation_deletes_previous_checkpoint(monkeypatch):
    deleted = []
    graph = SimpleNamespace(
        checkpointer=SimpleNamespace(
            delete_thread=lambda thread_id: deleted.append(thread_id)
        )
    )
    monkeypatch.setattr(ui.uuid, "uuid4", lambda: "new-thread")

    ui._new_conversation(graph, "old-thread")

    assert deleted == ["old-thread"]


@pytest.mark.parametrize(
    "history", [None, [], [{"role": "assistant", "content": "Hi"}]]
)
def test_respond_ignores_history_without_a_new_user_message(history):
    graph = FakeGraph(SimpleNamespace(next=(), interrupts=()))

    updates = list(ui._respond(graph, history, "thread-5"))

    assert len(updates) == 1
    assert "Ready" in updates[0][2]
    assert graph.invoke_calls == []
