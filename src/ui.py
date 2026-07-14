"""Gradio chat interface.

Responsibilities:
- Session isolation: every browser session gets its own UUID thread_id
  (held in gr.State), scoping the LangGraph checkpointer per session.
- Human-in-the-loop: after each run the graph snapshot is inspected; if
  the graph is paused at an interrupt, the UI shows the question, flags
  "waiting for input", and resumes with ``Command(resume=...)`` on the
  next user message.
- Status feedback: processing / success (with response time) / waiting /
  error states rendered above the chat.
- Input safety: oversized messages are rejected before model invocation.
- The user's message is appended to the chat immediately, before the
  LLM responds.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

# Keep the local customer-support demo from emitting Gradio usage telemetry by
# default. Deployments can explicitly opt in with GRADIO_ANALYTICS_ENABLED=True.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from langchain_core.messages import AIMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

MAX_USER_MESSAGE_LENGTH = 4_000

WELCOME = (
    "👋 Welcome to the Digital Music Store! I can help you browse our "
    "catalog (songs, albums, artists, genres) and, once you verify your "
    "identity, look up your invoices and purchases. How can I help?"
)


def _status(kind: str, detail: str = "") -> str:
    palette = {
        "idle": ("#6b7280", "Ready"),
        "processing": ("#2563eb", "Processing…"),
        "success": ("#16a34a", f"Done{detail}"),
        "waiting": ("#d97706", "Waiting for your input"),
        "error": ("#dc2626", f"Error{detail}"),
    }
    color, label = palette[kind]
    return (
        f"<div style='padding:6px 12px;border-radius:6px;display:inline-block;"
        f"background:{color}1a;color:{color};font-weight:600;font-size:0.9em'>"
        f"● {label}</div>"
    )


def _extract_reply(result: dict) -> str:
    """Pull the final assistant text out of a graph result."""
    for message in reversed(result.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue

        # ``AIMessage.content`` may be a string or a list of content blocks,
        # depending on the provider and LangChain output mode. ``text``
        # normalizes standard text blocks; the fallback also handles the
        # ``output_text`` shape returned by some OpenAI-compatible APIs.
        text = message.text
        if not text and isinstance(message.content, list):
            text = "".join(
                block.get("text", "")
                for block in message.content
                if isinstance(block, dict)
                and block.get("type") in {"text", "output_text"}
                and isinstance(block.get("text"), str)
            )
        if text.strip():
            return text
    return "I'm sorry — I wasn't able to generate a response. Please try again."


def _add_user_message(user_text: str, history: list) -> tuple[str, list]:
    history = history or []
    text = (user_text or "").strip()
    if text:
        if len(text) > MAX_USER_MESSAGE_LENGTH:
            history = history + [
                {
                    "role": "assistant",
                    "content": (
                        "That message is too long. Please shorten it to "
                        f"{MAX_USER_MESSAGE_LENGTH:,} characters or fewer."
                    ),
                }
            ]
        else:
            history = history + [{"role": "user", "content": text}]
    return "", history


def _has_pending_interrupt(snapshot: Any) -> bool:
    """Return whether a checkpoint is paused at a LangGraph interrupt."""
    return bool(getattr(snapshot, "interrupts", ()))


def _respond(
    graph: Any, history: list, thread_id: str | None
) -> Iterator[tuple[list, str | None, str]]:
    history = history or []
    if not history or history[-1]["role"] != "user":
        yield history, thread_id, _status("idle")
        return

    if not thread_id:
        thread_id = str(uuid.uuid4())
        logger.info("New session thread: %s", thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    user_text = history[-1]["content"]

    yield history, thread_id, _status("processing")

    start = time.perf_counter()
    try:
        snapshot = graph.get_state(config)
        if _has_pending_interrupt(snapshot):
            # Resume only a real LangGraph interrupt. A non-empty ``next``
            # can also represent recoverable work after an unrelated error.
            result = graph.invoke(Command(resume=user_text), config)
        else:
            result = graph.invoke({"user_input": user_text}, config)
        elapsed = time.perf_counter() - start

        interrupts = result.get("__interrupt__")
        if interrupts:
            # Paused again: surface the interrupt question to the user.
            reply = str(interrupts[0].value)
            status = _status("waiting")
        else:
            reply = _extract_reply(result)
            status = _status("success", f" in {elapsed:.1f}s")
    except Exception as exc:
        logger.exception("Graph invocation failed")
        reply = (
            "I'm sorry, something went wrong while processing your "
            "request. Please try again."
        )
        status = _status("error", f": {type(exc).__name__}")

    history = history + [{"role": "assistant", "content": reply}]
    yield history, thread_id, status


def _new_conversation(
    graph: Any | None = None, previous_thread_id: str | None = None
) -> tuple[str, list, str, str]:
    checkpointer = getattr(graph, "checkpointer", None)
    if previous_thread_id and hasattr(checkpointer, "delete_thread"):
        try:
            checkpointer.delete_thread(previous_thread_id)
        except Exception:  # pragma: no cover - defensive cleanup
            logger.exception("Could not delete the previous conversation checkpoint")
    thread_id = str(uuid.uuid4())
    logger.info("Conversation reset; new thread: %s", thread_id)
    return (
        "",
        [{"role": "assistant", "content": WELCOME}],
        thread_id,
        _status("idle"),
    )


def build_ui(graph) -> gr.Blocks:
    """Build the Gradio Blocks app around a compiled LangGraph graph."""

    def respond(history: list, thread_id: str | None):
        yield from _respond(graph, history, thread_id)

    def new_conversation(thread_id: str | None):
        return _new_conversation(graph, thread_id)

    with gr.Blocks(title="Digital Music Store — Customer Support") as demo:
        gr.Markdown(
            "# 🎵 Digital Music Store — Customer Support\n"
            "Multi-agent AI support: browse the catalog, check your "
            "invoices, and get personalized recommendations. Verify with "
            "your Customer ID, email, or phone to access account data."
        )
        status = gr.HTML(_status("idle"))
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME}],
            label="Support Chat",
            height=500,
        )
        with gr.Row():
            msg = gr.Textbox(
                placeholder=(
                    "Ask about music, your invoices, or verify your identity…"
                ),
                show_label=False,
                scale=5,
                autofocus=True,
                max_length=MAX_USER_MESSAGE_LENGTH,
            )
            send = gr.Button("Send", variant="primary", scale=1)
        reset = gr.Button("🔄 New Conversation")
        thread_state = gr.State(None)

        send.click(_add_user_message, [msg, chatbot], [msg, chatbot]).then(
            respond, [chatbot, thread_state], [chatbot, thread_state, status]
        )
        msg.submit(_add_user_message, [msg, chatbot], [msg, chatbot]).then(
            respond, [chatbot, thread_state], [chatbot, thread_state, status]
        )
        reset.click(
            new_conversation,
            [thread_state],
            [msg, chatbot, thread_state, status],
        )

    return demo
