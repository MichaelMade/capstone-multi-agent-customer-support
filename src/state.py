"""Shared LangGraph state schema.

Every node in the graph reads from and writes to this TypedDict. The
``messages`` field uses the ``add_messages`` reducer so node updates
append rather than overwrite. ``remaining_steps`` is a managed channel
that gives ReAct loops an execution budget, preventing infinite loops.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class InputState(TypedDict):
    """Untrusted input accepted by the compiled customer-support graph.

    A plain string is converted to a HumanMessage by the first graph node.
    This prevents callers from injecting internal fields or privileged message
    roles such as system/tool messages at the graph boundary.
    """

    user_input: str


class State(TypedDict, total=False):
    """Internal state shared by verification, memory, supervisor, and agents."""

    customer_id: str | None
    needs_verification: bool
    messages: Annotated[list[AnyMessage], add_messages]
    loaded_memory: str
    verification_message: str
    pending_memory_text: str
    user_input: str
    remaining_steps: RemainingSteps
