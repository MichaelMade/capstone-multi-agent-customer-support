"""Full multi-agent graph assembly.

Topology::

    START -> ingest_user_input -> verify_info --(verified)--> load_memory
                                      ^   \\--(unverified)--> human_input
                                      |______________________________|
    load_memory -> supervisor -> create_memory -> END

The supervisor is a langgraph-supervisor sub-graph coordinating the
hand-built music catalog ReAct agent and the prebuilt invoice ReAct
agent. Checkpointing (MemorySaver) gives per-thread conversation
history; InMemoryStore holds cross-thread preference memory.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor

from src.agents.invoice_agent import build_invoice_agent
from src.agents.memory import create_memory, load_memory
from src.agents.music_agent import build_music_agent
from src.agents.prompts import SUPERVISOR_PROMPT
from src.agents.verification import (
    human_input,
    route_after_verification,
    verify_info,
)
from src.llm import get_llm
from src.state import InputState, State

logger = logging.getLogger(__name__)


class _IgnoreManagedChannelWarning(logging.Filter):
    """Silence langgraph's cosmetic warning about the supervisor sub-graph
    echoing the managed ``remaining_steps`` channel back to the parent."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "unknown channel remaining_steps" not in record.getMessage()


logging.getLogger("langgraph").addFilter(_IgnoreManagedChannelWarning())


def ingest_user_input(state: State) -> dict:
    """Convert the graph's untrusted string input into a human message."""
    user_input = state.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-blank string")
    return {
        "user_input": "",
        "messages": [HumanMessage(content=user_input.strip())],
    }


def build_graph(checkpointer=None, store=None):
    """Compile the complete customer-support graph."""
    llm = get_llm()

    music_agent = build_music_agent(llm)
    invoice_agent = build_invoice_agent(llm)

    supervisor = create_supervisor(
        [invoice_agent, music_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        state_schema=State,
        output_mode="last_message",
    ).compile(name="supervisor")

    # Only plain user text is accepted from callers. Internal state fields and
    # privileged message roles are not injectable at the graph boundary.
    graph = StateGraph(State, input_schema=InputState)
    graph.add_node("ingest_user_input", ingest_user_input)
    graph.add_node("verify_info", verify_info)
    graph.add_node("human_input", human_input)
    graph.add_node("load_memory", load_memory)
    graph.add_node("supervisor", supervisor)
    graph.add_node("create_memory", create_memory)

    graph.add_edge(START, "ingest_user_input")
    graph.add_edge("ingest_user_input", "verify_info")
    graph.add_conditional_edges(
        "verify_info",
        route_after_verification,
        {"continue": "load_memory", "unverified": "human_input"},
    )
    graph.add_edge("human_input", "verify_info")
    graph.add_edge("load_memory", "supervisor")
    graph.add_edge("supervisor", "create_memory")
    graph.add_edge("create_memory", END)

    compiled = graph.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
        store=store if store is not None else InMemoryStore(),
    )
    logger.info("Multi-agent graph compiled")
    return compiled
