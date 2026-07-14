"""Music catalog sub-agent: a hand-built ReAct loop.

Demonstrates the ReAct pattern explicitly (per the capstone spec) rather
than using the prebuilt helper: an assistant node whose LLM has the five
music tools bound, a ToolNode that executes tool calls, and a
conditional edge that loops until the LLM stops calling tools.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.prompts import MUSIC_AGENT_PROMPT
from src.state import State
from src.tools import MUSIC_TOOLS

logger = logging.getLogger(__name__)

STEP_LIMIT_MESSAGE = (
    "I couldn't complete that catalog lookup within the available steps. "
    "Please try a more specific music question."
)


def build_music_agent(llm: BaseChatModel):
    """Compile the hand-built ReAct music catalog agent sub-graph."""
    llm_with_tools = llm.bind_tools(MUSIC_TOOLS)

    def music_assistant(state: State) -> dict:
        """LLM node: grounded system prompt + saved preferences + history."""
        memory = state.get("loaded_memory") or "None saved yet."
        system_prompt = MUSIC_AGENT_PROMPT.format(memory=memory)
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + list(state["messages"])
        )
        # Match the prebuilt ReAct agent's graceful step-budget behavior.  If
        # the model requests another tool when there is no room to execute it
        # and make a final response, replace the unresolved tool call with a
        # normal assistant message instead of raising GraphRecursionError.
        remaining_steps = state.get("remaining_steps")
        if (
            isinstance(response, AIMessage)
            and response.tool_calls
            and remaining_steps is not None
            and remaining_steps < 2
        ):
            logger.warning("Music agent exhausted its remaining step budget")
            response = AIMessage(
                id=response.id,
                content=STEP_LIMIT_MESSAGE,
                name="music_catalog_agent",
            )
        elif isinstance(response, AIMessage):
            response.name = "music_catalog_agent"
        return {"messages": [response]}

    def should_continue(state: State) -> str:
        """Loop to the tool node while the LLM keeps requesting tools."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(State)
    graph.add_node("music_assistant", music_assistant)
    graph.add_node("tools", ToolNode(MUSIC_TOOLS))
    graph.add_edge(START, "music_assistant")
    graph.add_conditional_edges(
        "music_assistant",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "music_assistant")
    return graph.compile(name="music_catalog_agent")
