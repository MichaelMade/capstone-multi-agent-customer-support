"""Invoice sub-agent built with LangChain's prebuilt agent helper.

The prompt callable injects the VERIFIED customer_id from graph state —
never parsed from user free text — satisfying the security requirement
that downstream agents only ever see the verified identity.
"""

from __future__ import annotations

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
    hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from typing_extensions import override

from src.agents.prompts import INVOICE_AGENT_PROMPT
from src.tools import INVOICE_TOOLS

INVOICE_MODEL_CALL_LIMIT = 6
INVOICE_STEP_LIMIT_MESSAGE = (
    "I couldn't complete that account lookup within the available steps. "
    "Please try a more specific invoice question."
)


class InvoiceAgentState(AgentState, total=False):
    """State channels required by the invoice agent."""

    customer_id: str | None


class _FriendlyModelCallLimitMiddleware(ModelCallLimitMiddleware):
    """Use a customer-facing message when the agent reaches its run limit."""

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update:
            return {
                **update,
                "messages": [
                    AIMessage(
                        content=INVOICE_STEP_LIMIT_MESSAGE,
                        name="invoice_agent",
                    )
                ],
            }
        return None


def _invoice_system_prompt(customer_id: str | None) -> str:
    """Build the invoice prompt from trusted, verified graph state.

    The customer_id comes from graph state, which only the verification
    node writes. The prompt carries it to the LLM in the exact
    "customer_id=<N>" format the agent prompt instructs it to trust.
    """
    identity = (
        f"VERIFIED IDENTITY: customer_id={customer_id}"
        if customer_id
        else "VERIFIED IDENTITY: none — the customer is NOT verified."
    )
    return f"{INVOICE_AGENT_PROMPT}\n\n{identity}"


@dynamic_prompt
def _invoice_prompt(request: ModelRequest) -> str:
    """Inject the state-bound identity into each invoice model call."""
    return _invoice_system_prompt(request.state.get("customer_id"))


def build_invoice_agent(llm: BaseChatModel):
    """Compile the invoice agent."""
    return create_agent(
        llm,
        tools=INVOICE_TOOLS,
        middleware=[
            _invoice_prompt,
            _FriendlyModelCallLimitMiddleware(
                run_limit=INVOICE_MODEL_CALL_LIMIT,
                exit_behavior="end",
            ),
        ],
        state_schema=InvoiceAgentState,
        name="invoice_agent",
    )
