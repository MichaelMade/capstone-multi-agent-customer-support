"""Deterministic contract tests for the capstone's highest-value rubric items."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.agents.memory as memory_module
import src.agents.verification as verification_module
import src.graph as graph_module
from src.agents.invoice_agent import build_invoice_agent
from src.agents.music_agent import build_music_agent
from src.agents.prompts import (
    GROUNDING_RULES,
    INVOICE_AGENT_PROMPT,
    MUSIC_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)
from src.models import PreferenceExtraction


def _normalized(text: str) -> str:
    return " ".join(text.split())


GROUNDING_RULE_CONTRACTS = (
    "Only provide catalog or account facts returned by your tools. "
    "Never infer, assume, or make up store data.",
    "Quote exact numbers, names, and amounts from tool results. "
    "Never round, estimate, or approximate.",
    "If a tool returns empty results or a not-found message, tell the customer "
    "exactly that. Never fabricate data to fill a gap.",
    "If a query falls outside your scope, say so explicitly and defer. "
    "Do not attempt to answer it.",
    "Never answer factual questions from your training data. Always call a tool first.",
    "If tool results are sampled or truncated, mention the total count and "
    "that more data exists.",
)


class _StructuredLLM:
    def __init__(self, result: PreferenceExtraction):
        self.result = result
        self.invocations = []

    def with_structured_output(self, schema):
        assert schema is PreferenceExtraction
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        return self.result


class _RecordingStore:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.put_calls = []

    def get(self, namespace, key):
        value = self.data.get((namespace, key))
        return SimpleNamespace(value=value) if value is not None else None

    def put(self, namespace, key, value):
        self.put_calls.append((namespace, key, value))
        self.data[(namespace, key)] = value


class _ConcurrentStructuredLLM:
    def __init__(self):
        self.barrier = threading.Barrier(2)

    def with_structured_output(self, schema):
        assert schema is PreferenceExtraction
        return self

    def invoke(self, messages):
        # Ensure both turns read the initially empty profile before either
        # finishes extraction and enters the serialized merge/write section.
        self.barrier.wait(timeout=5)
        customer_text = str(messages[-1].content).casefold()
        preference = "rock" if "rock" in customer_text else "jazz"
        return PreferenceExtraction(music_preferences=[preference])


def _saved_profile(customer_id: str, preferences: list[str]):
    return {
        (("memory_profile", customer_id), "user_memory"): {
            "customer_id": customer_id,
            "music_preferences": preferences,
        }
    }


def test_grounding_rules_define_all_six_required_safeguards():
    normalized_rules = _normalized(GROUNDING_RULES)

    for required_rule in GROUNDING_RULE_CONTRACTS:
        assert _normalized(required_rule) in normalized_rules


@pytest.mark.parametrize(
    "prompt",
    [MUSIC_AGENT_PROMPT, INVOICE_AGENT_PROMPT, SUPERVISOR_PROMPT],
    ids=["music", "invoice", "supervisor"],
)
def test_every_agent_prompt_includes_all_grounding_rules(prompt):
    assert _normalized(GROUNDING_RULES) in _normalized(prompt)


def test_supervisor_prompt_preserves_all_four_routing_contracts():
    prompt = _normalized(SUPERVISOR_PROMPT)

    assert "Music catalog questions -> music_catalog_agent" in prompt
    assert (
        "Invoice / purchase / billing / support-rep questions -> invoice_agent"
        in prompt
    )
    assert "call invoice_agent first, then music_catalog_agent" in prompt
    assert "OFF-TOPIC queries" in prompt
    assert "do NOT call any agent" in prompt


def test_specialist_agents_preserve_required_react_topologies():
    llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")

    music_graph = build_music_agent(llm).get_graph()
    music_edges = {
        (edge.source, edge.target, edge.conditional) for edge in music_graph.edges
    }
    assert set(music_graph.nodes) == {
        "__start__",
        "music_assistant",
        "tools",
        "__end__",
    }
    assert music_edges == {
        ("__start__", "music_assistant", False),
        ("music_assistant", "tools", True),
        ("music_assistant", "__end__", True),
        ("tools", "music_assistant", False),
    }

    invoice_graph = build_invoice_agent(llm).get_graph()
    assert {"model", "tools"} <= set(invoice_graph.nodes)
    assert any(
        edge.source == "model" and "after_model" in edge.target
        for edge in invoice_graph.edges
    )
    assert any(edge.source == "tools" for edge in invoice_graph.edges)


def test_production_graph_preserves_required_sequence(monkeypatch):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
    monkeypatch.setattr(graph_module, "get_llm", lambda: llm)

    graph = graph_module.build_graph().get_graph()
    actual_nodes = set(graph.nodes)
    actual_edges = {
        (edge.source, edge.target, edge.conditional) for edge in graph.edges
    }

    assert actual_nodes == {
        "__start__",
        "ingest_user_input",
        "verify_info",
        "human_input",
        "load_memory",
        "supervisor",
        "create_memory",
        "__end__",
    }
    assert actual_edges == {
        ("__start__", "ingest_user_input", False),
        ("ingest_user_input", "verify_info", False),
        ("verify_info", "load_memory", True),
        ("verify_info", "human_input", True),
        ("human_input", "verify_info", False),
        ("load_memory", "supervisor", False),
        ("supervisor", "create_memory", False),
        ("create_memory", "__end__", False),
    }


def test_memory_uses_exact_namespace_and_key_contract():
    assert memory_module._namespace("42") == ("memory_profile", "42")
    assert memory_module.MEMORY_KEY == "user_memory"


def test_verification_injects_customer_id_in_system_message(monkeypatch):
    customer = {
        "CustomerId": 5,
        "FirstName": "Test",
        "LastName": "Customer",
    }
    monkeypatch.setattr(
        verification_module, "_lookup_customer", lambda identifier: customer
    )

    update = verification_module.verify_info({"messages": [HumanMessage(content="5")]})

    assert update["customer_id"] == "5"
    assert update["needs_verification"] is False
    assert len(update["messages"]) == 1
    message = update["messages"][0]
    assert isinstance(message, SystemMessage)
    assert "Identity verified. customer_id=5 " in str(message.content)


def test_create_memory_merges_existing_and_new_preferences(monkeypatch):
    store = _RecordingStore(_saved_profile("7", ["Rock"]))
    llm = _StructuredLLM(PreferenceExtraction(music_preferences=["jazz"]))
    monkeypatch.setattr(memory_module, "get_store", lambda: store)
    monkeypatch.setattr(memory_module, "get_llm", lambda: llm)

    update = memory_module.create_memory(
        {
            "customer_id": "7",
            "messages": [HumanMessage(content="I love jazz")],
        }
    )

    assert update == {"pending_memory_text": ""}
    assert len(store.put_calls) == 1
    namespace, key, saved = store.put_calls[0]
    assert namespace == ("memory_profile", "7")
    assert key == "user_memory"
    assert saved == {
        "customer_id": "7",
        "music_preferences": ["Rock", "Jazz"],
    }


def test_create_memory_question_short_circuits_without_write(monkeypatch):
    initial = _saved_profile("7", ["Rock"])
    store = _RecordingStore(initial)
    monkeypatch.setattr(memory_module, "get_store", lambda: store)
    monkeypatch.setattr(
        memory_module,
        "get_llm",
        lambda: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    update = memory_module.create_memory(
        {
            "customer_id": "7",
            "messages": [HumanMessage(content="Do you have jazz music?")],
        }
    )

    assert update == {"pending_memory_text": ""}
    assert store.put_calls == []
    assert store.data == initial


def test_empty_extraction_preserves_existing_profile_without_write(monkeypatch):
    initial = _saved_profile("7", ["Rock"])
    store = _RecordingStore(initial)
    llm = _StructuredLLM(PreferenceExtraction(music_preferences=[]))
    monkeypatch.setattr(memory_module, "get_store", lambda: store)
    monkeypatch.setattr(memory_module, "get_llm", lambda: llm)

    update = memory_module.create_memory(
        {
            "customer_id": "7",
            "messages": [HumanMessage(content="I love jazz")],
        }
    )

    assert len(llm.invocations) == 1
    assert update == {"pending_memory_text": ""}
    assert store.put_calls == []
    assert store.data == initial


def test_concurrent_memory_writes_merge_without_losing_a_preference(monkeypatch):
    store = _RecordingStore()
    llm = _ConcurrentStructuredLLM()
    monkeypatch.setattr(memory_module, "get_store", lambda: store)
    monkeypatch.setattr(memory_module, "get_llm", lambda: llm)

    states = [
        {"customer_id": "7", "messages": [HumanMessage(content="I love rock")]},
        {"customer_id": "7", "messages": [HumanMessage(content="I love jazz")]},
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        updates = list(pool.map(memory_module.create_memory, states))

    saved = store.data[(("memory_profile", "7"), "user_memory")]
    assert updates == [{"pending_memory_text": ""}] * 2
    assert saved["customer_id"] == "7"
    assert set(saved["music_preferences"]) == {"Rock", "Jazz"}
