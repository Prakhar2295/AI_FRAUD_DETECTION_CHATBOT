"""LangGraph workflow builder for fraud intelligence orchestration."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config.settings import Settings
from app.graph.nodes.behavioral_node import create_behavioral_node
from app.graph.nodes.fraud_node import create_fraud_node
from app.graph.nodes.intent_node import create_intent_node
from app.graph.nodes.memory_node import create_memory_node
from app.graph.nodes.risk_node import create_risk_node
from app.graph.nodes.response_node import create_response_node
from app.graph.state import FraudWorkflowState
from app.services.llm_service import OllamaLLMService
from app.services.memory_service import MemoryService


def build_fraud_workflow(
    llm_service: OllamaLLMService,
    memory_service: MemoryService,
    settings: Settings,
):
    """Build a deterministic LangGraph fraud workflow."""
    workflow = StateGraph(FraudWorkflowState)

    workflow.add_node("intent", create_intent_node(llm_service))
    workflow.add_node("behavioral", create_behavioral_node(llm_service))
    workflow.add_node("fraud", create_fraud_node(llm_service))
    workflow.add_node("risk", create_risk_node(settings))
    workflow.add_node("response", create_response_node(llm_service))
    workflow.add_node("memory", create_memory_node(memory_service))

    workflow.set_entry_point("intent")
    workflow.add_edge("intent", "behavioral")
    workflow.add_edge("behavioral", "fraud")
    workflow.add_edge("fraud", "risk")
    workflow.add_edge("risk", "response")
    workflow.add_edge("response", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()
