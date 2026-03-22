"""
Multi-Agent Customer Operations System
Stack: LangGraph + Claude Haiku (claude-haiku-4-5-20251001)

KNOWN BUGS (intentional, for debugging exercises):
  Bug 1 (routing)  — queries containing "where" route to RefundAgent 30% of the time
  Bug 2 (memory)   — session memory is global; previous customer's context bleeds in
  Bug 3 (loop)     — OrderAgent retries LLM calls up to 5x on ambiguous queries
  Bug 4 (silent)   — ValidatorAgent always returns True regardless of response quality
"""

import random
import uuid
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class CustomerOpsState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    current_agent: str                        # specialist chosen by planner
    routing_decision: dict                    # {"agent": str, "confidence": float}
    tool_calls_made: list[str]                # log of every tool invoked this session
    session_id: str                           # unique ID per conversation
    memory: dict                              # cross-turn context


# ---------------------------------------------------------------------------
# MOCK TOOLS
# ---------------------------------------------------------------------------

@tool
def check_order_status(order_id: str) -> dict:
    """Return the current status of an order."""
    catalog: dict[str, dict] = {
        "ORD-001": {"status": "delivered",   "item": "Laptop Pro 15",   "delivered_on": "2024-01-15"},
        "ORD-002": {"status": "in_transit",  "item": "Smartphone X",    "estimated_delivery": "2024-01-22"},
        "ORD-003": {"status": "processing",  "item": "Wireless Headphones", "order_date": "2024-01-18"},
    }
    return catalog.get(order_id, {"status": "not_found", "order_id": order_id})


@tool
def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for a given order."""
    return {
        "refund_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
        "order_id": order_id,
        "status": "approved",
        "amount": "$149.99",
        "processing_days": "3-5 business days",
        "reason_recorded": reason,
    }


@tool
def get_tracking_info(order_id: str) -> dict:
    """Return carrier tracking details for an order."""
    catalog: dict[str, dict] = {
        "ORD-001": {"carrier": "UPS",   "tracking": "1ZMOCK000000000001", "status": "delivered",   "last_location": "Customer Address"},
        "ORD-002": {"carrier": "FedEx", "tracking": "MOCKFEDEX000000001", "status": "in_transit",  "last_location": "Chicago, IL"},
        "ORD-003": {"carrier": "USPS",  "tracking": "9400MOCK000000000001", "status": "label_created", "last_location": "Warehouse"},
    }
    return catalog.get(order_id, {"status": "not_found", "order_id": order_id})


@tool
def escalate_to_human(customer_id: str, issue_summary: str) -> dict:
    """Create a high-priority support ticket for human review."""
    return {
        "ticket_id": f"TKT-{uuid.uuid4().hex[:6].upper()}",
        "customer_id": customer_id,
        "priority": "high",
        "assigned_team": "customer_success",
        "estimated_response_hours": "2-4",
        "summary": issue_summary,
    }


ALL_TOOLS = [check_order_status, process_refund, get_tracking_info, escalate_to_human]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Session-scoped memory — keyed by session_id so sessions never bleed into each other
# ---------------------------------------------------------------------------

# FIX Bug 2: Dict is keyed by session_id. Each session gets its own sub-dict
# so _GLOBAL_MEMORY["last_query"] never exists at the top level, and a new
# session cannot see a prior session's context.
_GLOBAL_MEMORY: dict[str, Any] = {}


def _execute_tool_calls(response: AIMessage) -> tuple[list[ToolMessage], list[str]]:
    """Execute any tool_calls present in an AIMessage and return results + names."""
    results: list[ToolMessage] = []
    names: list[str] = []
    for tc in getattr(response, "tool_calls", []):
        fn = TOOLS_BY_NAME.get(tc["name"])
        if fn:
            output = fn.invoke(tc["args"])
            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
            names.append(tc["name"])
    return results, names


# ---------------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------------

def planner_agent(state: CustomerOpsState) -> dict:
    """Read the customer query + history and route to the correct specialist."""

    last_message = state["messages"][-1].content if state["messages"] else ""

    # -- ROUTING LOGIC --
    lower = last_message.lower()

    # FIX Bug 1: routing is now fully deterministic — no random branch.
    # Escalation keywords are checked FIRST so complaints that also mention
    # "order" or "package" are not silently swallowed by the order branch.
    if any(w in lower for w in ("complaint", "angry", "frustrated", "upset", "terrible",
                                 "worst", "awful", "horrible", "unacceptable", "manager")):
        routing = {"agent": "escalation_agent", "confidence": 0.95}

    elif any(w in lower for w in ("refund", "return", "money back", "charge", "reimburse")):
        routing = {"agent": "refund_agent", "confidence": 0.95}

    elif any(w in lower for w in ("where", "order", "status", "delivery", "track",
                                   "shipped", "arrive", "package")):
        routing = {"agent": "order_agent", "confidence": 0.90}

    else:
        routing = {"agent": "order_agent", "confidence": 0.50}

    system = (
        f"You are a customer operations planner. Briefly acknowledge the customer "
        f"and let them know a specialist ({routing['agent'].replace('_', ' ')}) "
        f"will assist them. Be concise and professional."
    )
    response = llm.invoke([SystemMessage(content=system), *state["messages"]])

    # FIX Bug 2: Write context under the session_id key so sessions are isolated.
    session_id = state["session_id"]
    session_memory = {"last_query": last_message, "routed_to": routing["agent"]}
    _GLOBAL_MEMORY[session_id] = session_memory

    return {
        "messages": [response],
        "current_agent": routing["agent"],
        "routing_decision": routing,
        "tool_calls_made": state.get("tool_calls_made", []),
        "session_id": session_id,
        "memory": session_memory,  # FIX Bug 2: only this session's data
    }


def refund_agent(state: CustomerOpsState) -> dict:
    """Handle refund and return requests."""
    system = (
        "You are a refund specialist. Help the customer process their refund request. "
        "Use the process_refund tool when you have an order ID and reason. "
        "Ask for the order ID if not provided. Be empathetic and professional."
    )
    response = llm_with_tools.invoke([SystemMessage(content=system), *state["messages"]])
    tool_messages, tool_names = _execute_tool_calls(response)

    return {
        "messages": [response, *tool_messages],
        "current_agent": "refund_agent",
        "routing_decision": state.get("routing_decision", {}),
        "tool_calls_made": state.get("tool_calls_made", []) + tool_names,
        "session_id": state["session_id"],
        "memory": state.get("memory", {}),
    }


def order_agent(state: CustomerOpsState) -> dict:
    """Handle order status and tracking queries."""
    system = (
        "You are an order status specialist. Use check_order_status and get_tracking_info "
        "to help the customer. Provide clear, accurate information. "
        "Ask for the order ID if not provided."
    )

    last_message = state["messages"][-1].content if state["messages"] else ""
    has_order_id = any(w in last_message.upper() for w in ("ORD-", "ORDER #", "ORDER NUMBER"))

    # FIX Bug 3: Single LLM call regardless of whether the query has an order ID.
    # No retry loop — one call is sufficient; if the agent needs more info it
    # will ask the customer in its response.
    response = llm_with_tools.invoke(
        [SystemMessage(content=system), *state["messages"]]
    )

    tool_messages, tool_names = _execute_tool_calls(response)  # type: ignore[arg-type]

    return {
        "messages": [response, *tool_messages],
        "current_agent": "order_agent",
        "routing_decision": state.get("routing_decision", {}),
        "tool_calls_made": state.get("tool_calls_made", []) + tool_names,
        "session_id": state["session_id"],
        "memory": state.get("memory", {}),
    }


def escalation_agent(state: CustomerOpsState) -> dict:
    """Handle complaints and issues requiring human review."""
    system = (
        "You are an escalation specialist. Respond with genuine empathy to the customer's "
        "frustration. Use escalate_to_human for serious complaints, providing the customer ID "
        "and a clear issue summary. Always give the customer a ticket ID and next steps."
    )
    response = llm_with_tools.invoke([SystemMessage(content=system), *state["messages"]])
    tool_messages, tool_names = _execute_tool_calls(response)

    return {
        "messages": [response, *tool_messages],
        "current_agent": "escalation_agent",
        "routing_decision": state.get("routing_decision", {}),
        "tool_calls_made": state.get("tool_calls_made", []) + tool_names,
        "session_id": state["session_id"],
        "memory": state.get("memory", {}),
    }


def validator_agent(state: CustomerOpsState) -> dict:
    """Quality-check the specialist's response before returning to the user."""

    # FIX Bug 4: Ask the LLM to review the last response and flag it if poor.
    system_review = (
        "You are a quality reviewer for customer service responses. "
        "Look at the conversation and assess the last agent reply. "
        "Reply with ONLY the word 'OK' if it fully and professionally addresses "
        "the customer's query, or 'IMPROVE' if it is incomplete, off-topic, or unhelpful."
    )
    review = llm.invoke([SystemMessage(content=system_review), *state["messages"]])
    # AIMessage.content may be a str or a list of content blocks depending on LangChain version
    review_text = (
        review.content
        if isinstance(review.content, str)
        else " ".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in review.content
        )
    )
    quality_ok = "IMPROVE" not in review_text.upper()

    if not quality_ok:
        system_improve = (
            "You are a quality reviewer for customer service responses. "
            "The previous response was inadequate. Rewrite it to fully and "
            "professionally address the customer's query."
        )
        improved = llm.invoke([SystemMessage(content=system_improve), *state["messages"]])
        return {**state, "messages": [improved]}

    return state


# ---------------------------------------------------------------------------
# GRAPH
# ---------------------------------------------------------------------------

def _route_from_planner(state: CustomerOpsState) -> str:
    return state.get("current_agent", "order_agent")


def build_graph() -> Any:
    g = StateGraph(CustomerOpsState)

    g.add_node("planner",          planner_agent)
    g.add_node("refund_agent",     refund_agent)
    g.add_node("order_agent",      order_agent)
    g.add_node("escalation_agent", escalation_agent)
    g.add_node("validator",        validator_agent)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "refund_agent":     "refund_agent",
            "order_agent":      "order_agent",
            "escalation_agent": "escalation_agent",
        },
    )
    g.add_edge("refund_agent",     "validator")
    g.add_edge("order_agent",      "validator")
    g.add_edge("escalation_agent", "validator")
    g.add_edge("validator",        END)

    return g.compile()


customer_ops_graph = build_graph()


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def run_query(query: str, session_id: str | None = None) -> dict:
    """Run a single customer query through the multi-agent graph."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    initial_state: CustomerOpsState = {
        "messages":        [HumanMessage(content=query)],
        "current_agent":   "",
        "routing_decision": {},
        "tool_calls_made": [],
        "session_id":      session_id,
        "memory":          {},  # BUG 2: starts empty, but _GLOBAL_MEMORY still persists
    }

    result = customer_ops_graph.invoke(initial_state)

    final_content = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            final_content = msg.content
            break

    return {
        "session_id":   session_id,
        "query":        query,
        "routing":      result.get("routing_decision", {}),
        "agent_used":   result.get("current_agent", ""),
        "tools_called": result.get("tool_calls_made", []),
        "response":     final_content,
    }
