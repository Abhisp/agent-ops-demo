# agent-ops-demo

A multi-agent customer operations system built with **LangGraph** and **Claude Haiku**, designed to explore agent routing patterns, tool use, and intentional bug injection for debugging exercises.

## Stack

- Python 3.11
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent graph orchestration
- [Claude Haiku](https://www.anthropic.com/claude) (`claude-haiku-4-5-20251001`) — LLM backbone
- [ZenML](https://zenml.io) — ML pipeline framework (scaffolded in `pipeline/`)

## Project Structure

```
agent-ops-demo/
├── agent/
│   ├── customer_ops_agent.py   # Multi-agent graph (planner + 3 specialists + validator)
│   └── run_agent.py            # CLI entry point
├── pipeline/                   # ZenML pipeline stubs
├── evals/                      # Evaluation harness (to be built)
├── data/
│   └── sample_queries.json     # 10 test queries across 4 categories
├── .env                        # ANTHROPIC_API_KEY (never commit this)
├── .gitignore
└── requirements.txt
```

## Setup

```bash
# 1. Clone / enter the project
cd agent-ops-demo

# 2. Create and activate the virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## Running

```bash
# 3 sample queries then interactive mode (default)
python agent/run_agent.py

# All 10 sample queries then interactive mode
python agent/run_agent.py --all

# Interactive mode only
python agent/run_agent.py --interactive
```

Each response prints a metadata line showing which agent handled the query, routing confidence, and tools called:

```
[order_agent | confidence=0.9 | tools=['check_order_status']]

Agent: Your order ORD-002 is currently in transit...
```

## Agent Graph

```
START
  └─► planner
        ├─► refund_agent     ─┐
        ├─► order_agent      ─┼─► validator ─► END
        └─► escalation_agent ─┘
```

| Agent | Responsibility |
|---|---|
| **PlannerAgent** | Reads the query, routes to a specialist |
| **RefundAgent** | Refund requests, return eligibility |
| **OrderAgent** | Order status, tracking, delivery |
| **EscalationAgent** | Complaints, angry customers, human handoff |
| **ValidatorAgent** | Quality-checks the specialist's response |

## Tools

All tools are mock implementations — no real APIs required.

| Tool | Description |
|---|---|
| `check_order_status(order_id)` | Returns mock order status |
| `process_refund(order_id, reason)` | Returns mock refund confirmation |
| `get_tracking_info(order_id)` | Returns mock carrier tracking data |
| `escalate_to_human(customer_id, issue_summary)` | Returns mock support ticket ID |

Sample order IDs: `ORD-001`, `ORD-002`, `ORD-003`

## Intentional Bugs

Four bugs are deliberately planted for debugging exercises. Each is marked with `# BUG N:` in the source.

| # | Type | Location | Description |
|---|---|---|---|
| 1 | Routing | `planner_agent()` | Queries containing `"where"` are misrouted to `refund_agent` ~30% of the time instead of `order_agent` |
| 2 | Memory leak | `_GLOBAL_MEMORY` + `planner_agent()` | Session memory is a module-level dict never cleared between conversations — prior customer context bleeds in |
| 3 | Retry loop | `order_agent()` | Ambiguous queries (no order ID) trigger up to 5 LLM API calls with no early exit, causing cost spikes |
| 4 | Silent fail | `validator_agent()` | `quality_ok` is hardcoded to `True` — the validator never inspects responses and always passes them through |

### Triggering the bugs

```
"Where is my stuff?"                          → Bug 1 (run several times to see misrouting)
"I have a problem with something I bought"    → Bug 3 (5 API calls fire)
Any two queries back-to-back                  → Bug 2 (second query sees first query's memory)
Any query with a poor/irrelevant response     → Bug 4 (validator silently passes it)
```

## Sample Queries

`data/sample_queries.json` contains 10 queries:

- 3 × order status
- 3 × refund requests
- 2 × complaints
- 2 × ambiguous (bug triggers)

## Next Steps

- `evals/` — add LLM-as-judge evals to catch routing errors at scale
- `pipeline/` — wrap the agent in a ZenML pipeline for reproducible runs and experiment tracking
- Fix the four bugs and verify with regression queries
