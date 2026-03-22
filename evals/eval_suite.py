"""
Eval suite for the Customer Operations Agent.

Four evals, one per intentional bug:
  Eval 1 — Routing correctness     → catches Bug 1 (30% "where" misroute)
  Eval 2 — Memory isolation        → catches Bug 2 (global memory bleed)
  Eval 3 — LLM call efficiency     → catches Bug 3 (5x retry loop)
  Eval 4 — Response quality gate   → catches Bug 4 (validator always True)
"""

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_QUERIES_PATH = DATA_DIR / "sample_queries.json"

sys.path.insert(0, str(AGENT_DIR))

from customer_ops_agent import _GLOBAL_MEMORY, run_query  # noqa: E402


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float          # 0.0 – 1.0
    threshold: float
    details: dict[str, Any]
    failures: list[dict[str, Any]] = field(default_factory=list)

    def summary_line(self) -> str:
        icon = "PASS ✓" if self.passed else "FAIL ✗"
        return (
            f"  [{icon}] {self.name}: "
            f"{self.score:.0%} (threshold ≥ {self.threshold:.0%})"
        )


@dataclass
class EvalReport:
    eval_results: list[EvalResult]
    overall_pass: bool
    summary: str

    @classmethod
    def from_results(cls, results: list[EvalResult]) -> "EvalReport":
        overall = all(r.passed for r in results)
        lines = ["", "=" * 56, "  EVAL REPORT", "=" * 56]
        for r in results:
            lines.append(r.summary_line())
            for f in r.failures[:2]:
                lines.append(f"      ↳ {f}")
        lines += ["", "─" * 56, f"  OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}", "=" * 56, ""]
        return cls(eval_results=results, overall_pass=overall, summary="\n".join(lines))


# ---------------------------------------------------------------------------
# LLM call counter (used by Eval 3)
# ---------------------------------------------------------------------------

@contextmanager
def _count_llm_calls() -> Generator[dict, None, None]:
    """Monkey-patch ChatAnthropic.invoke to count API calls."""
    import langchain_anthropic

    counter: dict[str, int] = {"count": 0}
    original = langchain_anthropic.ChatAnthropic.invoke

    def patched(self, *args: Any, **kwargs: Any) -> Any:
        counter["count"] += 1
        return original(self, *args, **kwargs)

    langchain_anthropic.ChatAnthropic.invoke = patched
    try:
        yield counter
    finally:
        langchain_anthropic.ChatAnthropic.invoke = original


# ---------------------------------------------------------------------------
# Eval 1 — Routing correctness
# Catches: Bug 1 ("where" queries 30% misrouted to refund_agent)
# ---------------------------------------------------------------------------

def eval_routing_correctness() -> EvalResult:
    """
    Run each sample query multiple times and check routing_decision per CATEGORY.
    The eval fails if any category's routing accuracy is below the threshold.

    Why per-category?  Bug 1 only affects "ambiguous" queries (~30% fail rate).
    Averaging across all 10 queries dilutes the signal to ~97% overall — which
    passes the threshold even with the bug present.  Checking the "ambiguous"
    bucket in isolation gives ~70% accuracy there, which clearly fails.

    Run counts: 2 per non-ambiguous query, 5 per ambiguous query.
    Pass threshold: every category >= 90% correct routing.
    """
    THRESHOLD = 0.90

    with open(SAMPLE_QUERIES_PATH) as f:
        queries = json.load(f)["queries"]

    # category → {correct, total}
    category_counts: dict[str, dict[str, int]] = {}
    failures: list[dict] = []

    for q in queries:
        cat = q.get("category", "unknown")
        runs = 5 if cat == "ambiguous" else 2
        category_counts.setdefault(cat, {"correct": 0, "total": 0})

        for _ in range(runs):
            result = run_query(q["query"])
            got = result["agent_used"]
            expected = q["expected_agent"]
            category_counts[cat]["total"] += 1
            if got == expected:
                category_counts[cat]["correct"] += 1
            else:
                failures.append({
                    "category": cat,
                    "query": q["query"][:60],
                    "expected": expected,
                    "got": got,
                    "confidence": result["routing"].get("confidence"),
                })

    # Score = worst-performing category (the pipeline only ships if all are healthy)
    category_scores = {
        cat: v["correct"] / v["total"] if v["total"] else 0.0
        for cat, v in category_counts.items()
    }
    worst_score = min(category_scores.values()) if category_scores else 0.0

    total_runs = sum(v["total"] for v in category_counts.values())
    total_correct = sum(v["correct"] for v in category_counts.values())

    return EvalResult(
        name="Routing Correctness",
        passed=worst_score >= THRESHOLD,
        score=worst_score,
        threshold=THRESHOLD,
        details={
            "total_runs": total_runs,
            "total_correct": total_correct,
            "per_category": {
                cat: f"{v['correct']}/{v['total']} ({v['correct']/v['total']:.0%})"
                for cat, v in category_counts.items()
            },
        },
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Eval 2 — Memory isolation
# Catches: Bug 2 (_GLOBAL_MEMORY never cleared between sessions)
# ---------------------------------------------------------------------------

def eval_memory_isolation() -> EvalResult:
    """
    Run two sequential conversations with different customer contexts.
    After session 1 completes, capture what _GLOBAL_MEMORY contains.
    If it holds session 1's data when session 2 starts, memory is bleeding.

    Pass threshold: zero memory bleed (0 leaked keys).
    """
    THRESHOLD = 1.0  # 100% isolation required

    # Reset state before test
    _GLOBAL_MEMORY.clear()

    # Session 1 — refund context
    session1_query = "I need a refund for order ORD-001, the laptop screen is cracked"
    run_query(session1_query, session_id="eval-session-001")

    # Capture what leaked BEFORE session 2 runs
    leaked: dict = dict(_GLOBAL_MEMORY)

    # Session 2 — completely different context (order tracking)
    session2_query = "Can you check the delivery status for order ORD-003?"
    run_query(session2_query, session_id="eval-session-002")

    # The bug: leaked["last_query"] == session1_query means session 2 SAW session 1's data
    bleed_detected = leaked.get("last_query") == session1_query

    failures: list[dict] = []
    if bleed_detected:
        failures.append({
            "description": "Session 1 context persisted in _GLOBAL_MEMORY before session 2 ran",
            "leaked_last_query": leaked.get("last_query", ""),
            "leaked_routed_to": leaked.get("routed_to", ""),
            "fix": "Clear _GLOBAL_MEMORY per session_id or use session-scoped state",
        })

    score = 0.0 if bleed_detected else 1.0
    return EvalResult(
        name="Memory Isolation",
        passed=not bleed_detected,
        score=score,
        threshold=THRESHOLD,
        details={"sessions_tested": 2, "leaked_keys": list(leaked.keys())},
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Eval 3 — LLM call efficiency
# Catches: Bug 3 (OrderAgent retries LLM 5x on ambiguous queries)
# ---------------------------------------------------------------------------

def eval_tool_call_efficiency() -> EvalResult:
    """
    Count actual LLM API calls made per query (not just tool calls, since Bug 3
    is a retry loop on the LLM layer, not the tool layer).

    Normal flow: planner (1 call) + specialist (1 call) = 2 calls max.
    With Bug 3:  planner (1) + order_agent retries (5) = 6 calls for ambiguous queries.

    Pass threshold: 80% of queries within the call budget (allows 1 noisy outlier).
    """
    MAX_LLM_CALLS = 3  # planner + 1 specialist + slack
    PASS_RATE_THRESHOLD = 0.80

    test_queries = [
        # Specific queries (have order ID) — should NOT trigger Bug 3
        ("What is the status of order ORD-001?", False),
        ("Track order ORD-002 for me please", False),
        # Ambiguous queries (no order ID) — trigger Bug 3 (5x retries)
        ("I have a problem with something I bought last week", True),
        ("Where is my stuff?", True),
        ("I need help with my recent purchase", True),
    ]

    results: list[bool] = []
    failures: list[dict] = []

    for query, is_ambiguous in test_queries:
        with _count_llm_calls() as counter:
            run_query(query)
        calls = counter["count"]
        ok = calls <= MAX_LLM_CALLS
        results.append(ok)
        if not ok:
            failures.append({
                "query": query[:60],
                "llm_calls_made": calls,
                "threshold": MAX_LLM_CALLS,
                "is_ambiguous": is_ambiguous,
                "excess_calls": calls - MAX_LLM_CALLS,
                "fix": "Add break condition to order_agent retry loop (Bug 3)",
            })

    score = sum(results) / len(results)
    return EvalResult(
        name="LLM Call Efficiency",
        passed=score >= PASS_RATE_THRESHOLD,
        score=score,
        threshold=PASS_RATE_THRESHOLD,
        details={
            "queries_tested": len(test_queries),
            "max_llm_calls_per_query": MAX_LLM_CALLS,
            "call_counts": {q: c for (q, _), c in zip(
                test_queries,
                [f["llm_calls_made"] for f in failures] + [MAX_LLM_CALLS] * (len(test_queries) - len(failures))
            )},
        },
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Eval 4 — Response quality gate
# Catches: Bug 4 (ValidatorAgent always returns True)
# ---------------------------------------------------------------------------

def eval_response_quality() -> EvalResult:
    """
    Have Claude Haiku score each response 1-10 on helpfulness, accuracy, and tone.
    Includes queries that trigger Bug 1 (misrouting) so bad responses surface —
    responses from the wrong specialist score poorly, and Bug 4 means the validator
    never catches and fixes them.

    Pass threshold: average score across all responses >= 7.0 / 10.
    """
    import anthropic

    SCORE_THRESHOLD = 0.70  # 7/10 normalised to 0.0–1.0
    MIN_AVG_SCORE = 7.0

    scorer = anthropic.Anthropic()

    # Mix of queries: include "where" queries that Bug 1 will misroute ~30% of the time
    eval_queries = [
        "Where is my stuff?",
        "Where is my order?",
        "What is the status of order ORD-002?",
        "I need a refund for order ORD-001, it arrived damaged",
        "Where is my package?",
    ]

    all_scores: list[float] = []
    failures: list[dict] = []

    for query in eval_queries:
        result = run_query(query)
        response_text = result.get("response", "")
        agent_used = result.get("agent_used", "unknown")
        tools_called = result.get("tools_called", [])

        # Ask Claude Haiku to score the response
        scoring_prompt = (
            f"Rate this customer service interaction. Return ONLY valid JSON, no other text.\n\n"
            f"Customer query: {query}\n"
            f"Agent that handled it: {agent_used}\n"
            f"Tools called: {tools_called}\n"
            f"Agent response: {response_text[:500]}\n\n"
            f"Score each dimension 1–10:\n"
            f"- helpfulness: Does the response address what the customer asked?\n"
            f"- accuracy: Is the content factually correct given the agent and tools used?\n"
            f"- tone: Is it professional, clear, and appropriately empathetic?\n\n"
            f'Return: {{"helpfulness": X, "accuracy": X, "tone": X, "note": "brief reason"}}'
        )

        try:
            scoring_response = scorer.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                messages=[{"role": "user", "content": scoring_prompt}],
            )
            raw = scoring_response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            avg = (scores["helpfulness"] + scores["accuracy"] + scores["tone"]) / 3
        except Exception as exc:
            # Scoring failure — count as zero, don't crash the eval
            avg = 0.0
            scores = {"helpfulness": 0, "accuracy": 0, "tone": 0, "note": f"scoring error: {exc}"}

        all_scores.append(avg)

        if avg < MIN_AVG_SCORE:
            failures.append({
                "query": query,
                "agent_used": agent_used,
                "avg_score": round(avg, 1),
                "scores": scores,
                "note": scores.get("note", ""),
                "fix": "Fix validator_agent() to actually evaluate response quality (Bug 4)",
            })

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    score_normalised = overall_avg / 10.0

    return EvalResult(
        name="Response Quality",
        passed=overall_avg >= MIN_AVG_SCORE,
        score=score_normalised,
        threshold=SCORE_THRESHOLD,
        details={
            "queries_scored": len(eval_queries),
            "average_score": round(overall_avg, 2),
            "individual_scores": [round(s, 1) for s in all_scores],
        },
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_evals() -> EvalReport:
    """Run all four evals and return a consolidated EvalReport."""
    print("\nRunning Eval 1 — Routing Correctness...")
    r1 = eval_routing_correctness()
    print(r1.summary_line())

    print("Running Eval 2 — Memory Isolation...")
    r2 = eval_memory_isolation()
    print(r2.summary_line())

    print("Running Eval 3 — LLM Call Efficiency...")
    r3 = eval_tool_call_efficiency()
    print(r3.summary_line())

    print("Running Eval 4 — Response Quality...")
    r4 = eval_response_quality()
    print(r4.summary_line())

    return EvalReport.from_results([r1, r2, r3, r4])


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    report = run_all_evals()
    print(report.summary)
    sys.exit(0 if report.overall_pass else 1)
