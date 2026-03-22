"""
ZenML pipeline: agent_pipeline

Steps (run in order):
  1. ingest_agent      — validate + extract agent metadata
  2. run_evals         — execute all four evals from eval_suite.py
  3. quality_gate      — fail pipeline if any eval did not pass
  4. deploy_agent      — copy files, write manifest, start FastAPI server
  5. setup_monitoring  — write monitoring/config.json + monitoring/monitor.py
"""

import datetime
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zenml import pipeline, step
from zenml.logger import get_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
sys.path.insert(0, str(PROJECT_ROOT / "evals"))

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentManifest:
    framework: str
    agents: list
    tools: list
    entry_point: str
    module_path: str
    load_error: str = ""


# EvalReport is imported from eval_suite inside the steps to avoid
# module-level side effects during pipeline graph construction.


# ---------------------------------------------------------------------------
# Step 1 — ingest_agent
# ---------------------------------------------------------------------------

@step
def ingest_agent() -> AgentManifest:
    """
    Load the agent module, verify it imports without errors, and extract
    the AgentManifest (framework, agent functions, tools, entry point).
    Raises RuntimeError if the module fails to load — failing the pipeline early.
    """
    module_path = PROJECT_ROOT / "agent" / "customer_ops_agent.py"

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("customer_ops_agent", str(module_path))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        raise RuntimeError(f"Agent module failed to load: {exc}") from exc

    agents = sorted(
        name for name in dir(mod)
        if name.endswith("_agent") and callable(getattr(mod, name)) and not name.startswith("_")
    )
    tools = sorted(mod.TOOLS_BY_NAME.keys()) if hasattr(mod, "TOOLS_BY_NAME") else []
    entry_point = "run_query" if hasattr(mod, "run_query") else "unknown"

    manifest = AgentManifest(
        framework="langgraph",
        agents=agents,
        tools=tools,
        entry_point=entry_point,
        module_path=str(module_path),
    )

    print("\nAgent Manifest")
    print(f"  Framework  : {manifest.framework}")
    print(f"  Agents     : {manifest.agents}")
    print(f"  Tools      : {manifest.tools}")
    print(f"  Entry point: {manifest.entry_point}")

    return manifest


# ---------------------------------------------------------------------------
# Step 2 — run_evals
# ---------------------------------------------------------------------------

@step
def run_evals(manifest: AgentManifest) -> Any:
    """
    Run the full eval suite (all four evals) and return an EvalReport.
    Accepts AgentManifest so ZenML orders this step after ingest_agent.
    """
    from eval_suite import EvalReport, run_all_evals  # noqa: PLC0415

    print(f"\nRunning eval suite  →  agent: {manifest.module_path}")
    report: EvalReport = run_all_evals()
    print(report.summary)
    return report


# ---------------------------------------------------------------------------
# Step 3 — quality_gate
# ---------------------------------------------------------------------------

@step
def quality_gate(report: Any) -> bool:
    """
    Inspect the EvalReport.  If overall_pass is False, print a plain-English
    explanation of every failing eval and raise RuntimeError — stopping the
    pipeline before deploy_agent runs.
    """
    if report.overall_pass:
        print("\n✓  Agent passed all quality checks. Safe to deploy.\n")
        return True

    lines = ["\n✗  Quality gate FAILED — the following checks did not pass:\n"]

    bug_hints = {
        "Routing Correctness": (
            "The planner is misrouting queries to the wrong specialist.\n"
            "    Root cause → Bug 1: 'where' keyword routes to refund_agent 30% of the time.\n"
            "    Fix: remove the random.random() < 0.30 branch in planner_agent()."
        ),
        "Memory Isolation": (
            "Customer context from a previous session is bleeding into new sessions.\n"
            "    Root cause → Bug 2: _GLOBAL_MEMORY is a module-level dict, never cleared.\n"
            "    Fix: key _GLOBAL_MEMORY by session_id or use local state per invocation."
        ),
        "LLM Call Efficiency": (
            "Agent is making excessive LLM API calls on ambiguous queries (cost spikes).\n"
            "    Root cause → Bug 3: order_agent() retries up to 5x with no early exit.\n"
            "    Fix: add `break` after the first successful LLM response in the retry loop."
        ),
        "Response Quality": (
            "Responses are passing the validator despite poor quality scores.\n"
            "    Root cause → Bug 4: validator_agent() hardcodes quality_ok = True.\n"
            "    Fix: implement real quality check logic; remove the hardcoded True."
        ),
    }

    for result in report.eval_results:
        if result.passed:
            continue
        lines.append(f"  • {result.name}")
        lines.append(f"    Score: {result.score:.0%}  (required ≥ {result.threshold:.0%})")
        lines.append(f"    {bug_hints.get(result.name, 'See eval_suite.py for details.')}")
        if result.failures:
            lines.append("    Failing examples:")
            for failure in result.failures[:2]:
                lines.append(f"      - {failure}")
        lines.append("")

    lines.append("Fix all failing checks and re-run the pipeline before deploying.")
    message = "\n".join(lines)
    print(message)
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Step 4 — deploy_agent
# ---------------------------------------------------------------------------

@step
def deploy_agent(report: Any, gate_passed: bool) -> str:
    """
    Copy agent files to deployed/, write deployment_manifest.json, and start
    a lightweight FastAPI server on port 8000.  Returns the endpoint URL.
    gate_passed is accepted as an input purely to create a ZenML data-flow
    dependency on quality_gate (so this step never runs if quality_gate fails).
    """
    import shutil

    if not gate_passed:
        return ""

    deployed_dir = PROJECT_ROOT / "deployed"
    deployed_dir.mkdir(exist_ok=True)

    # --- Copy agent source files ---
    for filename in ("customer_ops_agent.py", "run_agent.py"):
        shutil.copy2(PROJECT_ROOT / "agent" / filename, deployed_dir / filename)
    shutil.copy2(PROJECT_ROOT / ".env", deployed_dir / ".env")

    # --- Write deployment manifest ---
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    eval_scores = {r.name: round(r.score, 3) for r in report.eval_results}
    deployment_manifest = {
        "timestamp": timestamp,
        "agent_version": "1.0.0",
        "framework": "langgraph",
        "model": "claude-haiku-4-5-20251001",
        "eval_scores": eval_scores,
        "all_evals_passed": report.overall_pass,
        "endpoint": "http://localhost:8000",
    }
    manifest_path = deployed_dir / "deployment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(deployment_manifest, f, indent=2)

    # --- Write FastAPI server ---
    server_code = '''\
"""FastAPI server — Customer Operations Agent (auto-generated by deploy step)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from customer_ops_agent import run_query

app = FastAPI(title="Customer Ops Agent", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "claude-haiku-4-5-20251001"}


@app.post("/query")
def query_endpoint(req: QueryRequest) -> dict:
    return run_query(req.query, req.session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    server_path = deployed_dir / "server.py"
    server_path.write_text(server_code)

    endpoint = "http://localhost:8000"

    # --- Start server as background process ---
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    try:
        subprocess.run(
            [python_exe, "-m", "pip", "install", "fastapi", "uvicorn", "-q"],
            check=True, capture_output=True,
        )
        proc = subprocess.Popen(
            [python_exe, str(server_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(deployed_dir),
        )
        time.sleep(2)
        if proc.poll() is None:
            print(f"\n  Server started (PID {proc.pid}) → {endpoint}")
        else:
            print(f"\n  Server exited early. Start manually: python {server_path}")
    except Exception as exc:
        print(f"\n  Could not auto-start server ({exc}). Run: python {server_path}")

    print(f"\nDeployment complete")
    print(f"  Files     → {deployed_dir}")
    print(f"  Manifest  → {manifest_path}")
    print(f"  Endpoint  → {endpoint}")
    print(f"\n  Health check: curl {endpoint}/health")
    print(f"  Query:        curl -X POST {endpoint}/query -H 'Content-Type: application/json' \\")
    print(f"                     -d '{{\"query\": \"Status of order ORD-001?\"}}'")

    return endpoint


# ---------------------------------------------------------------------------
# Step 5 — setup_monitoring
# ---------------------------------------------------------------------------

@step
def setup_monitoring(endpoint: str, report: Any) -> None:
    """
    Write monitoring/config.json with alert thresholds, tool watch-list, and
    cost estimates derived from eval results.  Also write monitoring/monitor.py,
    a standalone script that scans agent run logs and prints alerts.
    """
    monitoring_dir = PROJECT_ROOT / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)

    # Cost estimate: Claude Haiku ≈ $1/M input + $5/M output tokens
    # ~500 tokens per query → $0.0005/query baseline
    # With Bug 3 (5x retries): $0.0025/query for ambiguous queries
    baseline_cost = round(0.0005 * 10, 4)  # per 10 queries

    config: dict[str, Any] = {
        "endpoint": endpoint,
        "model": "claude-haiku-4-5-20251001",
        "alert_thresholds": {
            "routing_accuracy_min": 0.90,
            "memory_bleed_tolerance": 0,
            "max_llm_calls_per_query": 3,
            "min_response_quality_score": 7.0,
        },
        "tools_to_monitor": [
            "check_order_status",
            "process_refund",
            "get_tracking_info",
            "escalate_to_human",
        ],
        "cost_estimate": {
            "baseline_usd_per_10_queries": baseline_cost,
            "bug3_cost_multiplier": 5,
            "note": (
                "Bug 3 retry loop can 5x cost for ambiguous queries. "
                "Fix: break on first good LLM response in order_agent()."
            ),
        },
        "eval_baselines": {r.name: round(r.score, 3) for r in report.eval_results},
    }

    config_path = monitoring_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # --- Write monitoring/monitor.py ---
    monitor_script = '''\
"""
Agent monitoring script.

Reads agent_runs.jsonl (one JSON object per line, written by your logging layer)
and alerts on routing anomalies, memory bleed, and excessive LLM calls.

Usage:
    python monitoring/monitor.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"
LOG_PATH    = Path(__file__).parent / "agent_runs.jsonl"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_recent_runs(n: int = 100) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    runs = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs[-n:]


def check_routing_health(runs: list[dict], min_accuracy: float) -> list[str]:
    if not runs:
        return []
    correct = sum(1 for r in runs if r.get("routing_correct", True))
    accuracy = correct / len(runs)
    if accuracy < min_accuracy:
        return [
            f"ALERT [routing]  accuracy={accuracy:.0%} < threshold={min_accuracy:.0%} "
            f"over last {len(runs)} runs — check Bug 1 (misroute on 'where' queries)."
        ]
    return []


def check_llm_call_frequency(runs: list[dict], max_calls: int) -> list[str]:
    alerts = []
    for r in runs:
        calls = r.get("llm_calls_made", 0)
        if isinstance(calls, int) and calls > max_calls:
            alerts.append(
                f"ALERT [cost]     query='{r.get('query', '')[:40]}' "
                f"used {calls} LLM calls (threshold={max_calls}) — "
                f"check Bug 3 (retry loop in order_agent)."
            )
    return alerts


def check_memory_bleed(runs: list[dict]) -> list[str]:
    seen: set[str] = set()
    for r in runs:
        mem = r.get("memory", {})
        last_q = mem.get("last_query", "")
        if last_q and last_q in seen:
            return [
                "ALERT [memory]   Previous session's last_query found in new session's memory. "
                "Check Bug 2 (_GLOBAL_MEMORY not cleared between sessions)."
            ]
        if last_q:
            seen.add(last_q)
    return []


def check_quality_scores(runs: list[dict], min_score: float) -> list[str]:
    scored = [r for r in runs if "quality_score" in r]
    if not scored:
        return []
    avg = sum(r["quality_score"] for r in scored) / len(scored)
    if avg < min_score:
        return [
            f"ALERT [quality]  average quality score={avg:.1f} < threshold={min_score} "
            f"— check Bug 4 (validator_agent always returns True)."
        ]
    return []


def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Agent monitor running...")

    try:
        config = load_config()
    except FileNotFoundError:
        print("Config not found. Run the pipeline first: python pipeline/run_pipeline.py")
        sys.exit(1)

    thresholds = config["alert_thresholds"]
    runs = load_recent_runs()

    if not runs:
        print(f"No logs found at {LOG_PATH}")
        print("Tip: append run_query() results as JSON lines to that file to enable monitoring.")
        return

    print(f"Analysing {len(runs)} recent runs...")

    alerts: list[str] = []
    alerts += check_routing_health(runs, thresholds["routing_accuracy_min"])
    alerts += check_llm_call_frequency(runs, thresholds["max_llm_calls_per_query"])
    alerts += check_memory_bleed(runs)
    alerts += check_quality_scores(runs, thresholds["min_response_quality_score"])

    if alerts:
        print(f"\\n{len(alerts)} alert(s):\\n")
        for a in alerts:
            print(f"  {a}")
    else:
        print("All checks healthy — no alerts.")

    costs = config.get("cost_estimate", {})
    print(f"\\nCost baseline: ${costs.get('baseline_usd_per_10_queries', '?')}/10 queries.")
    if any("cost" in a.lower() or "retry" in a.lower() for a in alerts):
        mult = costs.get("bug3_cost_multiplier", 5)
        print(f"  WARNING: retry loop detected — actual cost may be up to {mult}x higher.")


if __name__ == "__main__":
    main()
'''

    monitor_path = monitoring_dir / "monitor.py"
    monitor_path.write_text(monitor_script)

    print(f"\nMonitoring configured")
    print(f"  Config  → {config_path}")
    print(f"  Monitor → {monitor_path}")
    print(f"\n  Run anytime: python monitoring/monitor.py")


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@pipeline
def agent_pipeline() -> None:
    """
    Full agent CI/CD pipeline:
      ingest → evals → quality gate → deploy → monitoring

    Data-flow dependencies ensure sequential execution and guarantee that
    deploy_agent never runs if quality_gate raises an exception.
    """
    manifest   = ingest_agent()
    report     = run_evals(manifest)
    passed     = quality_gate(report)
    endpoint   = deploy_agent(report, passed)
    setup_monitoring(endpoint, report)
