"""
Monitoring step: setup_monitoring
"""

import json
from pathlib import Path
from typing import Any

from zenml import step

from .data import DriftReport
from .deployment import ModelVersion
from .evaluation import EvalReport

PROJECT_ROOT = Path(__file__).parent.parent.parent


@step
def setup_monitoring(endpoint: str, drift_report: DriftReport, version: ModelVersion) -> None:
    """Write monitoring/config.json with thresholds and canary promotion logic.
    Also writes monitoring/monitor.py — a standalone script that reads
    agent_runs.jsonl and alerts on routing anomalies, cost spikes, memory
    bleed, quality regressions, and canary health.
    """
    monitoring_dir = PROJECT_ROOT / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)

    def _get(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    config: dict[str, Any] = {
        "endpoint":        endpoint,
        "model":           "claude-haiku-4-5-20251001",
        "current_version": _get(version, "version"),
        "canary_percent":  _get(version, "traffic_percent"),
        "alert_thresholds": {
            "routing_accuracy_min":     0.90,
            "memory_bleed_tolerance":   0,
            "max_llm_calls_per_query":  3,
            "min_response_quality_score": 7.0,
            "max_error_rate":           0.05,
            "max_latency_p99_ms":       2000,
        },
        "canary_promotion": {
            "stages":                      [10, 50, 100],
            "min_healthy_minutes_per_stage": 30,
            "auto_rollback_on_breach":     True,
        },
        "drift_status": {
            "detected":           _get(drift_report, "drift_detected"),
            "severity":           _get(drift_report, "severity"),
            "drifted_categories": _get(drift_report, "drifted_categories"),
        },
        "tools_to_monitor": [
            "check_order_status",
            "process_refund",
            "get_tracking_info",
            "escalate_to_human",
        ],
        "eval_baselines": {
            name: round(score, 3)
            for name, score in (
                version["eval_scores"] if isinstance(version, dict) else version.eval_scores
            ).items()
        },
    }

    config_path = monitoring_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    monitor_script = '''\
"""
Agent monitor.

Reads agent_runs.jsonl (one JSON object per line) and alerts on:
  routing anomalies, memory bleed, excessive LLM calls,
  response quality regressions, and canary error rate.

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
            f"over last {len(runs)} runs"
        ]
    return []


def check_llm_call_frequency(runs: list[dict], max_calls: int) -> list[str]:
    alerts = []
    for r in runs:
        calls = r.get("llm_calls_made", 0)
        if isinstance(calls, int) and calls > max_calls:
            alerts.append(
                f"ALERT [cost]     query='{r.get('query', '')[:40]}' "
                f"used {calls} LLM calls (threshold={max_calls})"
            )
    return alerts


def check_memory_bleed(runs: list[dict]) -> list[str]:
    seen: set[str] = set()
    for r in runs:
        last_q = r.get("memory", {}).get("last_query", "")
        if last_q and last_q in seen:
            return [
                "ALERT [memory]   Previous session last_query found in new session — "
                "check _GLOBAL_MEMORY scoping."
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
            f"ALERT [quality]  avg score={avg:.1f} < threshold={min_score}"
        ]
    return []


def check_canary_health(runs: list[dict], max_error_rate: float) -> list[str]:
    if not runs:
        return []
    errors = sum(1 for r in runs if r.get("error"))
    rate = errors / len(runs)
    if rate > max_error_rate:
        return [
            f"ALERT [canary]   error rate={rate:.1%} > threshold={max_error_rate:.1%} "
            f"— consider rollback"
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
        print("Tip: append run_query() results as JSON lines to enable monitoring.")
        return

    version  = config.get("current_version", "unknown")
    canary   = config.get("canary_percent", 100)
    drift    = config.get("drift_status", {})

    print(f"Analysing {len(runs)} recent runs  |  version={version}  canary={canary}%")
    if drift.get("detected"):
        print(f"  ⚠  Drift: {drift['severity']} in {drift['drifted_categories']}")

    alerts: list[str] = []
    alerts += check_routing_health(runs,    thresholds["routing_accuracy_min"])
    alerts += check_llm_call_frequency(runs, thresholds["max_llm_calls_per_query"])
    alerts += check_memory_bleed(runs)
    alerts += check_quality_scores(runs,    thresholds["min_response_quality_score"])
    alerts += check_canary_health(runs,     thresholds["max_error_rate"])

    if alerts:
        print(f"\\n{len(alerts)} alert(s):\\n")
        for a in alerts:
            print(f"  {a}")
    else:
        print("  All checks healthy.")
        stages = config.get("canary_promotion", {}).get("stages", [10, 50, 100])
        if canary < 100:
            next_stage = next((s for s in stages if s > canary), 100)
            print(f"  Canary healthy → eligible to promote {canary}% → {next_stage}%")


if __name__ == "__main__":
    main()
'''

    monitor_path = monitoring_dir / "monitor.py"
    monitor_path.write_text(monitor_script)

    print(f"\nMonitoring configured")
    print(f"  Version      : {_get(version, 'version')}")
    print(f"  Canary       : {_get(version, 'traffic_percent')}%")
    print(f"  Drift status : {_get(drift_report, 'severity')}")
    if _get(drift_report, "drift_detected"):
        print(f"  ⚠  Drifted   : {_get(drift_report, 'drifted_categories')}")
    print(f"  Config       : {config_path}")
    print(f"  Monitor      : {monitor_path}")
