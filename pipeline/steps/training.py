"""
Training steps: tune_agent, ingest_agent

No gradient updates — "training" for an LLM agent means:
  - Analysing feature coverage to flag weak routing keyword lists
  - Selecting representative few-shot examples per category
  - Noting if detected drift warrants prompt or keyword updates
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

from zenml import step

from .data import DriftReport
from .features import FeatureSet, ROUTING_KEYWORDS

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    routing_keywords: dict      # {agent_name: [keywords]}
    few_shot_examples: dict     # {category: [example_queries]}
    module_path: str
    tuning_notes: list          # what was observed / recommended


@dataclass
class AgentManifest:
    framework: str
    agents: list
    tools: list
    entry_point: str
    module_path: str
    load_error: str = ""


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@step
def tune_agent(feature_set: FeatureSet, drift_report: DriftReport) -> AgentConfig:
    """Analyse feature coverage and drift; select few-shot examples per category."""
    module_path = str(PROJECT_ROOT / "agent" / "customer_ops_agent.py")
    tuning_notes: list[str] = []

    # Flag categories with low keyword coverage
    for cat, cov in feature_set.keyword_coverage.items():
        if cov < 0.5:
            tuning_notes.append(
                f"Low keyword coverage for '{cat}' ({cov:.0%}) — "
                f"consider expanding keyword list"
            )

    # Flag drift
    if drift_report.drift_detected:
        tuning_notes.append(
            f"Data drift detected ({drift_report.severity}) in "
            f"{drift_report.drifted_categories} — monitor routing accuracy closely"
        )

    if not tuning_notes:
        tuning_notes.append("No tuning changes required — coverage and distribution healthy")

    # Select few-shot examples: 2 shortest queries per category
    few_shot_examples: dict[str, list[str]] = {}
    for feat in feature_set.features:
        cat = feat["category"]
        few_shot_examples.setdefault(cat, []).append(feat["query"])

    for cat in few_shot_examples:
        few_shot_examples[cat] = sorted(few_shot_examples[cat], key=len)[:2]

    config = AgentConfig(
        routing_keywords=ROUTING_KEYWORDS,
        few_shot_examples=few_shot_examples,
        module_path=module_path,
        tuning_notes=tuning_notes,
    )

    print(f"\nAgent tuning")
    for note in tuning_notes:
        print(f"  • {note}")
    total_examples = sum(len(v) for v in few_shot_examples.values())
    print(f"  Few-shot examples selected: {total_examples} across {len(few_shot_examples)} categories")

    return config


@step
def ingest_agent(config: AgentConfig) -> AgentManifest:
    """Validate the agent module loads cleanly and extract its manifest."""
    module_path = config.module_path

    try:
        spec = importlib.util.spec_from_file_location("customer_ops_agent", module_path)
        mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
        spec.loader.exec_module(mod)                  # type: ignore[union-attr]
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
        module_path=module_path,
    )

    print(f"\nAgent manifest")
    print(f"  Framework  : {manifest.framework}")
    print(f"  Agents     : {manifest.agents}")
    print(f"  Tools      : {manifest.tools}")
    print(f"  Entry point: {manifest.entry_point}")

    return manifest
