"""
Data steps: ingest_data, detect_data_drift
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from zenml import step

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BASELINE_PATH = DATA_DIR / "baseline_distribution.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryDataset:
    queries: list
    train_queries: list
    eval_queries: list
    category_distribution: dict
    total_count: int


@dataclass
class DriftReport:
    baseline_distribution: dict
    current_distribution: dict
    drift_detected: bool
    drifted_categories: list
    severity: str              # "none" | "warning" | "critical"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@step
def ingest_data() -> QueryDataset:
    """Load sample_queries.json, validate schema, split 80/20 train/eval."""
    path = DATA_DIR / "sample_queries.json"

    with open(path) as f:
        raw = json.load(f)

    queries = raw["queries"]

    required_keys = {"query", "expected_agent", "category"}
    for q in queries:
        missing = required_keys - q.keys()
        if missing:
            raise ValueError(f"Query missing fields {missing}: {q}")

    category_distribution: dict[str, int] = {}
    for q in queries:
        cat = q["category"]
        category_distribution[cat] = category_distribution.get(cat, 0) + 1

    split_idx = int(len(queries) * 0.8)
    train_queries = queries[:split_idx]
    eval_queries = queries[split_idx:]

    dataset = QueryDataset(
        queries=queries,
        train_queries=train_queries,
        eval_queries=eval_queries,
        category_distribution=category_distribution,
        total_count=len(queries),
    )

    print(f"\nData ingestion")
    print(f"  Total queries : {dataset.total_count}")
    print(f"  Train / eval  : {len(train_queries)} / {len(eval_queries)}")
    print(f"  Categories    : {category_distribution}")

    return dataset


@step
def detect_data_drift(dataset: QueryDataset) -> DriftReport:
    """Compare current category distribution against stored baseline.

    On first run: saves current distribution as the baseline.
    Flags any category that shifted more than 15 percentage points.
    """
    total = dataset.total_count
    current_dist = {
        cat: round(count / total, 4)
        for cat, count in dataset.category_distribution.items()
    }

    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline_dist = json.load(f)
    else:
        baseline_dist = current_dist
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline_dist, f, indent=2)
        print("\n  No baseline found — saving current distribution as baseline.")

    drifted: list[str] = []
    for cat, current_pct in current_dist.items():
        baseline_pct = baseline_dist.get(cat, 0.0)
        if abs(current_pct - baseline_pct) > 0.15:
            drifted.append(cat)

    if not drifted:
        severity = "none"
    elif len(drifted) == 1:
        severity = "warning"
    else:
        severity = "critical"

    report = DriftReport(
        baseline_distribution=baseline_dist,
        current_distribution=current_dist,
        drift_detected=bool(drifted),
        drifted_categories=drifted,
        severity=severity,
    )

    print(f"\nDrift detection")
    print(f"  Severity      : {severity}")
    for cat, pct in current_dist.items():
        baseline = baseline_dist.get(cat, 0.0)
        marker = "  ← DRIFT" if cat in drifted else ""
        print(f"  {cat:22s}: {pct:.0%}  (baseline {baseline:.0%}){marker}")

    return report
