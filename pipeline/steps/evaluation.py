"""
Evaluation steps: run_evals, run_slice_evals, champion_challenger, quality_gate
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from zenml import step

from .training import AgentManifest
from .features import FeatureSet

PROJECT_ROOT = Path(__file__).parent.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "model_registry.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    eval_results: list
    overall_pass: bool
    summary: str


@dataclass
class SliceReport:
    slices: dict        # {slice_name: {score, correct, total, failures}}
    worst_slice: str
    worst_score: float
    passed: bool
    threshold: float = 0.80


@dataclass
class ComparisonResult:
    champion_scores: dict
    challenger_scores: dict
    regressions: list
    challenger_wins: bool
    is_first_run: bool


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@step
def run_evals(manifest: AgentManifest) -> EvalReport:
    """Run the full eval suite (all 4 evals)."""
    sys.path.insert(0, str(PROJECT_ROOT / "evals"))
    from eval_suite import run_all_evals, EvalReport as SuiteReport  # noqa: PLC0415

    print(f"\nRunning eval suite  →  agent: {manifest.module_path}")
    report: SuiteReport = run_all_evals()
    print(report.summary)

    return EvalReport(
        eval_results=report.eval_results,
        overall_pass=report.overall_pass,
        summary=report.summary,
    )


@step
def run_slice_evals(manifest: AgentManifest, feature_set: FeatureSet) -> SliceReport:
    """Break routing accuracy down by slice: length_bucket and has_order_id."""
    sys.path.insert(0, str(PROJECT_ROOT / "agent"))
    from customer_ops_agent import run_query  # noqa: PLC0415

    THRESHOLD = 0.80
    slices: dict[str, dict] = {}

    for feat in feature_set.features:
        result = run_query(feat["query"])
        got = result["agent_used"]
        expected = feat["expected_agent"]
        correct = int(got == expected)

        for slice_key in (
            f"length:{feat['length_bucket']}",
            f"order_id:{'yes' if feat['has_order_id'] else 'no'}",
        ):
            slices.setdefault(slice_key, {"correct": 0, "total": 0, "failures": []})
            slices[slice_key]["total"] += 1
            slices[slice_key]["correct"] += correct
            if not correct:
                slices[slice_key]["failures"].append({
                    "query":    feat["query"][:60],
                    "expected": expected,
                    "got":      got,
                })

    for data in slices.values():
        data["score"] = round(data["correct"] / data["total"], 3) if data["total"] else 0.0

    worst_slice = min(slices, key=lambda k: slices[k]["score"])
    worst_score = slices[worst_slice]["score"]

    report = SliceReport(
        slices=slices,
        worst_slice=worst_slice,
        worst_score=worst_score,
        passed=worst_score >= THRESHOLD,
        threshold=THRESHOLD,
    )

    print(f"\nSlice evals  (threshold ≥ {THRESHOLD:.0%})")
    for name, data in sorted(slices.items()):
        icon = "✓" if data["score"] >= THRESHOLD else "✗"
        print(f"  [{icon}] {name:28s}: {data['correct']}/{data['total']} ({data['score']:.0%})")

    return report


def _get(obj, key):
    """Access a field from either a dataclass instance or a plain dict (ZenML
    cloudpickle can deserialise dataclass instances as dicts across steps)."""
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


@step
def champion_challenger(eval_report: EvalReport, slice_report: SliceReport) -> ComparisonResult:
    """Load the current champion's scores from the registry and compare.
    Any metric that regresses more than 2 pp blocks the deploy.
    """
    challenger_scores = {_get(r, "name"): round(_get(r, "score"), 3) for r in eval_report.eval_results}
    challenger_scores["worst_slice"] = slice_report.worst_score

    if not REGISTRY_PATH.exists():
        print("\nNo registry found — first run, challenger becomes champion by default.")
        return ComparisonResult(
            champion_scores={},
            challenger_scores=challenger_scores,
            regressions=[],
            challenger_wins=True,
            is_first_run=True,
        )

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    champion_version = registry.get("champion")
    if not champion_version:
        return ComparisonResult(
            champion_scores={},
            challenger_scores=challenger_scores,
            regressions=[],
            challenger_wins=True,
            is_first_run=True,
        )

    champion_entry = next(
        (v for v in registry["versions"] if v["version"] == champion_version), None
    )
    champion_scores = champion_entry.get("eval_scores", {}) if champion_entry else {}

    regressions: list[str] = []
    for metric, chall_score in challenger_scores.items():
        champ_score = champion_scores.get(metric, 0.0)
        if chall_score < champ_score - 0.02:
            regressions.append(
                f"{metric}: {champ_score:.0%} → {chall_score:.0%}"
            )

    result = ComparisonResult(
        champion_scores=champion_scores,
        challenger_scores=challenger_scores,
        regressions=regressions,
        challenger_wins=not bool(regressions),
        is_first_run=False,
    )

    print(f"\nChampion / challenger  (champion: v{champion_version})")
    for metric, chall in challenger_scores.items():
        champ = champion_scores.get(metric, 0.0)
        arrow = "↑" if chall > champ + 0.001 else ("↓" if chall < champ - 0.001 else "=")
        print(f"  {metric:30s}: {champ:.0%} → {chall:.0%} {arrow}")
    if regressions:
        print(f"  Regressions : {regressions}")
    else:
        print(f"  Result      : challenger wins ✓")

    return result


@step
def quality_gate(
    eval_report: EvalReport,
    slice_report: SliceReport,
    comparison_result: ComparisonResult,
) -> bool:
    """Block deploy if core evals, slice evals, or champion/challenger checks fail."""
    all_passed = (
        _get(eval_report, "overall_pass")
        and _get(slice_report, "passed")
        and _get(comparison_result, "challenger_wins")
    )

    if all_passed:
        print("\n✓  All quality checks passed. Safe to deploy.\n")
        return True

    lines = ["\n✗  Quality gate FAILED:\n"]

    if not _get(eval_report, "overall_pass"):
        lines.append("  • Core evals failed — see eval report above")

    if not _get(slice_report, "passed"):
        lines.append(
            f"  • Slice eval: worst slice '{_get(slice_report, 'worst_slice')}' "
            f"scored {_get(slice_report, 'worst_score'):.0%} "
            f"(threshold ≥ {_get(slice_report, 'threshold'):.0%})"
        )

    if not _get(comparison_result, "challenger_wins"):
        lines.append("  • Champion/challenger: regressions detected")
        for r in _get(comparison_result, "regressions"):
            lines.append(f"      - {r}")

    lines.append("\nFix all failing checks and re-run before deploying.")
    message = "\n".join(lines)
    print(message)
    raise RuntimeError(message)
