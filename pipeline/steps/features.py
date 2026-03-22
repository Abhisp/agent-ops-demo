"""
Feature step: build_features
"""

from dataclasses import dataclass
from zenml import step

from .data import QueryDataset

# Mirrors the routing keywords in planner_agent — kept in sync manually
ROUTING_KEYWORDS: dict[str, list[str]] = {
    "escalation_agent": [
        "complaint", "angry", "frustrated", "upset", "terrible",
        "worst", "awful", "horrible", "unacceptable", "manager",
    ],
    "refund_agent": [
        "refund", "return", "money back", "charge", "reimburse",
    ],
    "order_agent": [
        "where", "order", "status", "delivery", "track",
        "shipped", "arrive", "package",
    ],
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureSet:
    features: list          # list of per-query feature dicts
    category_stats: dict    # {category: {avg_length, keyword_coverage, count}}
    avg_query_length: float
    keyword_coverage: dict  # {category: fraction of queries with ≥1 keyword hit}


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

@step
def build_features(dataset: QueryDataset) -> FeatureSet:
    """Compute per-query features: keyword hits, length bucket, order-ID flag,
    token estimate. Aggregate category-level keyword coverage stats.
    """
    features = []

    for q in dataset.queries:
        text = q["query"].lower()
        words = text.split()

        keyword_hits = {
            agent: [kw for kw in kws if kw in text]
            for agent, kws in ROUTING_KEYWORDS.items()
        }

        word_count = len(words)
        if word_count <= 6:
            length_bucket = "short"
        elif word_count <= 12:
            length_bucket = "medium"
        else:
            length_bucket = "long"

        features.append({
            "query":          q["query"],
            "category":       q["category"],
            "expected_agent": q["expected_agent"],
            "word_count":     word_count,
            "length_bucket":  length_bucket,
            "keyword_hits":   keyword_hits,
            "any_keyword_hit": any(hits for hits in keyword_hits.values()),
            "token_estimate": round(word_count * 1.3),
            "has_order_id":   any(
                w in q["query"].upper() for w in ("ORD-", "ORDER #", "ORDER NUMBER")
            ),
        })

    # Aggregate per-category stats
    category_stats: dict[str, dict] = {}
    for feat in features:
        cat = feat["category"]
        if cat not in category_stats:
            category_stats[cat] = {"lengths": [], "keyword_hits": 0, "count": 0}
        category_stats[cat]["lengths"].append(feat["word_count"])
        category_stats[cat]["keyword_hits"] += int(feat["any_keyword_hit"])
        category_stats[cat]["count"] += 1

    for cat, stats in category_stats.items():
        stats["avg_length"] = round(
            sum(stats["lengths"]) / len(stats["lengths"]), 1
        )
        stats["keyword_coverage"] = round(
            stats["keyword_hits"] / stats["count"], 3
        )

    keyword_coverage = {
        cat: stats["keyword_coverage"]
        for cat, stats in category_stats.items()
    }
    avg_query_length = round(
        sum(f["word_count"] for f in features) / len(features), 1
    )

    feature_set = FeatureSet(
        features=features,
        category_stats=category_stats,
        avg_query_length=avg_query_length,
        keyword_coverage=keyword_coverage,
    )

    print(f"\nFeature extraction")
    print(f"  Queries processed : {len(features)}")
    print(f"  Avg query length  : {avg_query_length} words")
    print(f"  Keyword coverage  :")
    for cat, cov in keyword_coverage.items():
        flag = "  ⚠ low" if cov < 0.5 else ""
        print(f"    {cat:22s}: {cov:.0%}{flag}")

    return feature_set
