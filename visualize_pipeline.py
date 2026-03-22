"""
Generates a DAG visualization of the agent_pipeline and saves it to pipeline_dag.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

edges = [
    ("ingest_data",        "detect_data_drift"),
    ("ingest_data",        "build_features"),
    ("detect_data_drift",  "tune_agent"),
    ("build_features",     "tune_agent"),
    ("build_features",     "run_slice_evals"),
    ("tune_agent",         "ingest_agent"),
    ("ingest_agent",       "run_evals"),
    ("ingest_agent",       "run_slice_evals"),
    ("run_evals",          "champion_challenger"),
    ("run_slice_evals",    "champion_challenger"),
    ("champion_challenger","quality_gate"),
    ("run_evals",          "quality_gate"),
    ("run_slice_evals",    "quality_gate"),
    ("quality_gate",       "register_model"),
    ("run_evals",          "register_model"),
    ("run_slice_evals",    "register_model"),
    ("champion_challenger","register_model"),
    ("register_model",     "deploy_agent"),
    ("ingest_agent",       "deploy_agent"),
    ("deploy_agent",       "setup_monitoring"),
    ("detect_data_drift",  "setup_monitoring"),
    ("register_model",     "setup_monitoring"),
]

# ---------------------------------------------------------------------------
# Stage layout — x_range defines the coloured background band
# ---------------------------------------------------------------------------

STAGES = {
    "Data":       {"color": "#4A90D9", "x": (0.0,  3.0)},
    "Features":   {"color": "#7B68EE", "x": (3.0,  5.2)},
    "Training":   {"color": "#F5A623", "x": (5.2,  8.4)},
    "Evaluation": {"color": "#E74C3C", "x": (8.4,  14.0)},
    "Deployment": {"color": "#27AE60", "x": (14.0, 17.2)},
    "Monitoring": {"color": "#95A5A6", "x": (17.2, 19.2)},
}

# Node positions — kept strictly within their stage band
POS = {
    # Data  (band 0–3)
    "ingest_data":          (0.9,  2.5),
    "detect_data_drift":    (2.1,  3.7),

    # Features  (band 3–5.2)
    "build_features":       (4.1,  1.3),

    # Training  (band 5.2–8.4)
    "tune_agent":           (6.0,  2.5),
    "ingest_agent":         (7.6,  2.5),

    # Evaluation  (band 8.4–14)
    "run_evals":            (9.3,  3.7),
    "run_slice_evals":      (9.3,  1.3),
    "champion_challenger":  (11.2, 2.5),
    "quality_gate":         (12.9, 2.5),

    # Deployment  (band 14–17.2)
    "register_model":       (14.8, 3.7),
    "deploy_agent":         (16.4, 2.5),

    # Monitoring  (band 17.2–19.2)
    "setup_monitoring":     (18.2, 2.5),
}

STAGE_NODES = {
    "Data":       ["ingest_data", "detect_data_drift"],
    "Features":   ["build_features"],
    "Training":   ["tune_agent", "ingest_agent"],
    "Evaluation": ["run_evals", "run_slice_evals", "champion_challenger", "quality_gate"],
    "Deployment": ["register_model", "deploy_agent"],
    "Monitoring": ["setup_monitoring"],
}

node_colour_map = {}
for stage, nodes in STAGE_NODES.items():
    for n in nodes:
        node_colour_map[n] = STAGES[stage]["color"]

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

G = nx.DiGraph()
G.add_edges_from(edges)

fig, ax = plt.subplots(figsize=(26, 8))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

Y_MIN, Y_MAX = 0.2, 5.0   # vertical extent of the stage bands

# 1. Stage background bands + headers
for stage, info in STAGES.items():
    x0, x1 = info["x"]
    color   = info["color"]

    # Semi-transparent background
    band = mpatches.FancyBboxPatch(
        (x0 + 0.05, Y_MIN), x1 - x0 - 0.1, Y_MAX - Y_MIN,
        boxstyle="round,pad=0.05",
        facecolor=color, alpha=0.10,
        edgecolor=color, linewidth=1.2,
    )
    ax.add_patch(band)

    # Stage header bar
    header = mpatches.FancyBboxPatch(
        (x0 + 0.1, Y_MAX - 0.55), x1 - x0 - 0.2, 0.48,
        boxstyle="round,pad=0.04",
        facecolor=color, alpha=0.80,
        edgecolor="none",
    )
    ax.add_patch(header)

    ax.text(
        (x0 + x1) / 2, Y_MAX - 0.31,
        stage,
        color="white", fontsize=9.5, fontweight="bold",
        ha="center", va="center",
    )

# 2. Edges
nx.draw_networkx_edges(
    G, POS,
    ax=ax,
    edge_color="#6C7A8D",
    arrows=True,
    arrowsize=16,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.06",
    width=1.4,
    min_source_margin=22,
    min_target_margin=22,
)

# 3. Nodes
node_colours = [node_colour_map.get(n, "#888") for n in G.nodes()]
nx.draw_networkx_nodes(
    G, POS,
    ax=ax,
    node_color=node_colours,
    node_size=3000,
    node_shape="s",
    alpha=0.93,
)

# 4. Labels
label_map = {n: n.replace("_", "\n") for n in G.nodes()}
nx.draw_networkx_labels(
    G, POS,
    labels=label_map,
    ax=ax,
    font_size=7,
    font_color="white",
    font_weight="bold",
)

# 5. Title
ax.set_title(
    "agent_pipeline  ·  Data → Features → Training → Evaluation → Deployment → Monitoring",
    color="white",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

ax.set_xlim(-0.3, 19.5)
ax.set_ylim(Y_MIN - 0.2, Y_MAX + 0.3)
ax.axis("off")
plt.tight_layout()

output = "pipeline_dag.png"
plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {output}")
