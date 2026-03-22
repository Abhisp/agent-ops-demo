"""
ZenML pipeline: agent_pipeline

  Data       — ingest_data, detect_data_drift
  Features   — build_features
  Training   — tune_agent, ingest_agent
  Evaluation — run_evals, run_slice_evals, champion_challenger, quality_gate
  Deployment — register_model, deploy_agent
  Monitoring — setup_monitoring
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "agent", PROJECT_ROOT / "evals"):
    sys.path.insert(0, str(p))

from zenml import pipeline
from zenml.logger import get_logger

from steps.data       import ingest_data, detect_data_drift
from steps.features   import build_features
from steps.training   import tune_agent, ingest_agent
from steps.evaluation import run_evals, run_slice_evals, champion_challenger, quality_gate
from steps.deployment import register_model, deploy_agent
from steps.monitoring import setup_monitoring

logger = get_logger(__name__)


@pipeline
def agent_pipeline() -> None:
    """
    Full agent CI/CD pipeline:
      Data → Features → Training → Evaluation → Deployment → Monitoring
    """
    # Data
    dataset      = ingest_data()
    drift_report = detect_data_drift(dataset)

    # Features
    feature_set  = build_features(dataset)

    # Training
    config       = tune_agent(feature_set, drift_report)
    manifest     = ingest_agent(config)

    # Evaluation
    eval_report  = run_evals(manifest)
    slice_report = run_slice_evals(manifest, feature_set)
    comparison   = champion_challenger(eval_report, slice_report)
    gate_passed  = quality_gate(eval_report, slice_report, comparison)

    # Deployment
    version      = register_model(eval_report, slice_report, comparison, gate_passed)
    endpoint     = deploy_agent(manifest, version, gate_passed)

    # Monitoring
    setup_monitoring(endpoint, drift_report, version)
