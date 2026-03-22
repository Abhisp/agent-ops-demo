"""
Entry point for the agent CI/CD pipeline.

Usage:
    python pipeline/run_pipeline.py

What it does:
  1. Initialises ZenML local store if not already set up
  2. Loads .env (ANTHROPIC_API_KEY)
  3. Runs all five pipeline steps in sequence
  4. Exits 0 on success, 1 if the quality gate blocks deployment
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure all project paths are importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "agent", PROJECT_ROOT / "evals", PROJECT_ROOT / "pipeline"):
    sys.path.insert(0, str(p))

os.chdir(PROJECT_ROOT)  # ZenML resolves relative paths from cwd

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY is not set. Add it to .env and retry.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Initialise ZenML local store (idempotent)
# ---------------------------------------------------------------------------


def _ensure_zenml_init() -> None:
    zen_dir = PROJECT_ROOT / ".zen"
    if zen_dir.exists():
        return
    print("Initialising ZenML local store...")
    try:
        subprocess.run(
            [sys.executable, "-m", "zenml", "init"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
        )
        print("ZenML initialised.\n")
    except subprocess.CalledProcessError:
        # zenml init can fail if already initialised at a parent level; continue
        pass


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_zenml_init()

    print("=" * 60)
    print("  Agent Ops Pipeline")
    print("  Steps: ingest → evals → quality gate → deploy → monitoring")
    print("=" * 60)

    from agent_pipeline import agent_pipeline  # noqa: PLC0415 (lazy import — after sys.path setup)

    try:
        agent_pipeline()
        print("\nPipeline completed successfully.")
        sys.exit(0)
    except Exception as exc:
        # RuntimeError from quality_gate surfaces here
        if "Quality gate FAILED" in str(exc):
            print("\nPipeline stopped at quality gate — see details above.")
        else:
            print(f"\nPipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
