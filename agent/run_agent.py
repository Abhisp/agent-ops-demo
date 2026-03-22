"""
CLI entry point for the Customer Operations Agent.

Usage:
    python agent/run_agent.py              # runs first 3 sample queries then interactive mode
    python agent/run_agent.py --all        # runs all 10 sample queries
    python agent/run_agent.py --interactive  # skip samples, go straight to interactive mode
"""

import json
import sys
from pathlib import Path

# Allow imports from the agent/ directory
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from customer_ops_agent import run_query  # noqa: E402 (must come after load_dotenv)

DATA_DIR = Path(__file__).parent.parent / "data"
DIVIDER = "─" * 60


def run_samples(count: int | None = 3) -> None:
    sample_path = DATA_DIR / "sample_queries.json"
    if not sample_path.exists():
        print("sample_queries.json not found — skipping sample run.")
        return

    with open(sample_path) as f:
        queries = json.load(f)["queries"]

    subset = queries if count is None else queries[:count]
    print(f"\nRunning {len(subset)} sample quer{'y' if len(subset) == 1 else 'ies'}...\n")

    for q in subset:
        print(DIVIDER)
        print(f"[{q['category'].upper()}] {q['query']}")
        if "note" in q:
            print(f"  note: {q['note']}")
        print()

        result = run_query(q["query"])

        print(f"  Agent     : {result['agent_used']}")
        print(f"  Confidence: {result['routing'].get('confidence', 'N/A')}")
        print(f"  Tools used: {result['tools_called'] or 'none'}")
        print(f"\n  Response:\n{result['response']}\n")


def interactive_mode() -> None:
    print("\n" + "=" * 60)
    print("  Interactive Mode  (type 'quit' or Ctrl-C to exit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nCustomer: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nProcessing...\n")
        try:
            result = run_query(user_input)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        tag = (
            f"[{result['agent_used']} | "
            f"confidence={result['routing'].get('confidence', '?')} | "
            f"tools={result['tools_called'] or 'none'}]"
        )
        print(tag)
        print(f"\nAgent: {result['response']}")


def main() -> None:
    args = sys.argv[1:]

    print("=" * 60)
    print("  Customer Operations Agent Demo")
    print("=" * 60)

    if "--interactive" in args:
        interactive_mode()
    elif "--all" in args:
        run_samples(count=None)
        interactive_mode()
    else:
        run_samples(count=3)
        interactive_mode()


if __name__ == "__main__":
    main()
