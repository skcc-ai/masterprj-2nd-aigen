import json
import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.insightgen.agent import InsightGenAgent
from tools.structsynth_tools import run_repository_analysis  # type: ignore


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run InsightGen agent to generate analysis artifacts.")
    parser.add_argument("--repo_path", default=".", help="Path to the target repository to analyze")
    parser.add_argument("--data_dir", default="./data", help="Directory to store/load StructSynth database")
    parser.add_argument("--artifacts_dir", default="./artifacts", help="Directory to store vector/index artifacts when running analysis")
    parser.add_argument("--run_analysis", action="store_true", help="Run StructSynth repository analysis before InsightGen")
    args = parser.parse_args()

    if args.run_analysis:
        try:
            run_repository_analysis(repo_path=args.repo_path, artifacts_dir=args.artifacts_dir, data_dir=args.data_dir)
        except Exception:
            # Proceed anyway; InsightGen may still use existing DB if present
            pass

    agent = InsightGenAgent(data_dir=args.data_dir, repo_path=args.repo_path)
    result = agent.analyze()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


