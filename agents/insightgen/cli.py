import json
import os
import sys
from dotenv import load_dotenv

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from agents.insightgen.agent import InsightGenAgent


def main():
    load_dotenv()
    agent = InsightGenAgent(data_dir="./data", repo_path=".")
    result = agent.analyze()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


