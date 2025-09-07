import json
import os
from datetime import datetime
from typing import Dict, Any


class TraceLogger:
    def __init__(self, base_dir: str = "./artifacts/eval"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.trace_path = os.path.join(self.base_dir, "trace.jsonl")

    def log(self, event: str, data: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "data": data,
        }
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


