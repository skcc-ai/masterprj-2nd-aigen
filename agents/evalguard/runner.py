import argparse
import glob
import json
import os
from typing import List

from .contracts import Criterion, LoopConfig
from .loop import eval_loop


DEFAULT_CRITERIA: List[Criterion] = [
    Criterion(name="경로/수치 정확도", type="rule", weight=1.2),
    Criterion(name="사실성(서술 근거 명시)", type="llm", weight=1.0),
    Criterion(name="근거 제시", type="llm", weight=1.0),
    Criterion(name="구성 명료성", type="llm", weight=1.0),
]


def main() -> None:
    p = argparse.ArgumentParser(description="EvalGuard runner")
    p.add_argument("--outputs_glob", default="./data/insightgen/*.md")
    p.add_argument("--output_path", default=None, help="Evaluate a single output file path")
    p.add_argument("--snapshots_dir", default="./artifacts/toolruns")
    p.add_argument("--criteria_json", default=None, help="Path to criteria meta JSON (planner output meta)")
    p.add_argument("--threshold", type=float, default=80.0)
    args = p.parse_args()

    cfg = LoopConfig(pass_threshold=args.threshold)

    # Single file mode
    if args.output_path:
        outputs = [args.output_path]
    else:
        outputs = sorted(glob.glob(args.outputs_glob))
    def _load_criteria_from_json(path: str, output_basename: str) -> list[Criterion] | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # direct list
            if isinstance(data, list):
                return [Criterion(**c) for c in data]
            # top-level key
            if isinstance(data, dict) and data.get("evaluation_criteria"):
                return [Criterion(**c) for c in data["evaluation_criteria"]]
            # selected_outputs 에서 name 매칭
            if isinstance(data, dict) and data.get("selected_outputs"):
                outs = data.get("selected_outputs") or []
                for o in outs:
                    nm = (o or {}).get("name")
                    if nm and output_basename.startswith(nm.replace(" ", "-")):
                        ecs = (o or {}).get("evaluation_criteria") or []
                        if ecs:
                            return [Criterion(**c) for c in ecs]
        except Exception:
            return None
        return None

    for out_path in outputs:
        # map output to snapshot by name heuristic
        base = os.path.splitext(os.path.basename(out_path))[0]
        snap_guess = os.path.join(args.snapshots_dir, f"{base}.json")
        if not os.path.exists(snap_guess):
            print(f"[WARN] snapshot not found for {out_path}: {snap_guess} — proceeding with empty snapshot")
        # Always use artifacts-relative ref so eval loop can resolve consistently
        ref = f"toolruns/{base}.json"
        # criteria resolution
        criteria = None
        if args.criteria_json and os.path.exists(args.criteria_json):
            criteria = _load_criteria_from_json(args.criteria_json, base)
        if not criteria:
            meta_guess = os.path.join(args.snapshots_dir, f"{base}.meta.json")
            if os.path.exists(meta_guess):
                criteria = _load_criteria_from_json(meta_guess, base)
        if not criteria:
            # try snapshot itself
            criteria = _load_criteria_from_json(snap_guess, base)
        final_criteria = criteria or DEFAULT_CRITERIA
        report = eval_loop(out_path, final_criteria, ref, cfg)
        print(json.dumps(report.dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


