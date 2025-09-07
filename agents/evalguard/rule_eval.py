from __future__ import annotations

import json
import re
from typing import Dict, Any, Tuple

from .contracts import EvaluationInput, CriterionResult


def _extract_number(s: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _safe_load_snapshot(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def score_file_count(output_text: str, snapshot: Dict[str, Any], tolerance_pct: float | None) -> Tuple[int, Dict[str, Any]]:
    expected = len((snapshot.get("files") or []))
    reported = _extract_number(output_text) or 0.0
    if expected <= 0:
        score = 0 if reported > 0 else 100
    else:
        diff = abs(reported - expected)
        pct = (diff / expected) * 100.0
        tol = tolerance_pct or 0.0
        score = 100 if pct <= tol else max(0, int(100 - pct))
    return score, {"expected": expected, "reported": reported, "tolerance_pct": tolerance_pct}


def score_paths_accuracy(output_text: str, snapshot: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    files = snapshot.get("files") or []
    listed = 0
    correct = 0
    for f in files[:50]:  # sample up to 50
        path = (f.get("path") or "").strip()
        if not path:
            continue
        listed += 1
        if path in output_text:
            correct += 1
    if listed == 0:
        return 50, {"evidence": "no files in snapshot"}
    score = int((correct / listed) * 100)
    return score, {"evidence": f"{correct}/{listed} paths matched"}


def rule_evaluate(ein: EvaluationInput, snapshot_path: str) -> Dict[str, CriterionResult]:
    snapshot = _safe_load_snapshot(snapshot_path)
    results: Dict[str, CriterionResult] = {}
    text = ein.output_text

    for c in ein.evaluation_criteria:
        if c.type != "rule":
            continue
        if "경로" in c.name or "수치 정확도" in c.name:
            s, extra = score_paths_accuracy(text, snapshot)
            results[c.name] = CriterionResult(score=int(s), evidence=extra.get("evidence"))
        else:
            results[c.name] = CriterionResult(score=50, evidence="unsupported rule criterion -> conservative score")

    return results


