from __future__ import annotations

import json
import os
from typing import Dict, Any

from .contracts import EvaluationInput, EvaluationReport, LoopConfig
from .rule_eval import rule_evaluate
from .llm_eval import llm_evaluate
from .improver import build_improvement
from .trace import TraceLogger


def _load_text(path: str) -> str:
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception:
        return ""


def _save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _weighted_average(per: Dict[str, Any], criteria: list) -> float:
    total, weight = 0.0, 0.0
    for c in criteria:
        r = per.get(c.name)
        if not r:
            continue
        total += float(r.score) * float(c.weight)
        weight += float(c.weight)
    return (total / weight) if weight > 0 else 0.0


def run_once(ein: EvaluationInput, cfg: LoopConfig, base_dir_eval: str = "./artifacts/eval") -> EvaluationReport:
    tracer = TraceLogger(base_dir_eval)
    snapshot_path = os.path.join("./artifacts", ein.snapshot_ref)
    tracer.log("start_once", ein.dict())

    # Rule first
    per_rule = rule_evaluate(ein, snapshot_path)
    # LLM for remaining criteria
    per_llm = llm_evaluate(ein, tracer)
    merged: Dict[str, Any] = {**per_rule, **per_llm}

    avg = _weighted_average(merged, ein.evaluation_criteria)
    report = EvaluationReport(
        name=ein.name,
        per_criterion=merged,
        average=round(avg, 2),
        passes=avg >= cfg.pass_threshold,
        iteration=ein.iteration,
        snapshot_ref=ein.snapshot_ref,
        llm_skipped=(len(per_llm) == 0),
        notice=("LLM 평가가 환경 미설정으로 건너뛰어졌습니다" if len(per_llm) == 0 else None),
    )

    # Write report
    safe_name = os.path.basename(ein.output_path)
    out_report = os.path.join(base_dir_eval, f"{safe_name}.report.json")
    os.makedirs(base_dir_eval, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(json.dumps(report.dict(), ensure_ascii=False, indent=2))

    tracer.log("report", {**report.dict(), "report_path": out_report})
    return report


def eval_loop(output_path: str, criteria: list, snapshot_ref: str, cfg: LoopConfig | None = None) -> EvaluationReport:
    cfg = cfg or LoopConfig()
    base_dir_eval = "./artifacts/eval"
    evalguard_out_dir = "./data/evalguard"
    tracer = TraceLogger(base_dir_eval)
    output_text = _load_text(output_path)
    name = os.path.splitext(os.path.basename(output_path))[0]

    iteration = 1

    while iteration <= cfg.max_iterations:
        ein = EvaluationInput(
            name=name,
            output_path=output_path,
            output_text=output_text,
            evaluation_criteria=criteria,
            snapshot_ref=snapshot_ref,
            iteration=iteration,
        )
        tracer.log("iteration", {"n": iteration})
        report = run_once(ein, cfg, base_dir_eval)
        # per-criterion minimum enforcement
        per_min_ok = all(r.score >= cfg.per_criterion_min for r in report.per_criterion.values())
        if report.passes and per_min_ok:
            tracer.log("pass", {"avg": report.average, "iter": iteration})
            # always persist the latest version to data/evalguard (even if no improvement was required)
            final_path = os.path.join("./data/evalguard", os.path.basename(output_path))
            _save_text(final_path, output_text)
            tracer.log("final_saved", {"path": final_path})
            return report

        # try improve for non-1st outputs
        snapshot_json = _load_text(os.path.join("./artifacts", snapshot_ref))
        tracer.log("improve_request", {"name": name, "avg": report.average})
        plan = build_improvement(output_text, report, snapshot_json, tracer)
        tracer.log("improve_plan", plan.dict())
        if not plan.should_improve or not plan.improved_text:
            # persist current as final even if we cannot improve
            final_path = os.path.join("./data/evalguard", os.path.basename(output_path))
            _save_text(final_path, output_text)
            tracer.log("final_saved", {"path": final_path, "reason": "no_improvement"})
            return report

        # write improved output to a separate location (do not overwrite original)
        output_text = plan.improved_text
        improved_path = os.path.join(evalguard_out_dir, os.path.basename(output_path))
        _save_text(improved_path, output_text)
        tracer.log("improved_saved", {"path": improved_path})

        iteration += 1

    # loop ended without early return; persist latest
    final_path = os.path.join("./data/evalguard", os.path.basename(output_path))
    _save_text(final_path, output_text)
    tracer.log("final_saved", {"path": final_path, "reason": "loop_exhausted"})
    return report


