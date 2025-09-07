from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class Criterion(BaseModel):
    name: str
    type: str  # "rule" | "llm"
    weight: float = 1.0
    tolerance_pct: Optional[float] = None


class EvaluationInput(BaseModel):
    name: str
    output_path: str
    output_text: str
    evaluation_criteria: List[Criterion]
    snapshot_ref: str  # artifacts/toolruns/<output_id>.json
    iteration: int = 1


class CriterionResult(BaseModel):
    score: int = Field(ge=0, le=100)
    evidence: Optional[str] = None
    expected: Optional[float] = None
    reported: Optional[float] = None
    tolerance_pct: Optional[float] = None
    llm_comment: Optional[str] = None


class EvaluationReport(BaseModel):
    name: str
    per_criterion: Dict[str, CriterionResult]
    average: float
    passes: bool
    iteration: int
    snapshot_ref: str
    llm_skipped: bool = False
    notice: Optional[str] = None


class ImprovementPlan(BaseModel):
    should_improve: bool
    reasons: List[str] = []
    improved_text: Optional[str] = None
    changes_delta: Optional[str] = None


class LoopConfig(BaseModel):
    pass_threshold: float = 80.0
    max_iterations: int = 10
    min_improvement_delta: float = 3.0
    early_stop_rounds: int = 2
    per_criterion_min: float = 70.0


class TraceEvent(BaseModel):
    event: str
    data: Dict[str, Any] = {}


