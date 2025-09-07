from __future__ import annotations

import os
import json
from typing import List
import openai

from .contracts import EvaluationReport, ImprovementPlan
from .trace import TraceLogger


IMPROVE_PROMPT = (
    "당신은 산출물 개선 전문가입니다. 목표는 '평가 점수의 상승'이며 다음 지침을 엄격히 따르세요.\n"
    "1) 낮은 점수 항목별로 원인을 명시하고 해당 부분을 구체적으로 수정합니다.\n"
    "2) ToolResults의 수치/경로/파일명을 정확히 인용하여 오차를 제거합니다.\n"
    "3) 사실성/근거/구성 명료성 항목은 예시/표/근거 인용을 통해 서술 품질을 끌어올립니다.\n"
    "4) 자리표시자/추정/추가 가정 금지. 반드시 제공된 스냅샷과 기존 텍스트를 근거로만 작성합니다.\n"
    "5) 변경된 부분이 명확히 드러나도록 문장/표를 재구성하고, 중복/장황함을 제거합니다.\n"
    "6) 최종에 '## Changes(Delta)' 섹션을 추가하여 항목별 변경 전/후를 요약합니다.\n"
)


def _get_client():
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("EVAL LLM API key not configured")
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
    return openai.OpenAI(api_key=api_key)


def build_improvement(output_text: str, report: EvaluationReport, snapshot_json: str, tracer: TraceLogger | None = None) -> ImprovementPlan:
    if report.average >= 80:
        return ImprovementPlan(should_improve=False)

    client = _get_client()
    # Force AZURE_OPENAI_DEPLOYMENT_GPT41 if present
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT41", os.getenv("EVAL_MODEL", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    user = (
        IMPROVE_PROMPT
        + "\n[현재 산출물]\n" + output_text[:10000]
        + "\n[평가 결과]\n" + json.dumps(report.dict(), ensure_ascii=False)
        + "\n[ToolResults 스냅샷]\n" + snapshot_json[:15000]
        + "\n요청: 수정된 전체 본문을 출력하고, 말미에 '## Changes(Delta)' 섹션을 추가."
    )
    if tracer:
        tracer.log("improve_prompt", {"name": report.name, "model": model, "temperature": 0.0, "prompt": user})
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Revise document deterministically. Return full markdown only."},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=3000,
    )
    improved = resp.choices[0].message.content or ""
    reasons: List[str] = [k for k, v in report.per_criterion.items() if v.score < 80]
    return ImprovementPlan(should_improve=True, reasons=reasons, improved_text=improved, changes_delta=None)


