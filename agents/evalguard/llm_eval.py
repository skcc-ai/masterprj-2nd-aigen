from __future__ import annotations

import os
from typing import Dict
import openai

from .contracts import EvaluationInput, CriterionResult
from .trace import TraceLogger


def _get_client():
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )
    return openai.OpenAI(api_key=api_key)


EVAL_PROMPT = (
    "당신은 코드 리포트 평가자입니다. 아래 출력 텍스트를 읽고 주어진 기준에 대해 0~100 정수 점수로 평가하세요.\n"
    "규칙: 외부 지식 금지, ToolResults의 수치만 신뢰, JSON 스키마 엄수.\n"
)


def llm_evaluate(ein: EvaluationInput, tracer: TraceLogger | None = None) -> Dict[str, CriterionResult]:
    client = _get_client()
    if client is None:
        if tracer:
            tracer.log("llm_eval_skipped", {"reason": "EVAL LLM API key not configured"})
        return {}
    # Force evaluation model to AZURE_OPENAI_DEPLOYMENT_GPT41 if present
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT41", os.getenv("EVAL_MODEL", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")))

    per: Dict[str, CriterionResult] = {}
    for c in ein.evaluation_criteria:
        if c.type != "llm":
            continue
        user = (
            EVAL_PROMPT
            + f"출력 텍스트:\n" + ein.output_text[:8000]
            + "\n평가 기준: " + c.name + "\n"
            + "반환 JSON 키: score(정수), llm_comment(선택)."
        )
        if tracer:
            tracer.log("llm_eval_prompt", {"criterion": c.name, "model": model, "temperature": 0.0, "prompt": user})
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Evaluate. Return JSON only."},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or "{}"
        if tracer:
            tracer.log("llm_eval_response", {"criterion": c.name, "raw": content})
        try:
            import json

            data = json.loads(content)
            score = int(data.get("score", 0))
            comment = data.get("llm_comment")
            per[c.name] = CriterionResult(score=max(0, min(100, score)), llm_comment=comment)
        except Exception:
            per[c.name] = CriterionResult(score=50, llm_comment="invalid JSON from LLM")

    return per


