import json
import os
from typing import Any, Dict, List

import openai


class LLMPlanner:
    """
    InsightGen용 LLM 기반 플래너.
    입력: DB 통계(stats)와 사용 가능한 tools 목록
    출력: 산출물 3가지에 대한 name/tools/reason/evaluation_criteria 구조
    """

    def __init__(self):
        self.client = self._init_client()
        self.model = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )

    def _init_client(self):
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM API key not configured")
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            # Azure OpenAI (proxies not supported in constructor; rely on env if needed)
            return openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            )
        # OpenAI
        return openai.OpenAI(api_key=api_key)

    def plan(self, stats: Dict[str, Any], tools: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """tools: [{name, description}]"""
        sys_prompt = (
            "당신은 코드 분석 제품의 기획/분석 전문가입니다. 주어진 코드베이스 통계와 사용 가능한 도구 목록을 바탕으로, "
            "가장 가치 있는 3가지 산출물을 선택하고 각 산출물에 대해 사용할 도구, 선택 근거, 평가 기준을 제안하세요. "
            "중요: chunk_id가 필요한 도구(예: analyze_chunk_llm, get_chunk, get_chunks_by_symbol)는 사용하지 마세요. "
            "반드시 JSON만 반환하세요. 불필요한 텍스트를 포함하지 마세요."
        )

        schema_example = {
            "outputs": [
                {
                    "name": "핵심 심볼 인덱스",
                    "tools": ["search_symbols_fts", "get_symbol"],
                    "reason": "심볼 분포/규모를 근거로 탐색 가속",
                    "evaluation_criteria": [
                        "검색 다양성", "정확성", "중복도"
                    ],
                }
            ]
        }

        user_prompt = (
            "[코드베이스 통계]\n" + json.dumps(stats, ensure_ascii=False) +
            "\n\n[사용 가능 도구]\n" + json.dumps(tools, ensure_ascii=False) +
            "\n\n[요구사항]\n- outputs 배열에 정확히 3개 항목만 포함\n- 각 항목: name, tools[], reason, evaluation_criteria[] 포함\n\n"
            + "[출력 JSON 스키마 예시]\n" + json.dumps(schema_example, ensure_ascii=False)
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = resp.choices[0].message.content
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])
            outputs = data.get("outputs", [])
            # 안전장치: 상위 3개만
            return outputs[:3]
        except Exception:
            raise RuntimeError("LLM planner returned invalid JSON")

    def plan_artifact_prompts(
        self,
        outputs: List[Dict[str, Any]],
        stats: Dict[str, Any],
        tools: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        각 산출물에 대해 사용할 tool 호출 계획(이름/파라미터/목적)과 산출물 생성용 프롬프트 템플릿을 생성.
        반환: [{name, file_ext, prompt_template, tool_invocations: [{name, params, alias, purpose}]}]
        """
        sys_prompt = (
            "당신은 코드 분석 결과 리포트를 자동 생성하는 시니어 프롬프트 엔지니어입니다. "
            "각 산출물에 대해 다음을 고품질로 설계하세요: 1) 사용할 도구 호출 계획, 2) 최종 산출물 생성을 위한 매우 구체적인 프롬프트 템플릿. "
            "프롬프트 템플릿은 다음을 포함해야 합니다: 목적/대상 사용자/산출물 구조(섹션 헤더)/포맷(마크다운 권장)/품질 체크리스트/길이 가이드. "
            "중요: chunk_id가 필요한 도구(예: analyze_chunk_llm, get_chunk, get_chunks_by_symbol)는 사용하지 마세요. "
            "템플릿은 자리표시자 대신 '다음 섹션의 ToolResults를 참고하여' 구체적으로 작성하라는 지시를 포함하세요. 반드시 JSON만 반환하세요."
        )

        schema_example = {
            "plans": [
                {
                    "name": "핵심 심볼 인덱스",
                    "file_ext": "md",
                    "tool_invocations": [
                        {"name": "search_symbols_fts", "params": {"query": "init", "top_k": 5}, "alias": "init_results", "purpose": "생성자 관련 심볼 수집"}
                    ],
                    "prompt_template": (
                        "[목적] 핵심 심볼 인덱스를 작성하여 신규 개발자가 빠르게 진입할 수 있도록 함.\n"
                        "[대상] 신규 팀원/코드리뷰어.\n"
                        "[출력형식] 마크다운.\n"
                        "[구성] # 개요 / ## 요약 / ## 핵심 심볼 목록 / ## 정렬 근거 / ## 다음 액션\n"
                        "[작성지침] 아래 ToolResults를 참고하여 각 심볼의 이름/타입/파일/요약/랭크를 표 형태로 정리. 랭크 순 정렬, 상단에 요약 3줄. 마지막에 3가지 권장 탐색 액션 제시.\n"
                    ),
                }
            ]
        }

        user_payload = {
            "selected_outputs": outputs,
            "stats": stats,
            "tools": tools,
            "requirements": [
                "각 산출물마다 file_ext를 md 또는 json 등으로 지정",
                "tool_invocations의 params는 간결한 기본값 사용",
                "prompt_template는 한국어로 작성",
                "총 plans는 selected_outputs와 동일한 순서/개수",
            ],
        }

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False) + "\n\n[스키마 예시]\n" + json.dumps(schema_example, ensure_ascii=False)},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        content = resp.choices[0].message.content
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])
            plans = data.get("plans", [])
            return plans
        except Exception:
            raise RuntimeError("LLM artifact prompt planning returned invalid JSON")


