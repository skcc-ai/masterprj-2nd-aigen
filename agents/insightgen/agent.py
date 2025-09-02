import json
from typing import Any, Dict, List

from tools import get_tools
from agents.insightgen.planner import LLMPlanner
import os


class InsightGenAgent:
    """
    간단한 InsightGen 에이전트: DB 규모/구조를 바탕으로 산출물 3가지를 선택하고,
    각 산출물별 사용할 tool과 근거를 결정한다. 실제 데이터는 tools를 호출해 요약을 생성.
    """

    def __init__(self, data_dir: str = "./data", repo_path: str = "."):
        self.data_dir = data_dir
        self.repo_path = repo_path
        self.tools = {t.name: t for t in get_tools()}
        self._llm_planner = None

    def _invoke(self, name: str, payload: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"tool_not_found:{name}"}
        try:
            return tool.invoke(payload)
        except Exception as e:
            return {"error": str(e), "tool": name}

    def analyze(self) -> Dict[str, Any]:
        # 1) 규모/구조 파악
        stats = self._invoke("get_database_stats", {"data_dir": self.data_dir})
        files = self._invoke("list_files", {"data_dir": self.data_dir})
        symbols = self._invoke("list_symbols", {"limit": 5000, "data_dir": self.data_dir})

        total_files = stats.get("total_files", 0) if isinstance(stats, dict) else 0
        total_symbols = stats.get("total_symbols", 0) if isinstance(stats, dict) else 0
        total_functions = 0
        if isinstance(stats, dict):
            st = stats.get("symbol_types", {}) or {}
            if isinstance(st, dict):
                total_functions = st.get("function", 0) or 0

        # 2) 산출물 3가지 선택
        # LLM이 결정하도록 시도 (환경변수/모델 설정 필요). 실패 시 안내 후 중단 (임의의 고정 답변 금지)
        outputs = []
        try:
            tool_specs = [
                {"name": n, "description": (getattr(t, "description", "") or "").strip()}
                for n, t in self.tools.items()
            ]
            if self._llm_planner is None:
                self._llm_planner = LLMPlanner()
            outputs = self._llm_planner.plan(stats if isinstance(stats, dict) else {}, tool_specs)
        except Exception as e:
            notice = (
                "LLM 기반 계획 수립(planner.py) 단계에서 실패했습니다. 환경변수/모델 설정을 확인하고 다시 시도해 주세요. "
                "임의의 고정 답변은 제공하지 않습니다."
            )
            return {
                "status": "planner_failed",
                "notice": notice,
                "error": str(e),
                "inputs": {"stats": stats},
                "selected_outputs": [],
                "drafts": {},
            }

        # 3) 각 산출물 초안 생성(도구 호출 샘플)
        # 핵심 심볼 인덱스: 검색어 몇 개로 샘플 인덱스 생성
        idx_samples: List[Any] = []
        for q in ["init", "data", "process"]:
            r = self._invoke("search_symbols_fts", {"query": q, "top_k": 5, "data_dir": self.data_dir})
            idx_samples.append({"query": q, "results": r})

        # 호출 그래프 개요: 첫 번째 함수형 심볼 기준으로 in/out 확인
        first_func = next((s for s in symbols if s.get("type") == "function"), None) if isinstance(symbols, list) else None
        calls_overview = {}
        if first_func and first_func.get("id"):
            fid = int(first_func["id"])  # type: ignore
            calls_overview = {
                "symbol": first_func,
                "outgoing": self._invoke("get_calls_from", {"caller_id": fid, "data_dir": self.data_dir}),
                "incoming": self._invoke("get_calls_to", {"callee_id": fid, "data_dir": self.data_dir}),
            }

        # 파일별 요약 카탈로그: 상위 2개 파일만 샘플 구성
        catalog: List[Dict[str, Any]] = []
        if isinstance(files, list):
            for f in files[:2]:
                file_symbols = [s for s in symbols if s.get("language") == f.get("language")] if isinstance(symbols, list) else []
                catalog.append({
                    "file": f,
                    "symbol_count_same_lang": len(file_symbols),
                })

        # 4) 산출물별 프롬프트/툴 호출 계획을 LLM이 설계
        try:
            tool_specs = [
                {"name": n, "description": (getattr(t, "description", "") or "").strip()}
                for n, t in self.tools.items()
            ]
            plans = self._llm_planner.plan_artifact_prompts(outputs, stats if isinstance(stats, dict) else {}, tool_specs)
        except Exception as e:
            return {
                "status": "artifact_planner_failed",
                "notice": "산출물 프롬프트/툴 계획 수립 실패. 환경설정 또는 응답 포맷을 확인하세요.",
                "error": str(e),
                "selected_outputs": outputs,
            }

        # 5) 계획에 따라 selected_outputs를 실제 산출물로 생성/저장
        artifacts_dir = os.path.join(self.data_dir, "insightgen")
        os.makedirs(artifacts_dir, exist_ok=True)

        artifact_paths: List[str] = []

        # LLM 요약 보강 헬퍼
        def _llm_summarize_json(title: str, payload: Any) -> str:
            try:
                # 간단히 search_symbols_fts를 재활용한지 여부와 무관하게 텍스트 생성
                from agents.structsynth.llm_analyzer import LLMAnalyzer  # type: ignore
                analyzer = LLMAnalyzer()
                # analyzer는 텍스트 기반이므로 JSON을 문자열로 변환하여 일반 청크 분석을 활용
                text = f"{title}\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
                return analyzer._get_llm_response(text)  # type: ignore
            except Exception:
                return ""

        # 계획된 각 산출물에 대해: 지정된 tool_invocations 실행 → ToolResults를 사용해 LLM 프롬프트 템플릿으로 생성
        for i, plan in enumerate(plans):
            name = plan.get("name", f"artifact_{i+1}")
            ext = plan.get("file_ext", "md")
            invocations = plan.get("tool_invocations", []) or []
            prompt_template = plan.get("prompt_template", "")

            tool_results = {}
            for inv in invocations:
                tname = inv.get("name")
                params = inv.get("params", {})
                alias = inv.get("alias", tname)
                if not tname:
                    continue
                result = self._invoke(tname, {**params, "data_dir": self.data_dir, "repo_path": self.repo_path})
                tool_results[alias] = result

            # 산출물 생성용 프롬프트를 LLM이 준 템플릿과 ToolResults로 구성하여 요약
            try:
                from agents.structsynth.llm_analyzer import LLMAnalyzer  # type: ignore
                analyzer = LLMAnalyzer()
                full_prompt = (
                    f"산출물: {name}\n\n[요구 템플릿]\n{prompt_template}\n\n[ToolResults]\n"
                    + json.dumps(tool_results, ensure_ascii=False, indent=2)
                )
                output_text = analyzer._get_llm_response(full_prompt)  # type: ignore
            except Exception:
                output_text = json.dumps({"template": prompt_template, "ToolResults": tool_results}, ensure_ascii=False, indent=2)

            filename = f"{i+1:02d}-{name}".replace(" ", "-")
            outpath = os.path.join(artifacts_dir, f"{filename}.{ext}")
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(output_text)
            artifact_paths.append(outpath)

        return {
            "inputs": {"stats": stats},
            "selected_outputs": outputs,
            "drafts": {
                "index_samples": idx_samples,
                "calls_overview": calls_overview,
                "file_catalog": catalog,
            },
            "artifacts": artifact_paths,
            "plans": plans,
        }


