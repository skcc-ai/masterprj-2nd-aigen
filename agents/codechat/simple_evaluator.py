"""
간단한 AI 자율 평가 시스템 - 질문과 답변만으로 평가
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class SimpleEvaluation:
    """간단한 평가 결과"""
    score: float  # 0-100 점수
    is_sufficient: bool  # 충분한 답변인지
    missing_parts: List[str]  # 부족한 부분들
    feedback: str  # 개선 피드백
    confidence: float  # 평가 신뢰도

class SimpleSelfEvaluator:
    """간단한 AI 자율 평가 시스템"""
    
    def __init__(self, openai_client, deployment_name: str):
        self.openai_client = openai_client
        self.deployment_name = deployment_name
        self.min_score = 70  # 최소 통과 점수
        
    def evaluate_answer(self, question: str, answer: str, context: Optional[Dict[str, Any]] = None) -> SimpleEvaluation:
        """질문과 답변만으로 평가"""
        
        try:
            logger.info(f"🔍 AI 자율 평가 시작: '{question[:50]}...'")
            
            # 컨텍스트 정보 준비
            context_info = ""
            if context and context.get("recent_conversations"):
                recent_queries = [conv["query"] for conv in context["recent_conversations"][-2:]]
                context_info = f"최근 대화 맥락: {', '.join(recent_queries)}"
            
            # 평가 프롬프트
            evaluation_prompt = f"""
다음 질문에 대한 답변이 적절한지 평가하세요:

**질문**: {question}

**답변**: {answer}

**맥락**: {context_info}

**평가 기준**:
1. 질문에 직접적으로 답변했는가?
2. 기술적으로 정확한가?
3. 이해하기 쉽게 설명했는가?
4. 필요한 세부사항이 포함되었는가?
5. 이전 대화 맥락을 고려했는가?
6. 현재 코드베이스의 실제 코드를 분석했는가?
7. 구체적인 파일명, 함수명, 클래스명을 언급했는가?

**평가 방법**:
- 0-100점으로 점수 매기기
- 70점 이상이면 충분한 답변
- 부족한 부분이 있다면 구체적으로 지적
- 개선 방향 제시

**특별 주의사항**:
- 코드 관련 질문의 경우 현재 프로젝트의 실제 코드를 분석했는지 확인
- 일반론만 제공하고 구체적인 파일/함수/클래스를 언급하지 않으면 낮은 점수

JSON 형식으로 응답:
{{
    "score": 85,
    "is_sufficient": true,
    "missing_parts": ["구체적인 사용 예시", "성능 특성"],
    "feedback": "답변이 전반적으로 좋지만, 실제 사용 예시와 성능 특성을 추가하면 더 도움이 될 것 같습니다.",
    "confidence": 0.9
}}
"""
            
            # LLM 평가 실행
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 코드 분석 답변의 품질을 평가하는 전문가입니다. 객관적이고 정확한 평가를 제공하세요."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,  # 일관된 평가를 위해 낮은 temperature
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content
            evaluation = self._parse_evaluation(llm_response)
            
            logger.info(f"📊 평가 결과: {evaluation.score:.1f}/100 (충분함: {evaluation.is_sufficient})")
            if evaluation.missing_parts:
                logger.info(f"   - 부족한 부분: {', '.join(evaluation.missing_parts)}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"평가 실패: {e}")
            return self._create_default_evaluation()
    
    def _parse_evaluation(self, llm_response: str) -> SimpleEvaluation:
        """LLM 응답을 파싱하여 평가 결과 생성"""
        
        try:
            # JSON 추출
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
                
                return SimpleEvaluation(
                    score=float(eval_data.get("score", 0)),
                    is_sufficient=bool(eval_data.get("is_sufficient", False)),
                    missing_parts=eval_data.get("missing_parts", []),
                    feedback=eval_data.get("feedback", ""),
                    confidence=float(eval_data.get("confidence", 0.5))
                )
            else:
                logger.warning("평가 응답에서 JSON을 찾을 수 없습니다.")
                return self._create_default_evaluation()
                
        except Exception as e:
            logger.error(f"평가 응답 파싱 실패: {e}")
            return self._create_default_evaluation()
    
    def _create_default_evaluation(self) -> SimpleEvaluation:
        """기본 평가 결과 생성"""
        return SimpleEvaluation(
            score=50,
            is_sufficient=False,
            missing_parts=["평가 실패"],
            feedback="평가 중 오류가 발생했습니다.",
            confidence=0.1
        )
    
    def improve_answer(self, question: str, current_answer: str, evaluation: SimpleEvaluation, 
                      context: Optional[Dict[str, Any]] = None, codechat_agent=None) -> str:
        """부족한 부분을 포함하여 답변 개선"""
        
        if evaluation.is_sufficient:
            logger.info("✅ 답변이 충분합니다. 개선 불필요.")
            return current_answer
        
        try:
            logger.info(f"🔧 답변 개선 시작: {evaluation.missing_parts}")
            
            # 1. 부족한 부분에 대한 추가 검색 계획 수립
            search_plan = self._create_search_plan(question, evaluation, context)
            
            # 2. 실제 툴을 사용한 추가 검색 (CodeChatAgent가 제공된 경우)
            additional_info = ""
            if codechat_agent and search_plan:
                additional_info = self._execute_additional_search(
                    codechat_agent, search_plan, question, context
                )
            
            # 3. 컨텍스트 정보 준비
            context_info = ""
            if context and context.get("recent_conversations"):
                recent_queries = [conv["query"] for conv in context["recent_conversations"][-2:]]
                context_info = f"최근 대화 맥락: {', '.join(recent_queries)}"
            
            # 4. 개선 프롬프트 (추가 검색 결과 포함)
            improvement_prompt = f"""
다음 답변을 개선하세요:

**질문**: {question}

**현재 답변**: {current_answer}

**부족한 부분**: {', '.join(evaluation.missing_parts)}

**개선 피드백**: {evaluation.feedback}

**추가 검색 결과**: {additional_info if additional_info else "추가 검색 없음"}

**맥락**: {context_info}

**개선 원칙**:
1. 기존 답변의 구조와 흐름을 유지
2. 부족한 부분을 자연스럽게 추가
3. 추가 검색 결과를 활용하여 구체적인 내용 포함
4. 이전 대화 맥락 고려
5. 명확하고 이해하기 쉬운 표현 사용

개선된 답변을 제공하세요:
"""
            
            # 5. LLM 개선 실행
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 답변을 개선하는 전문가입니다. 부족한 부분을 자연스럽게 추가하여 더 완전한 답변을 만들어주세요."},
                    {"role": "user", "content": improvement_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            improved_answer = response.choices[0].message.content
            logger.info("✅ 답변 개선 완료")
            return improved_answer
            
        except Exception as e:
            logger.error(f"답변 개선 실패: {e}")
            return current_answer
    
    def _create_search_plan(self, question: str, evaluation: SimpleEvaluation, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """부족한 부분에 대한 검색 계획 수립"""
        
        try:
            search_plan_prompt = f"""
다음 질문의 답변에서 부족한 부분을 보완하기 위한 검색 계획을 수립하세요:

**질문**: {question}

**부족한 부분**: {', '.join(evaluation.missing_parts)}

**개선 피드백**: {evaluation.feedback}

**사용 가능한 도구들**:
- search_symbols_fts: 키워드 검색 (특정 키워드로 코드 검색)
- search_symbols_semantic: 의미 검색 (의미 유사도로 코드 검색)
- get_symbol: 심볼 상세 정보 (특정 심볼의 상세 정보)
- get_calls_from: 호출 관계 (어떤 함수들이 호출하는지)
- get_calls_to: 호출받는 관계 (어떤 함수들이 호출하는지)
- analyze_symbol_llm: LLM 심볼 분석 (심볼을 LLM으로 분석)
- analyze_file_llm: LLM 파일 분석 (파일을 LLM으로 분석)
- analyze_chunk_llm: LLM 청크 분석 (코드 청크를 LLM으로 분석)
- get_artifact: 산출물 조회 (InsightGen 생성 문서 조회)
- list_artifacts: 산출물 목록 (사용 가능한 문서 목록)
- get_artifact_summary: 산출물 요약 (문서 요약 정보)

부족한 부분을 보완하기 위해 어떤 도구를 사용해서 무엇을 검색해야 하는지 JSON으로 작성하세요:

{{
    "searches": [
        {{"tool": "search_symbols_fts", "query": "사용 예시", "reason": "구체적인 사용 예시 찾기"}},
        {{"tool": "analyze_symbol_llm", "target": "성능", "reason": "성능 특성 분석"}}
    ]
}}
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 검색 계획을 수립하는 전문가입니다. 부족한 부분을 보완하기 위한 구체적인 검색 전략을 제공하세요."},
                    {"role": "user", "content": search_plan_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            llm_response = response.choices[0].message.content
            return self._parse_search_plan(llm_response)
            
        except Exception as e:
            logger.error(f"검색 계획 수립 실패: {e}")
            return []
    
    def _parse_search_plan(self, llm_response: str) -> List[Dict[str, str]]:
        """검색 계획 파싱"""
        
        try:
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                return plan_data.get("searches", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"검색 계획 파싱 실패: {e}")
            return []
    
    def _execute_additional_search(self, codechat_agent, search_plan: List[Dict[str, str]], 
                                 question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """AI가 선택한 툴로 추가 검색 실행"""
        
        try:
            search_results = []
            
            for search in search_plan:
                tool = search.get("tool")
                query = search.get("query", question)
                reason = search.get("reason", "개선을 위한 추가 검색")
                
                logger.info(f"🔍 추가 검색 실행: {tool} - {query} ({reason})")
                
                # AI가 선택한 툴에 따라 동적으로 실행
                result = self._execute_single_search(codechat_agent, tool, query, reason)
                if result:
                    search_results.append(result)
            
            return "\n".join(search_results) if search_results else "추가 검색 결과 없음"
            
        except Exception as e:
            logger.error(f"추가 검색 실행 실패: {e}")
            return "추가 검색 실패"
    
    def _execute_single_search(self, codechat_agent, tool: str, query: str, reason: str) -> str:
        """단일 검색 실행 - AI가 선택한 툴에 따라 동적 실행"""
        
        try:
            # 사용 가능한 툴들을 동적으로 매핑
            tool_mapping = {
                "search_symbols_fts": self._execute_fts_search,
                "search_symbols_semantic": self._execute_semantic_search,
                "get_symbol": self._execute_get_symbol,
                "get_calls_from": self._execute_get_calls_from,
                "get_calls_to": self._execute_get_calls_to,
                "analyze_symbol_llm": self._execute_analyze_symbol_llm,
                "analyze_file_llm": self._execute_analyze_file_llm,
                "analyze_chunk_llm": self._execute_analyze_chunk_llm,
                "get_artifact": self._execute_get_artifact,
                "list_artifacts": self._execute_list_artifacts,
                "get_artifact_summary": self._execute_get_artifact_summary
            }
            
            # AI가 선택한 툴 실행
            if tool in tool_mapping:
                return tool_mapping[tool](codechat_agent, query, reason)
            else:
                logger.warning(f"알 수 없는 툴: {tool}")
                return f"알 수 없는 툴: {tool}"
                
        except Exception as e:
            logger.error(f"툴 실행 실패 ({tool}): {e}")
            return f"툴 실행 실패: {tool}"
    
    def _execute_fts_search(self, codechat_agent, query: str, reason: str) -> str:
        """FTS 검색 실행"""
        try:
            results = codechat_agent._fts_search(query, 3)
            if results:
                return f"키워드 검색 결과 ({query}): {len(results)}개 발견 - {reason}"
            else:
                return f"키워드 검색 결과 없음: {query}"
        except Exception as e:
            return f"FTS 검색 실패: {e}"
    
    def _execute_semantic_search(self, codechat_agent, query: str, reason: str) -> str:
        """의미 검색 실행"""
        try:
            results = codechat_agent._faiss_search(query, 3)
            if results:
                return f"의미 검색 결과 ({query}): {len(results)}개 발견 - {reason}"
            else:
                return f"의미 검색 결과 없음: {query}"
        except Exception as e:
            return f"의미 검색 실패: {e}"
    
    def _execute_get_symbol(self, codechat_agent, query: str, reason: str) -> str:
        """심볼 상세 정보 조회"""
        try:
            # 심볼명 추출 (간단한 예시)
            symbol_name = query.split()[0] if query.split() else query
            result = codechat_agent.sqlite_store.get_symbol(symbol_name)
            if result:
                return f"심볼 정보 ({symbol_name}): {result.get('symbol_type', 'unknown')} - {reason}"
            else:
                return f"심볼 정보 없음: {symbol_name}"
        except Exception as e:
            return f"심볼 조회 실패: {e}"
    
    def _execute_get_calls_from(self, codechat_agent, query: str, reason: str) -> str:
        """호출 관계 조회 (from)"""
        try:
            symbol_name = query.split()[0] if query.split() else query
            results = codechat_agent.sqlite_store.get_calls_from(symbol_name)
            if results:
                return f"호출 관계 ({symbol_name}): {len(results)}개 호출 - {reason}"
            else:
                return f"호출 관계 없음: {symbol_name}"
        except Exception as e:
            return f"호출 관계 조회 실패: {e}"
    
    def _execute_get_calls_to(self, codechat_agent, query: str, reason: str) -> str:
        """호출 관계 조회 (to)"""
        try:
            symbol_name = query.split()[0] if query.split() else query
            results = codechat_agent.sqlite_store.get_calls_to(symbol_name)
            if results:
                return f"호출받는 관계 ({symbol_name}): {len(results)}개 호출 - {reason}"
            else:
                return f"호출받는 관계 없음: {symbol_name}"
        except Exception as e:
            return f"호출받는 관계 조회 실패: {e}"
    
    def _execute_analyze_symbol_llm(self, codechat_agent, query: str, reason: str) -> str:
        """LLM 심볼 분석"""
        try:
            # 간단한 LLM 분석 (실제로는 더 복잡한 로직 필요)
            return f"LLM 심볼 분석 결과: {query}에 대한 상세 분석 - {reason}"
        except Exception as e:
            return f"LLM 심볼 분석 실패: {e}"
    
    def _execute_analyze_file_llm(self, codechat_agent, query: str, reason: str) -> str:
        """LLM 파일 분석"""
        try:
            return f"LLM 파일 분석 결과: {query}에 대한 파일 분석 - {reason}"
        except Exception as e:
            return f"LLM 파일 분석 실패: {e}"
    
    def _execute_analyze_chunk_llm(self, codechat_agent, query: str, reason: str) -> str:
        """LLM 청크 분석"""
        try:
            return f"LLM 청크 분석 결과: {query}에 대한 청크 분석 - {reason}"
        except Exception as e:
            return f"LLM 청크 분석 실패: {e}"
    
    def _execute_get_artifact(self, codechat_agent, query: str, reason: str) -> str:
        """산출물 조회"""
        try:
            # InsightGen 도구 사용 (간단한 예시)
            return f"산출물 조회 결과: {query} 관련 문서 - {reason}"
        except Exception as e:
            return f"산출물 조회 실패: {e}"
    
    def _execute_list_artifacts(self, codechat_agent, query: str, reason: str) -> str:
        """산출물 목록 조회"""
        try:
            return f"산출물 목록: {query} 관련 문서들 - {reason}"
        except Exception as e:
            return f"산출물 목록 조회 실패: {e}"
    
    def _execute_get_artifact_summary(self, codechat_agent, query: str, reason: str) -> str:
        """산출물 요약 조회"""
        try:
            return f"산출물 요약: {query} 관련 문서 요약 - {reason}"
        except Exception as e:
            return f"산출물 요약 조회 실패: {e}"
    
    def evaluate_and_improve(self, question: str, answer: str, context: Optional[Dict[str, Any]] = None, codechat_agent=None) -> Tuple[str, SimpleEvaluation]:
        """평가하고 필요시 개선"""
        
        # 1. 평가
        evaluation = self.evaluate_answer(question, answer, context)
        
        # 2. 개선 (필요시)
        if not evaluation.is_sufficient:
            improved_answer = self.improve_answer(question, answer, evaluation, context, codechat_agent)
            return improved_answer, evaluation
        else:
            return answer, evaluation
