"""
AI 기반 컨텍스트 관리 - LLM이 대화 맥락을 학습하고 지능적으로 활용
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import time
from collections import deque
import hashlib
import re

logger = logging.getLogger(__name__)

@dataclass
class AIConversationContext:
    """AI 대화 컨텍스트"""
    session_id: str
    user_id: Optional[str]
    conversation_history: List[Dict[str, Any]]
    learned_patterns: Dict[str, Any]
    user_preferences: Dict[str, Any]
    context_insights: Dict[str, Any]
    created_at: float
    updated_at: float

@dataclass
class AIContextInsight:
    """AI 컨텍스트 인사이트"""
    insight_type: str  # pattern, preference, relationship, trend
    content: str
    confidence: float
    source: str
    timestamp: float

class AIContextManager:
    """AI 기반 컨텍스트 관리자"""
    
    def __init__(self, openai_client, deployment_name: str, max_context_size: int = 20):
        self.openai_client = openai_client
        self.deployment_name = deployment_name
        self.max_context_size = max_context_size
        self.contexts: Dict[str, AIConversationContext] = {}
        
    def get_or_create_context(self, session_id: str, user_id: Optional[str] = None) -> AIConversationContext:
        """세션 컨텍스트 가져오기 또는 생성"""
        if session_id not in self.contexts:
            self.contexts[session_id] = AIConversationContext(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                learned_patterns={},
                user_preferences={},
                context_insights={},
                created_at=time.time(),
                updated_at=time.time()
            )
            logger.info(f"새로운 AI 컨텍스트 생성: {session_id}")
        
        return self.contexts[session_id]
    
    def analyze_and_update_context(self, session_id: str, query: str, answer: str, 
                                 search_results: Optional[List[Any]] = None) -> Dict[str, Any]:
        """AI가 대화를 분석하고 컨텍스트 업데이트"""
        try:
            logger.info(f"🧠 AI 컨텍스트 분석 시작: {session_id}")
            
            context = self.get_or_create_context(session_id)
            
            # 대화 기록 추가
            conversation_entry = {
                "timestamp": time.time(),
                "query": query,
                "answer": answer,
                "query_length": len(query),
                "answer_length": len(answer),
                "search_results_count": len(search_results) if search_results else 0
            }
            context.conversation_history.append(conversation_entry)
            
            # 컨텍스트 크기 제한
            if len(context.conversation_history) > self.max_context_size:
                context.conversation_history = context.conversation_history[-self.max_context_size:]
            
            # AI 컨텍스트 분석
            context_analysis = self._ai_analyze_context(context, query, answer, search_results)
            
            # 학습된 패턴 업데이트
            self._update_learned_patterns(context, context_analysis)
            
            # 사용자 선호도 업데이트
            self._update_user_preferences(context, context_analysis)
            
            # 컨텍스트 인사이트 업데이트
            self._update_context_insights(context, context_analysis)
            
            context.updated_at = time.time()
            
            logger.info(f"✅ AI 컨텍스트 분석 완료: {session_id}")
            return context_analysis
            
        except Exception as e:
            logger.error(f"❌ AI 컨텍스트 분석 실패: {e}")
            return {}
    
    def get_context_for_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """질문에 대한 AI 컨텍스트 정보 반환"""
        context = self.get_or_create_context(session_id)
        
        # AI 컨텍스트 추천
        context_recommendations = self._ai_recommend_context(context, query)
        
        return {
            "session_id": session_id,
            "conversation_history": context.conversation_history[-5:],  # 최근 5개
            "learned_patterns": context.learned_patterns,
            "user_preferences": context.user_preferences,
            "context_insights": context.context_insights,
            "recommendations": context_recommendations,
            "context_age": time.time() - context.created_at
        }
    
    def _ai_analyze_context(self, context: AIConversationContext, query: str, 
                          answer: str, search_results: Optional[List[Any]]) -> Dict[str, Any]:
        """AI가 대화 컨텍스트를 분석"""
        try:
            # 대화 히스토리 요약
            history_summary = self._summarize_conversation_history(context.conversation_history)
            
            # AI 분석 프롬프트
            analysis_prompt = f"""
다음 대화 컨텍스트를 분석하여 사용자의 패턴과 선호도를 파악하세요.

현재 질문: {query}
현재 답변: {answer[:200]}...
검색 결과 수: {len(search_results) if search_results else 0}

대화 히스토리 요약:
{history_summary}

기존 학습된 패턴:
{json.dumps(context.learned_patterns, ensure_ascii=False, indent=2)}

기존 사용자 선호도:
{json.dumps(context.user_preferences, ensure_ascii=False, indent=2)}

분석해야 할 요소들:
1. 질문 유형 패턴 (정보 요청, 분석, 설명, 디버깅 등)
2. 선호하는 답변 스타일 (간결함, 상세함, 기술적, 비기술적)
3. 관심 있는 주제나 도메인
4. 질문의 복잡도 변화 추이
5. 만족도 지표 (답변 길이, 검색 결과 활용도)
6. 대화 흐름 패턴

JSON 형식으로 응답:
{{
    "question_patterns": {{
        "common_types": ["information", "analysis"],
        "complexity_trend": "increasing|decreasing|stable",
        "domain_focus": ["backend", "frontend", "database"]
    }},
    "preference_insights": {{
        "answer_style": "concise|detailed|technical|casual",
        "detail_level": "high|medium|low",
        "preferred_tools": ["search_symbols_fts", "analyze_symbol_llm"]
    }},
    "conversation_flow": {{
        "follow_up_likelihood": 0.8,
        "topic_consistency": 0.7,
        "complexity_progression": "linear|exponential|random"
    }},
    "satisfaction_indicators": {{
        "answer_length_preference": "short|medium|long",
        "search_result_utilization": 0.6,
        "technical_depth_preference": "high|medium|low"
    }},
    "recommendations": [
        "사용자가 상세한 설명을 선호하므로 답변을 더 구체적으로 제공",
        "백엔드 관련 질문이 많으므로 관련 도구 우선 사용"
    ]
}}
"""
            
            # LLM 분석 실행
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 대화 컨텍스트 분석 전문가입니다. 사용자의 패턴과 선호도를 정확히 파악하세요."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            llm_response = response.choices[0].message.content
            analysis_data = self._parse_context_analysis(llm_response)
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"AI 컨텍스트 분석 실패: {e}")
            return {}
    
    def _summarize_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """대화 히스토리 요약"""
        if not history:
            return "대화 히스토리가 없습니다."
        
        recent_queries = [entry["query"] for entry in history[-5:]]
        avg_query_length = sum(entry["query_length"] for entry in history) / len(history)
        avg_answer_length = sum(entry["answer_length"] for entry in history) / len(history)
        
        return f"""
최근 질문들: {', '.join(recent_queries)}
평균 질문 길이: {avg_query_length:.1f}자
평균 답변 길이: {avg_answer_length:.1f}자
총 대화 수: {len(history)}개
"""
    
    def _parse_context_analysis(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 컨텍스트 분석 데이터 추출"""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning("컨텍스트 분석 응답에서 JSON을 찾을 수 없음")
                return {}
                
        except Exception as e:
            logger.error(f"컨텍스트 분석 파싱 실패: {e}")
            return {}
    
    def _update_learned_patterns(self, context: AIConversationContext, analysis: Dict[str, Any]):
        """학습된 패턴 업데이트"""
        if "question_patterns" in analysis:
            patterns = analysis["question_patterns"]
            
            # 질문 유형 패턴 업데이트
            if "common_types" in patterns:
                if "question_types" not in context.learned_patterns:
                    context.learned_patterns["question_types"] = {}
                
                for qtype in patterns["common_types"]:
                    context.learned_patterns["question_types"][qtype] = \
                        context.learned_patterns["question_types"].get(qtype, 0) + 1
            
            # 복잡도 추이 업데이트
            if "complexity_trend" in patterns:
                context.learned_patterns["complexity_trend"] = patterns["complexity_trend"]
            
            # 도메인 포커스 업데이트
            if "domain_focus" in patterns:
                if "domains" not in context.learned_patterns:
                    context.learned_patterns["domains"] = {}
                
                for domain in patterns["domain_focus"]:
                    context.learned_patterns["domains"][domain] = \
                        context.learned_patterns["domains"].get(domain, 0) + 1
    
    def _update_user_preferences(self, context: AIConversationContext, analysis: Dict[str, Any]):
        """사용자 선호도 업데이트"""
        if "preference_insights" in analysis:
            preferences = analysis["preference_insights"]
            
            # 답변 스타일 선호도
            if "answer_style" in preferences:
                context.user_preferences["answer_style"] = preferences["answer_style"]
            
            # 상세도 선호도
            if "detail_level" in preferences:
                context.user_preferences["detail_level"] = preferences["detail_level"]
            
            # 선호하는 도구들
            if "preferred_tools" in preferences:
                if "tools" not in context.user_preferences:
                    context.user_preferences["tools"] = []
                
                for tool in preferences["preferred_tools"]:
                    if tool not in context.user_preferences["tools"]:
                        context.user_preferences["tools"].append(tool)
    
    def _update_context_insights(self, context: AIConversationContext, analysis: Dict[str, Any]):
        """컨텍스트 인사이트 업데이트"""
        if "recommendations" in analysis:
            context.context_insights["recommendations"] = analysis["recommendations"]
        
        if "conversation_flow" in analysis:
            context.context_insights["conversation_flow"] = analysis["conversation_flow"]
        
        if "satisfaction_indicators" in analysis:
            context.context_insights["satisfaction_indicators"] = analysis["satisfaction_indicators"]
    
    def _ai_recommend_context(self, context: AIConversationContext, query: str) -> Dict[str, Any]:
        """AI가 질문에 대한 컨텍스트 추천"""
        try:
            # AI 추천 프롬프트
            recommendation_prompt = f"""
다음 질문에 대해 기존 대화 컨텍스트를 바탕으로 추천사항을 제공하세요.

현재 질문: {query}

학습된 패턴:
{json.dumps(context.learned_patterns, ensure_ascii=False, indent=2)}

사용자 선호도:
{json.dumps(context.user_preferences, ensure_ascii=False, indent=2)}

컨텍스트 인사이트:
{json.dumps(context.context_insights, ensure_ascii=False, indent=2)}

추천해야 할 요소들:
1. 적절한 도구 선택
2. 답변 스타일 조정
3. 상세도 레벨
4. 관련 컨텍스트 활용
5. 후속 질문 제안

JSON 형식으로 응답:
{{
    "recommended_tools": ["search_symbols_fts", "analyze_symbol_llm"],
    "answer_style": "detailed",
    "detail_level": "high",
    "context_hints": ["이전에 백엔드 관련 질문을 많이 했음"],
    "follow_up_suggestions": ["관련 함수들의 호출 관계는?", "성능 최적화 방법은?"]
}}
"""
            
            # LLM 추천 실행
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 대화 컨텍스트 추천 전문가입니다. 사용자의 패턴을 바탕으로 최적의 추천을 제공하세요."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content
            return self._parse_context_analysis(llm_response)
            
        except Exception as e:
            logger.error(f"AI 컨텍스트 추천 실패: {e}")
            return {}
    
    def clear_context(self, session_id: str) -> None:
        """컨텍스트 초기화"""
        if session_id in self.contexts:
            del self.contexts[session_id]
        logger.info(f"AI 컨텍스트 초기화: {session_id}")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """컨텍스트 통계 반환"""
        total_sessions = len(self.contexts)
        active_sessions = sum(1 for ctx in self.contexts.values() 
                            if time.time() - ctx.updated_at < 3600)  # 1시간 이내
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "max_context_size": self.max_context_size
        }
