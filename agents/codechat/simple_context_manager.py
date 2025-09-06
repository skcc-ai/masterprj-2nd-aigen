"""
단순 대화 컨텍스트 관리 - 최근 대화 내용을 저장하고 맥락 파악에 활용
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class ConversationEntry:
    """대화 항목"""
    timestamp: float
    query: str
    answer: str
    search_results: List[Dict[str, Any]]
    query_type: str  # overview, specific
    tools_used: List[str]

@dataclass
class ConversationContext:
    """대화 컨텍스트"""
    session_id: str
    user_id: Optional[str]
    conversation_history: List[ConversationEntry]
    created_at: float
    updated_at: float

class SimpleContextManager:
    """단순 대화 컨텍스트 관리자"""
    
    def __init__(self, max_context_size: int = 10):
        self.max_context_size = max_context_size
        self.contexts: Dict[str, ConversationContext] = {}
        
    def get_or_create_context(self, session_id: str, user_id: Optional[str] = None) -> ConversationContext:
        """세션 컨텍스트 가져오기 또는 생성"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                created_at=time.time(),
                updated_at=time.time()
            )
            logger.info(f"새로운 대화 컨텍스트 생성: {session_id}")
        
        return self.contexts[session_id]
    
    def add_conversation(self, session_id: str, query: str, answer: str, 
                        search_results: List[Dict[str, Any]], query_type: str, 
                        tools_used: List[str]) -> None:
        """대화 내용 추가"""
        try:
            context = self.get_or_create_context(session_id)
            
            # 새로운 대화 항목 생성
            conversation_entry = ConversationEntry(
                timestamp=time.time(),
                query=query,
                answer=answer,
                search_results=search_results,
                query_type=query_type,
                tools_used=tools_used
            )
            
            # 대화 히스토리에 추가
            context.conversation_history.append(conversation_entry)
            
            # 컨텍스트 크기 제한
            if len(context.conversation_history) > self.max_context_size:
                context.conversation_history = context.conversation_history[-self.max_context_size:]
            
            context.updated_at = time.time()
            
            logger.info(f"대화 내용 추가: {session_id} - {len(context.conversation_history)}개 대화")
            
        except Exception as e:
            logger.error(f"대화 내용 추가 실패: {e}")
    
    def get_context_for_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """질문에 대한 컨텍스트 정보 반환"""
        context = self.get_or_create_context(session_id)
        
        # 최근 대화 내용 정리
        recent_conversations = []
        for entry in context.conversation_history[-5:]:  # 최근 5개 대화
            recent_conversations.append({
                "query": entry.query,
                "answer": entry.answer[:200] + "..." if len(entry.answer) > 200 else entry.answer,
                "query_type": entry.query_type,
                "tools_used": entry.tools_used,
                "timestamp": entry.timestamp
            })
        
        # 검색된 심볼/파일 정보 추출
        mentioned_symbols = []
        mentioned_files = []
        
        for entry in context.conversation_history[-3:]:  # 최근 3개 대화에서
            for result in entry.search_results:
                if isinstance(result, dict):
                    if "symbol_name" in result and result["symbol_name"]:
                        mentioned_symbols.append(result["symbol_name"])
                    if "file_path" in result and result["file_path"]:
                        mentioned_files.append(result["file_path"])
        
        # 중복 제거
        mentioned_symbols = list(set(mentioned_symbols))
        mentioned_files = list(set(mentioned_files))
        
        return {
            "session_id": session_id,
            "recent_conversations": recent_conversations,
            "mentioned_symbols": mentioned_symbols,
            "mentioned_files": mentioned_files,
            "total_conversations": len(context.conversation_history),
            "context_age": time.time() - context.created_at
        }
    
    def get_conversation_summary(self, session_id: str) -> str:
        """대화 요약 생성"""
        context = self.get_or_create_context(session_id)
        
        if not context.conversation_history:
            return "이전 대화가 없습니다."
        
        summary_parts = []
        summary_parts.append(f"총 {len(context.conversation_history)}개의 대화가 있었습니다.")
        
        # 최근 질문들
        recent_queries = [entry.query for entry in context.conversation_history[-3:]]
        summary_parts.append(f"최근 질문들: {', '.join(recent_queries)}")
        
        # 자주 언급된 심볼들
        all_symbols = []
        for entry in context.conversation_history:
            for result in entry.search_results:
                if isinstance(result, dict) and "symbol_name" in result:
                    all_symbols.append(result["symbol_name"])
        
        if all_symbols:
            from collections import Counter
            symbol_counts = Counter(all_symbols)
            common_symbols = [symbol for symbol, count in symbol_counts.most_common(3)]
            summary_parts.append(f"자주 언급된 심볼들: {', '.join(common_symbols)}")
        
        return "\n".join(summary_parts)
    
    def clear_context(self, session_id: str) -> None:
        """컨텍스트 초기화"""
        if session_id in self.contexts:
            del self.contexts[session_id]
        logger.info(f"대화 컨텍스트 초기화: {session_id}")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """컨텍스트 통계 반환"""
        total_sessions = len(self.contexts)
        active_sessions = sum(1 for ctx in self.contexts.values() 
                            if time.time() - ctx.updated_at < 3600)  # 1시간 이내
        
        total_conversations = sum(len(ctx.conversation_history) for ctx in self.contexts.values())
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_conversations": total_conversations,
            "max_context_size": self.max_context_size
        }
