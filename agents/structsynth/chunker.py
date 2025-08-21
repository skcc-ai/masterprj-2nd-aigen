"""
Code Chunker - AST 기반 코드 청킹
AST를 기반으로 코드를 의미있는 청크로 분할
주석 결합 및 토큰 제한을 고려한 정교한 청킹
"""

import logging
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """청킹 설정"""
    max_tokens: int = 1000
    min_tokens: int = 50
    overlap_tokens: int = 100
    include_comments: bool = True
    include_docstrings: bool = True
    max_comment_length: int = 200
    preserve_structure: bool = True


class CodeChunker:
    """AST 기반 코드 청킹 (주석 결합 + 토큰 제한)"""

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        logger.info(f"CodeChunker initialized with config: {self.config}")

    def create_chunks(self, ast_result: Dict[str, Any], file_id: int) -> List[Dict[str, Any]]:
        """AST 결과를 기반으로 청크 생성"""
        chunks = []
        
        # 파일 레벨 메타데이터
        file_info = ast_result.get("file", {})
        file_comments = self._extract_file_comments(ast_result)
        
        # 심볼별 청킹
        symbols = ast_result.get("symbols", [])
        for symbol in symbols:
            symbol_chunks = self._create_symbol_chunks(symbol, file_id, file_info, file_comments)
            chunks.extend(symbol_chunks)
        
        # 파일 레벨 요약 청크
        if file_comments:
            summary_chunk = self._create_summary_chunk(file_id, file_info, file_comments)
            chunks.append(summary_chunk)
        
        logger.info(f"✅ {len(chunks)}개 청크 생성 완료 (파일: {file_id})")
        return chunks

    def _create_symbol_chunks(self, symbol: Dict[str, Any], file_id: int, 
                            file_info: Dict[str, Any], file_comments: List[str]) -> List[Dict[str, Any]]:
        """심볼별 청크 생성"""
        chunks = []
        
        # 심볼 정보 추출
        symbol_name = symbol.get("name", "unknown")
        symbol_type = symbol.get("type", "unknown")
        symbol_body = symbol.get("body", {})
        
        # 심볼 관련 주석 수집
        symbol_comments = self._extract_symbol_comments(symbol, file_comments)
        
        # 본문 내용 추출
        body_content = self._extract_body_content(symbol_body)
        
        # 주석과 본문 결합
        combined_content = self._combine_comments_and_body(symbol_comments, body_content)
        
        # 토큰 제한에 따른 청킹
        if combined_content:
            content_chunks = self._split_by_tokens(combined_content, symbol_name)
            
            for i, chunk_content in enumerate(content_chunks):
                chunk = {
                    "file_id": file_id,
                    "symbol_id": f"{file_id}_{symbol_name}_{i}",
                    "chunk_type": f"{symbol_type}_chunk",
                    "symbol_name": symbol_name,
                    "symbol_type": symbol_type,
                    "content": chunk_content,
                    "line_start": symbol.get("location", {}).get("start_line", 0),
                    "line_end": symbol.get("location", {}).get("end_line", 0),
                    "tokens": self._count_tokens(chunk_content),
                    "chunk_index": i,
                    "total_chunks": len(content_chunks),
                    "metadata": {
                        "name": symbol_name,
                        "type": symbol_type,
                        "has_comments": bool(symbol_comments),
                        "comment_count": len(symbol_comments),
                        "preserves_structure": self.config.preserve_structure
                    }
                }
                chunks.append(chunk)
        
        return chunks

    def _extract_file_comments(self, ast_result: Dict[str, Any]) -> List[str]:
        """파일 레벨 주석 추출"""
        comments = []
        
        # 파일 상단 주석
        if self.config.include_comments:
            file_comments = ast_result.get("comments", [])
            for comment in file_comments:
                comment_text = comment.get("text", "").strip()
                if comment_text and len(comment_text) <= self.config.max_comment_length:
                    comments.append(comment_text)
        
        # 파일 docstring
        if self.config.include_docstrings:
            file_docstring = ast_result.get("file", {}).get("docstring", "")
            if file_docstring:
                comments.append(f"File Docstring: {file_docstring}")
        
        return comments

    def _extract_symbol_comments(self, symbol: Dict[str, Any], file_comments: List[str]) -> List[str]:
        """심볼 관련 주석 추출"""
        comments = []
        
        # 심볼 바로 위의 주석
        symbol_comments = symbol.get("comments", [])
        for comment in symbol_comments:
            comment_text = comment.get("text", "").strip()
            if comment_text and len(comment_text) <= self.config.max_comment_length:
                comments.append(comment_text)
        
        # 심볼 docstring
        symbol_docstring = symbol.get("docstring", "")
        if symbol_docstring:
            comments.append(f"Docstring: {symbol_docstring}")
        
        return comments

    def _extract_body_content(self, body: Dict[str, Any]) -> str:
        """본문 내용 추출"""
        content_parts = []
        
        # 노드 내용
        nodes = body.get("nodes", [])
        for node in nodes:
            node_content = node.get("content", "")
            if node_content:
                content_parts.append(node_content)
        
        # 기타 내용
        other_content = body.get("content", "")
        if other_content:
            content_parts.append(other_content)
        
        return "\n".join(content_parts)

    def _combine_comments_and_body(self, comments: List[str], body_content: str) -> str:
        """주석과 본문 결합"""
        combined = []
        
        # 주석 추가
        if comments:
            combined.append("## Comments and Documentation")
            for comment in comments:
                combined.append(f"# {comment}")
            combined.append("")
        
        # 본문 추가
        if body_content:
            combined.append("## Code Implementation")
            combined.append(body_content)
        
        return "\n".join(combined)

    def _split_by_tokens(self, content: str, symbol_name: str) -> List[str]:
        """토큰 제한에 따른 내용 분할"""
        if not content:
            return []
        
        # 토큰으로 분할
        tokens = content.split()
        chunks = []
        
        if len(tokens) <= self.config.max_tokens:
            # 한 번에 처리 가능
            chunks.append(content)
        else:
            # 여러 청크로 분할
            start = 0
            while start < len(tokens):
                end = min(start + self.config.max_tokens, len(tokens))
                
                # 청크 생성
                chunk_tokens = tokens[start:end]
                chunk_content = " ".join(chunk_tokens)
                
                # 청크가 너무 작지 않도록 조정
                if len(chunk_tokens) < self.config.min_tokens and start > 0:
                    # 이전 청크와 병합
                    if chunks:
                        chunks[-1] += " " + chunk_content
                    else:
                        chunks.append(chunk_content)
                else:
                    chunks.append(chunk_content)
                
                # 오버랩 고려
                start = max(start + 1, end - self.config.overlap_tokens)
        
        logger.debug(f"📝 {symbol_name}: {len(tokens)} 토큰을 {len(chunks)}개 청크로 분할")
        return chunks

    def _count_tokens(self, content: str) -> int:
        """토큰 수 계산 (간단한 공백 기준)"""
        if not content:
            return 0
        return len(content.split())

    def _create_summary_chunk(self, file_id: int, file_info: Dict[str, Any], 
                            file_comments: List[str]) -> Dict[str, Any]:
        """파일 요약 청크 생성"""
        summary_content = "## File Summary\n"
        summary_content += f"File: {file_info.get('path', 'unknown')}\n"
        summary_content += f"Language: {file_info.get('language', 'unknown')}\n"
        summary_content += f"Total Comments: {len(file_comments)}\n\n"
        
        if file_comments:
            summary_content += "## Key Comments\n"
            for i, comment in enumerate(file_comments[:3], 1):  # 상위 3개만
                summary_content += f"{i}. {comment}\n"
        
        return {
            "file_id": file_id,
            "symbol_id": f"{file_id}_summary",
            "chunk_type": "file_summary",
            "symbol_name": "file_summary",
            "symbol_type": "summary",
            "content": summary_content,
            "line_start": 1,
            "line_end": 1,
            "tokens": self._count_tokens(summary_content),
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {
                "name": "file_summary",
                "type": "summary",
                "has_comments": bool(file_comments),
                "comment_count": len(file_comments),
                "preserves_structure": True
            }
        }

    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """청킹 통계 정보"""
        if not chunks:
            return {"error": "청크가 없습니다"}
        
        total_tokens = sum(chunk.get("tokens", 0) for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        # 청크 타입별 통계
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "unknown")
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = {"count": 0, "total_tokens": 0}
            chunk_types[chunk_type]["count"] += 1
            chunk_types[chunk_type]["total_tokens"] += chunk.get("tokens", 0)
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "average_tokens_per_chunk": round(avg_tokens, 2),
            "chunk_types": chunk_types,
            "config": {
                "max_tokens": self.config.max_tokens,
                "min_tokens": self.config.min_tokens,
                "overlap_tokens": self.config.overlap_tokens,
                "include_comments": self.config.include_comments,
                "include_docstrings": self.config.include_docstrings
            }
        } 