"""
StructSynth Agent - 코드 구조 분석 및 AST 추출
코드베이스의 구조적 분석을 수행하고 벡터 데이터베이스에 저장
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .parser import CodeParser
from .chunker import CodeChunker
from .vector_store import VectorStore
from .llm_analyzer import LLMAnalyzer
from common.store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

class StructSynthAgent:
    """코드 구조 분석 및 AST 추출 에이전트"""
    
    def __init__(self, repo_path: str, artifacts_dir: str = "./artifacts", data_dir: str = "./data"):
        """
        StructSynthAgent 초기화
        
        Args:
            repo_path: 분석할 저장소 경로
            artifacts_dir: 결과물 저장 디렉토리
            data_dir: 데이터 저장 디렉토리
        """
        self.repo_path = Path(repo_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        
        # 디렉토리 생성
        self.artifacts_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # 컴포넌트 초기화
        self.parser = CodeParser()
        self.chunker = CodeChunker()
        self.vector_store = VectorStore(storage_dir=self.artifacts_dir)
        
        # LLM Analyzer 초기화 (환경변수 문제 시 None으로 설정)
        try:
            self.llm_analyzer = LLMAnalyzer()
            logger.info("LLM Analyzer 초기화 완료")
        except Exception as e:
            logger.warning(f"LLM Analyzer 초기화 실패 (LLM 기능 비활성화): {e}")
            self.llm_analyzer = None
        
        # 데이터베이스 초기화
        self.sqlite_store = SQLiteStore(self.data_dir / "structsynth_code.db")
        
        # 분석 결과 저장
        self.analysis_results = {}
        self.run_id = None
        
        logger.info(f"StructSynthAgent 초기화 완료: {self.repo_path}")
    
    def analyze_repository(self) -> Dict[str, Any]:
        """
        저장소 전체 분석 수행
        
        Returns:
            분석 결과 딕셔너리
        """
        logger.info(f"저장소 분석 시작: {self.repo_path}")
        
        try:
            # 실행 세션 시작
            self.run_id = self.sqlite_store.insert_run(
                agent_name="StructSynth",
                input_summary=f"Repository: {self.repo_path}",
                metadata={
                    "repo_path": str(self.repo_path),
                    "start_time": datetime.now().isoformat(),
                    "status": "running"
                }
            )
            
            # 1. 코드 파싱 및 AST 추출
            logger.info("1단계: 코드 파싱 및 AST 추출")
            parsed_data = self._parse_repository()
            
            # 2. 코드 청킹
            logger.info("2단계: 코드 청킹")
            chunked_data = self._chunk_code(parsed_data)
            
            # 3. 청킹 단위 LLM 분석
            logger.info("3단계: 청킹 단위 LLM 분석")
            chunked_data = self._perform_chunk_llm_analysis(chunked_data)
            
            # 4. 벡터 임베딩 생성 (LLM 분석된 청크들)
            logger.info("4단계: 벡터 임베딩 생성")
            self._create_embeddings(chunked_data)
            
            # 5. 데이터베이스 저장
            logger.info("5단계: 데이터베이스 저장")
            self._save_to_database(parsed_data, chunked_data)
            
            # 실행 완료
            self.sqlite_store.update_run_status(
                self.run_id, 
                "completed",
                finished_at=datetime.now().isoformat(),
                output_summary=f"Analysis completed: {parsed_data['total_files']} files, {parsed_data['total_symbols']} symbols"
            )
            
            # 결과 정리
            self.analysis_results = {
                "parsed_data": parsed_data,
                "chunked_data": chunked_data,
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("저장소 분석 완료")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"저장소 분석 실패: {e}")
            if self.run_id:
                self.sqlite_store.update_run_status(
                    self.run_id, 
                    "failed", 
                    finished_at=datetime.now().isoformat(),
                    output_summary=f"Analysis failed: {str(e)}"
                )
            raise
    
    def _parse_repository(self) -> Dict[str, Any]:
        """저장소 전체 파싱"""
        parsed_data = {
            "files": [],
            "symbols": [],
            "total_files": 0,
            "total_symbols": 0
        }
        
        # 지원하는 파일 확장자
        supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go'}
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in supported_extensions:
                try:
                    file_data = self.parser.parse_file(file_path)
                    if file_data:
                        parsed_data["files"].append(file_data)
                        parsed_data["symbols"].extend(file_data.get("symbols", []))
                        parsed_data["total_files"] += 1
                        parsed_data["total_symbols"] += len(file_data.get("symbols", []))
                        
                except Exception as e:
                    logger.warning(f"파일 파싱 실패 {file_path}: {e}")
        
        logger.info(f"파싱 완료: {parsed_data['total_files']}개 파일, {parsed_data['total_symbols']}개 심볼")
        return parsed_data
    
    def _chunk_code(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """코드 청킹 수행"""
        chunked_data = {
            "chunks": [],
            "total_chunks": 0
        }
        
        for file_data in parsed_data["files"]:
            try:
                file_chunks = self.chunker.chunk_file(file_data)
                chunked_data["chunks"].extend(file_chunks)
                chunked_data["total_chunks"] += len(file_chunks)
                
            except Exception as e:
                logger.warning(f"파일 청킹 실패 {file_data.get('file_path')}: {e}")
        
        logger.info(f"청킹 완료: {chunked_data['total_chunks']}개 청크")
        return chunked_data
    
    def _create_embeddings(self, chunked_data: Dict[str, Any]):
        """벡터 임베딩 생성"""
        try:
            for chunk in chunked_data["chunks"]:
                self.vector_store.add_chunk(chunk)
            
            self.vector_store.save()
            logger.info("벡터 임베딩 생성 완료")
            
        except Exception as e:
            logger.error(f"벡터 임베딩 생성 실패: {e}")
            raise
    
    def _perform_chunk_llm_analysis(self, chunked_data: Dict[str, Any]) -> Dict[str, Any]:
        """청킹 단위 LLM 분석 수행"""
        try:
            analyzed_chunks = []
            
            for i, chunk in enumerate(chunked_data["chunks"]):
                try:
                    # 청크별 LLM 분석
                    chunk_analysis = self.llm_analyzer.analyze_chunk(chunk)
                    
                    # 분석 결과를 청크에 추가
                    enriched_chunk = chunk.copy()
                    enriched_chunk["llm_analysis"] = chunk_analysis
                    enriched_chunk["chunk_id"] = i
                    
                    analyzed_chunks.append(enriched_chunk)
                    
                    logger.info(f"청크 {i} LLM 분석 완료: {chunk.get('symbol_name', 'unknown')}")
                    
                except Exception as e:
                    logger.warning(f"청크 {i} LLM 분석 실패: {e}")
                    # 분석 실패 시 원본 청크 유지
                    chunk["llm_analysis"] = {"error": str(e)}
                    analyzed_chunks.append(chunk)
            
            # 분석된 청크로 업데이트
            chunked_data["chunks"] = analyzed_chunks
            chunked_data["analyzed_chunks"] = len(analyzed_chunks)
            
            logger.info(f"청킹 단위 LLM 분석 완료: {len(analyzed_chunks)}개 청크")
            return chunked_data
            
        except Exception as e:
            logger.error(f"청킹 단위 LLM 분석 실패: {e}")
            return chunked_data
    
    def _save_to_database(self, parsed_data: Dict[str, Any], chunked_data: Dict[str, Any]):
        """SQLite 데이터베이스에 저장"""
        try:
            # 각 파일별로 AST 데이터 저장
            for file_data in parsed_data["files"]:
                try:
                    # 파일 데이터를 save_ast_data 형식에 맞게 변환
                    ast_data = {
                        "file": {
                            "path": file_data.get("file_path", ""),
                            "language": file_data.get("language", "unknown"),
                            "llm_summary": file_data.get("llm_summary")
                        },
                        "symbols": file_data.get("symbols", [])
                    }
                    
                    # AST 데이터 저장
                    self.sqlite_store.save_ast_data(ast_data)
                    
                except Exception as e:
                    logger.warning(f"파일 데이터 저장 실패 {file_data.get('file_path', 'unknown')}: {e}")
                    continue
            
            # 청크 데이터를 chunks 테이블에 저장
            chunks_saved = 0
            for chunk in chunked_data.get("chunks", []):
                try:
                    # symbol_id는 chunk에서 가져오거나 임시로 1 사용
                    symbol_id = chunk.get("symbol_id", 1)
                    chunk_type = chunk.get("chunk_type", "code")
                    content = chunk.get("content", "")
                    tokens = chunk.get("tokens", len(content.split()) if content else 0)
                    
                    # 청크 저장
                    chunk_id = self.sqlite_store.insert_chunk(
                        symbol_id=symbol_id,
                        chunk_type=chunk_type,
                        content=content,
                        tokens=tokens
                    )
                    
                    if chunk_id:
                        chunks_saved += 1
                        logger.debug(f"청크 저장 완료: ID {chunk_id}, 타입: {chunk_type}")
                    
                except Exception as e:
                    logger.warning(f"청크 저장 실패: {e}")
                    continue
            
            logger.info(f"데이터베이스 저장 완료: {chunks_saved}개 청크 저장됨")
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 실패: {e}")
            raise
    
    def _perform_llm_analysis(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용한 고급 분석"""
        try:
            # LLMAnalyzer의 통합 분석 메서드 호출
            analysis_results = self.llm_analyzer.perform_repository_analysis(parsed_data)
            
            logger.info("LLM 분석 완료")
            return analysis_results
            
        except Exception as e:
            logger.error(f"LLM 분석 실패: {e}")
            return {"error": str(e)}
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계 반환"""
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            logger.error(f"벡터 스토어 통계 조회 실패: {e}")
            return {}
    
    def search_symbols(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """심볼 검색"""
        try:
            return self.vector_store.search(query, top_k)
        except Exception as e:
            logger.error(f"심볼 검색 실패: {e}")
            return []
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        if not self.analysis_results:
            return {"status": "no_analysis_performed"}
        
        return {
            "status": "completed",
            "run_id": self.run_id,
            "files_analyzed": self.analysis_results.get("parsed_data", {}).get("total_files", 0),
            "symbols_found": self.analysis_results.get("parsed_data", {}).get("total_symbols", 0),
            "chunks_created": self.analysis_results.get("chunked_data", {}).get("total_chunks", 0),
            "timestamp": self.analysis_results.get("timestamp"),
            "vector_store_stats": self.get_vector_store_stats()
        }