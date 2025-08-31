"""
StructSynth Agent - 코드 구조 분석 및 AST 추출
코드베이스의 구조적 분석을 수행하고 벡터 데이터베이스에 저장
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .parser import CodeParser
from .chunker import CodeChunker
from .llm_analyzer import LLMAnalyzer
from common.store.sqlite_store import SQLiteStore
from common.store.faiss_store import FAISSStore

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
        
        # FAISS 벡터 스토어 사용 (고성능 벡터 검색)
        self.vector_store = FAISSStore(
            index_path=str(self.artifacts_dir / "faiss.index"),
            dimension=3072 
        )
        logger.info("FAISS 벡터 스토어 초기화 완료")
        
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
            
            # 2. 파일 및 심볼 레벨 LLM 분석
            logger.info("2단계: 파일 및 심볼 레벨 LLM 분석")
            parsed_data = self._perform_file_symbol_llm_analysis(parsed_data)
            
            # 3. 코드 청킹
            logger.info("3단계: 코드 청킹")
            chunked_data = self._chunk_code(parsed_data)
            
            # 4. 청킹 단위 LLM 분석
            logger.info("4단계: 청킹 단위 LLM 분석")
            chunked_data = self._perform_chunk_llm_analysis(chunked_data)
            
            # 5. 데이터베이스 저장
            logger.info("5단계: 데이터베이스 저장")
            self._save_to_database(parsed_data, chunked_data)
            
            # 5-1. JSON 파일 자동 저장
            logger.info("5-1단계: JSON 파일 자동 저장")
            self._save_metadata_to_json(parsed_data, chunked_data)
            
            # 6. 벡터 임베딩 생성
            logger.info("6단계: 벡터 임베딩 생성")
            try:
                self._create_embeddings(chunked_data)
                logger.info("벡터 임베딩 생성 완료")
            except Exception as e:
                logger.warning(f"벡터 임베딩 생성 실패 (분석은 완료됨): {e}")
                # 벡터 생성 실패해도 분석 결과는 이미 저장됨
            
            # 실행 완료 (벡터 생성 실패 여부와 관계없이)
            self.sqlite_store.update_run_status(
                self.run_id, 
                "completed",
                finished_at=datetime.now().isoformat(),
                output_summary=f"Analysis completed: {parsed_data['total_files']} files, {parsed_data['total_symbols']} symbols (벡터 생성: {'성공' if '벡터 임베딩 생성 완료' in locals() else '실패'})"
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
                logger.warning(f"파일 청킹 실패 {file_data.get('path')}: {e}")
        
        logger.info(f"청킹 완료: {chunked_data['total_chunks']}개 청크")
        return chunked_data
    
    def _perform_file_symbol_llm_analysis(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """파일 및 심볼 레벨 LLM 분석 수행"""
        if not self.llm_analyzer:
            logger.warning("LLM Analyzer가 초기화되지 않아 파일/심볼 분석을 건너뜁니다.")
            return parsed_data
        
        try:
            analyzed_files = []
            
            for file_data in parsed_data["files"]:
                try:
                    file_path = file_data.get("file_path", "")
                    logger.info(f"파일 LLM 분석 시작: {file_path}")
                    
                    # 파일 내용 읽기 (컨텍스트용)
                    file_context = ""
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_context = f.read()
                    except Exception as e:
                        logger.warning(f"파일 읽기 실패 {file_path}: {e}")
                    
                    # 파일 레벨 LLM 분석
                    file_llm_analysis = self.llm_analyzer.analyze_file(file_data, file_context)
                    
                    # 파일 데이터에 LLM 분석 결과 추가
                    enriched_file = file_data.copy()
                    enriched_file["llm_summary"] = file_llm_analysis.get("summary", "")
                    enriched_file["llm_analysis"] = file_llm_analysis
                    
                    # 심볼별 LLM 분석
                    enriched_symbols = []
                    for symbol in file_data.get("symbols", []):
                        try:
                            symbol_type = symbol.get("type", "")
                            
                            if symbol_type == "function":
                                symbol_llm_analysis = self.llm_analyzer.analyze_function(symbol, file_context)
                            elif symbol_type == "class":
                                symbol_llm_analysis = self.llm_analyzer.analyze_class(symbol, file_context)
                            else:
                                # 변수나 다른 타입의 심볼
                                symbol_llm_analysis = {
                                    "llm_summary": f"{symbol_type} 심볼",
                                    "responsibility": "변수 또는 기타 심볼",
                                    "design_notes": "기본 심볼",
                                    "collaboration": "기본 심볼",
                                    "llm_analysis": {}
                                }
                            
                            # 심볼에 LLM 분석 결과 추가 - symbols 테이블 스키마와 일치
                            enriched_symbol = symbol.copy()
                            enriched_symbol["llm_summary"] = symbol_llm_analysis.get("llm_summary", "")
                            enriched_symbol["responsibility"] = symbol_llm_analysis.get("responsibility", "")
                            enriched_symbol["design_notes"] = symbol_llm_analysis.get("design_notes", "")
                            enriched_symbol["collaboration"] = symbol_llm_analysis.get("collaboration", "")
                            enriched_symbol["llm_analysis"] = symbol_llm_analysis.get("llm_analysis", {})
                            
                            enriched_symbols.append(enriched_symbol)
                            
                        except Exception as e:
                            logger.warning(f"심볼 LLM 분석 실패 {symbol.get('name', 'unknown')}: {e}")
                            # 분석 실패 시 원본 심볼 유지
                            symbol["llm_summary"] = "분석 실패"
                            symbol["responsibility"] = "분석 실패"
                            symbol["design_notes"] = "분석 실패"
                            symbol["collaboration"] = "분석 실패"
                            symbol["llm_analysis"] = {"error": str(e)}
                            enriched_symbols.append(symbol)
                    
                    # 파일의 심볼을 분석된 심볼로 교체
                    enriched_file["symbols"] = enriched_symbols
                    analyzed_files.append(enriched_file)
                    
                    logger.info(f"파일 LLM 분석 완료: {file_path}")
                    
                except Exception as e:
                    logger.warning(f"파일 LLM 분석 실패 {file_data.get('path', 'unknown')}: {e}")
                    # 분석 실패 시 원본 파일 데이터 유지
                    analyzed_files.append(file_data)
            
            # 분석된 파일로 업데이트
            parsed_data["files"] = analyzed_files
            
            # 전체 심볼 목록도 업데이트
            all_symbols = []
            for file_data in analyzed_files:
                all_symbols.extend(file_data.get("symbols", []))
            parsed_data["symbols"] = all_symbols
            
            logger.info(f"파일 및 심볼 LLM 분석 완료: {len(analyzed_files)}개 파일, {len(all_symbols)}개 심볼")
            return parsed_data
            
        except Exception as e:
            logger.error(f"파일 및 심볼 LLM 분석 실패: {e}")
            return parsed_data
    
    def _create_embeddings(self, chunked_data: Dict[str, Any]):
        """벡터 임베딩 생성"""
        try:
            vectors = []
            doc_ids = []
            
            for i, chunk in enumerate(chunked_data["chunks"]):
                try:
                    # 청크를 텍스트로 변환
                    chunk_text = self._chunk_to_text(chunk)
                    
                    # 임베딩 생성 (LLM Analyzer 사용)
                    if self.llm_analyzer:
                        embedding = self.llm_analyzer.create_embedding(chunk_text)
                        if embedding:
                            vectors.append(embedding)
                            # 실제 chunk ID 사용 (enumerate 인덱스가 아닌)
                            chunk_id = chunk.get("id", i + 1)  # i+1로 1부터 시작
                            doc_ids.append(chunk_id)
                            logger.debug(f"청크 {chunk_id} 임베딩 생성 완료")
                        else:
                            logger.warning(f"청크 {i} 임베딩 생성 실패")
                    else:
                        logger.warning("LLM Analyzer가 없어 임베딩을 건너뜁니다")
                        break
                        
                except Exception as e:
                    logger.warning(f"청크 {i} 임베딩 생성 실패: {e}")
                    continue
            
            # FAISS 인덱스에 벡터 추가
            if vectors:
                self.vector_store.add_vectors(vectors, doc_ids)
                self.vector_store.save_index()
                
                # SQLite embeddings 테이블에 저장
                self._save_embeddings_to_sqlite(vectors, doc_ids)
                
                logger.info(f"벡터 임베딩 생성 완료: {len(vectors)}개")
            else:
                logger.warning("생성된 임베딩이 없습니다")
            
        except Exception as e:
            logger.error(f"벡터 임베딩 생성 실패: {e}")
            raise
    
    def _chunk_to_text(self, chunk: Dict[str, Any]) -> str:
        """청크를 텍스트로 변환"""
        text_parts = []
        
        # 청크 타입별 텍스트 구성
        if chunk.get("content"):
            text_parts.append(chunk["content"])
        
        if chunk.get("llm_analysis"):
            analysis = chunk["llm_analysis"]
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    if value and str(value) != "분석 실패":
                        text_parts.append(f"{key}: {value}")
            else:
                text_parts.append(str(analysis))
        
        if chunk.get("symbol_name"):
            text_parts.append(f"Symbol: {chunk['symbol_name']}")
        
        if chunk.get("chunk_type"):
            text_parts.append(f"Type: {chunk['chunk_type']}")
        
        return " | ".join(text_parts)
    
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
                    logger.warning(f"파일 데이터 저장 실패 {file_data.get('path', 'unknown')}: {e}")
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
            # 벡터 검색 수행
            search_results = self.vector_store.search(query, top_k)
            
            # 벡터 검색 결과가 유효하지 않은 경우 (API 키 없음, 결과 없음, 유사도가 너무 낮음)
            if not search_results or all(r.get("similarity", 0) < 0.1 for r in search_results):
                logger.info(f"벡터 검색 결과가 유효하지 않습니다. SQLite에서 직접 검색을 시도합니다.")
                # 벡터 검색이 실패하면 SQLite에서 직접 검색
                search_results = self._search_symbols_in_sqlite(query, top_k)
            
            if not search_results:
                logger.info(f"쿼리 '{query}'에 대한 검색 결과가 없습니다")
                return []
            
            # 검색 결과에 실제 심볼 정보 추가
            enriched_results = []
            for result in search_results:
                doc_id = result.get("doc_id")
                if doc_id:
                    # SQLite에서 심볼 정보 조회
                    symbol_info = self.sqlite_store.get_symbol_by_id(doc_id)
                    if symbol_info:
                        enriched_result = {
                            **result,
                            "symbol_info": symbol_info
                        }
                        enriched_results.append(enriched_result)
                    else:
                        # 심볼 정보가 없어도 기본 검색 결과는 포함
                        enriched_results.append(result)
                else:
                    enriched_results.append(result)
            
            logger.info(f"쿼리 '{query}'에 대한 {len(enriched_results)}개 검색 결과 반환")
            return enriched_results
            
        except Exception as e:
            logger.error(f"심볼 검색 실패: {e}")
            # 오류 발생 시에도 SQLite 검색 시도
            try:
                return self._search_symbols_in_sqlite(query, top_k)
            except Exception as fallback_error:
                logger.error(f"SQLite fallback 검색도 실패: {fallback_error}")
                return []
    
    def _search_symbols_in_sqlite(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """SQLite에서 직접 심볼 검색 (fallback)"""
        try:
            # 쿼리 정규화
            query_lower = query.lower().strip()
            logger.info(f"SQLite에서 직접 검색 시도: '{query}' -> '{query_lower}'")
            
            matching_chunks = []
            
            with sqlite3.connect(self.sqlite_store.db_path) as conn:
                # 1. symbols 테이블에서 직접 함수명/클래스명 검색
                cursor = conn.execute("""
                    SELECT id, name, type, file_path, start_line, end_line, content
                    FROM symbols 
                    WHERE LOWER(name) LIKE ? OR LOWER(content) LIKE ?
                    ORDER BY id
                    LIMIT ?
                """, (f'%{query_lower}%', f'%{query_lower}%', top_k))
                
                symbols = cursor.fetchall()
                logger.info(f"symbols 테이블에서 {len(symbols)}개 결과 발견")
                
                for symbol in symbols:
                    symbol_id, name, symbol_type, file_path, start_line, end_line, content = symbol
                    
                    # 정확한 매칭인 경우 높은 유사도 부여
                    similarity = 1.0 if query_lower in name.lower() else 0.8
                    
                    matching_chunks.append({
                        "doc_id": symbol_id,
                        "similarity": similarity,
                        "index": len(matching_chunks),
                        "symbol_info": {
                            "name": name,
                            "type": symbol_type,
                            "file_path": file_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "content": content
                        },
                        "chunk_content": content[:200] + "..." if len(content) > 200 else content
                    })
                
                # 2. chunks 테이블에서도 검색 (더 많은 컨텍스트)
                if len(matching_chunks) < top_k:
                    cursor = conn.execute("""
                        SELECT c.id, c.symbol_id, c.chunk_type, c.content, c.symbol_id
                        FROM chunks c
                        WHERE LOWER(c.content) LIKE ? OR LOWER(c.symbol_id) LIKE ?
                        ORDER BY c.id
                        LIMIT ?
                    """, (f'%{query_lower}%', f'%{query_lower}%', top_k - len(matching_chunks)))
                    
                    chunks = cursor.fetchall()
                    logger.info(f"chunks 테이블에서 {len(chunks)}개 추가 결과 발견")
                    
                    for chunk in chunks:
                        chunk_id, symbol_id, chunk_type, content, symbol_id_str = chunk
                        
                        # symbol_id에서 실제 심볼 이름 추출 (예: "1_SimpleClass" -> "SimpleClass")
                        if '_' in symbol_id_str:
                            actual_symbol_name = symbol_id_str.split('_', 1)[1]
                        else:
                            actual_symbol_name = symbol_id_str
                        
                        # 심볼 정보 조회 시도
                        symbol_info = None
                        try:
                            # symbol_id가 숫자인 경우 직접 조회
                            if symbol_id_str.isdigit():
                                symbol_info = self.sqlite_store.get_symbol_by_id(int(symbol_id_str))
                            else:
                                # symbol_id가 문자열인 경우 이름으로 검색
                                all_symbols = self.sqlite_store.get_all_symbols()
                                for symbol in all_symbols:
                                    if symbol.get('name') == actual_symbol_name:
                                        symbol_info = symbol
                                        break
                        except:
                            pass
                        
                        # 중복 제거 (이미 symbols에서 찾은 경우)
                        if not any(chunk["symbol_info"]["name"] == actual_symbol_name for chunk in matching_chunks):
                            matching_chunks.append({
                                "doc_id": chunk_id,
                                "similarity": 0.9,  # 높은 유사도 (직접 매칭)
                                "index": len(matching_chunks),
                                "symbol_info": symbol_info or {"name": actual_symbol_name, "type": chunk_type},
                                "chunk_content": content[:200] + "..." if len(content) > 200 else content
                            })
                
                # 3. 결과 정렬 (유사도 기준)
                matching_chunks.sort(key=lambda x: x["similarity"], reverse=True)
                matching_chunks = matching_chunks[:top_k]
                
                logger.info(f"SQLite 검색 완료: 총 {len(matching_chunks)}개 결과")
                return matching_chunks
                
        except Exception as e:
            logger.error(f"SQLite 직접 검색 실패: {e}")
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
    
    def _save_embeddings_to_sqlite(self, vectors: List[List[float]], doc_ids: List[int]):
        """임베딩을 SQLite embeddings 테이블에 저장"""
        try:
            embeddings_saved = 0
            
            for i, (vector, doc_id) in enumerate(zip(vectors, doc_ids)):
                try:
                    # 벡터를 바이너리로 변환
                    import numpy as np
                    vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                    
                    # embeddings 테이블에 저장
                    embedding_id = self.sqlite_store.insert_embedding(
                        object_type="chunk",
                        object_id=doc_id,
                        vector=vector_bytes,
                        dimension=len(vector)
                    )
                    
                    if embedding_id:
                        embeddings_saved += 1
                        logger.debug(f"임베딩 {i} 저장 완료: ID {embedding_id}, 차원: {len(vector)}")
                        
                        # chunks 테이블의 embedding_id 업데이트
                        try:
                            self.sqlite_store.update_chunk_embedding(doc_id, embedding_id)
                            logger.debug(f"청크 {doc_id}의 embedding_id 업데이트 완료: {embedding_id}")
                        except Exception as e:
                            logger.warning(f"청크 {doc_id}의 embedding_id 업데이트 실패: {e}")
                    
                except Exception as e:
                    logger.warning(f"임베딩 {i} 저장 실패: {e}")
                    continue
            
            logger.info(f"SQLite embeddings 저장 완료: {embeddings_saved}개 임베딩")
            
        except Exception as e:
            logger.error(f"SQLite embeddings 저장 실패: {e}")
            # 임베딩 저장 실패해도 전체 분석은 계속 진행
    
    def _save_metadata_to_json(self, parsed_data: Dict[str, Any], chunked_data: Dict[str, Any]):
        """분석 결과를 metadata.json 파일로 자동 저장"""
        try:
            # 사용자 요청 구조에 맞춘 metadata.json 생성
            metadata = []
            
            for symbol in parsed_data.get("symbols", []):
                # 파일 경로 찾기
                file_path = "unknown"
                for file_data in parsed_data.get("files", []):
                    if symbol in file_data.get("symbols", []):
                        file_path = file_data.get("file_path", "unknown")
                        break
                
                # 위치 정보
                location = {}
                if symbol.get("location"):
                    loc = symbol["location"]
                    if loc.get("start_line") and loc.get("end_line"):
                        location = {
                            "start_line": loc["start_line"],
                            "end_line": loc["end_line"]
                        }
                
                # LLM 분석 정보
                llm_analysis = {}
                if symbol.get("llm_summary"):
                    llm_analysis["summary"] = symbol["llm_summary"]
                if symbol.get("responsibility"):
                    llm_analysis["responsibility"] = symbol["responsibility"]
                if symbol.get("design_notes"):
                    llm_analysis["design_notes"] = symbol["design_notes"]
                if symbol.get("collaboration"):
                    llm_analysis["collaboration"] = symbol["collaboration"]
                
                # 텍스트 내용
                text_content = f"Symbol: {symbol.get('name', 'unknown')} | Type: {symbol.get('type', 'unknown')} | File: {file_path}"
                
                # 심볼 데이터 구성
                symbol_data = {
                    "file_path": file_path,
                    "symbol_type": symbol.get("type", "unknown"),
                    "symbol_name": symbol.get("name", "unknown"),
                    "location": location,
                    "llm_analysis": llm_analysis,
                    "text_content": text_content,
                    "timestamp": datetime.now().isoformat()
                }
                
                metadata.append(symbol_data)
            
            # metadata.json 파일로 저장
            output_path = self.artifacts_dir / "metadata.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ metadata.json 자동 저장 완료: {output_path}")
            logger.info(f"📊 총 {len(metadata)}개 심볼 정보 저장됨")
            
        except Exception as e:
            logger.error(f"❌ metadata.json 자동 저장 실패: {e}")
            # JSON 저장 실패해도 전체 분석은 계속 진행