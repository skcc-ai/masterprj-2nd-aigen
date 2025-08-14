"""
StructSynth Agent - Main Agent Class
코드 파싱 및 분석 준비 메인 에이전트

Input: Git 저장소 경로, 분석 옵션
Output: SQLite DB, FAISS 인덱스, AST JSON, 분석 결과 요약
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .parser import CodeParser
from .chunker import CodeChunker
from .indexer import CodeIndexer
from common.store.sqlite_store import SQLiteStore
from common.store.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class StructSynthAgent:
    """StructSynth Agent - 코드 파싱 및 분석 준비"""
    
    def __init__(self, 
                 repo_path: str,
                 artifacts_dir: str = "./artifacts",
                 data_dir: str = "./data",
                 supported_languages: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        """
        Args:
            repo_path: 분석할 Git 저장소 경로
            artifacts_dir: AST JSON 등 산출물 저장 경로
            data_dir: SQLite DB, FAISS 인덱스 저장 경로
            supported_languages: 지원 언어 목록 (기본: Python, Java, C/C++)
            exclude_patterns: 제외할 패턴 목록 (기본: .git, node_modules 등)
        """
        self.repo_path = Path(repo_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        
        # 디렉토리 생성
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.parser = CodeParser()
        self.chunker = CodeChunker()
        self.indexer = CodeIndexer()
        
        # 스토어 초기화
        self.sqlite_store = SQLiteStore(str(self.data_dir / "code.db"))
        self.faiss_store = FAISSStore(str(self.data_dir / "faiss.index"))
        
        # 지원 언어 설정
        self.supported_languages = supported_languages or {
            '.py': 'python',
            '.java': 'java', 
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.hxx': 'cpp'
        }
        
        # 제외 패턴 설정
        self.exclude_patterns = exclude_patterns or [
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'build', 'dist', 'target', 'bin', 'obj'
        ]
        
        logger.info(f"StructSynth Agent initialized for repo: {self.repo_path}")
    
    def analyze_repository(self) -> Dict[str, Any]:
        """
        저장소 전체 분석 수행
        
        Returns:
            Dict containing:
            - total_files: 총 파일 수
            - parsed_files: 파싱 성공 파일 수  
            - failed_files: 파싱 실패 파일 수
            - total_symbols: 총 심볼 수 (함수/클래스/메서드)
            - total_chunks: 총 청크 수
            - total_calls: 총 호출 관계 수
            - processing_time: 처리 시간
            - errors: 에러 목록
            - output_paths: 생성된 출력 파일 경로들
        """
        logger.info("Starting repository analysis...")
        
        start_time = datetime.now()
        results = {
            "total_files": 0,
            "parsed_files": 0,
            "failed_files": 0,
            "total_symbols": 0,
            "total_chunks": 0,
            "total_calls": 0,
            "processing_time": 0,
            "errors": [],
            "output_paths": {}
        }
        
        try:
            # 1. 파일 스캔
            files_to_analyze = self._scan_files()
            results["total_files"] = len(files_to_analyze)
            
            logger.info(f"Found {len(files_to_analyze)} files to analyze")
            
            # 2. 파일별 파싱 및 분석
            for file_path in files_to_analyze:
                try:
                    file_result = self._analyze_single_file(file_path)
                    results["parsed_files"] += 1
                    results["total_symbols"] += file_result.get("symbols_count", 0)
                    results["total_chunks"] += file_result.get("chunks_count", 0)
                    results["total_calls"] += file_result.get("calls_count", 0)
                    
                except Exception as e:
                    error_msg = f"Failed to analyze {file_path}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["failed_files"] += 1
            
            # 3. 인덱스 저장
            self.faiss_store.save_index()
            
            # 4. 결과 저장 및 출력 경로 정리
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time
            
            # 출력 경로 정리
            results["output_paths"] = {
                "sqlite_db": str(self.data_dir / "code.db"),
                "faiss_index": str(self.data_dir / "faiss.index"),
                "ast_dir": str(self.artifacts_dir / "ast"),
                "analysis_results": str(self.artifacts_dir / "analysis_results.json")
            }
            
            # 결과를 artifacts에 저장
            self._save_analysis_results(results)
            
            logger.info(f"Repository analysis completed in {processing_time:.2f}s")
            logger.info(f"Parsed: {results['parsed_files']}, Failed: {results['failed_files']}")
            logger.info(f"Symbols: {results['total_symbols']}, Chunks: {results['total_chunks']}, Calls: {results['total_calls']}")
            
        except Exception as e:
            error_msg = f"Repository analysis failed: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        return results
    
    def _scan_files(self) -> List[Path]:
        """분석할 파일 목록 스캔"""
        files = []
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and self._should_analyze_file(file_path):
                files.append(file_path)
        
        return files
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """파일 분석 여부 판단"""
        # 지원 언어 체크
        if file_path.suffix not in self.supported_languages:
            return False
        
        # 제외 패턴 체크
        file_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return False
        
        # 숨김 파일 제외
        if file_path.name.startswith('.'):
            return False
        
        return True
    
    def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """단일 파일 분석"""
        logger.info(f"Analyzing file: {file_path}")
        
        # 1. 파일 정보 저장
        file_id = self.sqlite_store.insert_file(
            repo=str(self.repo_path),
            commit_sha="HEAD",  # TODO: Git 커밋 해시 가져오기
            path=str(file_path.relative_to(self.repo_path)),
            lang=self.supported_languages[file_path.suffix],
            sha256=self._calculate_file_hash(file_path),
            lines=self._count_lines(file_path)
        )
        
        # 2. AST 파싱
        ast_result = self.parser.parse_file(file_path)
        
        # 3. 청크 생성
        chunks = self.chunker.create_chunks(ast_result, file_id)
        
        # 4. 임베딩 및 인덱싱
        chunk_ids = []
        vectors = []
        
        for chunk in chunks:
            # SQLite에 청크 저장
            chunk_id = self.sqlite_store.insert_chunk(
                file_id=file_id,
                symbol_id=chunk.get("symbol_id"),
                content=chunk["content"],
                tokens=chunk["tokens"]
            )
            chunk_ids.append(chunk_id)
            
            # 임베딩 생성
            embedding = self.indexer.create_embedding(chunk["content"])
            if embedding:
                vectors.append(embedding)
            else:
                # 임베딩 실패 시 더미 벡터
                vectors.append([0.0] * 1536)
        
        # FAISS에 벡터 추가
        if vectors:
            self.faiss_store.add_vectors(vectors, chunk_ids)
        
        # 5. 심볼 정보 저장
        symbols_count = 0
        symbols_map = {}  # name -> symbol_id 매핑
        
        for node in ast_result.get("nodes", []):
            if node.get("type") in ["function", "class", "method"]:
                symbol_id = self.sqlite_store.insert_symbol(
                    file_id=file_id,
                    name=node.get("name", ""),
                    kind=node.get("type", ""),
                    signature=node.get("signature", ""),
                    line_start=node.get("line_start", 0),
                    line_end=node.get("line_end", 0)
                )
                symbols_count += 1
                symbols_map[node.get("name", "")] = symbol_id
        
        # 6. 호출 관계 저장
        calls_count = 0
        for call in ast_result.get("calls", []):
            caller_name = call.get("caller_function", "")
            callee_name = call.get("callee_function", "")
            
            if caller_name in symbols_map and callee_name in symbols_map:
                self.sqlite_store.insert_call(
                    symbols_map[caller_name], 
                    symbols_map[callee_name]
                )
                calls_count += 1
        
        # 7. AST 결과를 artifacts에 저장
        ast_file_path = self.artifacts_dir / "ast" / f"{file_path.stem}_{file_id}.json"
        ast_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(ast_file_path, 'w', encoding='utf-8') as f:
            json.dump(ast_result, f, indent=2, ensure_ascii=False)
        
        return {
            "file_id": file_id,
            "symbols_count": symbols_count,
            "chunks_count": len(chunks),
            "calls_count": calls_count,
            "ast_file": str(ast_file_path)
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        import hashlib
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _count_lines(self, file_path: Path) -> int:
        """파일 라인 수 계산"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """분석 결과를 artifacts에 저장"""
        import json
        
        results_file = self.artifacts_dir / "analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis results saved to {results_file}")
    
    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        return {
            "repo_path": str(self.repo_path),
            "artifacts_dir": str(self.artifacts_dir),
            "data_dir": str(self.data_dir),
            "supported_languages": list(self.supported_languages.values()),
            "exclude_patterns": self.exclude_patterns,
            "sqlite_stats": self.sqlite_store.get_stats(),
            "faiss_stats": self.faiss_store.get_stats()
        }
    
    def close(self):
        """에이전트 정리"""
        self.faiss_store.close()
        logger.info("StructSynth Agent closed") 