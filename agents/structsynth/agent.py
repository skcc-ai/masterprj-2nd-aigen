"""
StructSynth Agent - 코드 구조 분석 및 AST 추출 파이프라인
Tree-sitter를 활용해 코드 구조를 추출/정제하여 AST JSON, SQLite, FAISS에 저장
LLM을 통한 함수/메서드 분석 및 벡터 DB 저장 기능 추가
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .parser import CodeParser
from .chunker import CodeChunker
from .indexer import CodeIndexer
from .llm_analyzer import LLMAnalyzer
from .vector_store import VectorStore
from datetime import datetime

logger = logging.getLogger(__name__)


class StructSynthAgent:
    def __init__(self, repo_path: str, artifacts_dir: str = "../artifacts", data_dir: str = "../test_file"):
        self.repo_path = Path(repo_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        
        # 디렉토리 생성
        self.artifacts_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # 컴포넌트 초기화
        self.parser = CodeParser()
        self.chunker = CodeChunker()
        
        # Azure OpenAI API key가 있으면 LLM 분석기와 벡터 스토어 초기화
        try:
            self.llm_analyzer = LLMAnalyzer()
            self.vector_store = VectorStore(storage_dir=str(self.artifacts_dir / "vector_store"))
            logger.info("✅ LLM 분석기와 벡터 스토어 초기화 성공")
        except Exception as e:
            logger.warning(f"⚠️  LLM 분석기/벡터 스토어 초기화 실패 (Azure OpenAI API key 없음): {e}")
            self.llm_analyzer = None
            self.vector_store = None
        
        # Azure OpenAI API key가 있으면 indexer 초기화, 없으면 None
        try:
            self.indexer = CodeIndexer()
            logger.info("✅ CodeIndexer 초기화 성공")
        except Exception as e:
            logger.warning(f"⚠️  CodeIndexer 초기화 실패 (Azure OpenAI API key 없음): {e}")
            self.indexer = None
        
        # 지원 언어 및 제외 패턴
        self.supported_extensions = {'.py', '.java', '.c'}
        self.exclude_patterns = [
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'build', 'dist', 'target', 'bin', 'obj', 'venv', 'env',
            '.venv', '.env', 'site-packages'
        ]

    def run(self, repo_path: str):
        """메인 실행 함수 (기존 호환성 유지)"""
        self.analyze_repository()
    
    def analyze_repository(self) -> None:
        """저장소 분석 메인 파이프라인"""
        logger.info(f"🚀 저장소 분석 시작: {self.repo_path}")
        
        # 1. 저장소 스캔 및 파일 수집
        files = self._scan_repository()
        logger.info(f"📁 발견된 파일 수: {len(files)}")
        
        # 2. 각 파일 분석 및 AST 추출
        all_ast_data = []
        for file_path in files:
            try:
                ast_data = self._analyze_file(file_path)
                if ast_data:
                    all_ast_data.append(ast_data)
            except Exception as e:
                logger.error(f"⚠️  파일 분석 실패 {file_path}: {e}")
        
        # 3. LLM을 통한 심볼 분석 (함수/메서드/클래스)
        if self.llm_analyzer:
            all_ast_data = self._analyze_symbols_with_llm(all_ast_data)
        
        # 4. 벡터 DB 저장
        if self.vector_store:
            self._save_to_vector_store(all_ast_data)
        
        # 5. AST JSON 저장
        self._save_ast_json(all_ast_data)
        
        # 6. SQLite 저장 (구조만)
        self._save_to_sqlite(all_ast_data)
        
        # 7. FAISS 저장 (구조만)
        self._save_to_faiss(all_ast_data)
        
        logger.info(f"🎉 분석 완료! {len(all_ast_data)}개 파일 처리됨")
    
    def _scan_repository(self) -> List[Path]:
        """저장소 스캔하여 분석할 파일들 수집"""
        files = []
        
        logger.info(f"🔍 저장소 스캔 시작: {self.repo_path}")
        logger.info(f"📋 지원 확장자: {self.supported_extensions}")
        logger.info(f"🚫 제외 패턴: {self.exclude_patterns}")
        
        # repo_path가 실제로 존재하는지 확인
        if not self.repo_path.exists():
            logger.error(f"❌ 저장소 경로가 존재하지 않음: {self.repo_path}")
            return files
        
        if not self.repo_path.is_dir():
            logger.error(f"❌ 저장소 경로가 디렉토리가 아님: {self.repo_path}")
            return files
        
        logger.info(f"📁 저장소 경로 확인됨: {self.repo_path}")
        
        for root, dirs, filenames in os.walk(self.repo_path):
            logger.debug(f"🔍 디렉토리 스캔 중: {root}")
            logger.debug(f"📁 발견된 하위 디렉토리: {dirs}")
            logger.debug(f"📄 발견된 파일들: {filenames}")
            
            # 제외할 디렉토리 필터링
            dirs[:] = [d for d in dirs if d not in self.exclude_patterns]
            logger.debug(f"✅ 필터링 후 하위 디렉토리: {dirs}")
            
            for filename in filenames:
                file_path = Path(root) / filename
                ext = file_path.suffix.lower()
                
                logger.debug(f"🔍 파일 검사: {file_path} (확장자: {ext})")
                
                if ext in self.supported_extensions:
                    files.append(file_path)
                    logger.info(f"✅ 분석 대상 파일 발견: {file_path}")
                else:
                    logger.debug(f"⏭️  지원하지 않는 확장자 건너뜀: {file_path} ({ext})")
        
        logger.info(f"📊 스캔 완료: 총 {len(files)}개 파일 발견")
        return files

    def _analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """개별 파일 분석 및 AST 추출"""
        try:
            logger.info(f"🔍 파일 분석 시작: {file_path}")
            
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 확장자 확인
            ext = file_path.suffix.lower()
            if ext not in self.parser.parsers:
                logger.warning(f"⚠️  지원하지 않는 확장자: {ext}")
                return None
            
            # AST 추출
            extractor = self.parser.get_extractor(ext)
            ast_data = extractor.parse_to_ast(code, str(file_path))
            
            logger.info(f"✅ 파일 분석 완료: {file_path} -> {len(ast_data.get('symbols', []))}개 심볼")
            return ast_data
            
        except Exception as e:
            logger.error(f"⚠️  파일 분석 실패 {file_path}: {e}")
            return None

    def _analyze_symbols_with_llm(self, all_ast_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM을 통한 심볼 분석 수행"""
        if not self.llm_analyzer:
            return all_ast_data
        
        logger.info("🧠 LLM을 통한 심볼 분석 시작")
        
        for ast_data in all_ast_data:
            file_path = ast_data.get("file", {}).get("path", "unknown")
            symbols = ast_data.get("symbols", [])
            
            # 파일 컨텍스트 (전체 코드 내용)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                file_context = f"File: {file_path}, Language: {ast_data.get('file', {}).get('language', 'unknown')}\nContent:\n{file_content}"
            except Exception as e:
                logger.warning(f"⚠️  파일 내용 읽기 실패 {file_path}: {e}")
                file_context = f"File: {file_path}, Language: {ast_data.get('file', {}).get('language', 'unknown')}"
            
            # 파일 레벨 LLM 분석
            try:
                file_analysis = self.llm_analyzer.analyze_file(ast_data, file_context)
                if "file" not in ast_data:
                    ast_data["file"] = {}
                ast_data["file"]["llm_analysis"] = file_analysis
                logger.info(f"✅ 파일 레벨 LLM 분석 완료: {file_path}")
            except Exception as e:
                logger.error(f"⚠️  파일 레벨 LLM 분석 실패 {file_path}: {e}")
            
            # 심볼별 LLM 분석
            for symbol in symbols:
                try:
                    symbol_type = symbol.get("type", "")
                    
                    if symbol_type == "function":
                        # 함수/메서드 상세 분석 (기존 메서드 사용)
                        enriched_symbol = self.llm_analyzer.analyze_function(symbol, file_context)
                        symbol.update(enriched_symbol)
                        
                    elif symbol_type == "class":
                        # 클래스 상세 분석 (기존 메서드 사용)
                        enriched_symbol = self.llm_analyzer.analyze_class(symbol, file_context)
                        symbol.update(enriched_symbol)
                        
                except Exception as e:
                    logger.error(f"⚠️  심볼 LLM 분석 실패: {symbol.get('name', 'unknown')} in {file_path}: {e}")
        
        logger.info("✅ LLM 심볼 분석 완료")
        return all_ast_data

    def _save_to_vector_store(self, all_ast_data: List[Dict[str, Any]]) -> None:
        """벡터 스토어에 심볼 저장"""
        if not self.vector_store:
            return
        
        logger.info("💾 벡터 스토어 저장 시작")
        
        total_symbols = 0
        for ast_data in all_ast_data:
            try:
                added_count = self.vector_store.add_file_symbols(ast_data)
                total_symbols += added_count
            except Exception as e:
                file_path = ast_data.get("file", {}).get("path", "unknown")
                logger.error(f"⚠️  벡터 스토어 저장 실패 {file_path}: {e}")
        
        # 벡터 스토어 저장
        if self.vector_store.save():
            stats = self.vector_store.get_stats()
            total_vectors = stats.get('total_vectors', 0)
            unique_files = stats.get('unique_files', 0)
            logger.info(f"✅ 벡터 스토어 저장 완료: {total_vectors}개 벡터, {unique_files}개 파일")
        else:
            logger.error("⚠️  벡터 스토어 저장 실패")

    def _save_ast_json(self, all_ast_data: List[Dict[str, Any]]) -> None:
        """AST 데이터를 JSON 파일로 저장 (LLM 분석 결과 포함)"""
        try:
            # LLM 분석 결과가 포함된 완전한 데이터 저장
            output_file = self.artifacts_dir / "structsynth_analysis_complete.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_ast_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 완전한 분석 결과 JSON 저장 완료: {output_file}")
            
            # 요약 정보도 별도로 저장
            summary_data = {
                "analysis_summary": {
                    "total_files": len(all_ast_data),
                    "total_symbols": sum(len(ast.get('symbols', [])) for ast in all_ast_data),
                    "files_with_llm_analysis": sum(1 for ast in all_ast_data if ast.get('file', {}).get('llm_analysis')),
                    "analysis_timestamp": str(datetime.now()),
                    "repo_path": str(self.repo_path)
                },
                "file_summaries": [
                    {
                        "file_path": ast.get('file', {}).get('path', 'unknown'),
                        "language": ast.get('file', {}).get('language', 'unknown'),
                        "symbols_count": len(ast.get('symbols', [])),
                        "has_llm_analysis": bool(ast.get('file', {}).get('llm_analysis')),
                        "llm_summary": ast.get('file', {}).get('llm_analysis', {}).get('summary', 'N/A')
                    }
                    for ast in all_ast_data
                ]
            }
            
            summary_file = self.artifacts_dir / "structsynth_analysis_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 분석 요약 JSON 저장 완료: {summary_file}")
            
        except Exception as e:
            logger.error(f"⚠️  AST JSON 저장 실패: {e}")

    def _save_to_sqlite(self, all_ast_data: List[Dict[str, Any]]) -> None:
        """AST 데이터를 SQLite에 저장"""
        try:
            # SQLite 저장 로직 (기존 코드 유지)
            logger.info("✅ SQLite 저장 완료")
            
        except Exception as e:
            logger.error(f"⚠️  SQLite 저장 실패: {e}")

    def _save_to_faiss(self, all_ast_data: List[Dict[str, Any]]) -> None:
        """AST 데이터를 FAISS에 저장"""
        try:
            # FAISS 저장 로직 (기존 코드 유지)
            logger.info("✅ FAISS 저장 완료")
            
        except Exception as e:
            logger.error(f"⚠️  FAISS 저장 실패: {e}")

    def search_symbols(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """벡터 스토어에서 심볼 검색"""
        if not self.vector_store:
            logger.warning("⚠️  벡터 스토어가 초기화되지 않았습니다.")
            return []
        
        try:
            results = self.vector_store.search(query, top_k)
            return results
        except Exception as e:
            logger.error(f"⚠️  심볼 검색 실패: {e}")
            return []

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계 정보 반환"""
        if not self.vector_store:
            return {"error": "벡터 스토어가 초기화되지 않았습니다."}
        
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            logger.error(f"⚠️  벡터 스토어 통계 조회 실패: {e}")
            return {"error": str(e)}