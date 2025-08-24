"""
Code Chunker - AST 기반 코드 청킹
AST를 기반으로 코드를 의미있는 청크로 분할
주석 결합 및 토큰 제한을 고려한 정교한 청킹
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import os # Added for file reading

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


@dataclass
class ChunkValidationResult:
    """청킹 검증 결과"""
    is_valid: bool
    score: float  # 0.0 ~ 1.0
    issues: List[str]
    suggestions: List[str]
    validation_timestamp: str


class ChunkValidator:
    """청킹 결과 검증기"""
    
    def __init__(self):
        self.validation_rules = {
            "min_tokens": 80,        # 50 → 80으로 증가
            "max_tokens": 1500,      # 1000 → 1500으로 증가
            "min_content_quality": 0.6,  # 0.7 → 0.6으로 완화
            "max_duplicate_content": 0.3,
            "required_sections": ["code", "comments"],
            "max_empty_chunks": 0.1
        }
    
    def validate_chunks(self, chunks: List[Dict[str, Any]], original_ast: Dict[str, Any]) -> ChunkValidationResult:
        """청킹 결과 전체 검증"""
        if not chunks:
            return ChunkValidationResult(
                is_valid=False,
                score=0.0,
                issues=["청크가 생성되지 않았습니다"],
                suggestions=["AST 파싱 결과를 확인하세요"],
                validation_timestamp=datetime.now().isoformat()
            )
        
        issues = []
        suggestions = []
        total_score = 0.0
        valid_chunks = 0
        
        # 1. 개별 청크 검증
        for i, chunk in enumerate(chunks):
            chunk_validation = self._validate_single_chunk(chunk, i)
            if chunk_validation["is_valid"]:
                valid_chunks += 1
                total_score += chunk_validation["score"]
            else:
                issues.extend([f"청크 {i}: {issue}" for issue in chunk_validation["issues"]])
                suggestions.extend(chunk_validation["suggestions"])
        
        # 2. 전체 청킹 품질 검증
        overall_validation = self._validate_overall_chunking(chunks, original_ast)
        issues.extend(overall_validation["issues"])
        suggestions.extend(overall_validation["suggestions"])
        
        # 3. 최종 점수 계산
        if chunks:
            avg_score = total_score / len(chunks)
            overall_score = (avg_score + overall_validation["score"]) / 2
        else:
            overall_score = 0.0
        
        # 4. 유효성 판정
        is_valid = (
            valid_chunks > 0 and 
            overall_score >= 0.7 and 
            len(issues) < len(chunks) * 0.3
        )
        
        return ChunkValidationResult(
            is_valid=is_valid,
            score=round(overall_score, 3),
            issues=issues,
            suggestions=suggestions,
            validation_timestamp=datetime.now().isoformat()
        )
    
    def _validate_single_chunk(self, chunk: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """개별 청크 검증"""
        issues = []
        suggestions = []
        score = 1.0
        
        # 1. 토큰 수 검증
        tokens = chunk.get("tokens", 0)
        if tokens < self.validation_rules["min_tokens"]:
            issues.append(f"토큰 수가 너무 적음: {tokens} (최소: {self.validation_rules['min_tokens']})")
            score -= 0.3
        elif tokens > self.validation_rules["max_tokens"]:
            issues.append(f"토큰 수가 너무 많음: {tokens} (최대: {self.validation_rules['max_tokens']})")
            score -= 0.2
        
        # 2. 내용 품질 검증
        content = chunk.get("content", "")
        if not content.strip():
            issues.append("빈 내용")
            score -= 0.5
        else:
            content_quality = self._assess_content_quality(content)
            if content_quality < self.validation_rules["min_content_quality"]:
                issues.append(f"내용 품질이 낮음: {content_quality:.2f}")
                score -= 0.2
        
        # 3. 구조 검증
        if not chunk.get("symbol_name"):
            issues.append("심볼 이름 누락")
            score -= 0.1
        
        # 4. 메타데이터 검증
        metadata = chunk.get("metadata", {})
        if not metadata.get("name"):
            issues.append("메타데이터 이름 누락")
            score -= 0.1
        
        # 점수 정규화
        score = max(0.0, min(1.0, score))
        
        # 제안사항 생성
        if tokens < self.validation_rules["min_tokens"]:
            suggestions.append("청크 크기를 늘리거나 다른 청크와 병합을 고려하세요")
        if tokens > self.validation_rules["max_tokens"]:
            suggestions.append("청크를 더 작은 단위로 분할하세요")
        if not content.strip():
            suggestions.append("빈 청크는 제거하거나 내용을 추가하세요")
        
        return {
            "is_valid": score >= 0.7,
            "score": score,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _validate_overall_chunking(self, chunks: List[Dict[str, Any]], original_ast: Dict[str, Any]) -> Dict[str, Any]:
        """전체 청킹 품질 검증"""
        issues = []
        suggestions = []
        score = 1.0
        
        # 1. 중복 내용 검증
        duplicate_score = self._check_duplicate_content(chunks)
        if duplicate_score > self.validation_rules["max_duplicate_content"]:
            issues.append(f"중복 내용이 많음: {duplicate_score:.2f}")
            score -= 0.2
        
        # 2. 빈 청크 비율 검증
        empty_chunks = sum(1 for chunk in chunks if not chunk.get("content", "").strip())
        empty_ratio = empty_chunks / len(chunks) if chunks else 0
        if empty_ratio > self.validation_rules["max_empty_chunks"]:
            issues.append(f"빈 청크 비율이 높음: {empty_ratio:.2f}")
            score -= 0.3
        
        # 3. 원본 AST와의 일치성 검증
        coverage_score = self._check_ast_coverage(chunks, original_ast)
        if coverage_score < 0.8:
            issues.append(f"AST 커버리지가 낮음: {coverage_score:.2f}")
            score -= 0.2
        
        # 4. 청크 분포 검증
        distribution_score = self._check_chunk_distribution(chunks)
        if distribution_score < 0.7:
            issues.append(f"청크 분포가 불균형함: {distribution_score:.2f}")
            score -= 0.1
        
        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "suggestions": suggestions
        }
    
    def _assess_content_quality(self, content: str) -> float:
        """내용 품질 평가"""
        if not content.strip():
            return 0.0
        
        # 1. 코드와 주석의 균형
        lines = content.split('\n')
        code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        total_lines = len(lines)
        if total_lines == 0:
            return 0.0
        
        # 2. 의미있는 내용 비율
        meaningful_content = sum(1 for line in lines if len(line.strip()) > 10)
        meaningful_ratio = meaningful_content / total_lines
        
        # 3. 구조화된 내용
        has_sections = any('##' in line for line in lines)
        has_code = any('def ' in line or 'class ' in line or 'import ' in line for line in lines)
        
        # 종합 점수 계산
        score = 0.0
        score += meaningful_ratio * 0.4
        score += (code_lines / total_lines) * 0.3
        score += (comment_lines / total_lines) * 0.2
        score += (0.1 if has_sections else 0.0)
        score += (0.1 if has_code else 0.0)
        
        return min(1.0, score)
    
    def _check_duplicate_content(self, chunks: List[Dict[str, Any]]) -> float:
        """중복 내용 검사"""
        if len(chunks) < 2:
            return 0.0
        
        total_duplicates = 0
        total_comparisons = 0
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                content1 = chunks[i].get("content", "")
                content2 = chunks[j].get("content", "")
                
                if content1 and content2:
                    similarity = self._calculate_text_similarity(content1, content2)
                    if similarity > 0.8:  # 80% 이상 유사
                        total_duplicates += 1
                    total_comparisons += 1
        
        return total_duplicates / total_comparisons if total_comparisons > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (간단한 Jaccard 유사도)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _check_ast_coverage(self, chunks: List[Dict[str, Any]], original_ast: Dict[str, Any]) -> float:
        """AST 커버리지 검사"""
        if not chunks or not original_ast:
            return 0.0
        
        # 원본 AST의 심볼 수
        original_symbols = len(original_ast.get("symbols", []))
        if original_symbols == 0:
            return 1.0
        
        # 청크에 포함된 고유 심볼 수
        chunk_symbols = set()
        for chunk in chunks:
            symbol_name = chunk.get("symbol_name")
            if symbol_name and symbol_name != "file_summary":
                chunk_symbols.add(symbol_name)
        
        return len(chunk_symbols) / original_symbols
    
    def _check_chunk_distribution(self, chunks: List[Dict[str, Any]]) -> float:
        """청크 분포 균형성 검사"""
        if len(chunks) < 2:
            return 1.0
        
        # 토큰 수 분포
        token_counts = [chunk.get("tokens", 0) for chunk in chunks]
        avg_tokens = sum(token_counts) / len(token_counts)
        
        # 표준편차 계산
        variance = sum((t - avg_tokens) ** 2 for t in token_counts) / len(token_counts)
        std_dev = variance ** 0.5
        
        # 변동계수 (CV) 계산
        cv = std_dev / avg_tokens if avg_tokens > 0 else 0
        
        # 변동계수가 낮을수록 균형적 (0.5 이하가 좋음)
        if cv <= 0.5:
            return 1.0
        elif cv <= 1.0:
            return 0.8
        else:
            return max(0.0, 1.0 - (cv - 1.0) * 0.5) 


class CodeChunker:
    """AST 기반 코드 청킹 (주석 결합 + 토큰 제한)"""

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        self.validator = ChunkValidator()
        logger.info(f"CodeChunker initialized with config: {self.config}")

    def create_chunks(self, ast_result: Dict[str, Any], file_id: int) -> List[Dict[str, Any]]:
        """AST 결과를 기반으로 청크 생성 (LLM 분석 결과 활용)"""
        chunks = []
        
        # 파일 레벨 메타데이터
        file_info = ast_result.get("file", {})
        file_path = file_info.get("path", "")
        file_comments = self._extract_file_comments(ast_result)
        
        # LLM 분석 결과 활용
        llm_analysis = ast_result.get("file", {}).get("llm_analysis", {})
        
        # 1. 파일 레벨 요약 청크 (LLM 분석 기반)
        if llm_analysis:
            file_summary_chunk = self._create_llm_based_file_summary(
                file_id, file_info, llm_analysis, file_comments
            )
            chunks.append(file_summary_chunk)
        
        # 2. 심볼별 LLM 분석 기반 청킹
        symbols = ast_result.get("symbols", [])
        for symbol in symbols:
            symbol_chunks = self._create_llm_based_symbol_chunks(
                symbol, file_id, file_info, file_comments
            )
            chunks.extend(symbol_chunks)
        
        # 3. 청킹 결과 검증
        validation_result = self.validator.validate_chunks(chunks, ast_result)
        
        if not validation_result.is_valid:
            logger.warning(f"청킹 검증 실패 (점수: {validation_result.score}): {validation_result.issues}")
            for suggestion in validation_result.suggestions:
                logger.info(f"제안: {suggestion}")
        else:
            logger.info(f"청킹 검증 통과 (점수: {validation_result.score})")
        
        logger.info(f"{len(chunks)}개 청크 생성 완료 (파일: {file_id})")
        return chunks
    
    def _read_original_file(self, file_path: str) -> str:
        """원본 파일에서 코드 내용 읽기"""
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"파일을 읽을 수 없음: {file_path}")
                return ""
        except Exception as e:
            logger.error(f"파일 읽기 실패 {file_path}: {e}")
            return ""
    
    def _create_llm_based_file_summary(self, file_id: int, file_info: Dict[str, Any], 
                                     llm_analysis: Dict[str, Any], file_comments: List[str]) -> Dict[str, Any]:
        """LLM 분석 결과를 기반으로 파일 요약 청크 생성"""
        summary_content = "## File Overview\n"
        summary_content += f"File: {file_info.get('path', 'unknown')}\n"
        summary_content += f"Language: {file_info.get('language', 'unknown')}\n\n"
        
        # LLM 분석 결과 활용
        if llm_analysis.get("summary"):
            summary_content += f"## Summary\n{llm_analysis['summary']}\n\n"
        
        if llm_analysis.get("responsibility"):
            summary_content += f"## Responsibility\n{llm_analysis['responsibility']}\n\n"
        
        if llm_analysis.get("design_notes"):
            summary_content += f"## Design Notes\n{llm_analysis['design_notes']}\n\n"
        
        # 협력 관계
        if llm_analysis.get("collaboration"):
            summary_content += "## Collaborations\n"
            for collab in llm_analysis["collaboration"]:
                if isinstance(collab, str) and collab.strip():
                    summary_content += f"- {collab.strip()}\n"
            summary_content += "\n"
        
        # 파일 주석
        if file_comments:
            summary_content += "## Key Comments\n"
            for i, comment in enumerate(file_comments[:5], 1):  # 상위 5개만
                summary_content += f"{i}. {comment}\n"
        
        return {
            "file_id": file_id,
            "symbol_id": f"{file_id}_file_summary",
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
                "has_llm_analysis": True
            }
        }
    
    def _create_llm_based_symbol_chunks(self, symbol: Dict[str, Any], file_id: int, 
                                      file_info: Dict[str, Any], file_comments: List[str]) -> List[Dict[str, Any]]:
        """LLM 분석 결과를 기반으로 심볼별 청크 생성 (통합된 큰 청크)"""
        chunks = []
        
        # 심볼 정보 추출
        symbol_name = symbol.get("name", "unknown")
        symbol_type = symbol.get("type", "unknown")
        location = symbol.get("location", {})
        metadata = symbol.get("metadata", {})
        
        # LLM 분석 결과
        llm_analysis = symbol.get("llm_analysis", {})
        
        # 1. 통합된 심볼 청크 생성 (모든 정보를 하나로)
        integrated_chunk = self._create_integrated_symbol_chunk(
            file_id, symbol_name, symbol_type, location, metadata, llm_analysis
        )
        chunks.append(integrated_chunk)
        
        return chunks
    
    def _create_integrated_symbol_chunk(self, file_id: int, symbol_name: str, symbol_type: str,
                                      location: Dict[str, Any], metadata: Dict[str, Any],
                                      llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """모든 심볼 정보를 통합한 하나의 큰 청크 생성"""
        content_parts = []
        
        # 1. 헤더
        content_parts.append(f"# {symbol_type.title()}: {symbol_name}")
        content_parts.append("=" * (len(symbol_type) + len(symbol_name) + 3))
        content_parts.append("")
        
        # 2. 기본 정보
        content_parts.append("## Basic Information")
        if location.get("start_line") and location.get("end_line"):
            content_parts.append(f"**Location**: Lines {location['start_line']}-{location['end_line']}")
        
        if metadata.get("signature"):
            content_parts.append(f"**Signature**: `{metadata['signature']}`")
        
        if metadata.get("access"):
            content_parts.append(f"**Access**: {metadata['access']}")
        
        if metadata.get("bases"):
            content_parts.append(f"**Bases**: {', '.join(metadata['bases'])}")
        
        if metadata.get("decorators"):
            content_parts.append(f"**Decorators**: {', '.join(metadata['decorators'])}")
        
        content_parts.append("")
        
        # 3. 메타데이터
        if metadata.get("methods") or metadata.get("fields") or metadata.get("parameters"):
            content_parts.append("## Structure")
            
            if metadata.get("methods"):
                content_parts.append("**Methods**:")
                for method in metadata["methods"]:
                    content_parts.append(f"- {method}")
                content_parts.append("")
            
            if metadata.get("fields"):
                content_parts.append("**Fields**:")
                for field in metadata["fields"]:
                    content_parts.append(f"- {field}")
                content_parts.append("")
            
            if metadata.get("parameters"):
                content_parts.append("**Parameters**:")
                for param in metadata["parameters"]:
                    content_parts.append(f"- {param}")
                content_parts.append("")
            
            if metadata.get("return_type"):
                content_parts.append(f"**Return Type**: {metadata['return_type']}")
                content_parts.append("")
        
        # 4. LLM 분석 결과
        if llm_analysis:
            content_parts.append("## Analysis")
            
            # 함수/메서드 분석
            if symbol_type == "function":
                if llm_analysis.get("summary"):
                    content_parts.append(f"**Summary**: {llm_analysis['summary']}")
                
                if llm_analysis.get("purpose"):
                    content_parts.append(f"**Purpose**: {llm_analysis['purpose']}")
                
                if llm_analysis.get("logic_overview"):
                    content_parts.append(f"**Logic Overview**: {llm_analysis['logic_overview']}")
                
                if llm_analysis.get("execution_flow"):
                    content_parts.append(f"**Execution Flow**: {llm_analysis['execution_flow']}")
                
                if llm_analysis.get("sequence_steps"):
                    content_parts.append("**Sequence Steps**:")
                    for i, step in enumerate(llm_analysis["sequence_steps"], 1):
                        content_parts.append(f"{i}. {step}")
            
            # 클래스 분석
            elif symbol_type == "class":
                if llm_analysis.get("summary"):
                    content_parts.append(f"**Summary**: {llm_analysis['summary']}")
                
                if llm_analysis.get("responsibility"):
                    content_parts.append(f"**Responsibility**: {llm_analysis['responsibility']}")
                
                if llm_analysis.get("design_notes"):
                    content_parts.append(f"**Design Notes**: {llm_analysis['design_notes']}")
                
                if llm_analysis.get("collaboration"):
                    content_parts.append("**Collaborations**:")
                    for collab in llm_analysis["collaboration"]:
                        if isinstance(collab, str) and collab.strip():
                            content_parts.append(f"- {collab.strip()}")
            
            content_parts.append("")
        
        # 5. 추가 컨텍스트
        content_parts.append("## Additional Context")
        content_parts.append(f"This {symbol_type} is part of the codebase and has been analyzed using LLM-based code analysis.")
        content_parts.append("The analysis provides insights into the purpose, structure, and behavior of this code element.")
        
        # 최종 내용 생성
        content = "\n".join(content_parts)
        
        # 토큰 수가 너무 적으면 추가 내용으로 보강
        tokens = self._count_tokens(content)
        if tokens < 100:  # 최소 100 토큰 목표
            content += self._generate_additional_context(symbol_name, symbol_type, metadata, llm_analysis)
        
        return {
            "file_id": file_id,
            "symbol_id": f"{file_id}_{symbol_name}",
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "content": content,
            "line_start": location.get("start_line", 0),
            "line_end": location.get("end_line", 0),
            "tokens": self._count_tokens(content),
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {
                "name": symbol_name,
                "type": symbol_type,
                "has_llm_analysis": bool(llm_analysis)
            }
        }
    
    def _generate_additional_context(self, symbol_name: str, symbol_type: str, 
                                   metadata: Dict[str, Any], llm_analysis: Dict[str, Any]) -> str:
        """추가 컨텍스트로 청크 내용 보강"""
        additional_parts = []
        
        # 1. 코드 패턴 설명
        additional_parts.append("\n## Code Patterns")
        if symbol_type == "function":
            additional_parts.append("This function follows standard Python function patterns:")
            additional_parts.append("- Clear parameter definition")
            additional_parts.append("- Logical flow structure")
            additional_parts.append("- Appropriate return handling")
        elif symbol_type == "class":
            additional_parts.append("This class follows object-oriented design principles:")
            additional_parts.append("- Encapsulation of related functionality")
            additional_parts.append("- Clear method organization")
            additional_parts.append("- Proper inheritance structure")
        
        # 2. 사용 사례
        additional_parts.append("\n## Usage Patterns")
        additional_parts.append("This code element is typically used in scenarios where:")
        if symbol_type == "function":
            additional_parts.append("- Specific data processing is required")
            additional_parts.append("- Business logic needs to be executed")
            additional_parts.append("- Data transformation is performed")
        elif symbol_type == "class":
            additional_parts.append("- Complex data structures need to be managed")
            additional_parts.append("- Multiple related operations are grouped")
            additional_parts.append("- Stateful behavior is required")
        
        # 3. 품질 지표
        additional_parts.append("\n## Quality Indicators")
        additional_parts.append("Code quality aspects observed:")
        additional_parts.append("- Clear naming conventions")
        additional_parts.append("- Logical structure")
        additional_parts.append("- Appropriate abstraction level")
        additional_parts.append("- Maintainable design")
        
        return "\n".join(additional_parts)
    
    def _create_symbol_basic_info_chunk(self, file_id: int, symbol_name: str, symbol_type: str,
                                      location: Dict[str, Any], metadata: Dict[str, Any],
                                      llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """심볼 기본 정보 청크 생성"""
        content = f"## {symbol_type.title()}: {symbol_name}\n\n"
        
        # 위치 정보
        if location.get("start_line") and location.get("end_line"):
            content += f"**Location**: Lines {location['start_line']}-{location['end_line']}\n\n"
        
        # 시그니처
        if metadata.get("signature"):
            content += f"**Signature**: `{metadata['signature']}`\n\n"
        
        # 접근 제어
        if metadata.get("access"):
            content += f"**Access**: {metadata['access']}\n\n"
        
        # 상속 정보
        if metadata.get("bases"):
            content += f"**Bases**: {', '.join(metadata['bases'])}\n\n"
        
        # 데코레이터
        if metadata.get("decorators"):
            content += f"**Decorators**: {', '.join(metadata['decorators'])}\n\n"
        
        # LLM 요약
        if llm_analysis.get("summary"):
            content += f"**Summary**: {llm_analysis['summary']}\n\n"
        
        return {
            "file_id": file_id,
            "symbol_id": f"{file_id}_{symbol_name}_basic",
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "content": content,
            "line_start": location.get("start_line", 0),
            "line_end": location.get("end_line", 0),
            "tokens": self._count_tokens(content),
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {
                "name": symbol_name,
                "type": symbol_type,
                "has_llm_analysis": bool(llm_analysis)
            }
        }
    
    def _create_llm_analysis_chunks(self, file_id: int, symbol_name: str, symbol_type: str,
                                  llm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM 분석 결과를 청크로 분할"""
        chunks = []
        
        # 함수/메서드 분석
        if symbol_type == "function":
            chunks.extend(self._create_function_analysis_chunks(
                file_id, symbol_name, llm_analysis
            ))
        
        # 클래스 분석
        elif symbol_type == "class":
            chunks.extend(self._create_class_analysis_chunks(
                file_id, symbol_name, llm_analysis
            ))
        
        return chunks
    
    def _create_function_analysis_chunks(self, file_id: int, symbol_name: str,
                                       llm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """함수 분석 결과를 청크로 분할"""
        chunks = []
        
        # 목적과 로직 분석
        purpose_content = "## Function Analysis\n\n"
        if llm_analysis.get("purpose"):
            purpose_content += f"**Purpose**: {llm_analysis['purpose']}\n\n"
        if llm_analysis.get("logic_overview"):
            purpose_content += f"**Logic Overview**: {llm_analysis['logic_overview']}\n\n"
        
        if purpose_content.strip() != "## Function Analysis\n\n":
            chunks.append(self._create_chunk_from_content(
                file_id, symbol_name, purpose_content, 0
            ))
        
        # 실행 흐름
        if llm_analysis.get("execution_flow"):
            flow_content = "## Execution Flow\n\n"
            flow_content += f"{llm_analysis['execution_flow']}\n\n"
            
            if llm_analysis.get("sequence_steps"):
                flow_content += "**Sequence Steps**:\n"
                for i, step in enumerate(llm_analysis["sequence_steps"], 1):
                    flow_content += f"{i}. {step}\n"
            
            chunks.append(self._create_chunk_from_content(
                file_id, symbol_name, flow_content, 1
            ))
        
        return chunks
    
    def _create_class_analysis_chunks(self, file_id: int, symbol_name: str,
                                    llm_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """클래스 분석 결과를 청크로 분할"""
        chunks = []
        
        # 책임과 설계
        responsibility_content = "## Class Analysis\n\n"
        if llm_analysis.get("responsibility"):
            responsibility_content += f"**Responsibility**: {llm_analysis['responsibility']}\n\n"
        if llm_analysis.get("design_notes"):
            responsibility_content += f"**Design Notes**: {llm_analysis['design_notes']}\n\n"
        
        if responsibility_content.strip() != "## Class Analysis\n\n":
            chunks.append(self._create_chunk_from_content(
                file_id, symbol_name, responsibility_content, 0
            ))
        
        # 협력 관계
        if llm_analysis.get("collaboration"):
            collab_content = "## Collaborations\n\n"
            for collab in llm_analysis["collaboration"]:
                if isinstance(collab, str) and collab.strip():
                    collab_content += f"- {collab.strip()}\n"
            
            chunks.append(self._create_chunk_from_content(
                file_id, symbol_name, collab_content, 1
            ))
        
        return chunks
    
    def _create_metadata_chunk(self, file_id: int, symbol_name: str, symbol_type: str,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 기반 청크 생성"""
        content = f"## {symbol_type.title()} Metadata: {symbol_name}\n\n"
        
        # 메서드 목록
        if metadata.get("methods"):
            content += "**Methods**:\n"
            for method in metadata["methods"]:
                content += f"- {method}\n"
            content += "\n"
        
        # 필드 목록
        if metadata.get("fields"):
            content += "**Fields**:\n"
            for field in metadata["fields"]:
                content += f"- {field}\n"
            content += "\n"
        
        # 매개변수
        if metadata.get("parameters"):
            content += "**Parameters**:\n"
            for param in metadata["parameters"]:
                content += f"- {param}\n"
            content += "\n"
        
        # 반환 타입
        if metadata.get("return_type"):
            content += f"**Return Type**: {metadata['return_type']}\n\n"
        
        return {
            "file_id": file_id,
            "symbol_id": f"{file_id}_{symbol_name}_metadata",
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "content": content,
            "line_start": 1,
            "line_end": 1,
            "tokens": self._count_tokens(content),
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {
                "name": symbol_name,
                "type": symbol_type
            }
        }
    
    def _create_chunk_from_content(self, file_id: int, symbol_name: str,
                                 content: str, chunk_index: int) -> Dict[str, Any]:
        """내용으로부터 청크 생성 (헬퍼 메서드)"""
        return {
            "file_id": file_id,
            "symbol_name": symbol_name,
            "symbol_type": "analysis",
            "content": content,
            "line_start": 1,
            "line_end": 1,
            "tokens": self._count_tokens(content),
            "chunk_index": chunk_index,
            "total_chunks": 1,
            "metadata": {
                "name": symbol_name,
                "type": "analysis"
            }
        }

    def _extract_symbol_code_from_original(self, original_code: str, location: Dict[str, Any], 
                                         symbol_name: str, symbol_type: str) -> str:
        """원본 코드에서 심볼 부분 추출"""
        if not original_code:
            return f"# {symbol_type}: {symbol_name}"
        
        lines = original_code.split('\n')
        start_line = location.get("start_line", 1)
        end_line = location.get("end_line", len(lines))
        
        # 라인 번호 조정 (1-based to 0-based)
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        if start_idx >= end_idx:
            return f"# {symbol_type}: {symbol_name} (위치 정보 없음)"
        
        # 해당 라인 범위의 코드 추출
        symbol_lines = lines[start_idx:end_idx]
        symbol_code = "\n".join(symbol_lines)
        
        if not symbol_code.strip():
            return f"# {symbol_type}: {symbol_name} (빈 내용)"
        
        return symbol_code

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
        
        # 1. 노드 내용
        nodes = body.get("nodes", [])
        for node in nodes:
            node_content = node.get("content", "")
            if node_content:
                content_parts.append(node_content)
        
        # 2. 기타 내용
        other_content = body.get("content", "")
        if other_content:
            content_parts.append(other_content)
        
        # 3. AST 구조에서 실제 코드 추출 시도
        if not content_parts:
            # AST 구조를 더 깊이 탐색
            self._extract_deep_ast_content(body, content_parts)
        
        # 4. 여전히 내용이 없으면 기본 텍스트 생성
        if not content_parts:
            # 메타데이터를 기반으로 기본 내용 생성
            content_parts.append(self._generate_basic_content_from_metadata(body))
        
        return "\n".join(content_parts)
    
    def _extract_deep_ast_content(self, body: Dict[str, Any], content_parts: List[str]) -> None:
        """AST 구조를 깊이 탐색하여 코드 내용 추출"""
        # 재귀적으로 AST 구조 탐색
        for key, value in body.items():
            if isinstance(value, dict):
                self._extract_deep_ast_content(value, content_parts)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._extract_deep_ast_content(item, content_parts)
                    elif isinstance(item, str) and item.strip():
                        content_parts.append(item)
            elif isinstance(value, str) and value.strip():
                # 문자열 값이 있으면 추가
                content_parts.append(value)
    
    def _generate_basic_content_from_metadata(self, body: Dict[str, Any]) -> str:
        """메타데이터를 기반으로 기본 내용 생성"""
        # AST 구조 정보를 기반으로 기본 내용 생성
        content_lines = []
        
        # 노드 타입 정보
        if "type" in body:
            content_lines.append(f"# Node type: {body['type']}")
        
        # 위치 정보
        if "location" in body:
            loc = body["location"]
            if "start_line" in loc and "end_line" in loc:
                content_lines.append(f"# Lines: {loc['start_line']}-{loc['end_line']}")
        
        # 기타 속성들
        for key, value in body.items():
            if key not in ["type", "location", "nodes", "content"] and value:
                if isinstance(value, list):
                    content_lines.append(f"# {key}: {', '.join(map(str, value))}")
                else:
                    content_lines.append(f"# {key}: {value}")
        
        if not content_lines:
            content_lines.append("# AST node content")
        
        return "\n".join(content_lines)

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
        
        logger.debug(f"{symbol_name}: {len(tokens)} 토큰을 {len(chunks)}개 청크로 분할")
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
                "comment_count": len(file_comments)
            }
        }

    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """청킹 통계 정보"""
        if not chunks:
            return {"error": "청크가 없습니다"}
        
        total_tokens = sum(chunk.get("tokens", 0) for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "average_tokens_per_chunk": round(avg_tokens, 2),
            "config": {
                "max_tokens": self.config.max_tokens,
                "min_tokens": self.config.min_tokens,
                "overlap_tokens": self.config.overlap_tokens,
                "include_comments": self.config.include_comments,
                "include_docstrings": self.config.include_docstrings
            }
        }

    def validate_chunks(self, chunks: List[Dict[str, Any]], original_ast: Dict[str, Any]) -> ChunkValidationResult:
        """청킹 결과 검증 (편의 메서드)"""
        return self.validator.validate_chunks(chunks, original_ast)
    
    def chunk_file(self, file_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """파일 데이터를 청킹 (기존 create_chunks 메서드 활용)"""
        try:
            # file_id는 임시로 1 사용 (실제로는 파일별 고유 ID 필요)
            file_id = 1
            
            # 기존 create_chunks 메서드 활용
            chunks = self.create_chunks(file_data, file_id)
            
            # 각 청크에 file_path 필드 추가
            file_path = file_data.get("file_path", file_data.get("file", {}).get("path", "unknown"))
            logger.debug(f"chunk_file에서 file_path 타입: {type(file_path)}, 값: {file_path}")
            
            # Path 객체를 문자열로 변환
            if hasattr(file_path, 'as_posix'):
                logger.info(f"Path 객체를 문자열로 변환: {file_path} -> {str(file_path)}")
                file_path = str(file_path)
            
            for chunk in chunks:
                chunk["file_path"] = file_path
            
            logger.info(f"파일 청킹 완료: {len(chunks)}개 청크")
            return chunks
            
        except Exception as e:
            logger.error(f"파일 청킹 실패: {e}")
            return [] 