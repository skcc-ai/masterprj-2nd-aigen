"""
Python 내장 ast 모듈 기반 코드 파서 및 AST 추출기
사용자 요청 구조에 맞춘 JSON 출력 생성
"""

import os
import logging
import re
import ast
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LANGUAGES = {
    ".py": "python",
    ".java": "java",
    ".c": "c",
}

class CodeParser:
    def __init__(self):
        self.parsers: Dict[str, object] = {}
        self.languages: Dict[str, str] = {}
        self._load_parsers()

    def _load_parsers(self):
        """파서 초기화"""
        logger.info("✅ Python 내장 ast 모듈 기반 파서 초기화")
        # 모든 언어에 대해 Python ast 모듈 사용
        for ext in LANGUAGES.keys():
            self.parsers[ext] = "ast"
            self.languages[ext] = LANGUAGES[ext]
        logger.info("✅ 모든 언어 파서 로드 완료")

    def parse(self, code: str, ext: str):
        """코드 파싱"""
        if ext not in self.parsers:
            raise ValueError(f"Unsupported extension: {ext}")
        
        if ext == ".py":
            # Python은 ast 모듈 사용
            try:
                tree = ast.parse(code)
                return tree
            except SyntaxError as e:
                logger.warning(f"⚠️ Python 코드 파싱 실패: {e}")
                return None
        else:
            # Java, C는 정규표현식 사용
            return self._parse_with_regex(code, ext)
    
    def _parse_with_regex(self, code: str, ext: str):
        """정규표현식 기반 파싱 (Java, C용)"""
        # 간단한 더미 트리 객체 생성
        class DummyTree:
            def __init__(self, code):
                self.code = code
                self.root_node = DummyNode(code)
        
        class DummyNode:
            def __init__(self, code):
                self.code = code
                self.type = "root"
                self.children = []
                self.start_line = 1
                self.end_line = len(code.split('\n'))
        
        return DummyTree(code)

    def get_extractor(self, ext: str):
        """언어별 AST 추출기 반환"""
        if ext not in self.parsers:
            raise ValueError(f"Unsupported extension: {ext}")
        
        lang_name = LANGUAGES.get(ext, "unknown")
        return ASTExtractor(ext, lang_name)


class ASTExtractor:
    """사용자 요청 구조에 맞춘 AST 추출기"""
    
    def __init__(self, file_ext: str, language: str):
        self.file_ext = file_ext
        self.language = language
        self.file_path: Optional[str] = None

    def parse_to_ast(self, code: str, file_path: str = None):
        """코드를 사용자 요청 구조의 AST로 변환"""
        self.file_path = file_path
        
        # 사용자 요청 구조에 맞춘 AST 생성
        ast_data = {
            "file": {
                "path": file_path,
                "language": self.language
            },
            "symbols": []
        }
        
        # 언어별 심볼 추출
        if self.language == "python":
            self._extract_python_symbols(code, ast_data)
        elif self.language == "java":
            self._extract_java_symbols(code, ast_data)
        elif self.language == "c":
            self._extract_c_symbols(code, ast_data)
        
        return ast_data

    def _extract_python_symbols(self, code: str, ast_data: dict):
        """Python 심볼 추출 (ast 모듈 사용)"""
        try:
            tree = ast.parse(code)
            
            # 함수와 클래스 찾기
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._extract_python_function(node, code, ast_data)
                elif isinstance(node, ast.ClassDef):
                    self._extract_python_class(node, code, ast_data)
                    
        except SyntaxError as e:
            logger.warning(f"⚠️ Python 코드 파싱 실패, 정규표현식 사용: {e}")
            self._extract_python_symbols_with_regex(code, ast_data)

    def _extract_python_function(self, node: ast.FunctionDef, code: str, ast_data: dict):
        """Python 함수 추출 (ast 노드 사용)"""
        # 함수명
        func_name = node.name
        
        # 매개변수
        params = []
        for arg in node.args.args:
            if arg.arg != 'self':  # self 제외
                params.append(arg.arg)
        
        # 데코레이터
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
        
        # 함수 심볼 생성
        function_symbol = {
            "type": "function",
            "name": func_name,
            "signature": f"def {func_name}({', '.join(params)})",
            "location": {
                "start_line": node.lineno,
                "end_line": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            },
            "metadata": {
                "decorators": decorators,
                "access": "public" if not decorators else "decorated",
                "parameters": params,
                "return_type": "Any"
            },
            "body": {
                "nodes": [],
                "comments": []
            }
        }
        
        ast_data["symbols"].append(function_symbol)
        logger.debug(f"✅ Python 함수 발견: {func_name}")

    def _extract_python_class(self, node: ast.ClassDef, code: str, ast_data: dict):
        """Python 클래스 추출 (ast 노드 사용)"""
        # 클래스명
        class_name = node.name
        
        # 상속 정보
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
        
        # 데코레이터
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
        
        # 클래스 내부 메서드와 필드 찾기
        methods = []
        fields = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        fields.append(target.id)
        
        # 클래스 심볼 생성
        class_symbol = {
            "type": "class",
            "name": class_name,
            "signature": f"class {class_name}({', '.join(bases)})" if bases else f"class {class_name}",
            "location": {
                "start_line": node.lineno,
                "end_line": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
            },
            "metadata": {
                "decorators": decorators,
                "access": "public" if not decorators else "decorated",
                "bases": bases,
                "methods": methods,
                "fields": fields
            },
            "body": {
                "nodes": [],
                "comments": []
            }
        }
        
        ast_data["symbols"].append(class_symbol)
        logger.debug(f"✅ Python 클래스 발견: {class_name}")

    def _extract_python_symbols_with_regex(self, code: str, ast_data: dict):
        """정규표현식으로 Python 심볼 추출 (fallback)"""
        lines = code.split('\n')
        
        # 함수 정의 찾기
        func_pattern = r'^\s*def\s+(\w+)\s*\([^)]*\)\s*:'
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match:
                func_name = match.group(1)
                function_symbol = {
                    "type": "function",
                    "name": func_name,
                    "signature": line.strip(),
                    "location": {"start_line": i + 1, "end_line": i + 1},
                    "metadata": {
                        "decorators": [],
                        "access": "public",
                        "parameters": [],
                        "return_type": "Any"
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(function_symbol)
                logger.debug(f"✅ Python 함수 발견 (정규표현식): {func_name}")
        
        # 클래스 정의 찾기
        class_pattern = r'^\s*class\s+(\w+)'
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                class_name = match.group(1)
                class_symbol = {
                    "type": "class",
                    "name": class_name,
                    "signature": line.strip(),
                    "location": {"start_line": i + 1, "end_line": i + 1},
                    "metadata": {
                        "decorators": [],
                        "access": "public",
                        "bases": [],
                        "methods": [],
                        "fields": []
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(class_symbol)
                logger.debug(f"✅ Python 클래스 발견 (정규표현식): {class_name}")

    def _extract_java_symbols(self, code: str, ast_data: dict):
        """Java 심볼 추출 (정규표현식 기반)"""
        lines = code.split('\n')
        
        # 클래스 추출
        class_pattern = r'^\s*(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)'
        for i, line in enumerate(lines):
            match = re.search(class_pattern, line)
            if match:
                class_name = match.group(1)
                class_symbol = {
                    "type": "class",
                    "name": class_name,
                    "signature": line.strip(),
                    "location": {"start_line": i + 1, "end_line": i + 1},
                    "metadata": {
                        "decorators": [],
                        "access": "public" if "public" in line else "package-private",
                        "bases": [],
                        "methods": [],
                        "fields": []
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(class_symbol)
                logger.debug(f"✅ Java 클래스 발견: {class_name}")
        
        # 메서드 추출
        method_pattern = r'^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:abstract\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)'
        for i, line in enumerate(lines):
            match = re.search(method_pattern, line)
            if match:
                return_type = match.group(1)
                method_name = match.group(2)
                
                # 생성자 제외
                if method_name in ["class", "interface", "enum"]:
                    continue
                
                method_symbol = {
                    "type": "function",
                    "name": method_name,
                    "signature": line.strip(),
                    "location": {"start_line": i + 1, "end_line": i + 1},
                    "metadata": {
                        "decorators": [],
                        "access": "public" if "public" in line else "private" if "private" in line else "protected" if "protected" in line else "package-private",
                        "parameters": [],
                        "return_type": return_type
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(method_symbol)
                logger.debug(f"✅ Java 메서드 발견: {method_name}")

    def _extract_c_symbols(self, code: str, ast_data: dict):
        """C 심볼 추출 (정규표현식 기반)"""
        lines = code.split('\n')
        
        # 함수 추출
        func_pattern = r'^\s*(\w+(?:\s+\*+)?)\s+(\w+)\s*\([^)]*\)\s*\{?'
        for i, line in enumerate(lines):
            match = re.search(func_pattern, line)
            if match:
                return_type = match.group(1)
                func_name = match.group(2)
                
                # main 함수나 특수 함수 제외
                if func_name in ["main", "if", "for", "while"]:
                    continue
                
                function_symbol = {
                    "type": "function",
                    "name": func_name,
                    "signature": line.strip(),
                    "location": {"start_line": i + 1, "end_line": i + 1},
                    "metadata": {
                        "decorators": [],
                        "access": "public",
                        "parameters": [],
                        "return_type": return_type
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(function_symbol)
                logger.debug(f"✅ C 함수 발견: {func_name}")
