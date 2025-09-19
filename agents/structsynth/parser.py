"""
Python 내장 ast 모듈 기반 코드 파서 및 AST 추출기
사용자 요청 구조에 맞춘 JSON 출력 생성
"""

import os
import logging
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 전용 파서 라이브러리들 (선택적 import)
try:
    import javalang
    JAVA_PARSER_AVAILABLE = True
except ImportError:
    JAVA_PARSER_AVAILABLE = False
    logger.warning("javalang not available - falling back to regex for Java")

try:
    import clang.cindex
    CLANG_PARSER_AVAILABLE = True
except ImportError:
    CLANG_PARSER_AVAILABLE = False
    logger.warning("libclang not available - falling back to regex for C++")

try:
    from pycparser import parse_file, c_parser
    PYC_PARSER_AVAILABLE = True
except ImportError:
    PYC_PARSER_AVAILABLE = False
    logger.warning("pycparser not available - falling back to regex for C")

LANGUAGES = {
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
}

class CodeParser:
    def __init__(self):
        self.parsers: Dict[str, object] = {}
        self.languages: Dict[str, str] = {}
        self._load_parsers()

    def _load_parsers(self):
        """파서 초기화"""
        logger.info(" Python 내장 ast 모듈 기반 파서 초기화")
        # 모든 언어에 대해 Python ast 모듈 사용
        for ext in LANGUAGES.keys():
            self.parsers[ext] = "ast"
            self.languages[ext] = LANGUAGES[ext]
        logger.info(" 모든 언어 파서 로드 완료")

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
    
    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """파일을 파싱하여 AST 데이터 반환"""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.warning(f"파일이 존재하지 않음: {file_path}")
                return None
            
            # 파일 확장자 확인
            ext = file_path_obj.suffix.lower()
            if ext not in self.parsers:
                logger.warning(f"지원하지 않는 파일 확장자: {ext}")
                return None
            
            # 파일 내용 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # AST 추출기 사용하여 파싱
            extractor = self.get_extractor(ext)
            ast_data = extractor.parse_to_ast(code, str(file_path))
            
            # file_path 필드를 올바르게 설정
            if ast_data and "file" in ast_data:
                ast_data["file"]["path"] = str(file_path)
                ast_data["file_path"] = str(file_path)  # 추가 필드
                ast_data["language"] = ast_data["file"]["language"]
            
            logger.info(f"파일 파싱 완료: {file_path} -> {len(ast_data.get('symbols', []))}개 심볼")
            return ast_data
            
        except Exception as e:
            logger.error(f"파일 파싱 실패 {file_path}: {e}")
            return None


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
        elif self.language == "cpp":
            self._extract_cpp_symbols(code, ast_data)
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
        
        # 함수 내부 호출 관계 추출
        calls = self._extract_function_calls(node)
        
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
            },
            "calls": calls  # 호출 관계 추가
        }
        
        ast_data["symbols"].append(function_symbol)
        logger.debug(f" Python 함수 발견: {func_name}")

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
            },
            "calls": []  # 클래스는 직접 호출하지 않음
        }
        
        ast_data["symbols"].append(class_symbol)
        logger.debug(f" Python 클래스 발견: {class_name}")

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
                logger.debug(f" Python 함수 발견 (정규표현식): {func_name}")
        
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
                logger.debug(f" Python 클래스 발견 (정규표현식): {class_name}")

    def _extract_java_symbols(self, code: str, ast_data: dict):
        """Java 심볼 추출 (javalang 우선, 정규표현식 fallback)"""
        if JAVA_PARSER_AVAILABLE:
            try:
                self._extract_java_symbols_with_javalang(code, ast_data)
                return
            except Exception as e:
                logger.warning(f"javalang 파싱 실패, 정규표현식 사용: {e}")
        
        # Fallback to regex parsing
        self._extract_java_symbols_with_regex(code, ast_data)
    
    def _extract_java_symbols_with_javalang(self, code: str, ast_data: dict):
        """javalang 라이브러리를 사용한 Java 심볼 추출"""
        try:
            tree = javalang.parse.parse(code)
            
            # 패키지 정보
            if tree.package:
                ast_data["file"]["package"] = tree.package.name
            
            # 임포트 정보
            if tree.imports:
                ast_data["file"]["imports"] = [imp.path for imp in tree.imports]
            
            # 클래스, 인터페이스, 열거형 추출
            for path, node in tree:
                if isinstance(node, javalang.tree.ClassDeclaration):
                    self._extract_javalang_class(node, ast_data, path)
                elif isinstance(node, javalang.tree.InterfaceDeclaration):
                    self._extract_javalang_interface(node, ast_data, path)
                elif isinstance(node, javalang.tree.EnumDeclaration):
                    self._extract_javalang_enum(node, ast_data, path)
                elif isinstance(node, javalang.tree.MethodDeclaration):
                    self._extract_javalang_method(node, ast_data, path)
                elif isinstance(node, javalang.tree.ConstructorDeclaration):
                    self._extract_javalang_constructor(node, ast_data, path)
                elif isinstance(node, javalang.tree.FieldDeclaration):
                    self._extract_javalang_field(node, ast_data, path)
            
            logger.info(f"✅ javalang으로 Java 파싱 완료: {len(ast_data['symbols'])}개 심볼")
            
        except Exception as e:
            logger.error(f"javalang 파싱 실패: {e}")
            raise
    
    def _extract_javalang_class(self, node: javalang.tree.ClassDeclaration, ast_data: dict, path: list):
        """javalang 클래스 노드 추출"""
        class_name = node.name
        
        # 어노테이션 추출
        annotations = []
        if node.annotations:
            for annotation in node.annotations:
                annotations.append(annotation.name)
        
        # 상속/구현 정보
        bases = []
        if node.extends:
            bases.append(node.extends.name)
        if node.implements:
            for impl in node.implements:
                bases.append(impl.name)
        
        # 메서드와 필드 목록
        methods = []
        fields = []
        for member in node.body:
            if hasattr(member, 'name'):
                if isinstance(member, javalang.tree.MethodDeclaration):
                    methods.append(member.name)
                elif isinstance(member, javalang.tree.FieldDeclaration):
                    for declarator in member.declarators:
                        fields.append(declarator.name)
        
        class_symbol = {
            "type": "class",
            "name": class_name,
            "signature": f"class {class_name}",
            "location": {
                "start_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1,
                "end_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1
            },
            "metadata": {
                "annotations": annotations,
                "access": "public" if "public" in node.modifiers else "package-private",
                "modifiers": list(node.modifiers) if node.modifiers else [],
                "bases": bases,
                "methods": methods,
                "fields": fields,
                "type": "class"
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(class_symbol)
        logger.debug(f"📁 javalang 클래스 발견: {class_name}")
    
    def _extract_javalang_interface(self, node: javalang.tree.InterfaceDeclaration, ast_data: dict, path: list):
        """javalang 인터페이스 노드 추출"""
        interface_name = node.name
        
        interface_symbol = {
            "type": "interface",
            "name": interface_name,
            "signature": f"interface {interface_name}",
            "location": {
                "start_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1,
                "end_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1
            },
            "metadata": {
                "access": "public" if "public" in node.modifiers else "package-private",
                "modifiers": list(node.modifiers) if node.modifiers else [],
                "type": "interface"
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(interface_symbol)
        logger.debug(f"🔗 javalang 인터페이스 발견: {interface_name}")
    
    def _extract_javalang_enum(self, node: javalang.tree.EnumDeclaration, ast_data: dict, path: list):
        """javalang 열거형 노드 추출"""
        enum_name = node.name
        
        enum_symbol = {
            "type": "enum",
            "name": enum_name,
            "signature": f"enum {enum_name}",
            "location": {
                "start_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1,
                "end_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1
            },
            "metadata": {
                "access": "public" if "public" in node.modifiers else "package-private",
                "type": "enum"
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(enum_symbol)
        logger.debug(f"📋 javalang 열거형 발견: {enum_name}")
    
    def _extract_javalang_method(self, node: javalang.tree.MethodDeclaration, ast_data: dict, path: list):
        """javalang 메서드 노드 추출"""
        method_name = node.name
        
        # 매개변수 추출
        parameters = []
        if node.parameters:
            for param in node.parameters:
                parameters.append({
                    "type": param.type.name if hasattr(param.type, 'name') else str(param.type),
                    "name": param.name
                })
        
        method_symbol = {
            "type": "function",
            "name": method_name,
            "signature": f"{method_name}({', '.join([p['name'] for p in parameters])})",
            "location": {
                "start_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1,
                "end_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1
            },
            "metadata": {
                "access": "public" if "public" in node.modifiers else "private" if "private" in node.modifiers else "protected" if "protected" in node.modifiers else "package-private",
                "modifiers": list(node.modifiers) if node.modifiers else [],
                "parameters": parameters,
                "return_type": node.return_type.name if hasattr(node.return_type, 'name') else str(node.return_type) if node.return_type else "void"
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(method_symbol)
        logger.debug(f"🔧 javalang 메서드 발견: {method_name}")
    
    def _extract_javalang_constructor(self, node: javalang.tree.ConstructorDeclaration, ast_data: dict, path: list):
        """javalang 생성자 노드 추출"""
        constructor_name = node.name
        
        # 매개변수 추출
        parameters = []
        if node.parameters:
            for param in node.parameters:
                parameters.append({
                    "type": param.type.name if hasattr(param.type, 'name') else str(param.type),
                    "name": param.name
                })
        
        constructor_symbol = {
            "type": "constructor",
            "name": constructor_name,
            "signature": f"{constructor_name}({', '.join([p['name'] for p in parameters])})",
            "location": {
                "start_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1,
                "end_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1
            },
            "metadata": {
                "access": "public" if "public" in node.modifiers else "private" if "private" in node.modifiers else "protected" if "protected" in node.modifiers else "package-private",
                "parameters": parameters,
                "return_type": "void",
                "is_constructor": True
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(constructor_symbol)
        logger.debug(f"🏗️ javalang 생성자 발견: {constructor_name}")
    
    def _extract_javalang_field(self, node: javalang.tree.FieldDeclaration, ast_data: dict, path: list):
        """javalang 필드 노드 추출"""
        for declarator in node.declarators:
            field_name = declarator.name
            
            field_symbol = {
                "type": "field",
                "name": field_name,
                "signature": f"{node.type.name if hasattr(node.type, 'name') else str(node.type)} {field_name}",
                "location": {
                    "start_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1,
                    "end_line": getattr(node.position, 'line', 1) if hasattr(node, 'position') and node.position else 1
                },
                "metadata": {
                    "access": "public" if "public" in node.modifiers else "private" if "private" in node.modifiers else "protected" if "protected" in node.modifiers else "package-private",
                    "modifiers": list(node.modifiers) if node.modifiers else [],
                    "field_type": node.type.name if hasattr(node.type, 'name') else str(node.type)
                },
                "body": {"nodes": [], "comments": []}
            }
            
            ast_data["symbols"].append(field_symbol)
            logger.debug(f"📝 javalang 필드 발견: {field_name}")
    
    def _extract_java_symbols_with_regex(self, code: str, ast_data: dict):
        """정규표현식을 사용한 Java 심볼 추출 (fallback)"""
        lines = code.split('\n')
        current_class = None
        brace_depth = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # 중괄호 깊이 추적
            brace_depth += stripped_line.count('{') - stripped_line.count('}')
            
            # 패키지 선언
            package_match = re.search(r'^\s*package\s+([\w\.]+);', stripped_line)
            if package_match:
                ast_data["file"]["package"] = package_match.group(1)
                continue
            
            # 임포트 문
            import_match = re.search(r'^\s*import\s+(?:static\s+)?([\w\.\*]+);', stripped_line)
            if import_match:
                if "imports" not in ast_data["file"]:
                    ast_data["file"]["imports"] = []
                ast_data["file"]["imports"].append(import_match.group(1))
                continue
            
            # 클래스/인터페이스/열거형 선언
            class_match = re.search(r'^\s*(?:(@\w+)\s+)?(?:(public|private|protected)\s+)?(?:(abstract|final|static)\s+)?(class|interface|enum)\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?', stripped_line)
            if class_match:
                annotation, access, modifier, type_keyword, class_name, extends, implements = class_match.groups()
                
                # 상속/구현 정보 파싱
                bases = []
                if extends:
                    bases.append(extends)
                if implements:
                    bases.extend([impl.strip() for impl in implements.split(',')])
                
                class_symbol = {
                    "type": "class" if type_keyword == "class" else type_keyword,
                    "name": class_name,
                    "signature": stripped_line,
                    "location": {"start_line": i + 1, "end_line": self._find_closing_brace(lines, i)},
                    "metadata": {
                        "annotations": [annotation] if annotation else [],
                        "access": access or "package-private",
                        "modifiers": [modifier] if modifier else [],
                        "bases": bases,
                        "methods": [],
                        "fields": [],
                        "type": type_keyword
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(class_symbol)
                current_class = class_name
                logger.debug(f"📁 Java {type_keyword} 발견: {class_name}")
                continue
            
            # 메서드 선언 (클래스 내부에서만)
            if current_class and brace_depth >= 1:
                method_match = re.search(r'^\s*(?:(@\w+)\s+)?(?:(public|private|protected)\s+)?(?:(static|final|abstract|synchronized)\s+)*(?:(<[^>]+>)\s+)?(\w+(?:<[^>]+>)?|void)\s+(\w+)\s*\(([^)]*)\)(?:\s+throws\s+[\w,\s]+)?', stripped_line)
                if method_match and not stripped_line.endswith(';'):  # 추상 메서드 제외
                    annotation, access, modifiers, generics, return_type, method_name, params = method_match.groups()
                    
                    # 생성자 구분
                    is_constructor = method_name == current_class
                    
                    # 매개변수 파싱
                    parameters = []
                    if params and params.strip():
                        for param in params.split(','):
                            param = param.strip()
                            if param:
                                param_parts = param.split()
                                if len(param_parts) >= 2:
                                    parameters.append({
                                        "type": " ".join(param_parts[:-1]),
                                        "name": param_parts[-1]
                                    })
                    
                    method_symbol = {
                        "type": "constructor" if is_constructor else "function",
                        "name": method_name,
                        "signature": stripped_line,
                        "location": {"start_line": i + 1, "end_line": self._find_method_end(lines, i)},
                        "metadata": {
                            "annotations": [annotation] if annotation else [],
                            "access": access or "package-private",
                            "modifiers": [m for m in (modifiers or "").split() if m],
                            "parameters": parameters,
                            "return_type": return_type if not is_constructor else "void",
                            "generics": generics,
                            "is_constructor": is_constructor,
                            "parent_class": current_class
                        },
                        "body": {"nodes": [], "comments": []}
                    }
                    ast_data["symbols"].append(method_symbol)
                    logger.debug(f"🔧 Java {'생성자' if is_constructor else '메서드'} 발견: {method_name}")
                    continue
            
            # 필드 선언
            if current_class and brace_depth >= 1:
                field_match = re.search(r'^\s*(?:(@\w+)\s+)?(?:(public|private|protected)\s+)?(?:(static|final|transient|volatile)\s+)*(\w+(?:<[^>]+>)?)\s+(\w+)(?:\s*=\s*[^;]+)?;', stripped_line)
                if field_match:
                    annotation, access, modifiers, field_type, field_name = field_match.groups()
                    
                    field_symbol = {
                        "type": "field",
                        "name": field_name,
                        "signature": stripped_line,
                        "location": {"start_line": i + 1, "end_line": i + 1},
                        "metadata": {
                            "annotations": [annotation] if annotation else [],
                            "access": access or "package-private",
                            "modifiers": [m for m in (modifiers or "").split() if m],
                            "field_type": field_type,
                            "parent_class": current_class
                        },
                        "body": {"nodes": [], "comments": []}
                    }
                    ast_data["symbols"].append(field_symbol)
                    logger.debug(f"📝 Java 필드 발견: {field_name}")

    def _extract_cpp_symbols(self, code: str, ast_data: dict):
        """C++ 심볼 추출 (libclang 우선, 정규표현식 fallback)"""
        if CLANG_PARSER_AVAILABLE:
            try:
                self._extract_cpp_symbols_with_clang(code, ast_data)
                return
            except Exception as e:
                logger.warning(f"libclang 파싱 실패, 정규표현식 사용: {e}")
        
        # Fallback to regex parsing
        self._extract_cpp_symbols_with_regex(code, ast_data)
    
    def _extract_cpp_symbols_with_clang(self, code: str, ast_data: dict):
        """libclang을 사용한 C++ 심볼 추출"""
        try:
            # 임시 파일 생성
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_file.flush()
                
                # clang index 생성
                index = clang.cindex.Index.create()
                
                # 파싱 옵션 설정
                args = ['-std=c++17', '-I/usr/include', '-I/usr/local/include']
                
                # 번역 단위 파싱
                tu = index.parse(tmp_file.name, args=args)
                
                if tu.diagnostics:
                    for diag in tu.diagnostics:
                        if diag.severity >= clang.cindex.Diagnostic.Error:
                            logger.warning(f"Clang 파싱 경고/에러: {diag}")
                
                # AST 순회하여 심볼 추출
                self._traverse_clang_cursor(tu.cursor, ast_data)
                
                logger.info(f"✅ libclang으로 C++ 파싱 완료: {len(ast_data['symbols'])}개 심볼")
                
            # 임시 파일 정리
            import os
            os.unlink(tmp_file.name)
            
        except Exception as e:
            logger.error(f"libclang 파싱 실패: {e}")
            raise
    
    def _traverse_clang_cursor(self, cursor, ast_data: dict, depth: int = 0):
        """clang cursor를 재귀적으로 순회하여 심볼 추출"""
        # 파일 외부 심볼은 제외 (표준 라이브러리 등)
        if cursor.location.file and not cursor.location.file.name.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
            return
        
        cursor_kind = cursor.kind
        
        if cursor_kind == clang.cindex.CursorKind.CLASS_DECL:
            self._extract_clang_class(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.STRUCT_DECL:
            self._extract_clang_struct(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.FUNCTION_DECL:
            self._extract_clang_function(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.CXX_METHOD:
            self._extract_clang_method(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.CONSTRUCTOR:
            self._extract_clang_constructor(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.DESTRUCTOR:
            self._extract_clang_destructor(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.FIELD_DECL:
            self._extract_clang_field(cursor, ast_data)
        elif cursor_kind == clang.cindex.CursorKind.NAMESPACE:
            self._extract_clang_namespace(cursor, ast_data)
        
        # 자식 노드들도 순회
        for child in cursor.get_children():
            self._traverse_clang_cursor(child, ast_data, depth + 1)
    
    def _extract_clang_class(self, cursor, ast_data: dict):
        """libclang 클래스 커서 추출"""
        class_name = cursor.spelling
        
        # 기본 클래스 찾기
        bases = []
        access_specifiers = []
        for child in cursor.get_children():
            if child.kind == clang.cindex.CursorKind.CXX_BASE_SPECIFIER:
                bases.append(child.type.spelling)
                access_specifiers.append(child.access_specifier.name.lower())
        
        class_symbol = {
            "type": "class",
            "name": class_name,
            "signature": f"class {class_name}",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": cursor.access_specifier.name.lower() if cursor.access_specifier else "public",
                "bases": bases,
                "base_access": access_specifiers,
                "is_template": self._is_template_class(cursor),
                "namespace": self._get_namespace(cursor)
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(class_symbol)
        logger.debug(f"🏗️ libclang 클래스 발견: {class_name}")
    
    def _extract_clang_struct(self, cursor, ast_data: dict):
        """libclang 구조체 커서 추출"""
        struct_name = cursor.spelling
        
        struct_symbol = {
            "type": "struct",
            "name": struct_name,
            "signature": f"struct {struct_name}",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": "public",  # struct는 기본적으로 public
                "namespace": self._get_namespace(cursor)
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(struct_symbol)
        logger.debug(f"📋 libclang 구조체 발견: {struct_name}")
    
    def _extract_clang_function(self, cursor, ast_data: dict):
        """libclang 함수 커서 추출"""
        func_name = cursor.spelling
        
        # 매개변수 추출
        parameters = []
        for arg in cursor.get_arguments():
            parameters.append({
                "type": arg.type.spelling,
                "name": arg.spelling
            })
        
        function_symbol = {
            "type": "function",
            "name": func_name,
            "signature": f"{func_name}({', '.join([p['name'] for p in parameters])})",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": "public",
                "parameters": parameters,
                "return_type": cursor.result_type.spelling,
                "namespace": self._get_namespace(cursor),
                "is_static": cursor.storage_class == clang.cindex.StorageClass.STATIC,
                "is_inline": cursor.is_inline_function()
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(function_symbol)
        logger.debug(f"⚙️ libclang 함수 발견: {func_name}")
    
    def _extract_clang_method(self, cursor, ast_data: dict):
        """libclang 메서드 커서 추출"""
        method_name = cursor.spelling
        
        # 매개변수 추출
        parameters = []
        for arg in cursor.get_arguments():
            parameters.append({
                "type": arg.type.spelling,
                "name": arg.spelling
            })
        
        method_symbol = {
            "type": "function",
            "name": method_name,
            "signature": f"{method_name}({', '.join([p['name'] for p in parameters])})",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": cursor.access_specifier.name.lower() if cursor.access_specifier else "private",
                "parameters": parameters,
                "return_type": cursor.result_type.spelling,
                "parent_class": self._get_parent_class(cursor),
                "is_virtual": cursor.is_virtual_method(),
                "is_static": cursor.is_static_method(),
                "is_const": cursor.is_const_method()
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(method_symbol)
        logger.debug(f"🔧 libclang 메서드 발견: {method_name}")
    
    def _extract_clang_constructor(self, cursor, ast_data: dict):
        """libclang 생성자 커서 추출"""
        constructor_name = cursor.spelling
        
        # 매개변수 추출
        parameters = []
        for arg in cursor.get_arguments():
            parameters.append({
                "type": arg.type.spelling,
                "name": arg.spelling
            })
        
        constructor_symbol = {
            "type": "constructor",
            "name": constructor_name,
            "signature": f"{constructor_name}({', '.join([p['name'] for p in parameters])})",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": cursor.access_specifier.name.lower() if cursor.access_specifier else "public",
                "parameters": parameters,
                "return_type": "void",
                "parent_class": self._get_parent_class(cursor),
                "is_constructor": True
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(constructor_symbol)
        logger.debug(f"🏗️ libclang 생성자 발견: {constructor_name}")
    
    def _extract_clang_destructor(self, cursor, ast_data: dict):
        """libclang 소멸자 커서 추출"""
        destructor_name = cursor.spelling
        
        destructor_symbol = {
            "type": "destructor",
            "name": destructor_name,
            "signature": destructor_name,
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": cursor.access_specifier.name.lower() if cursor.access_specifier else "public",
                "return_type": "void",
                "parent_class": self._get_parent_class(cursor),
                "is_destructor": True,
                "is_virtual": cursor.is_virtual_method()
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(destructor_symbol)
        logger.debug(f"🗑️ libclang 소멸자 발견: {destructor_name}")
    
    def _extract_clang_field(self, cursor, ast_data: dict):
        """libclang 필드 커서 추출"""
        field_name = cursor.spelling
        
        field_symbol = {
            "type": "field",
            "name": field_name,
            "signature": f"{cursor.type.spelling} {field_name}",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.location.line
            },
            "metadata": {
                "access": cursor.access_specifier.name.lower() if cursor.access_specifier else "private",
                "field_type": cursor.type.spelling,
                "parent_class": self._get_parent_class(cursor),
                "is_static": cursor.storage_class == clang.cindex.StorageClass.STATIC
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(field_symbol)
        logger.debug(f"📝 libclang 필드 발견: {field_name}")
    
    def _extract_clang_namespace(self, cursor, ast_data: dict):
        """libclang 네임스페이스 커서 추출"""
        namespace_name = cursor.spelling
        
        namespace_symbol = {
            "type": "namespace",
            "name": namespace_name,
            "signature": f"namespace {namespace_name}",
            "location": {
                "start_line": cursor.location.line,
                "end_line": cursor.extent.end.line
            },
            "metadata": {
                "access": "public"
            },
            "body": {"nodes": [], "comments": []}
        }
        
        ast_data["symbols"].append(namespace_symbol)
        logger.debug(f"📦 libclang 네임스페이스 발견: {namespace_name}")
    
    def _get_parent_class(self, cursor):
        """커서의 부모 클래스 이름 반환"""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL]:
                return parent.spelling
            parent = parent.semantic_parent
        return None
    
    def _get_namespace(self, cursor):
        """커서의 네임스페이스 반환"""
        namespaces = []
        parent = cursor.semantic_parent
        while parent:
            if parent.kind == clang.cindex.CursorKind.NAMESPACE:
                namespaces.append(parent.spelling)
            parent = parent.semantic_parent
        return "::".join(reversed(namespaces)) if namespaces else None
    
    def _is_template_class(self, cursor):
        """템플릿 클래스인지 확인"""
        for child in cursor.get_children():
            if child.kind == clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
                return True
        return False
    
    def _extract_cpp_symbols_with_regex(self, code: str, ast_data: dict):
        """정규표현식을 사용한 C++ 심볼 추출 (fallback)"""
        lines = code.split('\n')
        current_namespace = None
        current_class = None
        brace_depth = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # 중괄호 깊이 추적
            brace_depth += stripped_line.count('{') - stripped_line.count('}')
            
            # 전처리기 지시문 건너뛰기
            if stripped_line.startswith('#'):
                continue
            
            # 네임스페이스 선언
            namespace_match = re.search(r'^namespace\s+(\w+)', stripped_line)
            if namespace_match:
                current_namespace = namespace_match.group(1)
                continue
            
            # 인클루드 문
            include_match = re.search(r'^#include\s+[<"]([^>"]+)[">]', stripped_line)
            if include_match:
                if "includes" not in ast_data["file"]:
                    ast_data["file"]["includes"] = []
                ast_data["file"]["includes"].append(include_match.group(1))
                continue
            
            # 클래스/구조체 선언
            class_match = re.search(r'^(?:template\s*<[^>]*>\s+)?(?:(class|struct|union))\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?', stripped_line)
            if class_match:
                class_type, class_name, base_class = class_match.groups()
                
                class_symbol = {
                    "type": "class",
                    "name": class_name,
                    "signature": stripped_line,
                    "location": {"start_line": i + 1, "end_line": self._find_closing_brace(lines, i)},
                    "metadata": {
                        "access": "public" if class_type == "struct" else "private",
                        "class_type": class_type,
                        "bases": [base_class] if base_class else [],
                        "namespace": current_namespace,
                        "methods": [],
                        "fields": [],
                        "template": "template" in stripped_line
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(class_symbol)
                current_class = class_name
                logger.debug(f"🏗️ C++ {class_type} 발견: {class_name}")
                continue
            
            # 함수/메서드 선언
            func_match = re.search(r'^(?:template\s*<[^>]*>\s+)?(?:(virtual|static|inline|explicit)\s+)*(?:(\w+(?:\s*\*)*(?:\s*&)?)\s+)?(\w+|~\w+|operator\S+)\s*\(([^)]*)\)(?:\s*const)?(?:\s*override)?(?:\s*=\s*0)?', stripped_line)
            if func_match and not stripped_line.endswith(';'):
                modifiers, return_type, func_name, params = func_match.groups()
                
                # 생성자/소멸자 구분
                is_constructor = func_name == current_class
                is_destructor = func_name.startswith('~')
                is_operator = func_name.startswith('operator')
                
                # 매개변수 파싱
                parameters = []
                if params and params.strip():
                    for param in params.split(','):
                        param = param.strip()
                        if param and param != 'void':
                            param_parts = param.split()
                            if len(param_parts) >= 1:
                                parameters.append({
                                    "type": " ".join(param_parts[:-1]) if len(param_parts) > 1 else param_parts[0],
                                    "name": param_parts[-1] if len(param_parts) > 1 else ""
                                })
                
                symbol_type = "constructor" if is_constructor else "destructor" if is_destructor else "operator" if is_operator else "function"
                
                function_symbol = {
                    "type": symbol_type,
                    "name": func_name,
                    "signature": stripped_line,
                    "location": {"start_line": i + 1, "end_line": self._find_method_end(lines, i)},
                    "metadata": {
                        "access": "public",  # C++에서는 컨텍스트 필요
                        "modifiers": [m for m in (modifiers or "").split() if m],
                        "parameters": parameters,
                        "return_type": return_type or ("void" if is_constructor or is_destructor else "auto"),
                        "namespace": current_namespace,
                        "parent_class": current_class,
                        "is_constructor": is_constructor,
                        "is_destructor": is_destructor,
                        "is_operator": is_operator,
                        "template": "template" in stripped_line
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(function_symbol)
                logger.debug(f"⚙️ C++ {symbol_type} 발견: {func_name}")

    def _extract_c_symbols(self, code: str, ast_data: dict):
        """C 심볼 추출 (정규표현식 기반 - 향상된 버전)"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # 전처리기 지시문 건너뛰기
            if stripped_line.startswith('#'):
                # 인클루드 문 추출
                include_match = re.search(r'^#include\s+[<"]([^>"]+)[">]', stripped_line)
                if include_match:
                    if "includes" not in ast_data["file"]:
                        ast_data["file"]["includes"] = []
                    ast_data["file"]["includes"].append(include_match.group(1))
                continue
            
            # 구조체 선언
            struct_match = re.search(r'^(?:typedef\s+)?struct\s+(\w+)', stripped_line)
            if struct_match:
                struct_name = struct_match.group(1)
                
                struct_symbol = {
                    "type": "struct",
                    "name": struct_name,
                    "signature": stripped_line,
                    "location": {"start_line": i + 1, "end_line": self._find_closing_brace(lines, i)},
                    "metadata": {
                        "access": "public",
                        "fields": [],
                        "is_typedef": "typedef" in stripped_line
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(struct_symbol)
                logger.debug(f"📋 C struct 발견: {struct_name}")
                continue
            
            # 함수 선언
            func_match = re.search(r'^(?:static\s+|extern\s+|inline\s+)*(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)', stripped_line)
            if func_match and not stripped_line.endswith(';'):
                return_type, func_name, params = func_match.groups()
                
                # main 함수나 키워드 제외
                if func_name in ["main", "if", "for", "while", "switch", "return"]:
                    continue
                
                # 매개변수 파싱
                parameters = []
                if params and params.strip() and params.strip() != 'void':
                    for param in params.split(','):
                        param = param.strip()
                        if param:
                            param_parts = param.split()
                            if len(param_parts) >= 1:
                                parameters.append({
                                    "type": " ".join(param_parts[:-1]) if len(param_parts) > 1 else param_parts[0],
                                    "name": param_parts[-1] if len(param_parts) > 1 else ""
                                })
                
                function_symbol = {
                    "type": "function",
                    "name": func_name,
                    "signature": stripped_line,
                    "location": {"start_line": i + 1, "end_line": self._find_method_end(lines, i)},
                    "metadata": {
                        "access": "public",
                        "parameters": parameters,
                        "return_type": return_type,
                        "is_static": "static" in stripped_line,
                        "is_extern": "extern" in stripped_line,
                        "is_inline": "inline" in stripped_line
                    },
                    "body": {"nodes": [], "comments": []}
                }
                ast_data["symbols"].append(function_symbol)
                logger.debug(f"🔧 C 함수 발견: {func_name}")

    def _extract_function_calls(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """함수 내부의 호출 관계 추출"""
        calls = []
        
        for ast_node in ast.walk(node):
            if isinstance(ast_node, ast.Call):
                call_info = self._analyze_call_node(ast_node)
                if call_info:
                    calls.append(call_info)
            # Import 문에서 호출되는 모듈들도 분석
            elif isinstance(ast_node, ast.Import):
                for alias in ast_node.names:
                    calls.append({
                        "type": "import_call",
                        "name": alias.name,
                        "line": ast_node.lineno,
                        "confidence": 1.0
                    })
            elif isinstance(ast_node, ast.ImportFrom):
                if ast_node.module:
                    for alias in ast_node.names:
                        calls.append({
                            "type": "import_from_call",
                            "module": ast_node.module,
                            "name": alias.name,
                            "line": ast_node.lineno,
                            "confidence": 1.0
                        })
        
        return calls
    
    def _analyze_call_node(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        """호출 노드 분석"""
        try:
            # 함수 호출 (예: func())
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # 생성자 호출인지 확인 (대문자로 시작)
                if func_name[0].isupper():
                    return {
                        "type": "constructor_call",
                        "class_name": func_name,
                        "name": func_name,
                        "line": node.lineno,
                        "confidence": 1.0,
                        "args_count": len(node.args)
                    }
                else:
                    return {
                        "type": "function_call", 
                        "name": func_name,
                        "line": node.lineno,
                        "confidence": 1.0,
                        "args_count": len(node.args)
                    }
            
            # 메서드 호출 (예: obj.method())
            elif isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                
                if isinstance(node.func.value, ast.Name):
                    # 단순 메서드 호출 (obj.method())
                    return {
                        "type": "method_call",
                        "object": node.func.value.id,
                        "method": method_name,
                        "name": method_name,
                        "line": node.lineno,
                        "confidence": 1.0,
                        "args_count": len(node.args)
                    }
                elif isinstance(node.func.value, ast.Call):
                    # 체이닝된 호출 (예: obj.create().method())
                    return {
                        "type": "chained_call",
                        "method": method_name,
                        "name": method_name,
                        "line": node.lineno,
                        "confidence": 0.8,
                        "args_count": len(node.args)
                    }
                elif isinstance(node.func.value, ast.Attribute):
                    # 중첩된 속성 호출 (예: obj.attr.method())
                    return {
                        "type": "nested_method_call",
                        "method": method_name,
                        "name": method_name,
                        "line": node.lineno,
                        "confidence": 0.9,
                        "args_count": len(node.args)
                    }
                else:
                    # 기타 속성 호출
                    return {
                        "type": "attribute_call",
                        "method": method_name,
                        "name": method_name,
                        "line": node.lineno,
                        "confidence": 0.7,
                        "args_count": len(node.args)
                    }
            
            # 람다나 복잡한 표현식 호출
            else:
                return {
                    "type": "complex_call",
                    "name": "unknown_call",
                    "line": node.lineno,
                    "confidence": 0.5,
                    "args_count": len(node.args)
                }
                
        except Exception as e:
            logger.debug(f"호출 노드 분석 실패: {e}")
        
        return None
    
    def _find_closing_brace(self, lines: List[str], start_line: int) -> int:
        """중괄호 매칭을 통해 블록의 끝 라인 찾기"""
        brace_count = 0
        for i in range(start_line, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0 and '{' in lines[start_line]:
                return i + 1
        return start_line + 1
    
    def _find_method_end(self, lines: List[str], start_line: int) -> int:
        """메서드/함수의 끝 라인 찾기"""
        # 세미콜론으로 끝나는 경우 (선언만)
        if lines[start_line].strip().endswith(';'):
            return start_line + 1
        
        # 중괄호로 블록이 시작되는 경우
        return self._find_closing_brace(lines, start_line)
