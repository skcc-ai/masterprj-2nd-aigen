"""
Data Processor - 데이터 처리 및 변환 엔진
다양한 데이터 소스에서 데이터를 읽고, 검증하고, 변환하는 메인 클래스
"""

import json
import csv
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import re
import statistics
from collections import defaultdict, Counter
import threading
import queue
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """데이터 처리 메인 클래스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.processed_data = []
        self.error_log = []
        self.validation_rules = self._load_validation_rules()
        self.transformation_pipeline = self._setup_transformation_pipeline()
        self.data_cache = {}
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now(),
            'last_processed': None
        }
        
        # 스레드 안전을 위한 락
        self._lock = threading.Lock()
        self._processing_queue = queue.Queue()
        
        logger.info("DataProcessor 초기화 완료")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """데이터 검증 규칙 로드"""
        return {
            'required_fields': ['id', 'name', 'timestamp'],
            'field_types': {
                'id': str,
                'name': str,
                'timestamp': str,
                'value': (int, float),
                'category': str,
                'tags': list
            },
            'field_constraints': {
                'name': {'min_length': 1, 'max_length': 100},
                'value': {'min': 0, 'max': 1000000},
                'category': {'allowed_values': ['A', 'B', 'C', 'D']}
            },
            'custom_validators': {
                'timestamp': self._validate_timestamp,
                'email': self._validate_email,
                'phone': self._validate_phone
            }
        }
    
    def _setup_transformation_pipeline(self) -> List[callable]:
        """데이터 변환 파이프라인 설정"""
        return [
            self._normalize_strings,
            self._convert_data_types,
            self._add_derived_fields,
            self._clean_data,
            self._enrich_data
        ]
    
    def process_data_file(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """데이터 파일 처리 메인 함수"""
        try:
            logger.info(f"🔄 데이터 파일 처리 시작: {file_path}")
            
            # 파일 타입 자동 감지
            if not file_type:
                file_type = self._detect_file_type(file_path)
            
            # 파일 읽기
            raw_data = self._read_file(file_path, file_type)
            
            # 데이터 검증
            validated_data = self._validate_data(raw_data)
            
            # 데이터 변환
            transformed_data = self._transform_data(validated_data)
            
            # 결과 저장
            with self._lock:
                self.processed_data.extend(transformed_data)
                self.processing_stats['total_processed'] += len(transformed_data)
                self.processing_stats['successful'] += len(transformed_data)
                self.processing_stats['last_processed'] = datetime.now()
            
            # 캐시 업데이트
            self._update_cache(file_path, transformed_data)
            
            result = {
                'success': True,
                'file_path': file_path,
                'file_type': file_type,
                'records_processed': len(transformed_data),
                'processing_time': datetime.now().isoformat()
            }
            
            logger.info(f" 데이터 파일 처리 완료: {file_path} - {len(transformed_data)}개 레코드")
            return result
            
        except Exception as e:
            error_msg = f"데이터 파일 처리 실패 {file_path}: {e}"
            self.error_log.append(error_msg)
            logger.error(error_msg)
            
            with self._lock:
                self.processing_stats['failed'] += 1
            
            return {
                'success': False,
                'file_path': file_path,
                'error': str(e),
                'processing_time': datetime.now().isoformat()
            }
    
    def _detect_file_type(self, file_path: str) -> str:
        """파일 타입 자동 감지"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            return 'json'
        elif file_extension == '.csv':
            return 'csv'
        elif file_extension == '.txt':
            return 'text'
        elif file_extension in ['.xlsx', '.xls']:
            return 'excel'
        else:
            return 'unknown'
    
    def _read_file(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """파일 읽기"""
        try:
            if file_type == 'json':
                return self._read_json_file(file_path)
            elif file_type == 'csv':
                return self._read_csv_file(file_path)
            elif file_type == 'text':
                return self._read_text_file(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
        except Exception as e:
            logger.error(f"파일 읽기 실패 {file_path}: {e}")
            raise
    
    def _read_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """JSON 파일 읽기"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"예상치 못한 JSON 구조: {type(data)}")
    
    def _read_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """CSV 파일 읽기"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def _read_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """텍스트 파일 읽기 (간단한 파싱)"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # 간단한 key=value 형식 파싱
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        data.append({
                            'id': f"line_{line_num}",
                            'key': parts[0].strip(),
                            'value': parts[1].strip(),
                            'line_number': line_num
                        })
        return data
    
    def _validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 검증"""
        validated_data = []
        
        for record in data:
            try:
                if self._is_valid_record(record):
                    validated_data.append(record)
                else:
                    logger.warning(f"⚠️  유효하지 않은 레코드 건너뜀: {record}")
            except Exception as e:
                logger.error(f"레코드 검증 실패: {e}")
                continue
        
        logger.info(f" 데이터 검증 완료: {len(validated_data)}/{len(data)} 레코드 유효")
        return validated_data
    
    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        """개별 레코드 유효성 검사"""
        try:
            # 필수 필드 확인
            for field in self.validation_rules['required_fields']:
                if field not in record or record[field] is None:
                    return False
            
            # 필드 타입 확인
            for field, expected_type in self.validation_rules['field_types'].items():
                if field in record and record[field] is not None:
                    if not isinstance(record[field], expected_type):
                        return False
            
            # 필드 제약 조건 확인
            for field, constraints in self.validation_rules['field_constraints'].items():
                if field in record and record[field] is not None:
                    if not self._check_field_constraints(record[field], constraints):
                        return False
            
            # 커스텀 검증기 실행
            for field, validator in self.validation_rules['custom_validators'].items():
                if field in record and record[field] is not None:
                    if not validator(record[field]):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"레코드 검증 중 오류: {e}")
            return False
    
    def _check_field_constraints(self, value: Any, constraints: Dict[str, Any]) -> bool:
        """필드 제약 조건 확인"""
        try:
            if 'min_length' in constraints and isinstance(value, str):
                if len(value) < constraints['min_length']:
                    return False
            
            if 'max_length' in constraints and isinstance(value, str):
                if len(value) > constraints['max_length']:
                    return False
            
            if 'min' in constraints and isinstance(value, (int, float)):
                if value < constraints['min']:
                    return False
            
            if 'max' in constraints and isinstance(value, (int, float)):
                if value > constraints['max']:
                    return False
            
            if 'allowed_values' in constraints:
                if value not in constraints['allowed_values']:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_timestamp(self, timestamp: str) -> bool:
        """타임스탬프 검증"""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except Exception:
            return False
    
    def _validate_email(self, email: str) -> bool:
        """이메일 검증"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_phone(self, phone: str) -> bool:
        """전화번호 검증"""
        pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        return bool(re.match(pattern, phone))
    
    def _transform_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 변환 파이프라인 실행"""
        transformed_data = data.copy()
        
        for transform_func in self.transformation_pipeline:
            try:
                transformed_data = transform_func(transformed_data)
            except Exception as e:
                logger.error(f"데이터 변환 실패 {transform_func.__name__}: {e}")
        
        return transformed_data
    
    def _normalize_strings(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문자열 정규화"""
        for record in data:
            for key, value in record.items():
                if isinstance(value, str):
                    record[key] = value.strip().lower()
        return data
    
    def _convert_data_types(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 타입 변환"""
        for record in data:
            for key, value in record.items():
                try:
                    if key == 'value' and isinstance(value, str):
                        if '.' in value:
                            record[key] = float(value)
                        else:
                            record[key] = int(value)
                    elif key == 'timestamp' and isinstance(value, str):
                        record[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except Exception:
                    continue
        return data
    
    def _add_derived_fields(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """파생 필드 추가"""
        for record in data:
            # 해시 ID 생성
            if 'id' not in record:
                record['id'] = hashlib.md5(str(record).encode()).hexdigest()
            
            # 처리 시간 추가
            record['processed_at'] = datetime.now().isoformat()
            
            # 카테고리 그룹 추가
            if 'category' in record:
                record['category_group'] = self._get_category_group(record['category'])
        
        return data
    
    def _get_category_group(self, category: str) -> str:
        """카테고리 그룹 결정"""
        if category in ['A', 'B']:
            return 'high_priority'
        elif category in ['C']:
            return 'medium_priority'
        else:
            return 'low_priority'
    
    def _clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 정리"""
        cleaned_data = []
        
        for record in data:
            # 빈 값이 너무 많은 레코드 제거
            empty_fields = sum(1 for v in record.values() if v is None or v == '')
            if empty_fields < len(record) * 0.5:  # 50% 이상이 비어있으면 제거
                cleaned_data.append(record)
        
        return cleaned_data
    
    def _enrich_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 풍부화"""
        if not data:
            return data
        
        # 통계 정보 계산
        numeric_values = [r.get('value', 0) for r in data if isinstance(r.get('value'), (int, float))]
        
        if numeric_values:
            stats = {
                'mean': statistics.mean(numeric_values),
                'median': statistics.median(numeric_values),
                'std': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                'min': min(numeric_values),
                'max': max(numeric_values)
            }
            
            # 각 레코드에 통계 정보 추가
            for record in data:
                record['stats'] = stats
        
        return data
    
    def _update_cache(self, file_path: str, data: List[Dict[str, Any]]):
        """캐시 업데이트"""
        cache_key = hashlib.md5(file_path.encode()).hexdigest()
        self.data_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now(),
            'file_path': file_path
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        with self._lock:
            stats = self.processing_stats.copy()
            stats['current_time'] = datetime.now().isoformat()
            stats['uptime'] = (datetime.now() - stats['start_time']).total_seconds()
            stats['success_rate'] = stats['successful'] / max(stats['total_processed'], 1)
            stats['error_count'] = len(self.error_log)
        
        return stats
    
    def get_data_summary(self) -> Dict[str, Any]:
        """데이터 요약 정보"""
        if not self.processed_data:
            return {'message': '처리된 데이터가 없습니다'}
        
        # 카테고리별 통계
        categories = Counter(r.get('category', 'unknown') for r in self.processed_data)
        
        # 값 범위 분석
        values = [r.get('value', 0) for r in self.processed_data if isinstance(r.get('value'), (int, float))]
        
        summary = {
            'total_records': len(self.processed_data),
            'categories': dict(categories),
            'value_stats': {
                'count': len(values),
                'mean': statistics.mean(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0
            },
            'last_processed': self.processing_stats['last_processed'].isoformat() if self.processing_stats['last_processed'] else None
        }
        
        return summary
    
    def export_processed_data(self, output_file: str, format: str = 'json') -> bool:
        """처리된 데이터 내보내기"""
        try:
            if format == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.processed_data, f, ensure_ascii=False, indent=2, default=str)
            elif format == 'csv':
                if self.processed_data:
                    fieldnames = self.processed_data[0].keys()
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.processed_data)
            else:
                raise ValueError(f"지원하지 않는 출력 형식: {format}")
            
            logger.info(f"데이터 내보내기 완료: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"데이터 내보내기 실패: {e}")
            return False
    
    def clear_cache(self):
        """캐시 정리"""
        self.data_cache.clear()
        logger.info("캐시 정리 완료")
    
    def reset_stats(self):
        """통계 초기화"""
        with self._lock:
            self.processing_stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'start_time': datetime.now(),
                'last_processed': None
            }
        logger.info("통계 초기화 완료") 