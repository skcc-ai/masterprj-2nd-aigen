# -*- coding: utf-8 -*-
"""
Legacy Validation Utils
"""

import re
import datetime
from models.customer import CustomerType


class ValidationUtils:
    """레거시 검증 유틸리티"""
    
    def __init__(self):
        self.patterns = {
            'phone': re.compile(r'^01[016789]-?\d{3,4}-?\d{4}$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'business': re.compile(r'^\d{3}-?\d{2}-?\d{5}$'),
            'resident': re.compile(r'^\d{6}-?\d{7}$')
        }
        
    def validate_phone_number(self, phone_number):
        if not phone_number:
            return False
        return bool(self.patterns['phone'].match(phone_number.replace('-', '')))
        
    def validate_email(self, email):
        if not email:
            return False
        return bool(self.patterns['email'].match(email))
        
    def validate_registration_number(self, reg_number, customer_type):
        if not reg_number:
            return False
            
        clean_number = reg_number.replace('-', '')
        
        if customer_type == CustomerType.INDIVIDUAL:
            # 주민번호 검증 - 레거시 스타일로 간소화
            if len(clean_number) != 13 or not clean_number.isdigit():
                return False
            try:
                year = int(clean_number[:2])
                month = int(clean_number[2:4])
                day = int(clean_number[4:6])
                if month < 1 or month > 12 or day < 1 or day > 31:
                    return False
                return True
            except:
                return False
                
        elif customer_type == CustomerType.CORPORATE:
            # 사업자번호 검증 - 체크섬 계산
            if len(clean_number) != 10 or not clean_number.isdigit():
                return False
            check_array = [1, 3, 7, 1, 3, 7, 1, 3, 5]
            check_sum = 0
            for i in range(9):
                check_sum += int(clean_number[i]) * check_array[i]
            check_sum += int((int(clean_number[8]) * 5) / 10)
            check_digit = (10 - (check_sum % 10)) % 10
            return check_digit == int(clean_number[9])
        
        return False
        
    def validate_business_rules(self, data):
        """통합 비즈니스 룰 검증 - 레거시 스타일"""
        errors = []
        
        # 주문 금액 체크
        if 'order_amount' in data:
            amount = data['order_amount']
            if amount <= 0:
                errors.append("주문금액 오류")
            if amount > 10000000:
                errors.append("주문금액 초과")
        
        # 계약 기간 체크
        if 'contract_months' in data:
            months = data['contract_months']
            if months <= 0 or months > 36:
                errors.append("계약기간 오류")
        
        # 연령 체크
        if 'birth_date' in data:
            try:
                birth_date = datetime.datetime.strptime(data['birth_date'], '%Y-%m-%d').date()
                today = datetime.date.today()
                age = today.year - birth_date.year
                if age < 19 or age > 120:
                    errors.append("연령 오류")
            except:
                errors.append("생년월일 형식 오류")
        
        # 할인율 체크
        if 'discount_rate' in data and 'customer_grade' in data:
            rate = data['discount_rate']
            grade = data['customer_grade']
            max_rates = {"A": 30, "B": 20, "C": 10, "D": 5}
            max_rate = max_rates.get(grade, 5)
            if rate < 0 or rate > max_rate:
                errors.append("할인율 오류")
        
        return len(errors) == 0, errors
