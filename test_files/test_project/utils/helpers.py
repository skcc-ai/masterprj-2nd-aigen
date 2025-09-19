# -*- coding: utf-8 -*-
"""
Legacy Helper Utils
"""

import hashlib
import random
import string
import datetime
import json


class LegacyStringUtils:
    """문자열 유틸리티 - 레거시 스타일"""
    
    @staticmethod
    def generate_random_string(length):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))
    
    @staticmethod
    def mask_phone_number(phone_number):
        if not phone_number or len(phone_number) < 8:
            return phone_number
        return phone_number[:3] + "*" * (len(phone_number) - 6) + phone_number[-3:]
    
    @staticmethod
    def format_phone_number(phone_number):
        clean_number = phone_number.replace('-', '').replace(' ', '')
        if len(clean_number) == 11:
            return f"{clean_number[:3]}-{clean_number[3:7]}-{clean_number[7:]}"
        return phone_number
    
    @staticmethod
    def format_business_number(business_number):
        clean_number = business_number.replace('-', '')
        if len(clean_number) == 10:
            return f"{clean_number[:3]}-{clean_number[3:5]}-{clean_number[5:]}"
        return business_number


class LegacyNumberUtils:
    """숫자 유틸리티 - 레거시 스타일"""
    
    @staticmethod
    def format_currency(amount):
        return f"₩{amount:,.0f}"
    
    @staticmethod
    def calculate_discount_amount(original_price, discount_rate):
        return original_price * (discount_rate / 100)
    
    @staticmethod
    def calculate_tax(amount, tax_rate=0.1):
        return amount * tax_rate
    
    @staticmethod
    def round_currency(amount):
        return int(round(float(amount)))


class LegacyDateUtils:
    """날짜 유틸리티 - 레거시 스타일"""
    
    @staticmethod
    def get_current_timestamp():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def calculate_age(birth_date):
        today = datetime.date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    
    @staticmethod
    def add_months(date, months):
        # 간단한 월 추가 - 레거시 방식
        year = date.year
        month = date.month + months
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
        try:
            return datetime.date(year, month, date.day)
        except ValueError:
            # 말일 처리
            return datetime.date(year, month, 28)


class LegacySecurityUtils:
    """보안 유틸리티 - 레거시 스타일"""
    
    @staticmethod
    def hash_password(password, salt=None):
        if not salt:
            salt = LegacyStringUtils.generate_random_string(16)
        
        hash_object = hashlib.sha256((password + salt).encode())
        return {
            "hash": hash_object.hexdigest(),
            "salt": salt
        }
    
    @staticmethod
    def generate_token():
        return LegacyStringUtils.generate_random_string(32)
    
    @staticmethod
    def generate_otp():
        return ''.join(random.choices(string.digits, k=6))


class LegacyDataUtils:
    """데이터 유틸리티 - 레거시 스타일"""
    
    @staticmethod
    def dict_to_json(data):
        return json.dumps(data, ensure_ascii=False, default=str)
    
    @staticmethod
    def json_to_dict(json_str):
        return json.loads(json_str)
    
    @staticmethod
    def remove_none_values(data):
        return {k: v for k, v in data.items() if v is not None}
    
    @staticmethod
    def paginate_list(data, page, page_size):
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        return {
            "items": data[start_index:end_index],
            "total_count": len(data),
            "page": page,
            "page_size": page_size
        }


# 글로벌 캐시 - 레거시 스타일
_GLOBAL_CACHE = {}

def set_cache(key, value):
    """글로벌 캐시 설정"""
    _GLOBAL_CACHE[key] = value

def get_cache(key, default=None):
    """글로벌 캐시 조회"""
    return _GLOBAL_CACHE.get(key, default)

def clear_cache():
    """글로벌 캐시 클리어"""
    _GLOBAL_CACHE.clear()
