# -*- coding: utf-8 -*-
"""
Legacy Billing System
레거시 요금계산 시스템
"""

import datetime
from decimal import Decimal


class LegacyBillingEngine:
    """레거시 요금계산 엔진"""
    
    def __init__(self):
        self.tax_rate = 0.1  # 10% 부가세
        # 하드코딩된 요금표 - 레거시 스타일
        self.voice_rate = 3.0  # 초당 3원
        self.data_rate = 0.01  # MB당 0.01원
        self.sms_rate = 20     # 건당 20원
        self.basic_rates = {
            "BASIC": 30000,    # 기본 요금제 30,000원
            "PREMIUM": 55000,  # 프리미엄 요금제 55,000원
            "UNLIMITED": 89000 # 무제한 요금제 89,000원
        }
        
    def calculate_monthly_bill(self, customer_id, product_type, usage_data):
        """월 요금 계산 - 레거시 방식"""
        # 기본료
        basic_fee = self.basic_rates.get(product_type, 30000)
        
        # 사용료 계산
        voice_usage = usage_data.get('voice_seconds', 0)
        data_usage = usage_data.get('data_mb', 0)
        sms_usage = usage_data.get('sms_count', 0)
        
        # 무료 제공량 차감 (하드코딩)
        if product_type == "PREMIUM":
            voice_usage = max(0, voice_usage - 1800)  # 30분 무료
            data_usage = max(0, data_usage - 3072)    # 3GB 무료
            sms_usage = max(0, sms_usage - 200)       # 200건 무료
        elif product_type == "UNLIMITED":
            voice_usage = 0  # 무제한
            data_usage = 0   # 무제한
            sms_usage = 0    # 무제한
        
        # 사용료 계산
        voice_fee = voice_usage * self.voice_rate
        data_fee = data_usage * self.data_rate
        sms_fee = sms_usage * self.sms_rate
        
        usage_fee = voice_fee + data_fee + sms_fee
        
        # 할인 적용 (간단한 로직)
        discount = 0
        if self._is_vip_customer(customer_id):
            discount = (basic_fee + usage_fee) * 0.1  # 10% 할인
        
        subtotal = basic_fee + usage_fee - discount
        tax = subtotal * self.tax_rate
        total = subtotal + tax
        
        return {
            "customer_id": customer_id,
            "basic_fee": int(basic_fee),
            "usage_fee": int(usage_fee),
            "discount": int(discount),
            "tax": int(tax),
            "total": int(total),
            "details": {
                "voice_fee": int(voice_fee),
                "data_fee": int(data_fee),
                "sms_fee": int(sms_fee)
            }
        }
    
    def _is_vip_customer(self, customer_id):
        """VIP 고객 여부 - 하드코딩"""
        # 실제로는 DB에서 조회하지만 여기서는 간단히
        vip_customers = [1, 2, 3, 100, 200]  # 하드코딩된 VIP 고객 목록
        return customer_id in vip_customers


class LegacyPromotionEngine:
    """레거시 프로모션 엔진"""
    
    def __init__(self):
        # 하드코딩된 프로모션 목록
        self.active_promotions = [
            {
                "id": "NEWBIE_20",
                "name": "신규가입 20% 할인",
                "discount_rate": 20,
                "min_amount": 50000
            },
            {
                "id": "SUMMER_50K",
                "name": "여름 특가 5만원 할인",
                "discount_amount": 50000,
                "target_products": ["PREMIUM", "UNLIMITED"]
            }
        ]
    
    def apply_promotions(self, customer_id, order_amount, product_type):
        """프로모션 적용"""
        total_discount = 0
        applied_promos = []
        
        for promo in self.active_promotions:
            if self._is_eligible(customer_id, order_amount, product_type, promo):
                if "discount_rate" in promo:
                    discount = order_amount * (promo["discount_rate"] / 100)
                elif "discount_amount" in promo:
                    discount = promo["discount_amount"]
                else:
                    discount = 0
                    
                total_discount += discount
                applied_promos.append(promo["name"])
        
        return {
            "total_discount": int(total_discount),
            "applied_promotions": applied_promos
        }
    
    def _is_eligible(self, customer_id, order_amount, product_type, promo):
        """프로모션 자격 확인"""
        # 최소 금액 확인
        if promo.get("min_amount", 0) > order_amount:
            return False
        
        # 대상 상품 확인
        if "target_products" in promo and product_type not in promo["target_products"]:
            return False
        
        # 신규 고객 여부 (간단히 customer_id로 판단)
        if promo["id"] == "NEWBIE_20" and customer_id > 1000:
            return False
        
        return True


class LegacyCreditScorer:
    """레거시 신용평가 시스템"""
    
    def calculate_credit_score(self, customer_data):
        """신용점수 계산 - 간단한 룰 기반"""
        score = 500  # 기본 점수
        
        # 연령 점수
        age = customer_data.get("age", 30)
        if 25 <= age <= 45:
            score += 50
        elif 20 <= age < 25 or 45 < age <= 60:
            score += 20
        else:
            score -= 30
        
        # 고객 유형 점수
        customer_type = customer_data.get("customer_type", "INDIVIDUAL")
        if customer_type == "CORPORATE":
            score += 100
        elif customer_type == "POSTPAID":
            score += 30
        
        # 가입 기간 점수
        join_months = customer_data.get("join_months", 0)
        if join_months >= 24:
            score += 80
        elif join_months >= 12:
            score += 40
        elif join_months >= 6:
            score += 20
        
        # 주문 완료율 점수
        completion_rate = customer_data.get("order_completion_rate", 0.8)
        if completion_rate >= 0.95:
            score += 60
        elif completion_rate >= 0.8:
            score += 30
        
        # 점수 범위 조정
        score = max(300, min(850, score))
        
        # 등급 결정
        if score >= 750:
            grade = "A"
        elif score >= 650:
            grade = "B"
        elif score >= 550:
            grade = "C"
        else:
            grade = "D"
        
        return {
            "score": score,
            "grade": grade,
            "factors": {
                "age": age,
                "customer_type": customer_type,
                "join_months": join_months,
                "completion_rate": completion_rate
            }
        }


# 글로벌 인스턴스 - 레거시 방식
billing_engine = LegacyBillingEngine()
promotion_engine = LegacyPromotionEngine()
credit_scorer = LegacyCreditScorer()


def calculate_bill(customer_id, product_type, usage_data):
    """전역 함수 - 레거시 스타일"""
    return billing_engine.calculate_monthly_bill(customer_id, product_type, usage_data)


def apply_promotions(customer_id, order_amount, product_type):
    """전역 함수 - 레거시 스타일"""
    return promotion_engine.apply_promotions(customer_id, order_amount, product_type)


def get_credit_score(customer_data):
    """전역 함수 - 레거시 스타일"""
    return credit_scorer.calculate_credit_score(customer_data)
