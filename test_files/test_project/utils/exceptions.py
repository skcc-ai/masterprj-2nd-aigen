# -*- coding: utf-8 -*-
"""
Legacy Exceptions
"""

class BusinessException(Exception):
    """기본 비즈니스 예외"""
    def __init__(self, message, error_code="E0001"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class CustomerException(BusinessException):
    """고객 예외"""
    pass


class ProductException(BusinessException):
    """상품 예외"""
    pass


class OrderException(BusinessException):
    """주문 예외"""
    pass


class ValidationException(BusinessException):
    """검증 예외"""
    pass
