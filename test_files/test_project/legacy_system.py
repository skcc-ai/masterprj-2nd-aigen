# -*- coding: utf-8 -*-
"""
SKT Legacy Business System
통합 레거시 시스템
"""

import sqlite3
import datetime
import random
import string
import hashlib


# 글로벌 데이터베이스 연결
DB_NAME = "skt_legacy.db"
_db_connection = None


def get_db_connection():
    """데이터베이스 연결 반환"""
    global _db_connection
    if _db_connection is None:
        _db_connection = sqlite3.connect(DB_NAME, check_same_thread=False)
        _db_connection.row_factory = sqlite3.Row
        init_database()
    return _db_connection


def init_database():
    """데이터베이스 초기화"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 고객 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_number VARCHAR(20) UNIQUE,
            customer_name VARCHAR(100),
            customer_type VARCHAR(20),
            customer_status VARCHAR(20) DEFAULT 'ACTIVE',
            phone_number VARCHAR(20),
            email VARCHAR(100),
            registration_number VARCHAR(20),
            credit_grade VARCHAR(5) DEFAULT 'C',
            join_date DATETIME,
            birth_date DATE
        )
    """)
    
    # 상품 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_code VARCHAR(20) UNIQUE,
            product_name VARCHAR(200),
            product_type VARCHAR(20),
            price DECIMAL(10,2),
            status VARCHAR(20) DEFAULT 'ACTIVE'
        )
    """)
    
    # 주문 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_number VARCHAR(20) UNIQUE,
            customer_id INTEGER,
            product_id INTEGER,
            order_type VARCHAR(20),
            order_status VARCHAR(20) DEFAULT 'PENDING',
            order_date DATETIME,
            order_amount DECIMAL(10,2),
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
    """)
    
    conn.commit()


# 유틸리티 함수들
def generate_number(prefix, length=8):
    """번호 생성"""
    random_part = ''.join(random.choices(string.digits, k=length))
    return f"{prefix}{random_part}"


def validate_phone(phone_number):
    """전화번호 검증"""
    clean_phone = phone_number.replace('-', '').replace(' ', '')
    return len(clean_phone) == 11 and clean_phone.startswith('01')


def validate_business_number(business_number):
    """사업자번호 검증"""
    clean_number = business_number.replace('-', '')
    if len(clean_number) != 10 or not clean_number.isdigit():
        return False
    
    # 간단한 체크섬
    check_array = [1, 3, 7, 1, 3, 7, 1, 3, 5]
    check_sum = 0
    for i in range(9):
        check_sum += int(clean_number[i]) * check_array[i]
    check_sum += int((int(clean_number[8]) * 5) / 10)
    check_digit = (10 - (check_sum % 10)) % 10
    return check_digit == int(clean_number[9])


def hash_password(password):
    """비밀번호 해싱"""
    salt = ''.join(random.choices(string.ascii_letters, k=16))
    hash_obj = hashlib.sha256((password + salt).encode())
    return {"hash": hash_obj.hexdigest(), "salt": salt}


# 고객 관리 함수들
def create_customer(customer_name, customer_type, phone_number, email=None, registration_number=None):
    """고객 생성"""
    if not validate_phone(phone_number):
        return False, "잘못된 전화번호 형식"
    
    # 중복 확인
    if get_customer_by_phone(phone_number):
        return False, "이미 등록된 전화번호"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    customer_number = generate_number("C", 8)
    join_date = datetime.datetime.now()
    
    try:
        cursor.execute("""
            INSERT INTO customers 
            (customer_number, customer_name, customer_type, phone_number, email, 
             registration_number, join_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (customer_number, customer_name, customer_type, phone_number, 
              email, registration_number, join_date))
        
        conn.commit()
        return True, cursor.lastrowid
    except Exception as e:
        return False, str(e)


def get_customer_by_id(customer_id):
    """고객 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    result = cursor.fetchone()
    return dict(result) if result else None


def get_customer_by_phone(phone_number):
    """전화번호로 고객 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM customers WHERE phone_number = ?", (phone_number,))
    result = cursor.fetchone()
    return dict(result) if result else None


def search_customers(name=None, phone=None, customer_type=None):
    """고객 검색"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM customers WHERE 1=1"
    params = []
    
    if name:
        query += " AND customer_name LIKE ?"
        params.append(f"%{name}%")
    if phone:
        query += " AND phone_number LIKE ?"
        params.append(f"%{phone}%")
    if customer_type:
        query += " AND customer_type = ?"
        params.append(customer_type)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    return [dict(row) for row in results]


def update_customer_grade(customer_id, new_grade):
    """고객 등급 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE customers SET credit_grade = ? WHERE customer_id = ?",
        (new_grade, customer_id)
    )
    conn.commit()
    return cursor.rowcount > 0


# 상품 관리 함수들
def create_product(product_name, product_type, price):
    """상품 생성"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    product_code = generate_number("P", 6)
    
    try:
        cursor.execute("""
            INSERT INTO products (product_code, product_name, product_type, price) 
            VALUES (?, ?, ?, ?)
        """, (product_code, product_name, product_type, price))
        
        conn.commit()
        return True, cursor.lastrowid
    except Exception as e:
        return False, str(e)


def get_product_by_id(product_id):
    """상품 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
    result = cursor.fetchone()
    return dict(result) if result else None


def get_products_by_type(product_type):
    """상품 유형별 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM products WHERE product_type = ? AND status = 'ACTIVE'",
        (product_type,)
    )
    results = cursor.fetchall()
    return [dict(row) for row in results]


def get_all_products():
    """전체 상품 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM products WHERE status = 'ACTIVE'")
    results = cursor.fetchall()
    return [dict(row) for row in results]


# 주문 관리 함수들
def create_order(customer_id, product_id, order_type, order_amount):
    """주문 생성"""
    # 고객 존재 확인
    customer = get_customer_by_id(customer_id)
    if not customer:
        return False, "고객을 찾을 수 없음"
    
    # 상품 존재 확인
    product = get_product_by_id(product_id)
    if not product:
        return False, "상품을 찾을 수 없음"
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    order_number = generate_number("ORD", 10)
    order_date = datetime.datetime.now()
    
    try:
        cursor.execute("""
            INSERT INTO orders 
            (order_number, customer_id, product_id, order_type, order_date, order_amount) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (order_number, customer_id, product_id, order_type, order_date, order_amount))
        
        conn.commit()
        return True, cursor.lastrowid
    except Exception as e:
        return False, str(e)


def get_order_by_id(order_id):
    """주문 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT o.*, c.customer_name, p.product_name 
        FROM orders o 
        JOIN customers c ON o.customer_id = c.customer_id 
        JOIN products p ON o.product_id = p.product_id 
        WHERE o.order_id = ?
    """, (order_id,))
    
    result = cursor.fetchone()
    return dict(result) if result else None


def update_order_status(order_id, new_status):
    """주문 상태 변경"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE orders SET order_status = ? WHERE order_id = ?",
        (new_status, order_id)
    )
    conn.commit()
    return cursor.rowcount > 0


def get_orders_by_customer(customer_id):
    """고객별 주문 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT o.*, p.product_name 
        FROM orders o 
        JOIN products p ON o.product_id = p.product_id 
        WHERE o.customer_id = ? 
        ORDER BY o.order_date DESC
    """, (customer_id,))
    
    results = cursor.fetchall()
    return [dict(row) for row in results]


# 요금계산 함수들 (레거시 스타일)
def calculate_monthly_bill(customer_id, product_type, voice_seconds=0, data_mb=0, sms_count=0):
    """월 요금 계산"""
    # 하드코딩된 요금표
    basic_rates = {"BASIC": 30000, "PREMIUM": 55000, "UNLIMITED": 89000}
    voice_rate = 3.0  # 초당 3원
    data_rate = 0.01  # MB당 0.01원
    sms_rate = 20     # 건당 20원
    
    basic_fee = basic_rates.get(product_type, 30000)
    
    # 무료 제공량 차감
    if product_type == "PREMIUM":
        voice_seconds = max(0, voice_seconds - 1800)  # 30분 무료
        data_mb = max(0, data_mb - 3072)              # 3GB 무료
        sms_count = max(0, sms_count - 200)           # 200건 무료
    elif product_type == "UNLIMITED":
        voice_seconds = data_mb = sms_count = 0       # 무제한
    
    # 사용료 계산
    voice_fee = voice_seconds * voice_rate
    data_fee = data_mb * data_rate
    sms_fee = sms_count * sms_rate
    usage_fee = voice_fee + data_fee + sms_fee
    
    # VIP 할인 (고객ID가 100 이하면 VIP)
    discount = 0
    if customer_id <= 100:
        discount = (basic_fee + usage_fee) * 0.1
    
    subtotal = basic_fee + usage_fee - discount
    tax = subtotal * 0.1
    total = subtotal + tax
    
    return {
        "customer_id": customer_id,
        "basic_fee": int(basic_fee),
        "usage_fee": int(usage_fee),
        "discount": int(discount),
        "tax": int(tax),
        "total": int(total)
    }


def apply_promotion(customer_id, order_amount, product_type):
    """프로모션 적용"""
    discount = 0
    promotion_name = ""
    
    # 신규가입 할인 (고객ID 1000 이상)
    if customer_id >= 1000 and order_amount >= 50000:
        discount = order_amount * 0.2
        promotion_name = "신규가입 20% 할인"
    
    # 프리미엄 요금제 특가
    elif product_type in ["PREMIUM", "UNLIMITED"]:
        discount = 50000
        promotion_name = "프리미엄 5만원 할인"
    
    return {
        "discount_amount": int(discount),
        "promotion_name": promotion_name
    }


def calculate_credit_score(customer_id):
    """신용점수 계산"""
    customer = get_customer_by_id(customer_id)
    if not customer:
        return None
    
    score = 500  # 기본 점수
    
    # 고객 유형별 점수
    if customer.get("customer_type") == "CORPORATE":
        score += 100
    elif customer.get("customer_type") == "POSTPAID":
        score += 50
    
    # 가입기간별 점수
    if customer.get("join_date"):
        join_date = datetime.datetime.strptime(customer["join_date"], "%Y-%m-%d %H:%M:%S")
        months = (datetime.datetime.now() - join_date).days // 30
        if months >= 24:
            score += 80
        elif months >= 12:
            score += 40
    
    # 주문 이력별 점수
    orders = get_orders_by_customer(customer_id)
    if len(orders) > 10:
        score += 60
    elif len(orders) > 5:
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
    
    return {"score": score, "grade": grade}


# 배치 작업 함수들
def run_billing_batch():
    """요금계산 배치"""
    print("Starting billing batch...")
    customers = search_customers()
    processed = 0
    
    for customer in customers:
        try:
            bill = calculate_monthly_bill(
                customer["customer_id"], 
                "PREMIUM",  # 기본 요금제
                3600,       # 1시간 통화
                2048,       # 2GB 데이터
                100         # 100건 SMS
            )
            # 실제로는 청구서 테이블에 저장
            processed += 1
        except Exception as e:
            print(f"Error processing customer {customer['customer_id']}: {e}")
    
    print(f"Billing batch completed. Processed: {processed}")
    return processed


def run_grade_update_batch():
    """고객등급 업데이트 배치"""
    print("Starting grade update batch...")
    customers = search_customers()
    updated = 0
    
    for customer in customers:
        try:
            credit_info = calculate_credit_score(customer["customer_id"])
            if credit_info and credit_info["grade"] != customer.get("credit_grade"):
                update_customer_grade(customer["customer_id"], credit_info["grade"])
                updated += 1
        except Exception as e:
            print(f"Error updating grade for customer {customer['customer_id']}: {e}")
    
    print(f"Grade update batch completed. Updated: {updated}")
    return updated


def cleanup_old_orders():
    """오래된 주문 정리"""
    print("Starting order cleanup...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 90일 이전의 취소된 주문 삭제
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=90)
    
    cursor.execute("""
        DELETE FROM orders 
        WHERE order_status = 'CANCELLED' 
        AND order_date < ?
    """, (cutoff_date,))
    
    deleted_count = cursor.rowcount
    conn.commit()
    
    print(f"Order cleanup completed. Deleted: {deleted_count}")
    return deleted_count


# 통계 함수들
def get_customer_stats():
    """고객 통계"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT customer_status, COUNT(*) as count FROM customers GROUP BY customer_status")
    status_counts = {row["customer_status"]: row["count"] for row in cursor.fetchall()}
    
    cursor.execute("SELECT credit_grade, COUNT(*) as count FROM customers GROUP BY credit_grade")
    grade_counts = {row["credit_grade"]: row["count"] for row in cursor.fetchall()}
    
    return {"status_counts": status_counts, "grade_counts": grade_counts}


def get_order_stats():
    """주문 통계"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT order_status, COUNT(*) as count FROM orders GROUP BY order_status")
    status_counts = {row["order_status"]: row["count"] for row in cursor.fetchall()}
    
    cursor.execute("SELECT SUM(order_amount) as total_amount FROM orders WHERE order_status = 'COMPLETED'")
    total_revenue = cursor.fetchone()["total_amount"] or 0
    
    return {"status_counts": status_counts, "total_revenue": float(total_revenue)}


# 초기 샘플 데이터 생성
def create_sample_data():
    """샘플 데이터 생성"""
    print("Creating sample data...")
    
    # 샘플 고객 생성
    customers = [
        ("홍길동", "INDIVIDUAL", "010-1234-5678", "hong@example.com"),
        ("김철수", "INDIVIDUAL", "010-2345-6789", "kim@example.com"), 
        ("SK텔레콤", "CORPORATE", "010-3456-7890", "skt@skt.com"),
        ("삼성전자", "CORPORATE", "010-4567-8901", "samsung@samsung.com"),
        ("이영희", "INDIVIDUAL", "010-5678-9012", "lee@example.com")
    ]
    
    for name, ctype, phone, email in customers:
        create_customer(name, ctype, phone, email)
    
    # 샘플 상품 생성
    products = [
        ("5G 기본 요금제", "MOBILE_PLAN", 30000),
        ("5G 프리미엄 요금제", "MOBILE_PLAN", 55000),
        ("5G 무제한 요금제", "MOBILE_PLAN", 89000),
        ("갤럭시 S23", "DEVICE", 1200000),
        ("아이폰 14", "DEVICE", 1300000)
    ]
    
    for name, ptype, price in products:
        create_product(name, ptype, price)
    
    # 샘플 주문 생성
    orders = [
        (1, 1, "NEW_SUBSCRIPTION", 30000),
        (2, 2, "NEW_SUBSCRIPTION", 55000),
        (3, 3, "NEW_SUBSCRIPTION", 89000),
        (1, 4, "DEVICE_PURCHASE", 1200000),
        (2, 5, "DEVICE_PURCHASE", 1300000)
    ]
    
    for customer_id, product_id, otype, amount in orders:
        create_order(customer_id, product_id, otype, amount)
    
    print("Sample data created successfully!")


if __name__ == "__main__":
    # 데이터베이스 초기화
    init_database()
    
    # 샘플 데이터가 없으면 생성
    if not search_customers():
        create_sample_data()
    
    print("SKT Legacy System initialized!")
    print(f"Database: {DB_NAME}")
    print(f"Customers: {len(search_customers())}")
    print(f"Products: {len(get_all_products())}")
