# -*- coding: utf-8 -*-
"""
SKT Legacy API Server
간소화된 레거시 API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import datetime
from typing import Optional, List
import legacy_system as legacy

# FastAPI 앱 생성
app = FastAPI(
    title="SKT Legacy System API",
    description="SKT 레거시 시스템 API (간소화 버전)",
    version="1.0.0"
)

# Request 모델들
class CustomerRequest(BaseModel):
    customer_name: str
    customer_type: str
    phone_number: str
    email: Optional[str] = None
    registration_number: Optional[str] = None

class ProductRequest(BaseModel):
    product_name: str
    product_type: str
    price: float

class OrderRequest(BaseModel):
    customer_id: int
    product_id: int
    order_type: str
    order_amount: float

class BillingRequest(BaseModel):
    customer_id: int
    product_type: str
    voice_seconds: int = 0
    data_mb: int = 0
    sms_count: int = 0

# API 엔드포인트들
@app.get("/")
def root():
    return {"message": "SKT Legacy System API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.datetime.now()}

# 고객 관리 API
@app.post("/customers")
def create_customer(request: CustomerRequest):
    success, result = legacy.create_customer(
        request.customer_name,
        request.customer_type, 
        request.phone_number,
        request.email,
        request.registration_number
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=result)
    
    return {"success": True, "customer_id": result}

@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    customer = legacy.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@app.get("/customers")
def search_customers(name: Optional[str] = None, phone: Optional[str] = None, customer_type: Optional[str] = None):
    customers = legacy.search_customers(name, phone, customer_type)
    return {"customers": customers, "count": len(customers)}

@app.put("/customers/{customer_id}/grade")
def update_customer_grade(customer_id: int, grade: str):
    success = legacy.update_customer_grade(customer_id, grade)
    if not success:
        raise HTTPException(status_code=404, detail="Customer not found")
    return {"success": True}

# 상품 관리 API
@app.post("/products")
def create_product(request: ProductRequest):
    success, result = legacy.create_product(
        request.product_name,
        request.product_type,
        request.price
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=result)
    
    return {"success": True, "product_id": result}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    product = legacy.get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/products")
def get_products(product_type: Optional[str] = None):
    if product_type:
        products = legacy.get_products_by_type(product_type)
    else:
        products = legacy.get_all_products()
    return {"products": products, "count": len(products)}

# 주문 관리 API
@app.post("/orders")
def create_order(request: OrderRequest):
    success, result = legacy.create_order(
        request.customer_id,
        request.product_id,
        request.order_type,
        request.order_amount
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=result)
    
    return {"success": True, "order_id": result}

@app.get("/orders/{order_id}")
def get_order(order_id: int):
    order = legacy.get_order_by_id(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@app.put("/orders/{order_id}/status")
def update_order_status(order_id: int, status: str):
    success = legacy.update_order_status(order_id, status)
    if not success:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"success": True}

@app.get("/customers/{customer_id}/orders")
def get_customer_orders(customer_id: int):
    orders = legacy.get_orders_by_customer(customer_id)
    return {"orders": orders, "count": len(orders)}

# 요금계산 API
@app.post("/billing/calculate")
def calculate_bill(request: BillingRequest):
    bill = legacy.calculate_monthly_bill(
        request.customer_id,
        request.product_type,
        request.voice_seconds,
        request.data_mb,
        request.sms_count
    )
    return bill

@app.post("/billing/promotion")
def apply_promotion(customer_id: int, order_amount: float, product_type: str):
    promotion = legacy.apply_promotion(customer_id, order_amount, product_type)
    return promotion

@app.get("/billing/credit/{customer_id}")
def get_credit_score(customer_id: int):
    credit = legacy.calculate_credit_score(customer_id)
    if not credit:
        raise HTTPException(status_code=404, detail="Customer not found")
    return credit

# 배치 작업 API
@app.post("/batch/billing")
def run_billing_batch():
    processed = legacy.run_billing_batch()
    return {"message": "Billing batch completed", "processed": processed}

@app.post("/batch/grade-update")
def run_grade_update():
    updated = legacy.run_grade_update_batch()
    return {"message": "Grade update completed", "updated": updated}

@app.post("/batch/cleanup")
def run_cleanup():
    deleted = legacy.cleanup_old_orders()
    return {"message": "Cleanup completed", "deleted": deleted}

# 통계 API
@app.get("/stats/customers")
def get_customer_statistics():
    stats = legacy.get_customer_stats()
    return stats

@app.get("/stats/orders")
def get_order_statistics():
    stats = legacy.get_order_stats()
    return stats

# 초기화 API
@app.post("/init/sample-data")
def create_sample_data():
    legacy.create_sample_data()
    return {"message": "Sample data created"}

# 시작 시 데이터베이스 초기화
@app.on_event("startup")
async def startup():
    print("Initializing Legacy System...")
    legacy.init_database()
    
    # 샘플 데이터가 없으면 생성
    if not legacy.search_customers():
        legacy.create_sample_data()
    
    print("Legacy System ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
