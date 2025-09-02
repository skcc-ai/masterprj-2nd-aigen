# Source Analysis Report — acmeshop
- Symbols: 148
- Calls: 362
- Generated: 2025-09-01T12:00:00+09:00
- Provenance: run_id=run_20250901_1200, model=gpt-4o-mini, embed=text-embedding-3-large

## 1) 개요
이 코드베이스는 **API → Service → Repository**의 3계층 구조를 따른다. 주문·사용자·카탈로그·결제 모듈이 분리되어 있으며, 주문 생성 시 **결제 승인(ExternalHTTP)** 후 **저장소 반영(Repository)** 순으로 처리된다. 엔트리는 주로 `api.orders`, `api.users`, 일부 `cli.reindex`에서 시작된다.

## 2) 핵심 엔트리(Top 5)
- `py://acmeshop.api.orders#create_order` (file: `acmeshop/api/orders.py`, in_degree=0)
- `py://acmeshop.api.users#login`
- `py://acmeshop.api.orders#cancel_order`
- `py://acmeshop.cli.reindex#main`
- `py://acmeshop.api.catalog#list_products`

## 3) 핫스팟(연결도 Top 5)
- `py://acmeshop.service.orders#OrderService` (deg=54)
- `py://acmeshop.repo.orders#OrderRepository` (deg=41)
- `py://acmeshop.service.users#AuthService` (deg=33)
- `py://acmeshop.repo.users#UserRepository` (deg=29)
- `py://acmeshop.service.catalog#CatalogService` (deg=27)

## 4) 싱크 후보(출구/저장소 Top 5)
- `py://acmeshop.repo.orders#OrderRepository` (out_degree=0)
- `py://acmeshop.repo.users#UserRepository`
- `py://acmeshop.repo.catalog#CatalogRepository`
- `ExternalHTTP(PaymentGateway)`  *calls.external_http*
- `ExternalDB(Analytics)`  *calls.external_db*

## 5) 구조 하이라이트
- **주문 흐름**: API → `OrderService.create_order` → ExternalHTTP(PaymentGateway) 승인 → `OrderRepository.save`  
- **인증 흐름**: API → `AuthService.login` → `UserRepository.get_by_email` → 토큰 발급  
- **카탈로그 리빌드**: CLI → `CatalogService.reindex` → ExternalDB → `CatalogRepository.upsert_batch`

## 6) 잠재 리스크/개선 포인트
- **레이어 위반 1건**: `api.orders.cancel_order`가 `OrderRepository`를 **직접 호출**함 → Service를 경유하도록 수정 권장  
- **결제 실패 예외 전파**: `PaymentGateway` 오류에 대한 `retry/backoff` 정책 명시 부족  
- **핫스팟 집중**: `OrderService` 중심성 과다(deg=54) → 서브서비스 분리 검토

> 근거: calls/symbols 테이블, 심볼카드 요약. 상세는 `claims.json` 및 `provenance.json` 참조.