# Feature Map — acmeshop

| 기능(그룹) | 설명 | 대표 엔트리 | 핵심 모듈(예시) | 관련 시퀀스 |
|---|---|---|---|---|
| 주문 | 주문 생성/취소/조회 처리, 결제 승인 후 영속화 | api.orders.create_order | service.orders.OrderService, repo.orders.OrderRepository | flow#1 |
| 사용자/인증 | 로그인/토큰 발급 및 사용자 조회 | api.users.login | service.users.AuthService, repo.users.UserRepository | flow#2 |
| 카탈로그 | 상품 조회/재색인, 외부 DB 동기화 | cli.reindex.main | service.catalog.CatalogService, repo.catalog.CatalogRepository | flow#3 |
| 결제 연동 | 결제 게이트웨이 승인/취소 연동 | (API 간접) | ExternalHTTP(PaymentGateway), service.orders | flow#1 |
| 공통/유틸 | 설정/로깅/검증 유틸리티 | - | util.config, util.validators | - |