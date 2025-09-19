#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKT Legacy System Server
간소화된 실행 파일
"""

if __name__ == "__main__":
    import uvicorn
    from legacy_api import app
    
    print("=" * 50)
    print("SKT Legacy System")
    print("=" * 50)
    print("Starting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Sample Data: http://localhost:8000/init/sample-data")
    print("=" * 50)
    
    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
