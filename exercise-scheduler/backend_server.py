# backend_server.py
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from datetime import date
from decimal import Decimal

# --- FastAPI 앱 및 Pydantic 모델 정의 ---

app = FastAPI()

# API 응답의 데이터 구조를 정의하는 Pydantic 모델
# 이 모델을 사용하면 FastAPI가 자동으로 데이터 검증 및 JSON 변환을 처리합니다.
class LogEntry(BaseModel):
    user_id: str
    exercise_date: date
    completion_rate: float = Field(..., description="운동 완료율. Decimal에서 float으로 변환됨")
    memo: str

    class Config:
        # Pydantic v2 or higher
        from_attributes = True 
        # Pydantic v1
        # orm_mode = True

# --- 데이터베이스 연결 정보 및 함수 ---

# setup_db.py와 동일한 DB 연결 정보
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'synergym',
    'password': '1234',
    'host': '192.168.2.6',
    'port': '5432'
}

def get_db_connection():
    """PostgreSQL 데이터베이스 연결을 생성합니다."""
    return psycopg2.connect(**DB_CONFIG)

# --- API 엔드포인트 정의 ---

@app.get("/")
def read_root():
    return {"message": "SynergyM 운동 기록 분석 API 서버"}

# LangGraph가 호출할 API 엔드포인트
@app.get("/api/logs/user/{userId}", response_model=List[LogEntry])
def get_user_exercise_history(user_id: str):
    """특정 사용자의 운동 기록을 DB에서 조회하여 반환합니다."""
    conn = get_db_connection()
    # 결과를 딕셔너리 형태로 받기 위해 cursor_factory 사용
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # SQL 쿼리 실행 (SQL 인젝션 방지를 위해 파라미터 바인딩 사용)
    query = 'SELECT user_id, exercise_date, completion_rate, memo FROM "Exercise_Logs" WHERE user_id = %s'
    cursor.execute(query, (user_id,))
    
    # 조회된 모든 데이터를 가져옴
    logs = cursor.fetchall()
    
    # 연결 종료
    cursor.close()
    conn.close()
    
    # 조회된 데이터를 Pydantic 모델 리스트로 반환
    # FastAPI가 이 리스트를 JSON으로 자동 변환해줍니다.
    return logs