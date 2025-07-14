import psycopg2
from datetime import date
from decimal import Decimal

conn = psycopg2.connect(
    dbname='postgres',           # Spring 설정에 따라 DB 이름
    user='synergym',             # 사용자 이름
    password='1234',             # 비밀번호
    host='192.168.2.6',          # PostgreSQL 서버 주소
    port='5432'                  # 기본 포트
)
cursor = conn.cursor()

# 예제 데이터
sample_data = [
    ('user123', date(2025, 7, 7), Decimal('80.50'), '달리기 컨디션 좋음'),
    ('user123', date(2025, 7, 9), Decimal('75.25'), '조금 피곤했음'),
    ('user123', date(2025, 7, 11), Decimal('90.00'), '베스트 컨디션!'),
    ('user123', date(2025, 7, 13), Decimal('60.00'), '날씨가 너무 더웠음')
]

cursor.executemany('''
INSERT INTO "Exercise_Logs" (user_id, exercise_date, completion_rate, memo)
VALUES (%s, %s, %s, %s)
''', sample_data)

# 저장 및 종료
conn.commit()
conn.close()

print("✅ PostgreSQL에 Exercise_Logs 테이블 및 예제 데이터 삽입 완료!")
