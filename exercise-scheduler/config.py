# config.py
"""
프로젝트의 모든 설정 값을 중앙에서 관리하는 파일입니다.
API 엔드포인트, 모델 이름, 기본값 등을 정의합니다.
"""
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
# 이 파일의 위치를 기준으로 .env 파일을 찾습니다.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- API & Model Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = "gpt-4o-mini"
BACKEND_API_URL = "http://localhost:8081"
API_TIMEOUT = 5

# --- Test/Default Values ---
# 실제 운영 시에는 이 값을 동적으로 받아와야 합니다.
DEFAULT_USER_ID = ""
DEFAULT_JWT_TOKEN = ""