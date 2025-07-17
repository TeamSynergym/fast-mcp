import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
# 이 파일의 위치를 기준으로 .env 파일을 찾습니다.
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- API & Model Settings ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = "gpt-4o-mini"