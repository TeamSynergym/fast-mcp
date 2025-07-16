# app/services/api_client.py
import requests
import json
from typing import List, Dict, Any, Optional

# config.py에서 설정 값을 가져옵니다.
from config import BACKEND_API_URL, API_TIMEOUT

def get_email_from_backend(user_id: str, jwt_token: str) -> Optional[str]:
    """
    백엔드 API 서버를 호출하여 특정 사용자의 이메일을 가져옵니다.
    """
    api_endpoint = f"{BACKEND_API_URL}/api/users/{user_id}"
    headers = {"Authorization": f"Bearer {jwt_token}"}
    
    try:
        response = requests.get(api_endpoint, headers=headers, timeout=API_TIMEOUT)
        response.raise_for_status()
        user_data = response.json()
        return user_data.get("email")
    except requests.exceptions.RequestException as e:
        print(f"🚨 이메일 조회 API 호출 오류: {e}")
        return None

def get_history_from_backend(user_id: str, jwt_token: str) -> List[Dict[str, Any]]:
    """
    백엔드 API 서버를 호출하여 특정 사용자의 운동 기록을 가져옵니다.
    """
    api_endpoint = f"{BACKEND_API_URL}/api/logs/user/{user_id}"
    headers = {"Authorization": f"Bearer {jwt_token}"}
    
    try:
        response = requests.get(api_endpoint, headers=headers, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print("🚨 오류: 인증 실패 (401). JWT 토큰을 확인하세요.")
        else:
            print(f"🚨 운동 기록 조회 API 호출 오류: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"🚨 운동 기록 조회 API 호출 오류: {e}")
        return []