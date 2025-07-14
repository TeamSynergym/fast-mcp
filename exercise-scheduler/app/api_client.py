import requests
from typing import List, Dict, Any

# uvicorn으로 실행 중인 백엔드 서버의 주소
BACKEND_API_URL = "http://127.0.0.1:8000"

def get_history_from_backend(userId: str) -> List[Dict[str, Any]]:
    """
    백엔드 API 서버를 호출하여 특정 사용자의 운동 기록을 가져옵니다.

    Args:
        user_id (str): 조회할 사용자의 ID.

    Returns:
        List[Dict[str, Any]]: 사용자의 운동 기록 리스트.
                               API 호출 실패 시 빈 리스트를 반환합니다.
    """
    # 호출할 API 엔드포인트 주소 구성
    api_endpoint = f"{BACKEND_API_URL}/api/logs/user/{userId}"
    print(f"백엔드 API 호출: {api_endpoint}")

    try:
        # GET 요청 보내기
        response = requests.get(api_endpoint)

        # HTTP 상태 코드가 200번대가 아닐 경우 에러 발생
        response.raise_for_status()

        # 성공 시, JSON 응답 데이터를 파이썬 객체(리스트)로 변환하여 반환
        return response.json()

    except requests.exceptions.RequestException as e:
        # 네트워크 오류, 서버 오류 등 요청 중 발생한 모든 예외 처리
        print(f"🚨 백엔드 API 호출 중 오류가 발생했습니다: {e}")
        return []
