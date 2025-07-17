# # app/services/api_client.py
# import requests
# import json
# from typing import List, Dict, Any, Optional

# # config.py에서 설정 값을 가져옵니다.
# from config import BACKEND_API_URL, API_TIMEOUT

# def get_email_from_backend(user_id: str, jwt_token: str) -> Optional[str]:
#     """
#     백엔드 API 서버를 호출하여 특정 사용자의 이메일을 가져옵니다.
#     """
#     api_endpoint = f"{BACKEND_API_URL}/api/users/{user_id}"
#     headers = {"Authorization": f"Bearer {jwt_token}"}
   
#     try:
#         response = requests.get(api_endpoint, headers=headers, timeout=API_TIMEOUT)
#         response.raise_for_status()
#         user_data = response.json()
#         return user_data.get("email")
#     except requests.exceptions.RequestException as e:
#         print(f"🚨 이메일 조회 API 호출 오류: {e}")
#         return None

# def get_history_from_backend(user_id: str, jwt_token: str) -> List[Dict[str, Any]]:
#     """
#     백엔드 API 서버를 호출하여 특정 사용자의 운동 기록을 가져옵니다.
#     """
#     api_endpoint = f"{BACKEND_API_URL}/api/logs/user/{user_id}"
#     headers = {"Authorization": f"Bearer {jwt_token}"}
   
#     try:
#         response = requests.get(api_endpoint, headers=headers, timeout=API_TIMEOUT)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.HTTPError as e:
#         if e.response.status_code == 401:
#             print("🚨 오류: 인증 실패 (401). JWT 토큰을 확인하세요.")
#         else:
#             print(f"🚨 운동 기록 조회 API 호출 오류: {e}")
#         return []
#     except requests.exceptions.RequestException as e:
#         print(f"🚨 운동 기록 조회 API 호출 오류: {e}")
#         return []

# # --- 신규 추가된 함수 ---

# def get_comparison_stats_from_backend(user_id: str, jwt_token: str) -> Dict[str, Any]:
#     """
#     백엔드에서 익명화된 다른 사용자 비교 통계를 가져옵니다.
#     """
#     api_endpoint = f"{BACKEND_API_URL}/api/stats/comparison/{user_id}"
#     headers = {"Authorization": f"Bearer {jwt_token}"}
   
#     try:
#         response = requests.get(api_endpoint, headers=headers, timeout=API_TIMEOUT)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.HTTPError as e:
#         print(f"🚨 비교 통계 조회 API 호출 오류: {e}")
#         return {"error": "Failed to fetch comparison stats"}
#     except requests.exceptions.RequestException as e:
#         print(f"🚨 비교 통계 조회 API 호출 오류: {e}")
#         return {"error": "Request failed"}

# def save_achievement_to_backend(user_id: int, jwt_token: str, achievement_data: Dict[str, str]):
#     """
#     백엔드에 사용자의 달성 기록(뱃지, 트로피 등)을 저장합니다.
#     """
#     api_endpoint = f"{BACKEND_API_URL}/api/achievements/{user_id}"
#     headers = {"Authorization": f"Bearer {jwt_token}", "Content-Type": "application/json"}
   
#     try:
#         response = requests.post(api_endpoint, headers=headers, json=achievement_data, timeout=API_TIMEOUT)
#         response.raise_for_status()
        
#         print(f"✅ 업적({achievement_data.get('badge_name')}) 정보가 서버에 성공적으로 전송되었습니다.")
#         # 컨트롤러가 간단한 문자열을 반환하므로 .json() 대신 .text 사용 또는 상태만 확인
#         print(f"   서버 응답: {response.text}")
#         return {"status": "success"}

#     except requests.exceptions.HTTPError as e:
#         print(f"🚨 업적 저장 API 호출 오류: {e}")
#         return {"error": "Failed to save achievement"}
#     except requests.exceptions.RequestException as e:
#         print(f"🚨 업적 저장 API 호출 오류: {e}")
#         return {"error": "Request failed"}
    
    
# def save_goals_to_backend(user_id: str, jwt_token: str, goals: str):
#     """
#     백엔드에 최종 확정된 목표를 저장합니다.
#     """
#     print("... (API Call) 확정된 최종 목표 저장 요청 ...")
#     api_endpoint = f"{BACKEND_API_URL}/api/users/{user_id}/goals"
    
#     headers = {
#         "Authorization": f"Bearer {jwt_token}",
#         "Content-Type": "application/json"
#     }
    
#     response = requests.post(api_endpoint, headers=headers, data=goals, timeout=API_TIMEOUT)
#     response.raise_for_status()
    
#     print(f"✅ 사용자({user_id})의 새로운 목표가 성공적으로 저장되었습니다.")
#     return response.json()
