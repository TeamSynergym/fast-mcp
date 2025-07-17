# main.py
import sys
import os

# 현재 파일의 상위 디렉토리를 시스템 경로에 추가하여 모듈을 찾을 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph.builder import create_graph
from config import DEFAULT_USER_ID, DEFAULT_JWT_TOKEN, OPENAI_API_KEY

def run_workflow():
    """
    SynergyM 워크플로우를 설정하고 사용자와 상호작용하며 실행합니다.
    """
    # 1. API 키 존재 여부 확인
    if not OPENAI_API_KEY:
        print("🚨 OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일을 확인해주세요.")
        return

    print("=" * 60)
    print("🚀 SynergyM 운동 목표 제안 워크플로우를 시작합니다.")
    print("=" * 60)

    # 2. 사용자로부터 ID와 토큰 입력받기
    # 사용자 ID 입력 (입력하지 않으면 기본값 사용)
    user_id_input = input(f"👤 사용자 ID를 입력하세요 (기본값: {DEFAULT_USER_ID}): ")
    user_id = user_id_input if user_id_input else DEFAULT_USER_ID

    # JWT 토큰 입력 (입력하지 않으면 기본값 사용)
    # 실제 운영 환경에서는 더 안전한 인증 방식이 필요합니다.
    print("\n🔑 JWT 토큰을 입력하세요.")
    print("(테스트용 기본 토큰을 사용하려면 Enter를 누르세요)")
    jwt_token_input = input("> ")
    jwt_token = jwt_token_input if jwt_token_input else DEFAULT_JWT_TOKEN
    
    # 3. 그래프 생성 및 초기값 설정
    graph = create_graph()
    
    inputs = {
        "user_id": user_id,
        "jwt_token": jwt_token,
    }

    print("\n" + "=" * 60)
    print(f"   (대상 사용자 ID: {inputs['user_id']} 로 워크플로우를 진행합니다.)")
    print("=" * 60)

    # 4. 워크플로우 스트림 실행 및 결과 출력
    try:
        for s in graph.stream(inputs, stream_mode="values"):
            # 가장 마지막으로 업데이트된 상태 키를 가져옵니다.
            latest_update_key = list(s.keys())[-1]
            
            # 'interrupter' (피드백 대기 노드) 실행 직후에는 추가 출력을 하지 않아 중복 메시지를 피합니다.
            if latest_update_key == "feedback":
                continue
            
            # 그 외의 경우, 상태 업데이트 정보를 깔끔하게 출력합니다.
            latest_update_value = s[latest_update_key]
            print("\n[상태 업데이트] --------------------------------")
            print(f"-> {latest_update_key}: {latest_update_value}")
            print("---------------------------------------------")

    except Exception as e:
        print(f"\n🚨 워크플로우 실행 중 오류가 발생했습니다: {e}")

    finally:
        print("\n" + "=" * 60)
        print("✅ 워크플로우가 종료되었습니다.")
        print("=" * 60)

if __name__ == "__main__":
    run_workflow()