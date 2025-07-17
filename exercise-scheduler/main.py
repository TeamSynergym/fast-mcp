# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from app.graph.builder import create_graph
from app.graph.state import ExerciseState
from config import OPENAI_API_KEY

# -- Pydantic 모델 정의: API 요청/응답 형태를 강제 --

class InitialData(BaseModel):
    """워크플로우 시작에 필요한 초기 데이터 모델"""
    user_id: str
    jwt_token: str
    user_email: Optional[str]
    exercise_history: List[Dict[str, Any]]
    comparison_stats: Dict[str, Any]

class FeedbackData(BaseModel):
    """사용자 피드백을 전달하기 위한 모델"""
    feedback: Dict[str, str]
    # 이전 단계의 전체 상태를 받아야 워크플로우를 이어갈 수 있습니다.
    state: Dict[str, Any]

# -- FastAPI 애플리케이션 설정 --

app = FastAPI(
    title="SynergyM AI Goal Setting Service",
    description="LangGraph를 사용하여 사용자 맞춤형 운동 목표를 제안, 수정, 확정하는 워크플로우를 제공합니다.",
    version="1.0.0"
)

# 애플리케이션 시작 시 그래프를 한 번만 컴파일하여 재사용합니다.
graph = create_graph()

print("="*25, "Compiled Graph Structure", "="*25)

@app.on_event("startup")
async def startup_event():
    if not OPENAI_API_KEY:
        raise RuntimeError("🚨 OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# -- API 엔드포인트 정의 --

@app.post("/workflow/start", summary="목표 제안 워크플로우 시작")
async def start_workflow(initial_data: InitialData) -> Dict[str, Any]:
    """
    Spring 백엔드로부터 사용자 데이터를 받아 목표 제안 워크플로우를 시작합니다.
    사용자 피드백을 기다리는 지점(interrupter)까지 실행하고,
    제안된 목표와 현재까지의 상태(state)를 반환합니다.
    """
    inputs = initial_data.dict()
    
    for s in graph.stream(inputs, {"recursion_limit": 10}, stream_mode="values"):
        
        if "feedback" in s and s["feedback"] is not None:
            print("✅ 목표 제안 완료. 사용자 피드백 대기 중...")
            
            # 디버깅용 출력
            import json
            print("==================== 현재 상태(state) 객체 ====================")
            print(json.dumps(s, indent=2, ensure_ascii=False))
            print("============================================================")

            if 'suggested_goals' not in s or s['suggested_goals'] is None:
                raise HTTPException(status_code=500, detail="워크플로우 상태(state)에 'suggested_goals'가 생성되지 않았습니다.")

            response = {
                "message": "사용자 피드백 대기",
                "suggested_goals": s['suggested_goals'],
                "current_state": s
            }
            return response
            
    raise HTTPException(status_code=500, detail="워크플로우가 피드백 단계에 도달하지 못했습니다.")


@app.post("/workflow/resume", summary="피드백 기반 목표 수정 및 확정")
async def resume_workflow(feedback_data: FeedbackData) -> Dict[str, Any]:
    """
    사용자 피드백과 이전 상태(state)를 받아 워크플로우를 재개하고,
    최종 확정된 목표와 결과를 반환합니다.
    """
    current_state = feedback_data.state
    
    # 💡 [핵심 수정] stream_mode="values" 를 추가합니다.
    final_state = None
    for s in graph.stream(current_state, {"recursion_limit": 10}, stream_mode="values"):
        # 스트림의 마지막 상태가 최종 결과입니다.
        final_state = s

    if not final_state:
        raise HTTPException(status_code=500, detail="워크플로우가 정상적으로 종료되지 않았습니다.")

    # 최종 상태에서 필요한 정보를 추출합니다.
    final_goals = final_state.get("final_goals")
    is_goal_achieved = final_state.get("is_goal_achieved", False)
    
    # 목표 달성 여부에 따라 다른 응답을 반환합니다.
    if is_goal_achieved:
        print("✅ 워크플로우 최종 완료 (목표 달성)")
        return {
            "message": "목표 달성! 보상이 지급되었습니다.",
            "final_goals": final_goals,
            "generated_badge": final_state.get("generated_badge")
        }
    else:
        print("✅ 워크플로우 최종 완료 (목표 미달성 또는 설정만 완료)")
        return {
            "message": "목표 설정이 최종 완료되었습니다.",
            "final_goals": final_goals,
            "generated_badge": None
        }

# 이 파일을 직접 실행할 경우 uvicorn 서버로 실행되도록 설정
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
   
   
# localhost:8000/docs 테스트 데이터
# {
#   "user_id": "212",
#   "jwt_token": "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJqZW9uZ3lvb24yMDA0QG5hdmVyLmNvbSIsImF1dGgiOiJST0xFX01FTUJFUiIsImV4cCI6MTc1Mjc5NzYzOX0.UEVrFIW1kcc1ahaqX0M3-zhqY-vFTivvcHRaui7ku205Vn84SBbg7fdiM5lArhVyozO-SOYr9w_4bDHoV9vq7w",
#   "user_email": "jeongyoon2004@naver.com",
#   "exercise_history": [
#     {
#       "id": 1,
#       "createdAt": "2025-07-17T10:00:00",
#       "updatedAt": "2025-07-17T12:00:00",
#       "useYn": "Y",
#       "userId": 212,
#       "routineId": 201,
#       "exerciseDate": "2025-07-16",
#       "completionRate": 85.5,
#       "memo": "Great workout session!",
#       "routineIds": [201, 202, 203],
#       "routineNames": ["Morning Cardio", "Strength Training", "Yoga"]
#     },
# {
#       "id": 3,
#       "createdAt": "2025-07-17T10:00:00",
#       "updatedAt": "2025-07-17T12:00:00",
#       "useYn": "Y",
#       "userId": 212,
#       "routineId": 201,
#       "exerciseDate": "2025-07-17",
#       "completionRate": 85.5,
#       "memo": "Great workout session!",
#       "routineIds": [201, 202, 203],
#       "routineNames": ["Morning Cardio", "Strength Training", "Yoga"]
#     },
# {
#       "id": 2,
#       "createdAt": "2025-07-17T10:00:00",
#       "updatedAt": "2025-07-17T12:00:00",
#       "useYn": "Y",
#       "userId": 212,
#       "routineId": 201,
#       "exerciseDate": "2025-07-18",
#       "completionRate": 85.5,
#       "memo": "Great workout session!",
#       "routineIds": [201, 202, 203],
#       "routineNames": ["Morning Cardio", "Strength Training", "Yoga"]
#     }
#   ],
#   "comparison_stats": {
#     "frequencyPercentile": 75.5,
#     "comment": "You are doing better than 75% of users!"
#   }
# }


# {
#   "feedback": {
#     "choice": "1"
#   },
#   "state": {
#     "user_id": "212",
#     "jwt_token": "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJqZW9uZ3lvb24yMDA0QG5hdmVyLmNvbSIsImF1dGgiOiJST0xFX01FTUJFUiIsImV4cCI6MTc1Mjc5NzYzOX0.UEVrFIW1kcc1ahaqX0M3-zhqY-vFTivvcHRaui7ku205Vn84SBbg7fdiM5lArhVyozO-SOYr9w_4bDHoV9vq7w",
#     "user_email": "jeongyoon2004@naver.com",
#     "exercise_history": [
#       { "id": 1, "exerciseDate": "2025-06-23", "completionRate": 92.5, "memo": "Feeling strong!", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 2, "exerciseDate": "2025-06-25", "completionRate": 92.5, "memo": "Good session.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 3, "exerciseDate": "2025-06-27", "completionRate": 92.5, "memo": "Pushing my limits.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 4, "exerciseDate": "2025-06-28", "completionRate": 92.5, "memo": "Weekend workout complete.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 5, "exerciseDate": "2025-06-30", "completionRate": 92.5, "memo": "Start of the week.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 6, "exerciseDate": "2025-07-02", "completionRate": 92.5, "memo": "Felt a bit tired.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 7, "exerciseDate": "2025-07-04", "completionRate": 92.5, "memo": "Great energy today.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 8, "exerciseDate": "2025-07-05", "completionRate": 92.5, "memo": "Solid workout.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 9, "exerciseDate": "2025-07-07", "completionRate": 92.5, "memo": "Consistent effort.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 10, "exerciseDate": "2025-07-09", "completionRate": 92.5, "memo": "Another one done.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 11, "exerciseDate": "2025-07-11", "completionRate": 92.5, "memo": "Ready for the weekend.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 12, "exerciseDate": "2025-07-12", "completionRate": 92.5, "memo": "Kept the promise.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 13, "exerciseDate": "2025-07-14", "completionRate": 92.5, "memo": "New week, new goals.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 14, "exerciseDate": "2025-07-15", "completionRate": 92.5, "memo": "Great workout session!", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 15, "exerciseDate": "2025-07-16", "completionRate": 92.5, "memo": "Almost there.", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] },
#       { "id": 16, "exerciseDate": "2025-07-17", "completionRate": 92.5, "memo": "Monthly goal achieved!", "userId": 212, "routineId": 201, "routineNames": ["Strength Training"] }
#     ],
#     "coach_persona": "데이터를 중시하는 엄격한 트레이너",
#     "fatigue_analysis": {
#       "status": "normal",
#       "reason": "최근 운동 기록이 꾸준하며 완료율이 높아 상태가 양호함"
#     },
#     "slump_prediction": {
#       "risk": "low",
#       "reason": "일관된 높은 완료율을 유지하고 있어 슬럼프 위험이 낮음"
#     },
#     "comparison_stats": {
#       "frequencyPercentile": 75.5,
#       "comment": "You are doing better than 75% of users!"
#     },
#     "analysis_result": "지난 한 달간 총 16회 운동하셨고, 평균 완료율은 92.5% 입니다. 슬럼프 예측 분석 결과는 '일관된 높은 완료율을 유지하고 있어 슬럼프 위험이 낮음'이며, 다른 사용자들의 75%보다 더 꾸준히 운동하고 계십니다!\n\n🤖 AI 코멘트: 정말 대단한 성과입니다! 한 달간 16번의 운동과 92.5%라는 높은 평균 완료율은 엄청난 헌신을 보여줍니다. 지금의 페이스를 유지하며 계속 나아가세요. 당신의 꾸준함이 빛을 발하고 있습니다! 💪🔥",
#     "suggested_goals": "{\"weekly_goal\": {\"workouts\": 5, \"completion_rate\": 90}, \"monthly_goal\": {\"workouts\": 20, \"completion_rate\": 88}}",
#     "feedback": {
#       "choice": "1"
#     }
#   }
# }

