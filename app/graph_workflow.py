
from app.services.posture_analyzer import PostureAnalyzer
from app.services.radar_chart import extract_radar_scores, create_and_upload_radar_chart
from app.services.recommend import recommend_node
from app.services.exercise_vector_db import ExerciseVectorDB
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from app.services.ai_coach_service import ai_coach_interaction_service
import redis
import json
import os

posture_analyzer = PostureAnalyzer(model_path="models/yolopose_v1.pt")
vector_db = ExerciseVectorDB()
llm = ChatOpenAI(model="gpt-4o-mini")

# 1. 자세 분석 및 진단 생성 노드
def analyze_user_pose_node(state):
    image_url = state["image_url"]
    mode = state.get("mode", "front")
    print(f"[analyze_user_pose_node] image_url: {image_url}, mode: {mode}")
    analysis_result = posture_analyzer.analyze_posture(image_url, mode)
    print(f"[analyze_user_pose_node] analysis_result: {analysis_result}")
    if not analysis_result.get("success"):
        print("[analyze_user_pose_node] 자세 분석 실패!")
        return {"error": "자세 분석 실패", "analysis_result": analysis_result}
    pose_data_list = analysis_result.get("pose_data", [])
    if not pose_data_list or not pose_data_list[0]:
        print("[analyze_user_pose_node] 사람이 감지되지 않음! pose_data_list:", pose_data_list)
        return {"error": "사람이 감지되지 않았습니다.", "analysis_result": analysis_result}
    pose_data = pose_data_list[0]
    print(f"[analyze_user_pose_node] pose_data: {pose_data}")
    print(f"[analyze_user_pose_node] measurements: {pose_data.get('measurements')}")
    diagnosis = posture_analyzer.generate_ollama_diagnosis(pose_data, mode)
    print(f"[analyze_user_pose_node] diagnosis: {diagnosis}")
    return {
        "analysis_result": analysis_result,
        "pose_data": pose_data,
        "diagnosis": diagnosis
    }

# 2. Radar Chart 생성 및 업로드 노드
def radar_chart_node(state):
    pose_data = state.get("pose_data", {})
    scores = pose_data.get('scores', {}) if pose_data else {}
    print(f"[radar_chart_node] scores: {scores}")
    radar_scores = extract_radar_scores(scores)
    print(f"[radar_chart_node] radar_scores: {radar_scores}")
    radar_chart_url = None
    if len(radar_scores) in (3, 5) and any(v > 0 for v in radar_scores.values()):
        radar_chart_url = create_and_upload_radar_chart(radar_scores)
    else:
        print("[radar_chart_node] radar_scores가 3개/5개가 아니거나 모두 0입니다. 차트 생성/업로드 생략.")
    # diagnosis, pose_data도 함께 반환
    return {
        "radar_scores": radar_scores,
        "radar_chart_url": radar_chart_url,
        "scores": scores,
        "diagnosis": state.get("diagnosis", {}),
        "pose_data": pose_data
    }

# 3. 운동 추천 노드
def recommend_exercise_node(state):
    print("[recommend_exercise_node] state:", state)
    diagnosis = state.get("diagnosis", {})
    pose_data = state.get("pose_data", {})
    scores = state.get("scores", {})
    print("[recommend_exercise_node] diagnosis:", diagnosis)
    print("[recommend_exercise_node] pose_data:", pose_data)
    # 추천운동 생성 (서비스 함수 recommend_node를 명확히 호출)
    rec_result = recommend_node(diagnosis.get("korean", ""), vector_db, llm)
    if "error" in rec_result:
        print(f"[recommend_node] 오류: {rec_result['error']}")
        return {"error": rec_result["error"]}
    recommended_exercise = rec_result.get("recommended_exercise", {})
    search_query = rec_result.get("search_query", "")
    print(f"[recommend_exercise_node] 추천운동: {recommended_exercise}, search_query: {search_query}")
    return {
        "diagnosis": diagnosis,
        "recommended_exercise": recommended_exercise,
        "search_query": search_query,
        "pose_data": pose_data,
        "radar_chart_url": state.get("radar_chart_url"),
        "spineCurvScore": scores.get("척추굽음score", 0),
        "spineScolScore": scores.get("척추휨score", 0),
        "pelvicScore": scores.get("골반틀어짐score", 0),
        "neckScore": scores.get("거북목score", 0),
        "shoulderScore": scores.get("어깨score", 0)
    }

single_graph = StateGraph(dict)
single_graph.add_node("analyze_user_pose", analyze_user_pose_node)
single_graph.add_node("radar_chart", radar_chart_node)
single_graph.add_node("recommend_exercise", recommend_exercise_node)
single_graph.set_entry_point("analyze_user_pose")
single_graph.add_edge("analyze_user_pose", "radar_chart")
single_graph.add_edge("radar_chart", "recommend_exercise")
app_graph = single_graph.compile()

def analyze_user_merge_pose_node(state):
    front_url = state["front_image_url"]
    side_url = state["side_image_url"]
    # 1. 분석
    front_result = posture_analyzer.analyze_posture(front_url, mode='front')
    front_pose = front_result.get('pose_data', [{}])[0] if front_result.get('pose_data') else {}
    side_result = posture_analyzer.analyze_posture(side_url, mode='side')
    side_pose = side_result.get('pose_data', [{}])[0] if side_result.get('pose_data') else {}
    # 2. 점수 합치기
    front_scores = front_pose.get('scores', {}) if front_pose else {}
    side_scores = side_pose.get('scores', {}) if side_pose else {}
    all_scores = {
        "어깨score": front_scores.get("어깨score"),
        "골반틀어짐score": front_scores.get("골반틀어짐score"),
        "척추휨score": front_scores.get("척추휨score"),
        "척추굽음score": side_scores.get("척추굽음score"),
        "거북목score": side_scores.get("거북목score"),
    }
    # 3. 가장 낮은 점수 부위의 pose_data 선택
    min_part = None
    min_score = 101
    for key, val in all_scores.items():
        if val is not None and val < min_score:
            min_score = val
            min_part = key
    if min_part in ["어깨score", "골반틀어짐score", "척추휨score"]:
        pose_data_for_diag = front_pose
    else:
        pose_data_for_diag = side_pose
    return {
        "front_pose": front_pose,
        "side_pose": side_pose,
        "front_scores": front_scores,
        "side_scores": side_scores,
        "all_scores": all_scores,
        "pose_data_for_diag": pose_data_for_diag
    }

def radar_chart_merge_node(state):
    # 5개 부위 점수로 radar_scores 생성
    radar_scores = {
        "목": state["side_scores"].get("거북목score", 0),
        "어깨": state["front_scores"].get("어깨score", 0),
        "골반": state["front_scores"].get("골반틀어짐score", 0),
        "척추(정면)": state["front_scores"].get("척추휨score", 0),
        "척추(측면)": state["side_scores"].get("척추굽음score", 0)
    }
    radar_chart_url = None
    if all(v is not None for v in radar_scores.values()) and any(v > 0 for v in radar_scores.values()):
        radar_chart_url = create_and_upload_radar_chart(radar_scores)
    # 다음 노드에서 필요한 값도 모두 포함해서 반환
    return {
        "radar_scores": radar_scores,
        "radar_chart_url": radar_chart_url,
        "pose_data_for_diag": state["pose_data_for_diag"],
        "front_pose": state["front_pose"],
        "side_pose": state["side_pose"],
        "front_scores": state["front_scores"],
        "side_scores": state["side_scores"]
    }

def recommend_exercise_merge_node(state):
    diagnosis = posture_analyzer.generate_ollama_diagnosis(state["pose_data_for_diag"], "merge")
    rec_result = recommend_node(diagnosis.get("korean", ""), vector_db, llm)
    recommended_exercise = rec_result.get("recommended_exercise", {})
    search_query = rec_result.get("search_query", "")
    return {
        "diagnosis": diagnosis,
        "recommended_exercise": recommended_exercise,
        "search_query": search_query,
        "front_pose_data": state["front_pose"],
        "side_pose_data": state["side_pose"],
        "radar_scores": state["radar_scores"],
        "radar_chart_url": state["radar_chart_url"],
        "spineCurvScore": state["side_scores"].get("척추굽음score", 0),
        "spineScolScore": state["front_scores"].get("척추휨score", 0),
        "pelvicScore": state["front_scores"].get("골반틀어짐score", 0),
        "neckScore": state["side_scores"].get("거북목score", 0),
        "shoulderScore": state["front_scores"].get("어깨score", 0)
    }

# merge_graph 정의
merge_graph = StateGraph(dict)
merge_graph.add_node("analyze_user_merge_pose", analyze_user_merge_pose_node)
merge_graph.add_node("radar_chart_merge", radar_chart_merge_node)
merge_graph.add_node("recommend_exercise_merge", recommend_exercise_merge_node)
merge_graph.set_entry_point("analyze_user_merge_pose")
merge_graph.add_edge("analyze_user_merge_pose", "radar_chart_merge")
merge_graph.add_edge("radar_chart_merge", "recommend_exercise_merge")
app_merge_graph = merge_graph.compile()

redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "192.168.2.6"), port=6379, db=0, decode_responses=True)

def ai_coach_node(state):
    user_id = state.get("user_id")
    diagnosis = state.get("diagnosis", {})
    recommended_exercise = state.get("recommended_exercise", {})
    message = state.get("message")  # 사용자의 추가 메시지(선택)
    session_key = f"ai_coach_session:{user_id}"

    # 1. Redis에서 기존 대화 내역 불러오기 (없으면 빈 리스트)
    history = []
    if redis_client.exists(session_key):
        try:
            history = json.loads(redis_client.get(session_key))
        except Exception:
            history = []

    # 2. 사용자의 메시지가 있으면 대화 내역에 추가
    if message:
        history.append({"role": "user", "content": message})

    # 3. AI 코치 응답 생성
    diagnosis_text = diagnosis.get("korean", "")
    ai_coach_result = ai_coach_interaction_service(diagnosis_text, recommended_exercise, llm)
    if "error" in ai_coach_result:
        return {"error": ai_coach_result["error"]}
    ai_coach_message = ai_coach_result["ai_coach_response"]

    # 4. 대화 내역에 AI 코치 응답 추가
    history.append({"role": "ai_coach", "content": ai_coach_message})

    # 5. Redis에 24시간 TTL로 저장 (세션 1개만 유지)
    redis_client.set(session_key, json.dumps(history), ex=60*60*24)

    return {
        "type": "ai_coach",
        "response": ai_coach_message,
        "history": history,
        "recommended_exercise": recommended_exercise
    }

ai_coach_graph = StateGraph(dict)
ai_coach_graph.add_node("ai_coach", ai_coach_node)
ai_coach_graph.set_entry_point("ai_coach")
app_ai_coach_graph = ai_coach_graph.compile()