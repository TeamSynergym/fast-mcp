import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import RecommendationState
from app.services.exercise_vector_db import ExerciseVectorDB # VectorDB 임포트

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
vector_db = ExerciseVectorDB() # VectorDB 인스턴스 생성

def summarize_user_node(state: RecommendationState) -> dict:
    """사용자의 모든 데이터를 종합하여 자연어 프로필을 생성합니다."""
    print("--- [Rec Node 1] 사용자 프로필 요약 (체형 분석 포함) ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 전문 운동 코치이자 데이터 분석가입니다. 주어진 모든 사용자 데이터를 종합하여 운동 선호도, 현재 수준, 주된 목표, 그리고 신체적 강점과 약점을 요약하는 프로필을 한 문단으로 작성해주세요. 특히 체형 분석 점수를 해석하여 어떤 부위를 개선하거나 강화해야 하는지 명확히 언급해주세요."),
        ("human", """
        - 사용자 프로필: {user_profile}
        - 사용자 체형 분석 결과 (점수가 높을수록 좋음): {posture_analysis}
        - 운동 기록: {exercise_history}
        - 좋아요 한 운동: {liked_exercises}
        - 저장한 루틴: {user_routines}

        위 정보를 바탕으로 사용자의 특성을 종합적으로 요약해주세요:
        """)
    ])
    
    chain = prompt | llm
    summary = chain.invoke({
        "user_profile": state.get("user_profile"),
        "posture_analysis": state.get("posture_analysis"),
        "exercise_history": state.get("exercise_history"),
        "liked_exercises": state.get("liked_exercises"),
        "user_routines": state.get("user_routines"),
    }).content
    
    print(f"🧠 생성된 사용자 요약: {summary}")
    
    return {"user_summary": summary}

def vector_search_node(state: RecommendationState) -> dict:
    """
    Vector DB에서 운동을 검색하고, '이름'을 기준으로 중복 항목을 제외합니다.
    """
    print("--- [Rec Node 2] 벡터 검색 및 필터링 수행 ---")
    
    excluded_exercise_names = set()
    
    for liked in state.get("liked_exercises", []):
        exercise_name = liked.get("name")
        if exercise_name:
            excluded_exercise_names.add(exercise_name)

    # 운동 기록에서 운동 이름(name)을 추출합니다.
    for history_item in state.get("exercise_history", []):
        exercise_name = history_item.get("name")
        if exercise_name:
            excluded_exercise_names.add(exercise_name)
    
    print(f"🚫 제외할 운동 이름 목록: {excluded_exercise_names if excluded_exercise_names else '없음'}")

    # 벡터 검색을 수행합니다.
    user_summary = state['user_summary']
    initial_candidates = vector_db.search(query=user_summary, top_k=20)
    
    if initial_candidates:
        print(f"🔍 첫 번째 검색 후보 데이터 구조: {initial_candidates[0]}")

    filtered_recommendations = []
    for exercise in initial_candidates:
        exercise_name = exercise.get("name")
        # 운동 이름이 존재하고, 제외 목록에 없는 경우에만 추가합니다.
        if exercise_name and exercise_name not in excluded_exercise_names:
            filtered_recommendations.append(exercise)

    final_recommendations = filtered_recommendations[:3]

    print(f"✅ 최종 추천 목록 ({len(final_recommendations)}개): {[item.get('name', 'N/A') for item in final_recommendations]}")
    
    return {"recommendations": final_recommendations}

def generate_reason_node(state: RecommendationState) -> dict:
    """
    최종 추천된 운동 목록의 상태에 따라 적절한 추천 이유를 생성합니다.
    """
    print("--- [Rec Node 3] 추천 이유 생성 ---")
    recommendations = state.get("recommendations", [])
    
    if not recommendations:
        print("⚠️ 추천 목록이 비어있어, 대체 응답을 생성합니다.")
        reason = "최근 활동과 선호도를 바탕으로 분석한 결과, 지금은 새로운 운동보다는 꾸준히 해오신 운동에 집중하며 컨디션을 조절하는 것이 더 좋을 것 같아요. 이미 충분히 잘하고 계십니다!"
        return {"recommendation_reason": reason}

    print("✅ 추천 목록을 기반으로 맞춤 추천 이유를 생성합니다.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 코치입니다. 사용자의 특성과 추천된 운동 목록을 보고, 왜 이 운동들을 추천하는지 간결하고 설득력 있는 메시지를 작성해주세요. 반드시 추천 목록에 있는 운동 이름만 언급해야 합니다."),
        ("human", "사용자 특성: {user_summary}\n\n추천 운동 목록: {recommendations}\n\n추천 이유:")
    ])
    
    chain = prompt | llm
    reason = chain.invoke({
        "user_summary": state['user_summary'],
        "recommendations": [item.get('name', '알 수 없는 운동') for item in recommendations]
    }).content

    print(f"🤖 생성된 추천 이유: {reason}")
    return {"recommendation_reason": reason}