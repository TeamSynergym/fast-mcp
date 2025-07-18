import os
import json
import asyncio
from typing import TypedDict, Dict, Any, Optional, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
import cloudinary.uploader
import requests

# --- 서비스 및 노드 클래스 Import ---
from app.services.posture_analyzer import PostureAnalyzer
from app.agents.youtube_agent import graph as youtube_summary_agent
from app.nodes.chatbot_node import ChatbotActionNode
from app.services.exercise_vector_db import ExerciseVectorDB
from data_server import plot_radar_chart

load_dotenv()

cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

# --- 1. 서비스 초기화 ---
llm = ChatOpenAI(model="gpt-4o-mini")
posture_analyzer = PostureAnalyzer(model_path="models/yolopose_v1.pt")
chatbot_node = ChatbotActionNode()
vector_db = ExerciseVectorDB()

# --- 2. 그래프 상태 (Supervisor 패턴용) ---
class SupervisorGraphState(TypedDict):
    # 각 노드의 실행 결과를 메시지 형태로 누적하여 대화의 흐름을 관리
    messages: Annotated[list, add_messages]
    
    # 다음으로 호출할 노드의 이름을 저장
    next_agent: str
    
    # 원본 요청 데이터
    image_path: str
    analysis_mode: str
    
    # 재검색 횟수 추적
    search_retries: int
    
    # 노드별 결과 데이터
    pose_analysis_result: Dict[str, Any]
    diagnosis: Dict[str, str]
    recommended_exercise: Dict[str, Any]
    chatbot_result: Dict[str, Any]
    youtube_summary: Optional[Dict[str, Any]]
    comment_count: int
    final_output: Dict[str, Any]
    error: Optional[str]
    
    # 사용자 응답 관련 필드(댓글 요약여부 결정)
    user_response: Optional[str]  # 사용자의 응답
    youtube_thread_id: Optional[str]  # YouTube agent의 스레드 ID
    youtube_config: Optional[Dict[str, Any]]  # YouTube agent 설정
    
    # 재검색 시 제외할 URL 리스트
    tried_video_urls: list[str]
    
    # 레이더 차트 경로 (plot_radar_chart_node에서 반환)
    radar_chart_path: str

# --- 3. LangGraph 노드 함수 재정의 ---

# 각 노드는 이제 'messages'에 자신의 실행 결과를 HumanMessage 형태로 추가하여 슈퍼바이저에게 보고합니다.
def analyze_user_pose_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 1] 자세 분석 중...")
    try:
        
        analysis_result = posture_analyzer.analyze_posture(state["image_path"], mode=state["analysis_mode"])
        if not analysis_result.get("success") or not analysis_result.get("pose_data"):
            raise ValueError("자세 분석 실패")
            
        person_analysis = analysis_result["pose_data"][0]
        diagnosis_texts = posture_analyzer.generate_ollama_diagnosis(person_analysis, state["analysis_mode"])
        
        message = HumanMessage(content=f"자세 분석 완료. 진단: {diagnosis_texts['korean']}")
        return {"pose_analysis_result": person_analysis, "diagnosis": diagnosis_texts, "messages": [message]}
    except Exception as e:
        return {"error": f"자세 분석 노드 오류: {e}"}

def plot_radar_chart_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node X] 자세 분석 결과 레이더 차트 생성 중...")
    try:
        scores = state["pose_analysis_result"]["scores"]
        radar_values = {
            "목": scores["거북목score"],
            "어깨": scores["어깨score"],
            "골반": scores["골반틀어짐score"],
            "척추(정면)": scores["척추휨score"],
            "척추(측면)": scores["척추굽음score"]
        }
        output_path = plot_radar_chart(radar_values)
        print(f"  > 레이더 차트 저장: {output_path}")
        # 기존 state에 radar_chart_path를 추가해서 반환
        return {
            **state,
            "radar_chart_path": output_path,
            "messages": [HumanMessage(content="레이더 차트 생성 완료")]
        }
    except Exception as e:
        print("plot_radar_chart_node 에러:", e)
        return {**state, "error": f"레이더 차트 노드 오류: {e}"}

def recommend_exercise_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 2] 맞춤 운동 추천 중 (from VectorDB)...")
    if state.get("error"): return {}
    try:
        diagnosis_text = state["diagnosis"]["korean"]
        
        # LLM을 사용해 진단 내용에서 핵심 키워드를 추출하여 검색 쿼리 생성
        prompt = f"""아래의 자세 진단 내용에 가장 적합한 '단 한 가지'의 검색어을 추천해줘.
        ~난이도, ~효과를 가진, ~부위의, ~운동의 순서로 검색어를 작성해야해.
        VectorDB 검색에 사용할 키워드 문장 오직 한개만 간결하게 한 줄로 답해줘.
        
        [진단 내용]
        {diagnosis_text}
        [출력 예시]
        - 중급 난이도의 유연성을 높이는 효과를 가진 골반 부위의 스트레칭 운동
        [생성된 검색어]
        """
        llm_query = llm.invoke(prompt).content.strip()
        print(f"  > LLM 생성 검색어: '{llm_query}'")
        
        recommended_list = vector_db.search(llm_query, top_k=1)
        
        if not recommended_list:
            raise ValueError("VectorDB에서 추천 운동을 찾지 못했습니다.")
        
        # VectorDB에서 찾은 실제 운동 객체를 변수에 저장
        retrieved_exercise = recommended_list[0]
        print(f"  > VectorDB 검색 결과 운동명: '{retrieved_exercise['name']}'")
        
        message = HumanMessage(content=f"DB 기반 운동 추천 완료: {retrieved_exercise['name']}")
        
        # 상태(state)에 DB에서 직접 찾은 운동 객체를 저장
        return {"recommended_exercise": retrieved_exercise, "messages": [message]}
        
    except Exception as e:
        return {"error": f"운동 추천 노드 오류: {e}"}

def ai_coach_interaction_node(state: SupervisorGraphState) -> Dict[str, Any]:
    """AI 코치와의 대화 노드"""
    print("[Node - AI 코치] 사용자와 대화 중...")
    try:
        diagnosis_text = state["diagnosis"]["korean"]
        recommended_exercise = state["recommended_exercise"]["name"]
        
        prompt = f"""
        당신은 AI 피트니스 코치입니다. 아래 정보를 바탕으로 사용자와 대화를 시작하세요.
        
        [진단 내용]
        {diagnosis_text}
        
        [추천 운동]
        {recommended_exercise}
        
        사용자에게 운동의 중요성과 자세 교정의 필요성을 설명하고, 동기부여를 제공하세요.
        """
        response = llm.invoke(prompt).content.strip()
        print(f"  > AI 코치 응답: {response}")
        
        message = HumanMessage(content=f"AI 코치 대화 완료: {response}")
        return {"messages": [message]}
    except Exception as e:
        return {"error": f"AI 코치 노드 오류: {e}"}

def video_search_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print(f"[Node 3 - 시도 {state['search_retries'] + 1}] 보충 영상 검색 중 (Youtube)...")
    if state.get("error"): return {}
    try:
        # search_retries 값에 따라 분기 로직 추가
        if state.get("search_retries", 0) > 0:
            # 재검색일 경우 (search_retries > 0)
            print("   > 영상 검증 실패로 재검색을 실행합니다. 이전 영상은 제외됩니다.")
            tried_urls = state.get("tried_video_urls", [])
        else:
            # 초기 검색일 경우 (search_retries == 0)
            print("   > 초기 영상 검색을 실행합니다.")
            tried_urls = []
            
        exercise_name = state["recommended_exercise"]["name"]
        if "자세" in exercise_name or "스트레칭" in exercise_name:
            search_query = f"{exercise_name}"
        else:
            search_query = f"{exercise_name} 운동"
            
        print(f"  > 유튜브 검색어: '{search_query}'")
        
        result = asyncio.run(chatbot_node.run(prompt=search_query, exclude_urls=tried_urls))
        
        new_url = result.get("youtube_url")
        
        if not result.get("youtube_url") or "No video found" in result.get("youtube_url"):
            raise ValueError("추천 유튜브 영상을 찾지 못했습니다.")

        message = HumanMessage(content=f"유튜브 영상 검색 완료. URL: {result.get('youtube_url')}")
        updated_tried_urls = tried_urls + [new_url]
        return {
            "chatbot_result": result, 
            "messages": [message], 
            "search_retries": state["search_retries"] + 1,
            "tried_video_urls": updated_tried_urls # 상태 업데이트
        }
    except Exception as e:
        return {"error": f"챗봇 액션 노드 오류: {e}"}

def summarize_video_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 4] 유튜브 영상 요약 중...")
    if state.get("error"): return {}
    try:
        summary_result = youtube_summary_agent.invoke({"url": state["chatbot_result"]["youtube_url"]})
        if summary_result.get("error"):
            raise ValueError(f"유튜브 요약 실패: {summary_result['error']}")
        
        summary = summary_result.get("script_summary")
        
        comment_count = summary_result.get("comment_count", 0)  # 댓글 수 반환(기본값 0)
        
        message = HumanMessage(content=f"영상 요약 완료. 댓글 수: {comment_count}")
        return {
            "youtube_summary": summary, 
            "comment_count": comment_count,
            "messages": [message]
            }
    except Exception as e:
        return {"error": f"유튜브 요약 노드 오류: {e}"}
    
class ValidationResult(BaseModel):
    is_relevant: bool = Field(description="요약 내용이 건강이나 운동과 관련이 있는지 여부")
    reason: str = Field(description="관련이 있거나 없는지에 대한 간략한 이유")

def validate_summary_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 5-1] 영상 요약 검증 중...")
    if state.get("error"): return {}

    summary_dict = state.get("youtube_summary", {})
    summary_text = json.dumps(summary_dict)
    diagnosis_text = state["diagnosis"]["korean"]
    recommended_exercise = state["recommended_exercise"]["name"]

    # 요약이 너무 짧거나 없는 경우, 바로 부적합 판정
    if not summary_dict or len(summary_text) < 50:
        print("  > 검증 실패: 요약 내용이 너무 짧거나 없습니다.")
        message = HumanMessage(content="요약 검증 실패: 내용 부실")
        return {"messages": [message]}

    # LLM을 통한 관련성 검증
    structured_validator = llm.with_structured_output(ValidationResult)
    prompt = f"""
당신은 사용자의 자세 교정을 위한 운동 영상을 필터링하는 AI 전문가입니다.

[분석할 정보]
- 자세 진단: '{diagnosis_text}'
- 추천 운동: '{recommended_exercise}'
- 추천 운동의 효과: '{state["recommended_exercise"].get("description", "효과 정보 없음")}'
- 영상 요약: '{summary_text}'

[수행할 작업]
- '자세 진단'을 개선하고 '추천 운동'을 수행하는 데 '영상 요약'의 내용이 적합한지 판단해주세요.
- 가장 먼저 제외조건에 해당하는 내용이 있는지 살펴보고, 해당할 시 반드시 관련없음으로 판단해야 합니다.
- 또한 동영상의 내용과 '추천 운동의 효과'가 서로 관련있는지도 판단해야 합니다.

[제외 조건]
- **스포츠 전문 훈련:** 특정 스포츠(예: 축구, 야구, 골프, 수영)의 기술 향상을 위한 훈련은 '관련 없음'으로 처리합니다.

[출력 형식]
아래 JSON 형식에 따라, 판단 결과(`is_relevant`)와 구체적인 이유(`reason`)를 작성해주세요.

{{
  "is_relevant": <true 또는 false>,
  "reason": "<판단 이유를 간단히 작성합니다.>"
}}
"""
    
    validation: ValidationResult = structured_validator.invoke(prompt)
    
    if validation.is_relevant:
        print(f"  > 검증 성공: {validation.reason}")
        message = HumanMessage(content="요약 검증 성공")
    else:
        print(f"  > 검증 실패: {validation.reason}")
        message = HumanMessage(content="요약 검증 실패: 관련성 부족")
        
    return {"messages": [message]}

def ask_user_response_node(state: SupervisorGraphState) -> Dict[str, Any]:
    """사용자에게 댓글 요약 관심 여부를 묻는 노드"""
    print("[Node 5-2] 사용자 응답 요청 중...")
    
    # 콘솔에서 사용자 입력 받기
    print("\n--- 추가 정보 제공 ---")
    print("영상 스크립트 요약이 완료되었습니다!")
    print("영상에 대한 댓글 반응도 궁금하시다면 알려드릴게요!")
    
    user_input = input("응답해주세요 (예: '응', '네', '보여줘' 또는 '괜찮아', '아니'): ").strip()
    
    message = HumanMessage(content=f"사용자 응답 수집 완료: {user_input}")
    return {
        "user_response": user_input,
        "youtube_thread_id": f"thread_{hash(state['chatbot_result']['youtube_url'])}",
        "youtube_config": {"configurable": {"thread_id": f"thread_{hash(state['chatbot_result']['youtube_url'])}"}},
        "messages": [message]
    }
    
def comment_summary_unavailable_node(state: SupervisorGraphState) -> Dict[str, Any]:
    """댓글 수가 적어 요약 제공이 불가능함을 알리는 노드"""
    print("[Node 5-3] 댓글 요약 제공 불가 안내")
    
    # 최종 결과에 표시될 메시지를 youtube_summary에 추가
    updated_youtube_summary = state.get("youtube_summary", {})
    if isinstance(updated_youtube_summary, dict):
         updated_youtube_summary["comment_summary"] = "댓글 개수가 10개 미만으로 댓글 요약을 제공하지 않습니다."
    
    message = HumanMessage(content="댓글 요약 제공 불가: 댓글 수 부족")
    return {
        "messages": [message],
        "youtube_summary": updated_youtube_summary
    }
    
def rerun_youtube_agent_node(state: SupervisorGraphState) -> Dict[str, Any]:
    """사용자 응답을 바탕으로 YouTube agent를 재실행하는 노드"""
    print("[Node 5-4] YouTube Agent 재실행 중...")
    
    try:
        # YouTube agent의 메모리 그래프 사용
        youtube_state = {
            "url": state["chatbot_result"]["youtube_url"],
            "reply": state["user_response"],
            "script_summary": state.get("youtube_summary", {})
        }
        
        # continue_with_memory 함수 사용하여 댓글 요약 실행
        from app.agents.youtube_agent import graph_memory, continue_with_memory
        
        result = continue_with_memory(
            graph_memory, 
            youtube_state, 
            state["youtube_config"], 
            {"reply": state["user_response"], "url": youtube_state["url"]}
        )
        
        # 댓글 요약 결과 추가
        updated_youtube_summary = state.get("youtube_summary", {})
        if result.get("comment_summary"):
            updated_youtube_summary["comment_summary"] = result["comment_summary"]
        
        message = HumanMessage(content="YouTube 댓글 요약 완료")
        return {
            "youtube_summary": updated_youtube_summary,
            "messages": [message]
        }
        
    except Exception as e:
        message = HumanMessage(content=f"YouTube 재실행 실패: {str(e)}")
        return {"messages": [message]}

def present_final_result_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("✅ 최종 결과 생성 중...")
    if state.get("error"):
        final_output = {"success": False, "error_message": state["error"]}
    else:
        final_output = {
            "success": True,
            "analysis": {
                "diagnosis": state.get("diagnosis", {}).get("korean"),
                "details": state.get("pose_analysis_result")
            },
            "primary_recommendation": state.get("recommended_exercise"), 
            "supplementary_video": { 
                "search_phrase": state.get("chatbot_result", {}).get("search_phrase"),
                "youtube_url": state.get("chatbot_result", {}).get("youtube_url"),
                "video_summary": state.get("youtube_summary") or {}
            },            
        }
    print("\n--- 최종 결과 (JSON) ---")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    return {"final_output": final_output}

# --- 4. Supervisor 노드 (재검색 로직 추가) ---
def supervisor_node(state: SupervisorGraphState) -> Dict[str, str]:
    print("[Supervisor] 다음 작업 결정 중...")
    last_message = state['messages'][-1].content
    
    if "자세 분석 완료" in last_message:
        return {"next_agent": "plot_radar_chart"}
    elif "레이더 차트 생성 완료" in last_message:
        return {"next_agent": "recommend_exercise"}
    elif "DB 기반 운동 추천 완료" in last_message:
        user_choice = input("운동 추천 후 다음 단계 선택 (1: AI 코치와 대화, 2: 유튜브 영상 추천): ").strip()
        if user_choice == "1":
            return {"next_agent": "ai_coach_interaction"}
        elif user_choice == "2":
            return {"next_agent": "video_search"}
        else:
            print("잘못된 입력입니다. 기본적으로 유튜브 영상 추천으로 진행합니다.")
            return {"next_agent": "video_search"}
    elif "AI 코치 대화 완료" in last_message:
        return {"next_agent": "present_final_result"}
    elif "유튜브 영상 추천 완료" in last_message:
        return {"next_agent": "summarize_video"}
    elif "유튜브 영상 검색 완료" in last_message:
        return {"next_agent": "summarize_video"}
    elif "영상 요약 완료" in last_message:
        return {"next_agent": "validate_summary"}
    elif "요약 검증 성공" in last_message:
        # 검증 성공 후 댓글 수에 따라 분기
        comment_count = state.get("comment_count", 0)
        if comment_count >= 10:
            print(f"  > 댓글 수({comment_count})가 10개 이상이므로 사용자에게 질문합니다.")
            return {"next_agent": "ask_user_response"}
        else:
            print(f"  > 댓글 수({comment_count})가 10개 미만이므로 요약을 제공하지 않습니다.")
            return {"next_agent": "comment_summary_unavailable"}
    elif "사용자 응답 수집 완료" in last_message:
        # 사용자 응답에 따라 분기
        user_response = state.get("user_response", "").lower()
        
        # 긍정적 응답인지 확인
        positive_responses = ["응", "네", "보여줘", "궁금해", "그래", "좋아", "yes", "y"]
        if any(pos in user_response for pos in positive_responses):
            return {"next_agent": "rerun_youtube_agent"}
        else:
            # 부정적 응답이면 바로 최종 결과로
            return {"next_agent": "present_final_result"}
    elif "YouTube 댓글 요약 완료" in last_message:
        return {"next_agent": "present_final_result"}
    elif "댓글 요약 제공 불가" in last_message: 
        return {"next_agent": "present_final_result"}
    elif "YouTube 재실행 실패" in last_message:
        return {"next_agent": "present_final_result"}
    elif "요약 검증 실패" in last_message:
        if state["search_retries"] >= 2:
            return {"next_agent": "present_final_result"}
        else:
            return {"next_agent": "video_search"}
    else:
        return {"next_agent": "END"}
    
    
# --- 5. 그래프 구성 (Supervisor 패턴) ---
workflow = StateGraph(SupervisorGraphState)

workflow.add_node("analyze_user_pose", analyze_user_pose_node)
workflow.add_node("plot_radar_chart", plot_radar_chart_node)
workflow.add_node("recommend_exercise", recommend_exercise_node)
workflow.add_node("video_search", video_search_node)
workflow.add_node("summarize_video", summarize_video_node)
workflow.add_node("validate_summary", validate_summary_node)
workflow.add_node("present_final_result", present_final_result_node)
workflow.add_node("ask_user_response", ask_user_response_node)
workflow.add_node("rerun_youtube_agent", rerun_youtube_agent_node)
workflow.add_node("comment_summary_unavailable", comment_summary_unavailable_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("ai_coach_interaction", ai_coach_interaction_node)

workflow.set_entry_point("analyze_user_pose")
workflow.add_edge("analyze_user_pose", "supervisor")
workflow.add_edge("plot_radar_chart", "supervisor")
workflow.add_edge("recommend_exercise", "supervisor")
workflow.add_edge("video_search", "supervisor")
workflow.add_edge("summarize_video", "supervisor")
workflow.add_edge("validate_summary", "supervisor")
workflow.add_edge("ask_user_response", "supervisor")
workflow.add_edge("rerun_youtube_agent", "supervisor")
workflow.add_edge("comment_summary_unavailable", "supervisor")
workflow.add_edge("ai_coach_interaction", "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "plot_radar_chart": "plot_radar_chart",
        "recommend_exercise": "recommend_exercise",
        "video_search": "video_search",
        "summarize_video": "summarize_video",
        "validate_summary": "validate_summary",
        "ask_user_response": "ask_user_response",  
        "rerun_youtube_agent": "rerun_youtube_agent", 
        "comment_summary_unavailable": "comment_summary_unavailable",
        "present_final_result": "present_final_result",
        "END": END
    }
)
workflow.add_edge("present_final_result", END)
workflow_app = workflow.compile()  # LangGraph 워크플로우 인스턴스

# --- 분석/레이더차트만 포함하는 워크플로우 정의 ---
analysis_workflow = StateGraph(SupervisorGraphState)
analysis_workflow.add_node("analyze_user_pose", analyze_user_pose_node)
analysis_workflow.add_node("plot_radar_chart", plot_radar_chart_node)
analysis_workflow.set_entry_point("analyze_user_pose")
analysis_workflow.add_edge("analyze_user_pose", "plot_radar_chart")
analysis_workflow.add_edge("plot_radar_chart", END)
analysis_workflow_app = analysis_workflow.compile()

# --- FastAPI 인스턴스 생성 ---
app = FastAPI()

class AnalysisRequest(BaseModel):
    image_url: str
    analysis_mode: str = "front"

# --- 분석/레이더차트만 실행하는 엔드포인트 ---
@app.post("/analyze-graph")
async def analyze_graph_endpoint(request: AnalysisRequest):
    print("==== /analyze-graph endpoint called ====")
    print(f"Received image_url: {request.image_url}, analysis_mode: {request.analysis_mode}")
    # 1. 분석/레이더차트만 실행하는 워크플로우 상태 초기화
    initial_state = {
        "messages": [HumanMessage(content="자세 분석을 시작합니다.")],
        "image_path": request.image_url,
        "analysis_mode": request.analysis_mode,
        "search_retries": 0,
        "comment_count": 0,
        "user_response": None,
        "youtube_thread_id": None,
        "youtube_config": None,
        "radar_chart_path": None # 초기화
    }
    # 2. 분석/레이더차트만 실행
    result = analysis_workflow_app.invoke(initial_state)
    print("result keys:", result.keys())
    print("radar_chart_path in result:", result.get("radar_chart_path"))
    # 3. 레이더 차트 Cloudinary 업로드
    radar_chart_path = result.get("radar_chart_path")
    print("레이더 차트 경로:", radar_chart_path)
    radar_chart_url = None
    if radar_chart_path:
        try:
            upload_result = cloudinary.uploader.upload(radar_chart_path, folder="radar_charts/")
            radar_chart_url = upload_result["secure_url"]
            print("Cloudinary 업로드 결과:", radar_chart_url)
        except Exception as e:
            print("Cloudinary 업로드 에러:", e)
    # 4. 분석 점수 추출
    pose_data = result.get("pose_analysis_result", {})
    scores = pose_data.get("scores", {})
    feedback = pose_data.get("feedback", {})
    measurements = pose_data.get("measurements", {})
    return {
        "diagnosis": result.get("diagnosis", {}),  # 전체 diagnosis 객체 반환 (korean 키 포함)
        "radar_chart_url": radar_chart_url,
        "spineCurvScore": scores.get("척추굽음score"),
        "spineScolScore": scores.get("척추휨score"),
        "pelvicScore": scores.get("골반틀어짐score"),
        "neckScore": scores.get("거북목score"),
        "shoulderScore": scores.get("어깨score"),
        "feedback": feedback,
        "measurements": measurements
    }

class ChatbotRequest(BaseModel):
    type: str  # "ai_coach" or "recommend_video"
    user_id: int
    history_id: int
    message: Optional[str] = None

class ChatbotResponse(BaseModel):
    type: str
    response: str
    video_url: Optional[str] = None
    video_title: Optional[str] = None

def get_analysis_history_from_spring(history_id: int):
    SPRING_API_URL = "http://localhost:8081/api/analysis-histories"
    try:
        resp = requests.get(f"{SPRING_API_URL}/{history_id}")
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        print(f"Spring API 호출 오류: {e}")
        return None

@app.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_endpoint(req: ChatbotRequest):
    # 1. 분석 결과 Spring에서 조회
    history = get_analysis_history_from_spring(req.history_id)
    if not history:
        raise HTTPException(status_code=404, detail="분석 이력을 찾을 수 없습니다.")

    # 2. diagnosis 등 state 구성
    diagnosis = {}
    if history.get("diagnosis"):
        try:
            diagnosis = json.loads(history["diagnosis"])
        except Exception:
            diagnosis = {"korean": history["diagnosis"]}

    # 3. 추천 운동을 recommend_exercise_node로부터 받아옴
    rec_state = {
        "diagnosis": diagnosis,
        "search_retries": 0,
        "tried_video_urls": []
    }
    rec_result = recommend_exercise_node(rec_state)
    recommended_exercise = rec_result.get("recommended_exercise", {"name": "목 스트레칭"})

    # 4. 기존 노드 활용
    if req.type == "ai_coach":
        state = {
            "diagnosis": diagnosis,
            "recommended_exercise": recommended_exercise
        }
        result = ai_coach_interaction_node(state)
        response_text = result["messages"][-1].content if "messages" in result and result["messages"] else "AI 코치 응답이 없습니다."
        return ChatbotResponse(type="ai_coach", response=response_text)

    elif req.type == "recommend_video":
        state = {
            "recommended_exercise": recommended_exercise,
            "search_retries": 0,
            "tried_video_urls": []
        }
        # 1. 영상 추천
        result = video_search_node(state)
        chatbot_result = result.get("chatbot_result", {})
        video_url = chatbot_result.get("youtube_url")
        video_title = chatbot_result.get("video_title")
        if not video_url or "No video found" in str(video_url):
            return ChatbotResponse(
                type="recommend_video",
                response="추천 영상을 찾지 못했습니다. 다시 시도해 주세요.",
                video_url=None,
                video_title=None
            )
        # 2. 영상 요약
        state.update({"chatbot_result": chatbot_result})
        result = summarize_video_node(state)
        summary = result.get("youtube_summary")
        comment_count = result.get("comment_count", 0)
        # 3. 요약 검증
        state.update({"youtube_summary": summary, "comment_count": comment_count})
        result = validate_summary_node(state)
        # 4. 댓글 수에 따라 분기
        if comment_count >= 10:
            # 실제 서비스에서는 프론트에서 추가 요청을 받아야 하므로, 여기서는 안내 메시지 반환
            return ChatbotResponse(
                type="recommend_video",
                response="영상 요약이 완료되었습니다. 댓글 요약을 원하시면 추가로 요청해 주세요.",
                video_url=video_url,
                video_title=video_title
            )
        else:
            # 댓글 요약 불가 안내
            result = comment_summary_unavailable_node(state)
            return ChatbotResponse(
                type="recommend_video",
                response="영상 요약이 완료되었으나, 댓글 수가 적어 댓글 요약을 제공하지 않습니다.",
                video_url=video_url,
                video_title=video_title
            )
    else:
        return ChatbotResponse(type="error", response="지원하지 않는 type입니다.")
