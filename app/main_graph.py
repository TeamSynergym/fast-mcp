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

# --- 서비스 및 노드 클래스 Import ---
from app.services.posture_analyzer import PostureAnalyzer
from app.agents.youtube_agent import graph as youtube_summary_agent
from app.nodes.chatbot_node import ChatbotActionNode
from app.services.exercise_vector_db import ExerciseVectorDB

load_dotenv()

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
    final_output: Dict[str, Any]
    error: Optional[str]
    
    # 사용자 응답 관련 필드(댓글 요약여부 결정)
    user_response: Optional[str]  # 사용자의 응답
    youtube_thread_id: Optional[str]  # YouTube agent의 스레드 ID
    youtube_config: Optional[Dict[str, Any]]  # YouTube agent 설정

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

def recommend_exercise_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 2] 맞춤 운동 추천 중 (from VectorDB)...")
    if state.get("error"): return {}
    try:
        diagnosis_text = state["diagnosis"]["korean"]
        
        # LLM을 사용해 진단 내용에서 핵심 키워드를 추출하여 검색 쿼리 생성
        prompt = f"""아래의 자세 진단 내용에 가장 적합한 '단 한 가지'의 검색어을 추천해줘. 
        ~난이도, ~효과를 가진, ~운동의 순서로 검색어를 작성해야해.
        VectorDB 검색에 사용할 키워드 문장 오직 한개만 간결하게 한 줄로 답해줘.
        
        [진단 내용]
        {diagnosis_text}
        [출력 예시]
        - 중급 난이도의 유연성을 높이는 효과를 가진 스트레칭 운동
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

def video_search_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print(f"[Node 3 - 시도 {state['search_retries'] + 1}] 보충 영상 검색 중 (Youtube)...")
    if state.get("error"): return {}
    try:
        exercise_name = state["recommended_exercise"]["name"]
        if "자세" in exercise_name or "스트레칭" in exercise_name:
            search_query = f"{exercise_name} 하는 법"
        else:
            search_query = f"{exercise_name} 운동 자세"
            
        print(f"  > 원본 검색어: '{exercise_name}'")
        print(f"  > ✅ 최종 유튜브 검색어: '{search_query}'")
        
        # 가공된 검색어로 챗봇 노드 실행
        result = asyncio.run(chatbot_node.run(prompt=search_query))
        
        if not result.get("youtube_url") or "No video found" in result.get("youtube_url"):
            raise ValueError("추천 유튜브 영상을 찾지 못했습니다.")

        message = HumanMessage(content=f"유튜브 영상 검색 완료. URL: {result.get('youtube_url')}")
        return {"chatbot_result": result, "messages": [message], "search_retries": state["search_retries"] + 1}
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
        message = HumanMessage(content="영상 요약 완료.")
        return {"youtube_summary": summary, "messages": [message]}
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

    # 요약이 너무 짧거나 없는 경우, 바로 부적합 판정
    if not summary_dict or len(summary_text) < 50:
        print("  > 검증 실패: 요약 내용이 너무 짧거나 없습니다.")
        message = HumanMessage(content="요약 검증 실패: 내용 부실")
        return {"messages": [message]}

    # LLM을 통한 관련성 검증
    structured_validator = llm.with_structured_output(ValidationResult)
    prompt = f"""사용자의 자세 진단은 '{diagnosis_text}'입니다. 아래 유튜브 영상 요약이 이 진단과 관련이 있습니까?
    
    [영상 요약]
    {summary_text}
    
    판단 결과와 이유를 JSON 형식으로 답해주세요."""
    
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
    
def rerun_youtube_agent_node(state: SupervisorGraphState) -> Dict[str, Any]:
    """사용자 응답을 바탕으로 YouTube agent를 재실행하는 노드"""
    print("[Node 5-3] YouTube Agent 재실행 중...")
    
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
            "primary_recommendation": state.get("recommended_exercise"), # DB 기반 추천
            "supplementary_video": { # 유튜브 기반 추천
                "search_phrase": state.get("chatbot_result", {}).get("search_phrase"),
                "youtube_url": state.get("chatbot_result", {}).get("youtube_url"),
                "video_summary": state.get("youtube_summary"),
                "comment_summary": state.get("youtube_summary", {}).get("comment_summary", None)
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
        return {"next_agent": "recommend_exercise"}
    elif "DB 기반 운동 추천 완료" in last_message:
        return {"next_agent": "video_search"}
    elif "유튜브 영상 검색 완료" in last_message:
        return {"next_agent": "summarize_video"}
    elif "영상 요약 완료" in last_message:
        return {"next_agent": "validate_summary"}
    elif "요약 검증 성공" in last_message:
        # 검증 성공 후 사용자 응답 요청
        return {"next_agent": "ask_user_response"}
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
workflow.add_node("recommend_exercise", recommend_exercise_node)
workflow.add_node("video_search", video_search_node)
workflow.add_node("summarize_video", summarize_video_node)
workflow.add_node("validate_summary", validate_summary_node)
workflow.add_node("present_final_result", present_final_result_node)
workflow.add_node("ask_user_response", ask_user_response_node)
workflow.add_node("rerun_youtube_agent", rerun_youtube_agent_node)
workflow.add_node("supervisor", supervisor_node)

workflow.set_entry_point("analyze_user_pose")
workflow.add_edge("analyze_user_pose", "supervisor")
workflow.add_edge("recommend_exercise", "supervisor")
workflow.add_edge("video_search", "supervisor")
workflow.add_edge("summarize_video", "supervisor")
workflow.add_edge("validate_summary", "supervisor")
workflow.add_edge("ask_user_response", "supervisor")
workflow.add_edge("rerun_youtube_agent", "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "recommend_exercise": "recommend_exercise",
        "video_search": "video_search",
        "summarize_video": "summarize_video",
        "validate_summary": "validate_summary",
        "ask_user_response": "ask_user_response",  
        "rerun_youtube_agent": "rerun_youtube_agent", 
        "present_final_result": "present_final_result",
        "END": END
    }
)
workflow.add_edge("present_final_result", END)
app = workflow.compile()

# --- 6. 실행 예시 ---
if __name__ == "__main__":
    initial_state = {
        "messages": [HumanMessage(content="자세 분석을 시작합니다.")],
        "image_path": "app/services/images/test_front.jpg",
        "analysis_mode": "front",
        "search_retries": 0,
        "user_response": None,
        "youtube_thread_id": None,
        "youtube_config": None
    }
    
    print("🚀 AI 피트니스 코치 워크플로우를 시작합니다 (Interactive Mode).")
    print(f"   - 입력 이미지: {initial_state['image_path']}")
    print(f"   - 분석 모드: {initial_state['analysis_mode']}")
    print("-" * 50)

    app.invoke(initial_state)

    print("\n" + "-"*50)
    print("🏁 워크플로우 실행 완료.")