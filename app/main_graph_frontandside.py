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
from data_server import plot_radar_chart

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
    image_paths: list[str]
    analysis_modes: list[str]
    
    # 재검색 횟수 추적
    search_retries: int
    
    # 노드별 결과 데이터
    pose_analysis_results: list[Dict[str, Any]]
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

# --- 3. LangGraph 노드 함수 재정의 ---

# 각 노드는 이제 'messages'에 자신의 실행 결과를 HumanMessage 형태로 추가하여 슈퍼바이저에게 보고합니다.
def analyze_both_poses_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 1] 정면/측면 자세 분석 중...")
    results = []
    messages = []
    for image_path, mode in zip(state["image_paths"], state["analysis_modes"]):
        analysis_result = posture_analyzer.analyze_posture(image_path, mode=mode)
        if not analysis_result.get("success") or not analysis_result.get("pose_data"):
            messages.append(HumanMessage(content=f"{mode} 분석 실패"))
            results.append(None)
        else:
            person_analysis = analysis_result["pose_data"][0]
            diagnosis_texts = posture_analyzer.generate_ollama_diagnosis(person_analysis, mode)
            messages.append(HumanMessage(content=f"{mode} 분석 완료. 진단: {diagnosis_texts['korean']}"))
            results.append({
                "person_analysis": person_analysis,
                "diagnosis": diagnosis_texts,
                "mode": mode,
                "image_path": image_path
            })
    messages.append(HumanMessage(content="두 사진 분석 완료"))
    return {
        "pose_analysis_results": results,
        "messages": messages
    }

def plot_avg_radar_chart_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node X] 두 사진 score 평균 레이더 차트 생성 중...")
    try:
        results = [r for r in state["pose_analysis_results"] if r is not None]
        keys = ["목score", "어깨score", "골반score", "척추(정면)score", "척추(측면)score"]
        avg_scores = {}
        for key in keys:
            vals = [r["person_analysis"]["scores"].get(key) for r in results if r["person_analysis"]["scores"].get(key) is not None]
            if vals:
                avg_scores[key] = sum(vals) / len(vals)
            else:
                avg_scores[key] = None
        radar_values = {
            "목": avg_scores["목score"],
            "어깨": avg_scores["어깨score"],
            "골반": avg_scores["골반score"],
            "척추(정면)": avg_scores["척추(정면)score"],
            "척추(측면)": avg_scores["척추(측면)score"]
        }
        # 가장 낮은 score 부위 찾기 (None 제외)
        min_part = min(
            (k for k, v in radar_values.items() if v is not None),
            key=lambda k: radar_values[k]
        )
        output_path = plot_radar_chart(radar_values)
        print(f"  > 평균 레이더 차트 저장: {output_path}")
        return {
            "radar_chart_path": output_path,
            "lowest_score_part": min_part,  # 추가!
            "messages": [HumanMessage(content="평균 레이더 차트 생성 완료")]
        }
    except Exception as e:
        return {"error": f"평균 레이더 차트 노드 오류: {e}"}

def recommend_exercise_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print("[Node 2] 맞춤 운동 추천 중 (from VectorDB)...")
    if state.get("error"): return {}
    try:
        # 두 장의 진단 합치기
        diagnosis_texts = []
        for r in state["pose_analysis_results"]:
            if r is not None and "diagnosis" in r:
                diag = r["diagnosis"].get("korean", "")
                if diag:
                    diagnosis_texts.append(diag)
        diagnosis_text = " ".join(diagnosis_texts)
        # 가장 낮은 score 부위명
        lowest_score_part = state.get("lowest_score_part", "")
        prompt = f'''아래의 자세 진단 내용과 집중해야 할 부위를 참고해서, 가장 적합한 '단 한 가지'의 검색어를 추천해줘.
        ~난이도, ~효과를 가진, ~부위의, ~운동의 순서로 검색어를 작성해야해.
        VectorDB 검색에 사용할 키워드 문장 오직 한개만 간결하게 한 줄로 답해줘.

        [진단 내용]
        {diagnosis_text}
        [집중해야 할 부위]
        {lowest_score_part}
        [출력 예시]
        - 중급 난이도의 유연성을 높이는 효과를 가진 골반 부위의 스트레칭 운동
        [생성된 검색어]
        '''
        llm_query = llm.invoke(prompt).content.strip()
        print(f"  > LLM 생성 검색어: '{llm_query}'")
        recommended_list = vector_db.search(llm_query, top_k=1)
        if not recommended_list:
            raise ValueError("VectorDB에서 추천 운동을 찾지 못했습니다.")
        retrieved_exercise = recommended_list[0]
        print(f"  > VectorDB 검색 결과 운동명: '{retrieved_exercise['name']}'")
        message = HumanMessage(content=f"DB 기반 운동 추천 완료: {retrieved_exercise['name']}")
        return {"recommended_exercise": retrieved_exercise, "messages": [message]}
    except Exception as e:
        return {"error": f"운동 추천 노드 오류: {e}"}

def video_search_node(state: SupervisorGraphState) -> Dict[str, Any]:
    print(f"[Node 3 - 시도 {state['search_retries'] + 1}] 보충 영상 검색 중 (Youtube)...")
    if state.get("error"): return {}
    try:
        if state.get("search_retries", 0) > 0:
            print("  > 영상 검증 실패로 재검색을 실행합니다. 이전 영상은 제외됩니다.")
            tried_urls = state.get("tried_video_urls", [])
        else:
            print("  > 초기 영상 검색을 실행합니다.")
            tried_urls = []
        # VectorDB에서 추천된 운동명 기반으로 검색어 생성
        exercise_name = state["recommended_exercise"]["name"]
        # --- 스포츠 전문 훈련 키워드 제외 ---
        exclude_keywords = ["축구", "야구", "골프", "수영", "농구", "테니스", "탁구", "배구"]
        exclude_str = " ".join([f'-{kw}' for kw in exclude_keywords])
        # 운동/스트레칭이 포함되어 있으면 그대로, 아니면 '운동' 붙임
        if any(x in exercise_name for x in ["운동", "스트레칭"]):
            search_query = f"{exercise_name} {exclude_str}"
        else:
            search_query = f"{exercise_name} 운동 {exclude_str}"
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
            "tried_video_urls": updated_tried_urls
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
    # 두 장의 진단 합치기
    diagnosis_texts = []
    for r in state["pose_analysis_results"]:
        if r is not None and "diagnosis" in r:
            diag = r["diagnosis"].get("korean", "")
            if diag:
                diagnosis_texts.append(diag)
    diagnosis_text = " ".join(diagnosis_texts)
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
        # 가장 점수 낮은 부위명
        lowest_score_part = state.get("lowest_score_part")
        part_to_diag_keyword = {
            "목": "거북목",
            "어깨": "어깨",
            "골반": "골반",
            "척추(정면)": "척추",
            "척추(측면)": "척추"
        }
        diag_keyword = part_to_diag_keyword.get(lowest_score_part, "")
        # pose_analysis_results에서 해당 부위 진단만 추출 (더 유연하게)
        diagnosis_text = ""
        for r in state["pose_analysis_results"]:
            if r is not None and "diagnosis" in r:
                diag = r["diagnosis"]["korean"]
                for line in diag.splitlines():
                    if diag_keyword in line:
                        diagnosis_text = line.strip()
                        break
                if diagnosis_text:
                    break
                if ":" in diag:
                    after_colon = diag.split(":", 1)[1]
                    if diag_keyword in after_colon:
                        diagnosis_text = diag.strip()
                        break
                for sent in diag.split('.'):
                    if diag_keyword in sent:
                        diagnosis_text = sent.strip()
                        break
            if diagnosis_text:
                break  # 하나만 추출
        if not diagnosis_text:
            diagnosis_text = f"{diag_keyword}에 대한 구체적 진단이 없습니다."
        # details 리스트 만들기 (기존과 동일)
        details = []
        for r in state["pose_analysis_results"]:
            if r is not None and "person_analysis" in r:
                pa = r["person_analysis"]
                details.append({
                    "mode": r.get("mode"),
                    "num_keypoints": pa.get("num_keypoints"),
                    "scores": pa.get("scores"),
                    "measurements": pa.get("measurements")
                })
        # --- front/side 점수 평균 합친 avg_details 생성 ---
        # keys: 거북목score, 어깨score, 골반틀어짐score, 척추휨score, 척추굽음score
        score_keys = ["거북목score", "어깨score", "골반틀어짐score", "척추휨score", "척추굽음score"]
        score_sums = {k: [] for k in score_keys}
        for r in state["pose_analysis_results"]:
            if r is not None and "person_analysis" in r:
                scores = r["person_analysis"].get("scores", {})
                for k in score_keys:
                    v = scores.get(k)
                    if v is not None:
                        score_sums[k].append(v)
        avg_scores = {k: (sum(vs)/len(vs) if vs else None) for k, vs in score_sums.items()}
        avg_details = {"avg_scores": avg_scores}
        # --- avg_diagnosis: 가장 낮은 평균 score 부위에 대한 진단 ---
        # radar_values와 동일하게 매핑
        radar_part_map = {
            "목": "거북목score",
            "어깨": "어깨score",
            "골반": "골반틀어짐score",
            "척추(정면)": "척추휨score",
            "척추(측면)": "척추굽음score"
        }
        # 평균 score 중 가장 낮은 부위
        min_avg_part = None
        min_avg_score = float('inf')
        for part, key in radar_part_map.items():
            val = avg_scores.get(key)
            if val is not None and val < min_avg_score:
                min_avg_score = val
                min_avg_part = part
        avg_diag_keyword = part_to_diag_keyword.get(min_avg_part, "")
        avg_diagnosis_text = ""
        for r in state["pose_analysis_results"]:
            if r is not None and "diagnosis" in r:
                diag = r["diagnosis"]["korean"]
                for line in diag.splitlines():
                    if avg_diag_keyword in line:
                        avg_diagnosis_text = line.strip()
                        break
                if avg_diagnosis_text:
                    break
                if ":" in diag:
                    after_colon = diag.split(":", 1)[1]
                    if avg_diag_keyword in after_colon:
                        avg_diagnosis_text = diag.strip()
                        break
                for sent in diag.split('.'):
                    if avg_diag_keyword in sent:
                        avg_diagnosis_text = sent.strip()
                        break
            if avg_diagnosis_text:
                break
        if not avg_diagnosis_text:
            avg_diagnosis_text = f"{avg_diag_keyword}에 대한 구체적 진단이 없습니다."
        # --- avg_feedback: front/side 통합 feedback ---
        feedbacks = []
        for r in state["pose_analysis_results"]:
            if r is not None and "person_analysis" in r:
                fb = r["person_analysis"].get("feedback", {})
                overall = fb.get("overall")
                if overall and overall not in feedbacks:
                    feedbacks.append(overall)
        avg_feedback = "\n".join(feedbacks) if feedbacks else "통합 피드백이 없습니다."
        # --- 최종 결과 생성 ---
        final_output = {
            "success": True,
            "analysis": {
                # diagnosis 필드 완전 삭제
                "details": details,
                "avg_details": avg_details,
                "avg_diagnosis": avg_diagnosis_text,
                "avg_feedback": avg_feedback
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
    
    if "두 사진 분석 완료" in last_message:
        return {"next_agent": "plot_avg_radar_chart"}
    elif "평균 레이더 차트 생성 완료" in last_message:
        return {"next_agent": "recommend_exercise"}
    elif "DB 기반 운동 추천 완료" in last_message:
        return {"next_agent": "video_search"}
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

workflow.add_node("analyze_both_poses", analyze_both_poses_node)
workflow.add_node("plot_avg_radar_chart", plot_avg_radar_chart_node)
workflow.add_node("recommend_exercise", recommend_exercise_node)
workflow.add_node("video_search", video_search_node)
workflow.add_node("summarize_video", summarize_video_node)
workflow.add_node("validate_summary", validate_summary_node)
workflow.add_node("present_final_result", present_final_result_node)
workflow.add_node("ask_user_response", ask_user_response_node)
workflow.add_node("rerun_youtube_agent", rerun_youtube_agent_node)
workflow.add_node("comment_summary_unavailable", comment_summary_unavailable_node)
workflow.add_node("supervisor", supervisor_node)

workflow.set_entry_point("analyze_both_poses")
workflow.add_edge("analyze_both_poses", "supervisor")
workflow.add_edge("plot_avg_radar_chart", "supervisor")
workflow.add_edge("recommend_exercise", "supervisor")
workflow.add_edge("video_search", "supervisor")
workflow.add_edge("summarize_video", "supervisor")
workflow.add_edge("validate_summary", "supervisor")
workflow.add_edge("ask_user_response", "supervisor")
workflow.add_edge("rerun_youtube_agent", "supervisor")
workflow.add_edge("comment_summary_unavailable", "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "plot_avg_radar_chart": "plot_avg_radar_chart",
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
app = workflow.compile()

# --- 6. 실행 예시 ---
if __name__ == "__main__":
    initial_state = {
        "messages": [HumanMessage(content="자세 분석을 시작합니다.")],
        "image_paths": ["app/services/images/test_front2.jpg", "app/services/images/test_side2.jpg"],
        "analysis_modes": ["front", "side"],
        "search_retries": 0,
        "comment_count": 0,
        "user_response": None,
        "youtube_thread_id": None,
        "youtube_config": None
    }
    
    print("🚀 AI 피트니스 코치 워크플로우를 시작합니다 (Interactive Mode).")
    print(f"   - 입력 이미지: {initial_state['image_paths']}")
    print(f"   - 분석 모드: {initial_state['analysis_modes']}")
    print("-" * 50)

    app.invoke(initial_state)

    print("\n" + "-"*50)
    print("🏁 워크플로우 실행 완료.")