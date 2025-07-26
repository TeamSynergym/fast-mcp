
from app.services.posture_analyzer import PostureAnalyzer
from app.services.radar_chart import extract_radar_scores, create_and_upload_radar_chart
from app.services.recommend import recommend_node
from app.services.exercise_vector_db import ExerciseVectorDB
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from app.services.ai_coach_service import ai_coach_interaction_service
from app.nodes.chatbot_node import ChatbotActionNode
from app.agents.youtube_agent import graph as youtube_summary_agent
import redis
import json
import os
import asyncio

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



# 비디오 검색 노드
def video_search_node(state):
    print("[video_search_node] 비디오 검색 시작")
    recommended_exercise = state.get("recommended_exercise", {})
    exercise_name = recommended_exercise.get("name", "")
    
    if not exercise_name:
        return {"error": "추천 운동 이름이 없습니다."}
    
    try:
        # search_retries 값에 따라 분기 로직
        search_retries = state.get("search_retries", 0)
        if search_retries > 0:
            print("   > 영상 검증 실패로 재검색을 실행합니다. 이전 영상은 제외됩니다.")
            tried_urls = state.get("tried_video_urls", [])
        else:
            print("   > 초기 영상 검색을 실행합니다.")
            tried_urls = []
            
        # 검색어 생성
        if "자세" in exercise_name or "스트레칭" in exercise_name:
            search_query = f"{exercise_name}"
        else:
            search_query = f"{exercise_name} 운동"
            
        print(f"  > 유튜브 검색어: '{search_query}'")
        
        # 챗봇 노드를 통해 영상 검색
        chatbot_node = ChatbotActionNode()
        result = chatbot_node.run(prompt=search_query, exclude_urls=tried_urls)
        new_url = result.get("youtube_url")
        video_title = result.get("video_title")
        
        if not new_url or "No video found" in new_url:
            raise ValueError("추천 유튜브 영상을 찾지 못했습니다.")

        print(f"  > 검색 완료. URL: {new_url}, 제목: {video_title}")
        
        # 새로운 URL을 tried_urls에 추가
        updated_tried_urls = tried_urls + [new_url] if new_url else tried_urls
        
        return {
            "chatbot_result": result,
            "search_retries": search_retries + 1,
            "tried_video_urls": updated_tried_urls,
            "diagnosis": state.get("diagnosis"),
            "recommended_exercise": recommended_exercise,
            "pose_data": state.get("pose_data"),
            "radar_chart_url": state.get("radar_chart_url")
        }
        
    except Exception as e:
        print(f"  > 검색 실패: {e}")
        return {"error": f"비디오 검색 실패: {e}"}

# 영상 요약 노드
def summarize_video_node(state):
    print("[summarize_video_node] 유튜브 영상 요약 시작")
    try:
        youtube_url = state.get("chatbot_result", {}).get("youtube_url")
        if not youtube_url:
            return {"error": "YouTube URL이 없습니다."}
            
        summary_result = youtube_summary_agent.invoke({"url": youtube_url})
        if summary_result.get("error"):
            raise ValueError(f"유튜브 요약 실패: {summary_result['error']}")
        
        summary = summary_result.get("script_summary")
        if not isinstance(summary, dict):
            summary = {"summary": summary}
        comment_count = summary_result.get("comment_count", 0)
        
        print(f"  > 요약 완료. 댓글 수: {comment_count}")
        
        return {
            "youtube_summary": summary,
            "comment_count": comment_count,
            "chatbot_result": state.get("chatbot_result"),
            "diagnosis": state.get("diagnosis"),
            "recommended_exercise": state.get("recommended_exercise"),
            "pose_data": state.get("pose_data"),
            "radar_chart_url": state.get("radar_chart_url")
        }
        
    except Exception as e:
        return {"error": f"영상 요약 실패: {e}"}

# 요약 검증 노드
def validate_summary_node(state):
    print("[validate_summary_node] 영상 요약 검증 시작")
    try:
        summary_dict = state.get("youtube_summary", {})
        summary_text = json.dumps(summary_dict)
        diagnosis_text = state.get("diagnosis", {}).get("korean", "")
        recommended_exercise = state.get("recommended_exercise", {}).get("name", "")

        # 요약이 너무 짧거나 없는 경우, 바로 부적합 판정
        if not summary_dict or len(summary_text) < 50:
            print("  > 검증 실패: 요약 내용이 너무 짧거나 없습니다.")
            comment_count = state.get("comment_count", 0)
            if comment_count >= 10:
                return {
                    "error": "요약 내용이 부실합니다.",
                    "youtube_summary": state.get("youtube_summary"),
                    "comment_count": comment_count,
                    "chatbot_result": state.get("chatbot_result"),
                    "diagnosis": state.get("diagnosis"),
                    "recommended_exercise": state.get("recommended_exercise"),
                    "pose_data": state.get("pose_data"),
                    "radar_chart_url": state.get("radar_chart_url"),
                    "next_step": "comment_summary"
                }
            else:
                return {
                    "error": "요약 내용이 부실합니다.",
                    "youtube_summary": state.get("youtube_summary"),
                    "comment_count": comment_count,
                    "chatbot_result": state.get("chatbot_result"),
                    "diagnosis": state.get("diagnosis"),
                    "recommended_exercise": state.get("recommended_exercise"),
                    "pose_data": state.get("pose_data"),
                    "radar_chart_url": state.get("radar_chart_url"),
                    "next_step": "comment_summary_unavailable"
                }

        # LLM을 통한 관련성 검증
        prompt = f"""
당신은 사용자의 자세 교정을 위한 운동 영상을 필터링하는 AI 전문가입니다.

[분석할 정보]
- 자세 진단: '{diagnosis_text}'
- 추천 운동: '{recommended_exercise}'
- 추천 운동의 효과: '{state.get("recommended_exercise", {}).get("description", "효과 정보 없음")}'
- 영상 요약: '{summary_text}'

[수행할 작업]
- '자세 진단'을 개선하고 '추천 운동'을 수행하는 데 '영상 요약'의 내용이 적합한지 판단해주세요.
- 가장 먼저 제외조건에 해당하는 내용이 있는지 살펴보고, 해당할 시 반드시 관련없음으로 판단해야 합니다.
- 또한 동영상의 내용과 '추천 운동의 효과'가 서로 관련있는지도 판단해야 합니다.

[제외 조건]
- **스포츠 전문 훈련:** 특정 스포츠(예: 축구, 야구, 골프, 수영)의 기술 향상을 위한 훈련은 '관련 없음'으로 처리합니다.

관련이 있으면 '적합', 없으면 '부적합'으로만 답해주세요.
"""
        
        validation_result = llm.invoke(prompt).content.strip()
        
        if "적합" in validation_result:
            print("  > 검증 성공: 영상이 적합합니다.")
            # 댓글 수에 따라 분기 로직 추가
            comment_count = state.get("comment_count", 0)
            if comment_count >= 10:
                print(f"  > 댓글 수({comment_count})가 10개 이상이므로 댓글 요약을 진행합니다.")
                return {
                    "validation_result": "success",
                    "youtube_summary": state.get("youtube_summary"),
                    "comment_count": state.get("comment_count"),
                    "chatbot_result": state.get("chatbot_result"),
                    "diagnosis": state.get("diagnosis"),
                    "recommended_exercise": state.get("recommended_exercise"),
                    "pose_data": state.get("pose_data"),
                    "radar_chart_url": state.get("radar_chart_url"),
                    "next_step": "comment_summary"
                }
            else:
                print(f"  > 댓글 수({comment_count})가 10개 미만이므로 댓글 요약을 제공하지 않습니다.")
                return {
                    "validation_result": "success",
                    "youtube_summary": state.get("youtube_summary"),
                    "comment_count": state.get("comment_count"),
                    "chatbot_result": state.get("chatbot_result"),
                    "diagnosis": state.get("diagnosis"),
                    "recommended_exercise": state.get("recommended_exercise"),
                    "pose_data": state.get("pose_data"),
                    "radar_chart_url": state.get("radar_chart_url"),
                    "next_step": "comment_summary_unavailable"
                }
        else:
            print("  > 검증 실패: 영상이 부적합합니다.")
            return {
                "error": "영상이 부적합합니다.",
                "youtube_summary": state.get("youtube_summary"),
                "comment_count": state.get("comment_count"),
                "chatbot_result": state.get("chatbot_result"),
                "diagnosis": state.get("diagnosis"),
                "recommended_exercise": state.get("recommended_exercise"),
                "pose_data": state.get("pose_data"),
                "radar_chart_url": state.get("radar_chart_url"),
                "next_step": "comment_summary_unavailable"
            }
            
    except Exception as e:
        return {
            "error": f"요약 검증 실패: {e}",
            "youtube_summary": state.get("youtube_summary"),
            "comment_count": state.get("comment_count"),
            "chatbot_result": state.get("chatbot_result"),
            "diagnosis": state.get("diagnosis"),
            "recommended_exercise": state.get("recommended_exercise"),
            "pose_data": state.get("pose_data"),
            "radar_chart_url": state.get("radar_chart_url"),
            "next_step": "comment_summary_unavailable"
        }

# 댓글 요약 불가 노드
def comment_summary_unavailable_node(state):
    print("[comment_summary_unavailable_node] 댓글 요약 제공 불가 안내")
    
    # 최종 결과에 표시될 메시지를 youtube_summary에 추가
    updated_youtube_summary = state.get("youtube_summary", {})
    if isinstance(updated_youtube_summary, dict):
         updated_youtube_summary["comment_summary"] = "댓글 개수가 10개 미만으로 댓글 요약을 제공하지 않습니다."
    
    return {
        "youtube_summary": updated_youtube_summary,
        "comment_count": state.get("comment_count"),
        "chatbot_result": state.get("chatbot_result"),
        "diagnosis": state.get("diagnosis"),
        "recommended_exercise": state.get("recommended_exercise"),
        "pose_data": state.get("pose_data"),
        "radar_chart_url": state.get("radar_chart_url")
    }

# YouTube Agent 재실행 노드
def rerun_youtube_agent_node(state):
    print("[rerun_youtube_agent_node] YouTube Agent 재실행 시작")
    
    try:
        # YouTube agent의 메모리 그래프 사용
        youtube_state = {
            "url": state["chatbot_result"]["youtube_url"],
            "reply": state.get("user_response", ""),
            "script_summary": state.get("youtube_summary", {})
        }
        
        # continue_with_memory 함수 사용하여 댓글 요약 실행
        from app.agents.youtube_agent import graph_memory, continue_with_memory
        
        result = continue_with_memory(
            graph_memory, 
            youtube_state, 
            {"configurable": {"thread_id": f"thread_{hash(state['chatbot_result']['youtube_url'])}"}}, 
            {"reply": state.get("user_response", ""), "url": youtube_state["url"]}
        )
        
        # 댓글 요약 결과 추가
        updated_youtube_summary = state.get("youtube_summary", {})
        if result.get("comment_summary"):
            updated_youtube_summary["comment_summary"] = result["comment_summary"]
        
        print("  > 댓글 요약 완료")
        
        return {
            "youtube_summary": updated_youtube_summary,
            "comment_count": state.get("comment_count"),
            "chatbot_result": state.get("chatbot_result"),
            "diagnosis": state.get("diagnosis"),
            "recommended_exercise": state.get("recommended_exercise"),
            "pose_data": state.get("pose_data"),
            "radar_chart_url": state.get("radar_chart_url")
        }
        
    except Exception as e:
        return {"error": f"YouTube 재실행 실패: {e}"}

# 최종 결과 생성 노드
def present_final_result_node(state):
    print("최종 결과 생성 중...")
    if state.get("error"):
        final_output = {"success": False, "error_message": state["error"]}
    else:
        diagnosis = state.get("diagnosis")
        diagnosis_korean = diagnosis.get("korean") if diagnosis else None

        chatbot_result = state.get("chatbot_result") or {}
        search_phrase = chatbot_result.get("search_phrase") if chatbot_result else None
        youtube_url = chatbot_result.get("youtube_url") if chatbot_result else None

        final_output = {
            "success": True,
            "analysis": {
                "diagnosis": diagnosis_korean,
                "details": state.get("pose_data")
            },
            "primary_recommendation": state.get("recommended_exercise"),
            "supplementary_video": {
                "search_phrase": search_phrase,
                "youtube_url": youtube_url,
                "video_summary": state.get("youtube_summary") or {}
            },
        }
    print("\n--- 최종 결과 (JSON) ---")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    return {
        "final_output": final_output,
        "diagnosis": state.get("diagnosis"),
        "recommended_exercise": state.get("recommended_exercise"),
        "chatbot_result": state.get("chatbot_result"),
        "youtube_summary": state.get("youtube_summary"),
        "comment_count": state.get("comment_count"),
        "pose_data": state.get("pose_data"),
        "radar_chart_url": state.get("radar_chart_url")
    }


# YouTube 전용 그래프
youtube_graph = StateGraph(dict)
youtube_graph.add_node("video_search", video_search_node)
youtube_graph.add_node("summarize_video", summarize_video_node)
youtube_graph.add_node("validate_summary", validate_summary_node)
youtube_graph.add_node("comment_summary_unavailable", comment_summary_unavailable_node)
youtube_graph.add_node("rerun_youtube_agent", rerun_youtube_agent_node)
youtube_graph.add_node("present_final_result", present_final_result_node)
youtube_graph.set_entry_point("video_search")
youtube_graph.add_edge("video_search", "summarize_video")
youtube_graph.add_edge("summarize_video", "validate_summary")

# 조건부 엣지: 댓글 수에 따라 분기
def route_after_validation(state):
    next_step = state.get("next_step", "comment_summary_unavailable")
    return next_step

youtube_graph.add_conditional_edges(
    "validate_summary",
    route_after_validation,
    {
        "comment_summary": "rerun_youtube_agent",
        "comment_summary_unavailable": "comment_summary_unavailable"
    }
)

youtube_graph.add_edge("comment_summary_unavailable", "present_final_result")
youtube_graph.add_edge("rerun_youtube_agent", "present_final_result")
app_youtube_graph = youtube_graph.compile()


