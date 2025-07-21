from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import json
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

router = APIRouter(prefix="/youtube", tags=["YouTube"])

class YoutubeRequest(BaseModel):
    userId: int
    historyId: int
    message: str = None
    videoUrl: str = None
    type: str = None  # 'recommend', 'summary', 'comment_summary'
    diagnosis: dict = None  # Spring에서 진단 결과를 직접 전달받음
    recommended_exercise: dict = None  # Spring에서 추천운동을 직접 전달받음

class YoutubeResponse(BaseModel):
    type: str
    response: str
    videoUrl: str = None
    videoTitle: str = None

@router.post("", response_model=YoutubeResponse)
async def youtube_endpoint(req: YoutubeRequest):
    session_key = f"chat_session:{req.userId}"
    # SPRING_API_URL 삭제
    # 진단/추천운동 등은 req에서 직접 받음
    diagnosis = req.diagnosis or {"korean": "진단 정보가 없습니다."}
    recommended_exercise = req.recommended_exercise or {"name": "목 스트레칭"}
    search_query = req.message or ""
    from app.analyze_router import recommend_exercise_node
    from app.main_graph import video_search_node, summarize_video_node
    if req.type == "recommend":
        state = {
            "diagnosis": diagnosis,
            "recommended_exercise": recommended_exercise,
            "user_message": req.message,
            "search_query": search_query
        }
        result = await video_search_node(state)
        if "error" in result:
            return YoutubeResponse(type="error", response=f"영상 추천 오류: {result['error']}")
        chatbot_result = result.get("chatbot_result", {})
        video_url = chatbot_result.get("youtube_url")
        video_title = chatbot_result.get("video_title")
        redis_client.rpush(session_key, json.dumps({"type": "youtube_recommend", "content": video_url}))
        return YoutubeResponse(type="youtube_recommend", response="추천 영상입니다.", videoUrl=video_url, videoTitle=video_title)
    elif req.type == "summary":
        state = {
            "diagnosis": diagnosis,
            "recommended_exercise": recommended_exercise,
            "user_message": req.message,
            "search_query": search_query,
            "chatbot_result": {"youtube_url": req.videoUrl}
        }
        result = await summarize_video_node(state)
        if "error" in result:
            return YoutubeResponse(type="error", response=f"영상 요약 오류: {result['error']}")
        summary = result.get("youtube_summary")
        redis_client.rpush(session_key, json.dumps({"type": "youtube_summary", "content": summary}))
        return YoutubeResponse(type="youtube_summary", response=str(summary))
    elif req.type == "comment_summary":
        from app.agents.youtube_agent import graph as youtube_summary_agent
        video_url = req.videoUrl
        summary_result = youtube_summary_agent.invoke({"url": video_url, "reply": "댓글 요약해주세요"})
        if summary_result.get("error"):
            return YoutubeResponse(type="error", response=f"댓글 요약 오류: {summary_result['error']}")
        comment_summary = summary_result.get("comment_summary", {})
        redis_client.rpush(session_key, json.dumps({"type": "youtube_comment_summary", "content": comment_summary}))
        return YoutubeResponse(type="youtube_comment_summary", response=str(comment_summary))
    else:
        return YoutubeResponse(type="error", response="지원하지 않는 type입니다.") 