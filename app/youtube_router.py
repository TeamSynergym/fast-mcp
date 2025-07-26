from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import json
import redis
import os
from dotenv import load_dotenv
from app.graph_workflow import app_youtube_graph
import time

load_dotenv()
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "192.168.2.6"), port=6379, db=0, decode_responses=True)

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
    sessionId: str = None  # sessionId 필드 추가
    videoUrl: str = None
    videoTitle: str = None
    youtubeSummary: dict = None
    commentCount: int = None

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

@router.post("", response_model=YoutubeResponse)
async def youtube_endpoint(req: YoutubeRequest):
    try:
        # 요청 타입에 따라 처리
        if req.type == "recommend":
            # 운동 추천 후 YouTube 영상 검색 및 요약
            input_data = {
                "recommended_exercise": req.recommended_exercise,
                "diagnosis": req.diagnosis,
                "search_retries": 0,
                "tried_video_urls": []
            }
            
            result = app_youtube_graph.invoke(input_data)
            
            print(f"[DEBUG] YouTube 그래프 결과: {result}")
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            # 결과를 Redis에 저장
            session_key = f"youtube_session:{req.userId}:{req.historyId}"
            redis_client.set(session_key, json.dumps(result), ex=60*60*24)
            
            # 세션 ID 생성 (기존 세션 ID가 있으면 사용, 없으면 새로 생성)
            session_id = f"youtube_{req.userId}_{req.historyId}_{int(time.time())}"
            
            # 결과에서 필요한 정보 추출
            chatbot_result = result.get("chatbot_result") or {}
            video_url = chatbot_result.get("youtube_url") or ""
            video_title = chatbot_result.get("video_title") or ""
            youtube_summary = result.get("youtube_summary", {})
            comment_count = result.get("comment_count")
            if comment_count is None:
                comment_count = 0
            
            return YoutubeResponse(
                type="recommend",
                response="YouTube 영상 추천 완료",
                sessionId=session_id,  # sessionId 추가
                videoUrl=video_url,
                videoTitle=video_title,
                youtubeSummary=youtube_summary,
                commentCount=comment_count
            )
            
        elif req.type == "comment_summary":
            # 댓글 요약만 실행 (댓글 수가 10개 이상일 때만)
            input_data = {
                "message": req.message,
                "videoUrl": req.videoUrl,
                "recommended_exercise": req.recommended_exercise,
                "diagnosis": req.diagnosis
            }
            
            # Redis에서 기존 요약 데이터 가져오기
            session_key = f"youtube_session:{req.userId}:{req.historyId}"
            if redis_client.exists(session_key):
                try:
                    cached_data = json.loads(redis_client.get(session_key))
                    input_data.update(cached_data)
                    
                    # 댓글 수 확인
                    chatbot_result = cached_data.get("chatbot_result", {})
                    youtube_summary = cached_data.get("youtube_summary", {})
                    comment_count = cached_data.get("comment_count", 0)
                    if comment_count < 10:
                        return YoutubeResponse(
                            type="comment_summary",
                            response="댓글 수가 10개 미만으로 댓글 요약을 제공하지 않습니다.",
                            sessionId=f"youtube_{req.userId}_{req.historyId}",  # sessionId 추가
                            videoUrl=chatbot_result.get("youtube_url"),
                            youtubeSummary=youtube_summary,
                            commentCount=comment_count
                        )
                except Exception:
                    pass
            
            # 댓글 요약 실행
            from app.graph_workflow import rerun_youtube_agent_node
            result = rerun_youtube_agent_node(input_data)
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            # 결과를 Redis에 저장
            redis_client.set(session_key, json.dumps(result), ex=60*60*24)
            
            # 결과에서 필요한 정보 추출
            chatbot_result = result.get("chatbot_result", {})
            youtube_summary = result.get("youtube_summary", {})
            comment_count = result.get("comment_count", 0)
            
            return YoutubeResponse(
                type="comment_summary",
                response="댓글 요약 완료",
                sessionId=f"youtube_{req.userId}_{req.historyId}",  # sessionId 추가
                videoUrl=chatbot_result.get("youtube_url"),
                youtubeSummary=youtube_summary,
                commentCount=comment_count
            )
            
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 요청 타입입니다.")
            
    except Exception as e:
        print(f"[ERROR] YouTube 엔드포인트 에러: {str(e)}")
        import traceback
        print(f"[ERROR] 상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))