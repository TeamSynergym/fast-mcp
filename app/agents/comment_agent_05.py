# %%
import os
import json
from typing import TypedDict, List
from pydantic import BaseModel, Field
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
import googleapiclient.discovery

class Sentiment(BaseModel):
    """전반적인 감성 분석 결과"""
    description: str = Field(description="댓글의 전반적인 분위기와 여론을 1~2문장으로 요약")
    positive_percentage: int = Field(description="긍정적인 반응의 비율 (0-100 사이의 정수)")

class CommentSummary(BaseModel):
    """
    유튜브 댓글 요약을 위한 최종 데이터 구조.
    LLM은 이 구조에 맞춰서만 답변해야 합니다.
    """
    overall_sentiment: Sentiment = Field(description="댓글의 전반적인 감성 분석 결과")
    key_topics: List[str] = Field(description="댓글에서 가장 자주 언급되는 핵심 주제 목록 (이모지 포함)")
    user_tips: List[str] = Field(description="사용자들이 공유하는 유용한 팁 목록")
    faq: List[str] = Field(description="사용자들이 자주 묻는 질문(FAQ) 목록")

# %%
# 상태(state) 정의: 총 댓글 수 필드 추가
class CommentState(TypedDict):
    url: str
    video_id: str
    total_comment_count: int
    comments: List[str]
    comment_summary: CommentSummary
    error: str

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
structured_llm = llm.with_structured_output(CommentSummary)

api_key = os.getenv("YOUTUBE_API_KEY")
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
# %%
# video_id 추출 함수
def extract_video_id(url: str) -> str:
    """
    URL에서 ID를 추출
    """
    if not isinstance(url, str):
        return None

    video_id = url.split('watch?v=')[-1]
    print(f"  -> ID 추출 성공! (결과: {video_id})")
    return video_id

# %%
# 노드 1: 영상 ID 추출 및 총 댓글 수 확인
def get_video_stats(state: CommentState) -> dict:
    """URL에서 영상 ID를 추출하고, 영상의 총 댓글 수를 확인합니다."""
    print("➡️ [1] 영상 정보 확인 시작...")
    url = state.get("url")
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("유효한 유튜브 URL에서 Video ID를 추출할 수 없습니다.")
        print(f"  - 영상 ID 추출 성공: {video_id}")

        # videos().list API를 호출하여 통계 정보(댓글 수) 가져오기
        request = youtube.videos().list(part="statistics", id=video_id)
        response = request.execute()

        if not response.get("items"):
            raise ValueError("API로부터 영상 정보를 가져올 수 없습니다.")

        stats = response["items"][0].get("statistics", {})
        total_comment_count = int(stats.get("commentCount", 0))
        print(f"  - ✅ 총 댓글 수 확인: {total_comment_count}개")

        return {"video_id": video_id, "total_comment_count": total_comment_count}

    except Exception as e:
        error_message = f"ERROR: 영상 정보 확인 중 오류 발생 - {e}"
        print(f"  - 🚨 {error_message}")
        return {"error": error_message}

# %%
# 조건부 라우터: 댓글 수에 따라 분기
def check_comment_count(state: CommentState) -> str:
    """총 댓글 수에 따라 다음 단계를 결정합니다."""
    print("➡️ [2] 댓글 수 확인 및 분기...")
    if state.get("error"):
        print("  - 🚨 오류가 감지되어 프로세스를 종료합니다.")
        return "end" # 오류 발생 시 종료

    total_comment_count = state.get("total_comment_count", 0)
    if total_comment_count < 10:
        print(f"  - 💬 댓글 수가 {total_comment_count}개로 10개 미만입니다. 요약을 제공하지 않습니다.")
        return "no_summary" # 10개 미만이면 요약 안 함
    else:
        print(f"  - ✅ 댓글 수가 {total_comment_count}개입니다. 댓글 내용 수집을 시작합니다.")
        return "fetch_comments" # 10개 이상이면 댓글 수집

# %%
# 노드 2: 댓글 내용 수집
def fetch_comments(state: CommentState) -> dict:
    """commentThreads API를 사용해 실제 댓글 내용을 가져옵니다."""
    print("➡️ [3] 댓글 내용 수집 중...")
    video_id = state.get("video_id")
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=100, order="relevance"
        )
        response = request.execute()
        comments = [
            item['snippet']['topLevelComment']['snippet']['textDisplay']
            for item in response['items']
        ]
        if not comments:
            raise ValueError("댓글이 없습니다.") # 댓글 수는 10개 이상인데 가져온 내용이 없는 경우
        print(f"  - ✅ 댓글 {len(comments)}개 수집 성공")
        return {"comments": comments}
    except Exception as e:
        error_message = f"ERROR: 댓글 내용 수집 중 오류 발생 - {e}"
        print(f"  - 🚨 {error_message}")
        return {"error": error_message}

# %%
# 노드 3: 요약 불가 처리
def handle_no_summary(state: CommentState) -> dict:
    """댓글 수가 적어 요약을 제공하지 않음을 처리합니다."""
    summary_message = "댓글 개수가 10개 미만으로 댓글 요약을 제공하지 않습니다."
    return {"comment_summary": summary_message}

# %%
# 댓글 요약 생성
def summarize_comments_node(state: CommentState) -> dict:
    """
    댓글을 분석하여 Pydantic 모델 형식의 구조화된 요약을 생성합니다.
    """
    print("➡️ [Comment Agent] 댓글 요약 생성 시작...")
    comments_text = state.get("comments")

    prompt = f"""
    당신은 유튜브 영상의 댓글들을 분석하여 유용한 정보를 추출하고 구조화된 JSON 형식으로 요약하는 전문가입니다.
    아래 댓글 모음을 바탕으로, 요청된 JSON 스키마에 맞춰 각 항목을 채워주세요.

    [분석할 댓글 모음]
    ---
    {comments_text}
    ---
    """
    try:
        # LLM을 호출하면, LangChain이 자동으로 Pydantic 모델 객체를 반환
        summary_result: CommentSummary = structured_llm.invoke([
            SystemMessage(content="You are an expert at summarizing YouTube comments into a structured JSON format."),
            HumanMessage(content=prompt)
        ])
        print("✅ 댓글 요약 생성 및 구조화 성공!")
        return {"comment_summary": summary_result.dict()}

    except Exception as e:
        error_message = f"ERROR: 댓글 요약 생성 중 오류 발생 - {e}"
        print(f"🚨 {error_message}")
        return {"error": error_message}

builder = StateGraph(CommentState)

# 노드
builder.add_node("get_video_stats", get_video_stats)
builder.add_node("fetch_comments", fetch_comments)
builder.add_node("handle_no_summary", handle_no_summary)
builder.add_node("summarize_comments", summarize_comments_node)

# 엣지(연결)
builder.set_entry_point("get_video_stats")

builder.add_conditional_edges(
    "get_video_stats",
    check_comment_count,
    {
        "fetch_comments": "fetch_comments", # 댓글 수 많으면 -> 내용 수집
        "no_summary": "handle_no_summary",   # 댓글 수 적으면 -> 요약 불가 처리
        "end": END                         # 에러 나면 -> 종료
    }
)

builder.add_edge("fetch_comments", "summarize_comments")
builder.add_edge("handle_no_summary", END)
builder.add_edge("summarize_comments", END)

graph = builder.compile()


# %%
# 결과 출력 함수
def run_agent(url: str):
    """
    에이전트를 실행하고 최종 결과를 출력합니다.
    - 초기 입력을 'url' 필드에 담아 전달합니다.
    - 최종 결과는 'comment_summary' 필드에서 가져옵니다.
    - 'error' 필드를 확인하여 오류를 처리합니다.
    """
    inputs = {"url": url}
    final_state = graph.invoke(inputs)

    if final_state.get("error"):
        print("\n" + "="*30)
        print("❌ 최종 실행 중 오류 발생:")
        print("="*30)
        print(final_state["error"])
        return

    content = final_state.get("comment_summary", "")

    print("\n" + "="*30)
    print("✅ 최종 요약 (JSON 형식):")
    print("="*30)

    if not content or not content.strip():
        print("⚠️ 요약 결과가 비어 있습니다. content 값:", repr(content))
        return

    # "댓글 개수가 10개 미만..." 메시지는 JSON 파싱 없이 바로 출력
    if "댓글 개수가 10개 미만" in content:
        print(content)
        return

    # LLM 응답에서 마크다운 코드 블록 제거
    if content.strip().startswith("```json"):
        start_index = content.find('{')
        end_index = content.rfind('}')
        if start_index != -1 and end_index != -1:
            content = content[start_index : end_index + 1]

    try:
        parsed_json = json.loads(content)
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("⚠️ JSON 파싱 실패. 원본 content 출력:")
        print(content)