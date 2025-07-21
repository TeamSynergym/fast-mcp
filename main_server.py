from dotenv import load_dotenv
load_dotenv()
import os

from fastapi import FastAPI
import uvicorn

from app.analyze_router import router as analyze_router
from app.ai_coach_router import router as ai_coach_router
from app.youtube_router import router as youtube_router
from app.goal_setting_router import router as goal_setting_router
from app.recommendation_router import router as recommendation_router



app = FastAPI(
    title="SynergyM 통합 AI 서비스",
    version="1.0.0"
)

app.include_router(analyze_router)
app.include_router(ai_coach_router)
app.include_router(youtube_router)
app.include_router(goal_setting_router)
app.include_router(recommendation_router)

@app.get("/", summary="Health Check")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)