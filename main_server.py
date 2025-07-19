from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

from app.goal_setting_router import router as goal_setting_router
from app.recommendation_router import router as recommendation_router

load_dotenv()

app = FastAPI(
    title="SynergyM 통합 AI 서비스",
    version="1.0.0"
)

app.include_router(goal_setting_router)
app.include_router(recommendation_router)

@app.get("/", summary="Health Check")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)