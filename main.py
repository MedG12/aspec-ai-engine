import logging

# Konfigurasi Logging Dasar ke level INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from api.routes import predict, chat, insert
from services.background import start_scheduler, stop_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    start_scheduler()
    yield
    # Shutdown events
    stop_scheduler()

app = FastAPI(
    title="ASPEC AI Engine API",
    description="Microservice for Asset Predictive Maintenance (Machine Learning & NLP)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup for Next.js frontend
origins = [
    "http://localhost:3000",
    # Add production frontend URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["Predict"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(insert.router, prefix="/api", tags=["Vector DB"])

@app.get("/")
def read_root():
    return {"message": "Welcome to ASPEC AI Engine API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
