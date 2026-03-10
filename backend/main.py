"""
FastAPI Backend — News Crew Summarizer

Endpoints:
  GET  /           → health check
  GET  /health     → detailed health check with env validation
  POST /summarize  → run the CrewAI RAG pipeline
"""

import os
import time
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure backend directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-warm the embedding model to avoid first-request lag."""
    print("🚀 News Crew API starting up...")
    try:
        from rag_chain import get_embeddings
        get_embeddings()
        print("✅ Embedding model loaded.")
    except Exception as e:
        print(f"⚠️  Could not pre-warm embeddings: {e}")
    yield
    print("🛑 News Crew API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="News Crew Summarizer API",
    description=(
        "Production-ready AI news summarization using CrewAI + ChromaDB RAG + Gemini"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="News topic to search and summarize",
        examples=["Latest AI breakthroughs 2024"],
    )


class SummarizeResponse(BaseModel):
    status: str
    query: str
    summary: str
    duration_seconds: float
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    tavily_key_set: bool
    gemini_key_set: bool
    chroma_dir_exists: bool
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"message": "News Crew Summarizer API is running 🚀", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    tavily = bool(os.getenv("TAVILY_API_KEY"))
    gemini = bool(os.getenv("GEMINI_API_KEY"))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_dir = os.path.join(base_dir, "data", "chromadb")

    all_good = tavily and gemini
    return HealthResponse(
        status="healthy" if all_good else "degraded",
        tavily_key_set=tavily,
        gemini_key_set=gemini,
        chroma_dir_exists=os.path.exists(chroma_dir),
        message=(
            "All systems operational ✅"
            if all_good
            else "⚠️ Missing API keys — check your .env file"
        ),
    )


@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarize"])
async def summarize_news(request: SummarizeRequest):
    """
    Runs the full CrewAI RAG pipeline:
      1. Agent 1 searches Tavily and stores results in ChromaDB
      2. Agent 2 retrieves from ChromaDB and generates a Gemini summary

    This endpoint may take 30–90 seconds depending on news volume.
    """
    # Validate env keys early for better error messages
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "TAVILY_API_KEY not configured. "
                "Add it to your .env file. Free tier: https://tavily.com"
            ),
        )
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY not configured. "
                "Add it to your .env file. Free tier: https://aistudio.google.com"
            ),
        )

    start = time.time()

    try:
        from agent import run_news_crew
        result = run_news_crew(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {e}")

    duration = round(time.time() - start, 2)

    if result["status"] == "error":
        return SummarizeResponse(
            status="error",
            query=request.query,
            summary="",
            duration_seconds=duration,
            error=result["error"],
        )

    return SummarizeResponse(
        status="success",
        query=request.query,
        summary=result["summary"],
        duration_seconds=duration,
        error=None,
    )
