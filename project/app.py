"""
FastAPI application entry point.

Exposes:
  GET  /health  — readiness probe
  POST /chat    — stateless conversational assessment recommender

Production configuration:
  - CORS enabled for all origins (configure as needed for production)
  - Structured JSON logging
  - Exception handlers for 422 (validation) and 500 (internal)
  - Startup event builds FAISS index if not present
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

# Load .env file before any project imports that might read env vars
load_dotenv()

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.chat_logic import process_chat
from models.schemas import ChatRequest, ChatResponse, HealthResponse
from utils.helpers import (
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    get_logger,
    load_catalog,
)

_log = get_logger("app")


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    On startup: pre-load catalog and build FAISS index if missing.
    """
    _log.info("=" * 60)
    _log.info("SHL Assessment Recommender — starting up")
    _log.info("=" * 60)

    # Pre-load catalog to warm the cache
    try:
        catalog = load_catalog()
        _log.info("Catalog pre-loaded: %d assessments.", len(catalog))
    except Exception as e:
        _log.error("Failed to load catalog: %s", e)

    # Build FAISS index if not present
    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        _log.info("FAISS index not found — building now...")
        try:
            from embeddings.build_index import build_index
            build_index()
        except Exception as e:
            _log.warning(
                "Could not build FAISS index: %s. "
                "Keyword-only search will be used as fallback.",
                e,
            )
    else:
        _log.info("FAISS index found at %s.", FAISS_INDEX_PATH)
        # Warm the retriever
        try:
            from agent.retriever import _load_index
            _load_index()
        except Exception as e:
            _log.warning("Retriever warm-up failed: %s", e)

    _log.info("Startup complete. Ready to serve requests.")
    yield
    _log.info("Shutting down SHL Assessment Recommender.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL assessments from the "
        "official SHL product catalog through multi-turn dialogue."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for evaluator compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc: Exception):
    """Return structured error for schema validation failures."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": str(exc),
        },
    )


@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: Exception):
    """Return structured error for unexpected server failures."""
    _log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )


# ---------------------------------------------------------------------------
# Middleware: request logging + timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each request with method, path, status, and elapsed time."""
    t_start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t_start) * 1000
    _log.info(
        "%s %s → %d (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Readiness probe",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """
    Readiness probe.

    Returns HTTP 200 with `{"status": "ok"}` when the service is ready.
    The evaluator allows up to 2 minutes for cold-start wake-up.
    """
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational assessment recommender",
    tags=["Chat"],
    responses={
        200: {
            "description": "Agent reply with optional assessment recommendations",
            "content": {
                "application/json": {
                    "example": {
                        "reply": "Here are 5 assessments for a mid-level Java developer.",
                        "recommendations": [
                            {
                                "name": "Core Java (Advanced Level) (New)",
                                "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                                "test_type": "K",
                            }
                        ],
                        "end_of_conversation": False,
                    }
                }
            },
        },
        422: {"description": "Request schema validation error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.

    Every call must include the **full conversation history** in `messages`.
    The service reconstructs all context from history on each call.

    - `recommendations` is `[]` when clarifying or refusing.
    - `recommendations` contains 1–10 items when recommending.
    - `end_of_conversation` is `true` when the agent considers the task complete.
    """
    try:
        return process_chat(request)
    except Exception as e:
        _log.error("chat() unhandled exception: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
