"""
main.py
-------
FastAPI application entry point for the Healthcare Symptom Checker.

Responsibilities:
- Instantiate the FastAPI app with metadata.
- Load environment variables from a .env file (if present) via python-dotenv.
- Verify MongoDB connectivity and indexes during application startup.
- Register CORS middleware.
- Include the API router.
- Mount the frontend directory as static files so index.html is served at /.
"""

from contextlib import asynccontextmanager
import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

# Ensure project-root imports work when this file is executed directly

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


from backend.database import close_database, ensure_indexes, ping_database
from backend.router import router                                           


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Try MongoDB startup work, but let the API boot even if it is down."""
    try:
        await ping_database()
        await ensure_indexes()
        logger.info("MongoDB connection verified and indexes ensured.")
    except RuntimeError as exc:
        logger.warning(
            "MongoDB unavailable during startup. Continuing in degraded mode: %s",
            exc,
        )
    try:
        yield
    finally:
        await close_database()
        logger.info("MongoDB client closed.")

app = FastAPI(
    lifespan=lifespan,
    title="Healthcare Symptom Checker",
    description=(
        "An educational tool that analyses user-reported symptoms using an AI "
        "language model and returns probable conditions with recommended next steps. "
        "NOT a substitute for professional medical advice."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend",
)

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    logger.info("Frontend mounted from: %s", _FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found at: %s — UI will not be served.", _FRONTEND_DIR)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
