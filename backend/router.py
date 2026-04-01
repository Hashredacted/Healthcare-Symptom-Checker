import logging
from typing import Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from pymongo import DESCENDING

from backend.database import get_database_status, get_query_history_collection
from backend.models import build_query_history_document, serialize_query_history
from backend.safety import classify_risk
from backend.llm import get_symptom_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Symptom Checker"])

DISCLAIMER = (
    "This information is for educational purposes only and is NOT a substitute "
    "for professional medical advice, diagnosis, or treatment. Always seek the "
    "advice of a qualified healthcare provider with any questions you may have "
    "regarding a medical condition."
)


class SymptomRequest(BaseModel):
    symptoms: str

    @field_validator("symptoms")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Symptom text must not be empty.")
        if len(stripped) < 3:
            raise ValueError("Please describe your symptoms in more detail.")
        return stripped


class AnalysisResponse(BaseModel):
    level: str
    response: str
    disclaimer: str
    timestamp: str


class HistoryRecord(BaseModel):
    id: str
    timestamp: str
    user_input: str
    llm_response: str
    risk_level: str


@router.get("/health", summary="Health check")
async def health_check():
    database_status = get_database_status()
    return {
        "status": "ok",
        "database": "available" if database_status["available"] else "unavailable",
        "database_message": database_status["message"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post(
    "/check-symptoms",
    response_model=AnalysisResponse,
    summary="Analyse user symptoms",
    description=(
        "Accepts a symptom description, classifies risk level, optionally calls "
        "the Gemini LLM, persists the interaction, and returns an educational analysis."
    ),
)
async def check_symptoms(
    request: SymptomRequest,
    collection: Any = Depends(get_query_history_collection),
) -> AnalysisResponse:
    symptoms = request.symptoms
    logger.info("Received symptom query (length=%d)", len(symptoms))

    risk = classify_risk(symptoms)
    level = risk["level"]

    if level == "EMERGENCY":
        logger.warning("EMERGENCY keyword detected. Bypassing LLM.")
        final_response = risk["message"]
    else:
        try:
            llm_text = await get_symptom_analysis(symptoms, risk_level=level)
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    "Unable to reach the analysis service. Please check your "
                    "GEMINI_API_KEY and try again."
                ),
            ) from exc

        if level == "URGENT":
            final_response = risk["message"] + llm_text
        else:
            final_response = llm_text

    record = build_query_history_document(
        user_input=symptoms,
        llm_response=final_response,
        risk_level=level,
    )
    try:
        result = await collection.insert_one(record)
    except Exception as exc:
        logger.exception("Database write failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to save your request history right now.",
        ) from exc

    logger.info("Saved query_history id=%s (level=%s)", result.inserted_id, level)

    return AnalysisResponse(
        level=level,
        response=final_response,
        disclaimer=DISCLAIMER,
        timestamp=record["timestamp"].isoformat(),
    )


@router.get(
    "/history",
    response_model=list[HistoryRecord],
    summary="Get recent query history",
    description="Returns the 20 most recent symptom-check interactions.",
)
async def get_history(
    collection: Any = Depends(get_query_history_collection),
) -> list[HistoryRecord]:
    try:
        cursor = collection.find({}).sort("timestamp", DESCENDING).limit(20)
        return [
            HistoryRecord(**serialize_query_history(document))
            async for document in cursor
        ]
    except Exception as exc:
        logger.exception("Database read failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to load query history right now.",
        ) from exc
