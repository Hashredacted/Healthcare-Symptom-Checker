from datetime import datetime, timezone
from typing import Any


def build_query_history_document(
    user_input: str,
    llm_response: str,
    risk_level: str,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp or datetime.now(timezone.utc),
        "user_input": user_input,
        "llm_response": llm_response,
        "risk_level": risk_level,
    }


def serialize_query_history(document: dict[str, Any]) -> dict[str, str]:
    timestamp = document.get("timestamp")
    if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    timestamp_text = timestamp.isoformat() if isinstance(timestamp, datetime) else ""

    return {
        "id": str(document.get("_id", "")),
        "timestamp": timestamp_text,
        "user_input": str(document.get("user_input", "")),
        "llm_response": str(document.get("llm_response", "")),
        "risk_level": str(document.get("risk_level", "")),
    }
