import re
from typing import TypedDict


class RiskResult(TypedDict):
    level: str
    message: str | None


EMERGENCY_KEYWORDS: list[str] = [
    "heart attack", "cardiac arrest", "chest pain", "chest tightness",
    "chest pressure", "heart racing", "palpitations and fainted",
    "can't breathe", "cannot breathe", "not breathing", "stopped breathing",
    "difficulty breathing", "choking", "shortness of breath and chest pain",
    "stroke", "face drooping", "arm weakness and speech", "sudden numbness",
    "sudden severe headache", "worst headache of my life", "seizure", "unconscious",
    "unresponsive", "passed out", "blacked out",
    "bleeding profusely", "bleeding heavily", "uncontrolled bleeding",
    "severe bleeding", "blood everywhere", "deep wound",
    "suicide", "suicidal", "kill myself", "end my life",
    "want to die", "self harm", "self-harm", "overdose", "took too many pills",
    "poisoning", "poisoned", "anaphylaxis", "allergic reaction and throat",
    "severe allergic", "throat closing", "can't swallow",
    "diabetic coma", "loss of consciousness",
]

URGENT_COMBOS: list[list[str]] = [
    ["sudden", "severe", "pain"],
    ["sudden", "weakness"],
    ["sudden", "confusion"],
    ["severe", "chest"],
    ["severe", "abdominal", "pain"],
    ["high", "fever", "stiff", "neck"],
    ["vomiting", "blood"],
    ["coughing", "blood"],
    ["can't", "move"],
    ["cannot", "move"],
    ["extreme", "pain"],
]

EMERGENCY_RESPONSE = (
    "🚨 **EMERGENCY DETECTED — Please call 911 (or your local emergency number) immediately.** 🚨\n\n"
    "Based on the symptoms you described, you may be experiencing a medical emergency. "
    "**Do NOT wait or rely on this tool.** Please:\n\n"
    "1. **Call 911** (USA) / **999** (UK) / **112** (EU) or your local emergency services NOW.\n"
    "2. If you're alone, unlock your front door so responders can enter.\n"
    "3. Stay on the line with the dispatcher until help arrives.\n\n"
    "---\n"
    "⚠️ *This tool is for educational purposes only and is NOT a substitute for "
    "professional medical advice or emergency services.*"
)

URGENT_BANNER = (
    "⚠️ **URGENT CARE RECOMMENDED** ⚠️\n\n"
    "Your symptoms suggest you may need prompt medical attention. "
    "Please consider visiting an urgent care clinic or emergency room soon, "
    "or call your doctor immediately.\n\n"
    "---\n\n"
)


def classify_risk(text: str) -> RiskResult:
    normalised = text.lower().strip()

    for keyword in EMERGENCY_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, normalised):
            return RiskResult(level="EMERGENCY", message=EMERGENCY_RESPONSE)

    for combo in URGENT_COMBOS:
        if all(word in normalised for word in combo):
            return RiskResult(level="URGENT", message=URGENT_BANNER)

    return RiskResult(level="NORMAL", message=None)
