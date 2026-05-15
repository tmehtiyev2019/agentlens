import os
from typing import Literal

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a safety classifier for an AI research and product evaluation pipeline.

Your job is to classify whether an input idea is safe to process through an automated multi-agent analysis system.

Classification categories:
- "safe": The idea has clear legitimate uses and no meaningful harm potential.
- "borderline": The idea involves dual-use technology, sensitive industries, or edgy framing, but has plausible legitimate commercial or research applications. Err toward this over "harmful".
- "harmful": The idea is explicitly and unambiguously designed to cause harm with no plausible legitimate framing.

Examples of HARMFUL (block):
- Designing biological agents optimized for human lethality
- Building a system to stalk or surveil a specific private individual without consent
- Creating content explicitly intended to incite violence against an identifiable group
- Step-by-step synthesis routes for chemical or radiological weapons

Examples of BORDERLINE (allow with warning):
- Autonomous weapon systems for national defense
- AI-assisted surveillance for law enforcement with legal oversight
- Dual-use biotech with both therapeutic and weaponizable potential
- Offensive cybersecurity tools positioned as red-team products

Examples of SAFE (allow):
- Nuclear energy startups
- Drone delivery logistics
- AI-powered threat detection
- Defense contracting software
- Biosecurity research platforms

Critical rule: err strongly on the side of allowing. Only classify as "harmful" when the idea is explicitly and unambiguously framed to cause harm. Legitimate commercial, research, or national-security framing — even in sensitive domains — should be classified as "safe" or "borderline", not "harmful"."""


class IntentClassification(BaseModel):
    category: Literal["safe", "borderline", "harmful"]
    reason: str
    specific_concern: str | None = None


def classify_intent(idea: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    is_safe=True  → idea can proceed through pipeline
    is_safe=False → pipeline stops; reason explains why
    """
    try:
        llm = ChatAnthropic(
            model=os.getenv("CLASSIFIER_MODEL", "claude-haiku-4-5-20251001"),
            temperature=0,
        )
        structured = llm.with_structured_output(IntentClassification)
        result: IntentClassification = structured.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=idea)]
        )

        log = logger.bind(category=result.category, reason=result.reason)

        if result.category == "harmful":
            log.warning(
                "intent_classifier.blocked",
                specific_concern=result.specific_concern,
            )
            return False, result.reason

        if result.category == "borderline":
            log.info(
                "intent_classifier.borderline",
                specific_concern=result.specific_concern,
            )
            return True, f"borderline: {result.reason}"

        log.info("intent_classifier.safe")
        return True, result.reason

    except Exception as exc:
        logger.error(
            "intent_classifier.unavailable",
            error=str(exc),
            exc_info=True,
        )
        # Fail open — a broken classifier must not block legitimate pipeline runs.
        return True, "classifier_unavailable"
