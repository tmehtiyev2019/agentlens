"""
Intent classifier — combined safety + input-type classification in one LLM call.

Returns (is_safe, reason, input_type) where input_type distinguishes between:
  "idea"  → a moonshot/product concept to run through the full TEA pipeline
  "chat"  → a greeting, question, or anything that is NOT an idea to evaluate
"""

import os
from typing import Literal

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are a dual-purpose classifier for AgentLens — a techno-economic analysis system.

Classify the user input along TWO dimensions simultaneously:

DIMENSION 1 — Safety:
  "safe"       → clear legitimate uses, no meaningful harm potential
  "borderline" → dual-use or sensitive framing, but plausible commercial/research use
  "harmful"    → explicitly and unambiguously designed to cause harm

DIMENSION 2 — Input type:
  "idea" → any description of a product, technology, business, or research concept to
            evaluate — even a rough or vague 1-sentence pitch counts as an idea
  "chat" → a greeting, question, casual message, or anything that is NOT proposing an
            idea to evaluate (e.g. "hi", "how does this work?", "give me an example",
            "what is a moonshot?", "thanks", "hello")

Rules:
- Err strongly toward "safe" for safety.
- Any concept worth evaluating — however speculative or incomplete — is "idea" not "chat".
- Only classify as "chat" when there is genuinely NO idea being proposed.\
"""


class InputAnalysis(BaseModel):
    safety_category: Literal["safe", "borderline", "harmful"]
    input_type: Literal["idea", "chat"]
    safety_reason: str
    specific_concern: str | None = None


def classify_intent(idea: str) -> tuple[bool, str, Literal["idea", "chat"]]:
    """
    Returns (is_safe, reason, input_type).

    is_safe=True    → proceed through pipeline
    is_safe=False   → pipeline stops; reason explains why
    input_type      → "idea" runs full TEA pipeline; "chat" routes to chat_response_node
    """
    try:
        llm = ChatOpenAI(
            model=os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        result: InputAnalysis = llm.with_structured_output(InputAnalysis).invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=idea)]
        )

        log = logger.bind(
            safety=result.safety_category,
            input_type=result.input_type,
            reason=result.safety_reason,
        )

        if result.safety_category == "harmful":
            log.warning("intent_classifier.blocked", specific_concern=result.specific_concern)
            return False, result.safety_reason, "idea"

        if result.safety_category == "borderline":
            log.info("intent_classifier.borderline", specific_concern=result.specific_concern)
            return True, f"borderline: {result.safety_reason}", result.input_type

        log.info("intent_classifier.safe")
        return True, result.safety_reason, result.input_type

    except Exception as exc:
        logger.error("intent_classifier.unavailable", error=str(exc), exc_info=True)
        # Fail open — broken classifier must not block legitimate pipeline runs.
        return True, "classifier_unavailable", "idea"
