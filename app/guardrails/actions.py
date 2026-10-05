import re
from typing import Optional

from nemoguardrails.actions import action


# Regex patterns for PII / secrets. Checked against the raw user message before
# intent matching — embeddings can't tell a card number from any other number.
PII_PATTERNS = {
    "email":   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone":   r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "ssn":     r"\b\d{3}-\d{2}-\d{4}\b",
    "api_key": r"(api[_\s-]?key|token|secret)[:\s]+[A-Za-z0-9_\-]{10,}",
}

# Card numbers: 13–19 digits, optionally grouped by spaces/dashes, Luhn-validated
# to avoid flagging arbitrary long numbers (IDs, timestamps).
CARD_CANDIDATE = re.compile(r"\b\d(?:[\s-]?\d){12,18}\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def find_pii(text: str) -> list[str]:
    """Return the PII types found in text (empty list if clean)."""
    found = [ptype for ptype, pat in PII_PATTERNS.items()
             if re.search(pat, text, re.IGNORECASE)]

    for match in CARD_CANDIDATE.finditer(text):
        if _luhn_valid(re.sub(r"\D", "", match.group())):
            found.append("credit_card")
            break

    return found


@action(is_system_action=True)
async def detect_pii_in_input(context: Optional[dict] = None):
    """NeMo input-rail action: returns PII types found, or [] (falsy) if clean."""
    user_message = context.get("user_message", "") if context else ""
    return find_pii(user_message)
