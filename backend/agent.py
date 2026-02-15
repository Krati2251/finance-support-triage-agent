"""
Ticket analysis using LangChain + Groq (Llama 3.1 70B).

OPTIMISATION:  Analysis + draft reply are produced in a **single** LLM call
so each email costs only 1 API request (Groq free tier = 30 req/min).
"""

import os
import hashlib
from functools import lru_cache
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from schemas import TicketAnalysis, TicketAnalysisWithDraft

# ── Load env ──
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file: GROQ_API_KEY=gsk_..."
    )

# ────────────────────── LLM Setup ──────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,            # deterministic for classification
    max_tokens=2048,          # enough for analysis + full draft
    request_timeout=60,
)

# Structured output — forces the LLM to return valid JSON matching
# our Pydantic schema (analysis + draft combined).
structured_llm = llm.with_structured_output(TicketAnalysisWithDraft)

# Also keep an analysis-only structured LLM for the /analyze endpoint
structured_llm_analysis_only = llm.with_structured_output(TicketAnalysis)

# ────────────────────── Combined Prompt ──────────────────────
# One prompt that asks for BOTH analysis AND draft reply in a single call.

COMBINED_SYSTEM_PROMPT = """\
You are a senior financial support triage agent AND a professional customer \
support writer. You will analyse an incoming customer email and produce:

A) STRUCTURED TRIAGE ANALYSIS
B) A PERSONALISED DRAFT REPLY

═══ PART A — ANALYSIS RULES ═══

SENTIMENT — Exactly one of: Positive, Negative, Neutral, Urgent
  • Positive — satisfied or sending thanks
  • Negative — frustrated, angry, or dissatisfied
  • Neutral  — informational or routine enquiry
  • Urgent   — immediate danger, fraud, or threats

INTENT — Short phrase (5-10 words) describing what the customer wants.

ENTITIES — Extract (set null if not found):
  • customer_name  — sender's full name
  • transaction_id — any transaction/reference ID (e.g. TXN-12345)
  • amount         — monetary amount (e.g. "$500.00")

PRIORITY:
  • High   — fraud, theft, unauthorised transactions, account compromise
  • Medium — billing errors, payment failures, refund requests, disputes
  • Low    — general enquiries, account info requests, feedback

CATEGORY:
  • Fraud         — fraud, theft, unauthorised access, suspicious activity
  • Payment Issue — billing errors, refund requests, payment failures, disputes
  • General       — all other enquiries

SUMMARY — Concise 1-2 sentence summary for the support agent.

═══ PART B — DRAFT REPLY RULES ═══

Write a professional, empathetic plain-text email reply (80-150 words).

1. GREETING — "Dear <customer_name>," (or "Dear Valued Customer," if unknown).

2. TONE BY CATEGORY:
   • Fraud:
     – Express urgent concern and empathy
     – Assure account security is top priority
     – State fraud/security team is investigating
     – Advise "We have temporarily secured your account"
     – Provide hotline: 1-800-FRAUD-HELP
   • Payment Issue:
     – Acknowledge inconvenience with empathy
     – Payment/billing team is reviewing
     – Reference number: [REF-XXXXXX]
     – Resolution: 2-3 business days
   • General:
     – Polite, warm, professional
     – Helpful and informative
     – Offer further assistance

3. CLOSING — End with:
   "Best regards,
   Finance Support Team
   finance-support@company.com"

4. Do NOT use markdown. Plain text only.

Return ALL fields in a single JSON response.
"""

combined_prompt = ChatPromptTemplate.from_messages([
    ("system", COMBINED_SYSTEM_PROMPT),
    ("human", "Analyse and draft a reply for the following customer email:\n\n{email_body}"),
])

# Analysis-only prompt (for the /analyze endpoint that doesn't need a draft)
ANALYSIS_ONLY_SYSTEM_PROMPT = """\
You are a senior financial support triage agent. Analyse the incoming \
customer email and return a structured JSON report.

SENTIMENT — Exactly one of: Positive, Negative, Neutral, Urgent
INTENT — Short phrase (5-10 words) describing what the customer wants.
ENTITIES — Extract customer_name, transaction_id, amount (null if not found).
PRIORITY — High (fraud/theft), Medium (billing/payment), Low (general).
CATEGORY — Fraud, Payment Issue, or General.
SUMMARY — 1-2 sentence summary.
"""

analysis_only_prompt = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_ONLY_SYSTEM_PROMPT),
    ("human", "Analyse the following customer email:\n\n{email_body}"),
])

# ────────────────────── Chains ──────────────────────

combined_chain = combined_prompt | structured_llm
analysis_only_chain = analysis_only_prompt | structured_llm_analysis_only

# ────────────────────── In-memory cache ──────────────────────
# Prevents re-analysing the exact same email body within one server session.
_cache: dict[str, TicketAnalysisWithDraft] = {}


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ────────────────────── Public API ──────────────────────

def analyze_and_draft(email_body: str) -> TicketAnalysisWithDraft:
    """
    Analyse a customer email AND generate a draft reply in **one** LLM call.

    Returns:
        TicketAnalysisWithDraft — contains all analysis fields + draft_response.
    """
    if not email_body or not email_body.strip():
        raise ValueError("email_body cannot be empty.")

    clean = email_body.strip()
    key = _cache_key(clean)

    if key in _cache:
        return _cache[key]

    result: TicketAnalysisWithDraft = combined_chain.invoke({"email_body": clean})
    _cache[key] = result
    return result


def analyze_ticket(email_body: str) -> TicketAnalysis:
    """
    Analyse-only (no draft). Used by the /analyze endpoint.

    For endpoints that also need a draft, prefer analyze_and_draft() to
    save an API call.
    """
    if not email_body or not email_body.strip():
        raise ValueError("email_body cannot be empty.")

    clean = email_body.strip()
    key = _cache_key(clean)

    # If we already have a combined result cached, reuse the analysis part
    if key in _cache:
        cached = _cache[key]
        return TicketAnalysis(
            sentiment=cached.sentiment,
            intent=cached.intent,
            entities=cached.entities,
            priority=cached.priority,
            category=cached.category,
            summary=cached.summary,
        )

    return analysis_only_chain.invoke({"email_body": clean})


def generate_draft_response(analysis: TicketAnalysis) -> str:
    """
    Backward-compatible wrapper. If the analysis came from analyze_and_draft(),
    return its draft. Otherwise fall back to a simple template (no extra API call).
    """
    if isinstance(analysis, TicketAnalysisWithDraft):
        return analysis.draft_response

    # Fallback: generate a template-based draft without an API call
    customer = analysis.entities.customer_name or "Valued Customer"
    cat = analysis.category.value

    if cat == "Fraud":
        body = (
            f"Dear {customer},\n\n"
            "Thank you for alerting us to this matter. We take the security of your "
            "account very seriously. Our fraud investigation team has been immediately "
            "notified and is actively reviewing the suspicious activity you reported.\n\n"
            "As a precaution, we have temporarily secured your account to prevent any "
            "further unauthorized transactions. Please do not hesitate to contact our "
            "dedicated fraud hotline at 1-800-FRAUD-HELP for immediate assistance.\n\n"
            "We will keep you updated on the progress of our investigation.\n\n"
            "Best regards,\nFinance Support Team\nfinance-support@company.com"
        )
    elif cat == "Payment Issue":
        body = (
            f"Dear {customer},\n\n"
            "Thank you for reaching out regarding your payment concern. We sincerely "
            "apologize for the inconvenience this has caused.\n\n"
            "Our payment and billing team is currently reviewing your case. Your "
            "reference number is [REF-XXXXXX]. You can expect a resolution within "
            "2-3 business days.\n\n"
            "If you have any further questions, please don't hesitate to reach out.\n\n"
            "Best regards,\nFinance Support Team\nfinance-support@company.com"
        )
    else:
        body = (
            f"Dear {customer},\n\n"
            "Thank you for contacting us. We have received your enquiry and our team "
            "is looking into it.\n\n"
            "We will get back to you shortly with a detailed response. If you need "
            "immediate assistance, please feel free to reach out again.\n\n"
            "Best regards,\nFinance Support Team\nfinance-support@company.com"
        )
    return body


# ────────────────────── Quick test ──────────────────────

if __name__ == "__main__":
    sample_email = (
        "Hi, my name is Rajesh Kumar. I noticed an unauthorized "
        "transaction of $500 on my account. The transaction ID is TXN-98432. "
        "I did not make this purchase and I'm very worried someone has access "
        "to my account. Please investigate this immediately and freeze my card."
    )

    print("🔍 Analysing sample email (single combined call)...\n")
    result = analyze_and_draft(sample_email)

    print("=" * 60)
    print("          TICKET ANALYSIS + DRAFT RESULT")
    print("=" * 60)
    print(f"  Sentiment   : {result.sentiment.value}")
    print(f"  Intent      : {result.intent}")
    print(f"  Priority    : {result.priority.value}")
    print(f"  Category    : {result.category.value}")
    print(f"  Summary     : {result.summary}")
    print()
    print("  Entities:")
    print(f"    Name      : {result.entities.customer_name or 'N/A'}")
    print(f"    Txn ID    : {result.entities.transaction_id or 'N/A'}")
    print(f"    Amount    : {result.entities.amount or 'N/A'}")
    print()
    print("  Draft Response:")
    print(f"    {result.draft_response[:200]}...")
    print("=" * 60)
