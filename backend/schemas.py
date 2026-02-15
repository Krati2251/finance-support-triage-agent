"""
Pydantic schemas for the ticket analysis pipeline.

These schemas serve two purposes:
1. Structured output parsing — the LLM is forced to return data matching these shapes.
2. FastAPI request / response models — automatic validation & OpenAPI docs.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# -------------------- Enums --------------------

class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Category(str, Enum):
    FRAUD = "Fraud"
    PAYMENT_ISSUE = "Payment Issue"
    GENERAL = "General"


class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"
    URGENT = "Urgent"


# -------------------- Extracted Entities --------------------

class ExtractedEntities(BaseModel):
    """Key entities pulled from the email body."""
    customer_name: Optional[str] = Field(
        default=None,
        description="Full name of the customer, if mentioned in the email.",
    )
    transaction_id: Optional[str] = Field(
        default=None,
        description="Transaction or reference ID, if mentioned (e.g. TXN-12345).",
    )
    amount: Optional[str] = Field(
        default=None,
        description="Monetary amount mentioned in the email (e.g. '$500.00').",
    )


# -------------------- Full Analysis Result --------------------

class TicketAnalysis(BaseModel):
    """Complete analysis output returned by the triage agent."""
    sentiment: Sentiment = Field(
        description="Overall sentiment of the email.",
    )
    intent: str = Field(
        description=(
            "A short phrase describing what the customer wants "
            "(e.g. 'Report unauthorized transaction', 'Request refund')."
        ),
    )
    entities: ExtractedEntities = Field(
        description="Key entities extracted from the email body.",
    )
    priority: Priority = Field(
        description=(
            "Ticket priority: High for fraud/theft/unauthorized access, "
            "Medium for billing errors/payment failures, Low for general queries."
        ),
    )
    category: Category = Field(
        description=(
            "Ticket category: 'Fraud' for fraud/theft/unauthorized, "
            "'Payment Issue' for billing/refund/payment errors, "
            "'General' for everything else."
        ),
    )
    summary: str = Field(
        description="A concise 1–2 sentence summary of the email for the support agent.",
    )


class TicketAnalysisWithDraft(TicketAnalysis):
    """
    Combined analysis + draft in a single model.
    This allows a single LLM call to produce both the triage analysis
    and the draft reply — halving API usage.
    """
    draft_response: str = Field(
        description=(
            "A professional, empathetic plain-text email reply (80-150 words). "
            "Start with 'Dear <customer_name>,' and end with "
            "'Best regards,\\nFinance Support Team\\nfinance-support@company.com'. "
            "For Fraud: express urgent concern, mention fraud team is investigating, "
            "provide hotline 1-800-FRAUD-HELP. "
            "For Payment Issue: acknowledge inconvenience, mention 2-3 business days resolution, "
            "include reference [REF-XXXXXX]. "
            "For General: be warm and helpful. "
            "Do NOT use markdown formatting."
        ),
    )


# -------------------- API Request / Response Bodies --------------------

class AnalyzeRequest(BaseModel):
    """POST body for the /analyze and /process_ticket endpoints."""
    email_body: str = Field(
        ...,
        min_length=10,
        description="The raw email text to analyze.",
        json_schema_extra={
            "example": (
                "Hi, my name is Rajesh Kumar. I noticed an unauthorized "
                "transaction of $500 on my account. Transaction ID: TXN-98432. "
                "Please investigate immediately."
            )
        },
    )


class ProcessTicketResponse(BaseModel):
    """Full response from the /process_ticket endpoint."""
    ticket_id: str = Field(
        description="UUID of the saved ticket in the database.",
    )
    analysis: TicketAnalysis = Field(
        description="Structured analysis of the email.",
    )
    draft_response: str = Field(
        description="AI-generated email reply draft.",
    )
    extracted_text: Optional[str] = Field(
        default=None,
        description="Text extracted via OCR from an uploaded image (None if text was provided directly).",
    )
    message: str = Field(
        default="Ticket processed and saved successfully.",
        description="Status message.",
    )
