from contextlib import asynccontextmanager
from typing import Optional, List
import os, re, smtplib, logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import engine, Base, get_db
from models import Ticket, TicketStatus, TicketPriority, TicketCategory
from schemas import AnalyzeRequest, TicketAnalysis, ProcessTicketResponse
from agent import analyze_ticket, generate_draft_response, analyze_and_draft
from ocr import extract_text_from_image, SUPPORTED_CONTENT_TYPES

load_dotenv()

logger = logging.getLogger("email_sender")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")


def _extract_recipient_email(email_body: str) -> str | None:
    """Pull the sender's email address from the stored email body (From: line)."""
    if not email_body:
        return None
    for line in email_body.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("from:"):
            value = stripped.split(":", 1)[1].strip()
            match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', value)
            return match.group(0) if match else None
    return None


def _extract_subject(email_body: str) -> str:
    """Pull the subject from the stored email body."""
    if not email_body:
        return "Finance Support Response"
    for line in email_body.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("subject:"):
            return "Re: " + stripped.split(":", 1)[1].strip()
    return "Finance Support Response"


def send_reply_email(to_email: str, subject: str, body: str) -> bool:
    """Send the draft response to the customer via Gmail SMTP."""
    if not EMAIL_USER or not EMAIL_PASSWORD:
        logger.error("EMAIL_USER / EMAIL_PASSWORD not set — cannot send.")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        html = (
            '<div style="font-family:Arial,sans-serif;font-size:14px;'
            'line-height:1.7;color:#333;">'
            + body.replace("\n", "<br>") +
            '<br><br><hr style="border:none;border-top:1px solid #ddd;">'
            '<small style="color:#888;">This is an automated response from '
            'Finance Support Triage Agent.</small></div>'
        )
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, to_email, msg.as_string())

        logger.info(f"✅ Reply sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send email to {to_email}: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup (if they don't exist)."""
    print("📦 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables are ready.")
    yield


app = FastAPI(
    title="Finance Support Triage Agent",
    description="An AI-powered finance support triage agent",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (port 8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Finance Agent is Running"}


@app.post("/analyze", response_model=TicketAnalysis)
def analyze_email(request: AnalyzeRequest):
    """
    Analyse a customer support email and return structured triage data.

    Accepts the raw email text and returns sentiment, intent,
    extracted entities, priority, category, and a summary.
    """
    try:
        result = analyze_ticket(request.email_body)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


# ---------- Map schema enums → ORM enums ----------

_PRIORITY_MAP = {
    "High": TicketPriority.HIGH,
    "Medium": TicketPriority.MEDIUM,
    "Low": TicketPriority.LOW,
}

_CATEGORY_MAP = {
    "Fraud": TicketCategory.FRAUD,
    "Payment Issue": TicketCategory.PAYMENT_ISSUE,
    "General": TicketCategory.GENERAL,
}


@app.post("/process_ticket", response_model=ProcessTicketResponse)
def process_ticket(request: AnalyzeRequest, db: Session = Depends(get_db)):
    """
    End-to-end ticket processing pipeline:

    1. **Analyse** — Run AI analysis on the email (sentiment, intent, entities,
       priority, category, summary).
    2. **Draft** — Generate a personalised email reply based on the analysis.
    3. **Save** — Persist the ticket with all data to the PostgreSQL database.
    4. **Return** — Send back the ticket ID, full analysis, and draft response.
    """
    # ---- Step 1 + 2: Analyse the email AND generate draft (single LLM call) ----
    try:
        result = analyze_and_draft(request.email_body)
        analysis = result   # TicketAnalysisWithDraft extends TicketAnalysis
        draft = result.draft_response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )

    # ---- Step 3: Save to database ----
    try:
        ticket = Ticket(
            customer_name=analysis.entities.customer_name or "Unknown",
            email_body=request.email_body,
            status="New",
            priority=_PRIORITY_MAP.get(analysis.priority.value, TicketPriority.MEDIUM),
            category=_CATEGORY_MAP.get(analysis.category.value, TicketCategory.GENERAL),
            sentiment=analysis.sentiment.value,
            intent=analysis.intent,
            summary=analysis.summary,
            transaction_id=analysis.entities.transaction_id,
            amount=analysis.entities.amount,
            draft_response=draft,
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database save failed: {str(e)}",
        )

    # ---- Step 4: Return result ----
    return ProcessTicketResponse(
        ticket_id=str(ticket.id),
        analysis=analysis,
        draft_response=draft,
        message="Ticket processed and saved successfully.",
    )


@app.post("/process_ticket_image", response_model=ProcessTicketResponse)
async def process_ticket_image(
    file: UploadFile = File(..., description="Image file of the customer email (PNG, JPG, TIFF, BMP, WebP)"),
    db: Session = Depends(get_db),
):
    """
    End-to-end ticket processing from an **image upload**:

    1. **OCR** — Extract text from the uploaded image.
    2. **Analyse** — Run AI analysis on the extracted text.
    3. **Draft** — Generate a personalised email reply.
    4. **Save** — Persist the ticket to the database.
    5. **Return** — Send back the ticket ID, extracted text, analysis, and draft.
    """
    # ---- Validate file type ----
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(SUPPORTED_CONTENT_TYPES))}"
            ),
        )

    # ---- Step 1: OCR — extract text from image ----
    try:
        image_bytes = await file.read()
        extracted_text = extract_text_from_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR extraction failed: {str(e)}",
        )

    # ---- Step 2 + 3: Analyse AND generate draft (single LLM call) ----
    try:
        result = analyze_and_draft(extracted_text)
        analysis = result
        draft = result.draft_response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )

    # ---- Step 4: Save to database ----
    try:
        ticket = Ticket(
            customer_name=analysis.entities.customer_name or "Unknown",
            email_body=extracted_text,
            status="New",
            priority=_PRIORITY_MAP.get(analysis.priority.value, TicketPriority.MEDIUM),
            category=_CATEGORY_MAP.get(analysis.category.value, TicketCategory.GENERAL),
            sentiment=analysis.sentiment.value,
            intent=analysis.intent,
            summary=analysis.summary,
            transaction_id=analysis.entities.transaction_id,
            amount=analysis.entities.amount,
            draft_response=draft,
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database save failed: {str(e)}",
        )

    # ---- Step 5: Return result ----
    return ProcessTicketResponse(
        ticket_id=str(ticket.id),
        analysis=analysis,
        draft_response=draft,
        extracted_text=extracted_text,
        message="Image processed via OCR, analysed, and saved successfully.",
    )


# =====================================================================
#  TICKET CRUD ENDPOINTS (used by the Streamlit frontend)
# =====================================================================

def _ticket_to_dict(ticket: Ticket) -> dict:
    """Serialise a Ticket ORM object to a plain dict."""
    return {
        "id": str(ticket.id),
        "customer_name": ticket.customer_name,
        "email_body": ticket.email_body,
        "status": ticket.status.value if ticket.status else "New",
        "priority": ticket.priority.value if ticket.priority else "Medium",
        "category": ticket.category.value if ticket.category else "General",
        "sentiment": ticket.sentiment,
        "intent": ticket.intent,
        "summary": ticket.summary,
        "transaction_id": ticket.transaction_id,
        "amount": ticket.amount,
        "draft_response": ticket.draft_response,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
    }


@app.get("/tickets")
def list_tickets(
    status: Optional[str] = Query(None, description="Filter by status: New, In Progress, Resolved, Closed"),
    db: Session = Depends(get_db),
):
    """Return all tickets, optionally filtered by status."""
    query = db.query(Ticket).order_by(Ticket.created_at.desc())

    if status:
        try:
            status_enum = TicketStatus(status)
            query = query.filter(Ticket.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of: New, In Progress, Resolved, Closed",
            )

    tickets = query.all()
    return [_ticket_to_dict(t) for t in tickets]


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """Get a single ticket by ID."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return _ticket_to_dict(ticket)


@app.patch("/tickets/{ticket_id}/approve")
def approve_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """Approve the AI draft — marks ticket as 'In Progress'."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = TicketStatus.IN_PROGRESS
    db.commit()
    db.refresh(ticket)
    return {"message": "Ticket approved. Draft response sent.", "ticket": _ticket_to_dict(ticket)}


@app.post("/approve_ticket/{ticket_id}")
def approve_and_close_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """Approve the AI draft, send it via email, and close the ticket."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # --- Actually send the reply email ---
    email_sent = False
    recipient = _extract_recipient_email(ticket.email_body)
    if recipient and ticket.draft_response:
        subject = _extract_subject(ticket.email_body)
        email_sent = send_reply_email(recipient, subject, ticket.draft_response)
    elif not recipient:
        logger.warning(f"No recipient email found in ticket {ticket_id}")

    ticket.status = TicketStatus.CLOSED
    db.commit()
    db.refresh(ticket)

    msg = (
        f"Ticket approved, response sent to {recipient}, and ticket closed."
        if email_sent
        else "Ticket approved and closed, but email could not be sent."
    )
    return {
        "message": msg,
        "email_sent": email_sent,
        "recipient": recipient,
        "ticket": _ticket_to_dict(ticket),
    }


@app.patch("/tickets/{ticket_id}/reject")
def reject_ticket(ticket_id: str, db: Session = Depends(get_db)):
    """Reject the AI draft — marks ticket as 'Closed'."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.status = TicketStatus.CLOSED
    db.commit()
    db.refresh(ticket)
    return {"message": "Ticket rejected and closed.", "ticket": _ticket_to_dict(ticket)}


# =====================================================================
#  EMAIL INGESTION ENDPOINT (triggered from the Streamlit frontend)
# =====================================================================

@app.post("/fetch_emails")
def fetch_emails_endpoint(
    db: Session = Depends(get_db),
    include_read: bool = Query(False, description="Also fetch already-read emails"),
    max_emails: int = Query(5, ge=1, le=50, description="Max emails to process"),
):
    """
    Connect to Gmail via IMAP, pull emails, analyse each one
    with the AI agent, save tickets, and return a summary.

    - By default only UNSEEN (unread) emails are fetched.
    - Set include_read=true to re-fetch ALL recent emails (useful for testing).
    """
    import os
    import imaplib
    import email as email_lib
    from email.header import decode_header as _decode_header
    import time as _time

    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    if not EMAIL_USER or not EMAIL_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="EMAIL_USER and EMAIL_PASSWORD must be set in the .env file.",
        )

    _start = _time.time()
    print(f"📧 fetch_emails called  include_read={include_read}  max_emails={max_emails}")

    # ── Helper: decode MIME headers ──
    def _decode_hdr(value: str) -> str:
        if not value:
            return ""
        parts = []
        for part, charset in _decode_header(value):
            if isinstance(part, bytes):
                parts.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                parts.append(part)
        return " ".join(parts)

    # ── Helper: extract plain-text body ──
    def _extract_body(msg) -> str:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                cd = str(part.get("Content-Disposition", ""))
                if "attachment" in cd:
                    continue
                if ct == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        break
                elif ct == "text/html" and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        import re
                        html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        body = re.sub(r"<[^>]+>", " ", html)
                        body = re.sub(r"\s+", " ", body).strip()
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        return body.strip()

    # ── Connect to Gmail ──
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select("INBOX")
        print(f"  ✅ Connected to Gmail as {EMAIL_USER}")
    except Exception as e:
        print(f"  ❌ IMAP connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"IMAP connection failed: {e}")

    # ── Search for emails ──
    try:
        # Use date-based search instead of UNSEEN to catch emails that
        # were auto-marked as read by Gmail / phone within seconds.
        # This way we never miss emails. Duplicates are filtered by
        # the email_body check against the DB below.
        from datetime import datetime as _dt, timedelta as _td
        if include_read:
            search_criteria = "ALL"
        else:
            # Fetch emails from the last 2 days (IMAP SINCE uses date only, no time)
            since_date = (_dt.now() - _td(days=2)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE "{since_date}")'
        status, messages = mail.search(None, search_criteria)
        print(f"  🔍 Search criteria: {search_criteria}  status: {status}")

        if status != "OK":
            raise HTTPException(status_code=500, detail="Could not search mailbox.")

        email_ids = messages[0].split()
        # Take only the most recent N emails (last items = newest)
        email_ids = email_ids[-max_emails:] if len(email_ids) > max_emails else email_ids
        print(f"  📬 Found {len(email_ids)} email(s) to process")

        results = []
        errors = []
        skipped_dupes = 0

        for eid in email_ids:
            try:
                st_fetch, msg_data = mail.fetch(eid, "(RFC822)")
                if st_fetch != "OK":
                    continue
                raw = msg_data[0][1]
                msg = email_lib.message_from_bytes(raw)

                subject = _decode_hdr(msg.get("Subject", "(No Subject)"))
                sender = _decode_hdr(msg.get("From", "(Unknown)"))
                body = _extract_body(msg)

                if not body or len(body.strip()) < 10:
                    mail.store(eid, "+FLAGS", "\\Seen")
                    continue

                full_text = f"From: {sender}\nSubject: {subject}\n\n{body}"

                # ── Skip duplicates: check if an identical email_body already exists ──
                existing = db.query(Ticket).filter(
                    Ticket.email_body == full_text
                ).first()
                if existing:
                    skipped_dupes += 1
                    mail.store(eid, "+FLAGS", "\\Seen")
                    continue

                print(f"  📩 Processing: {subject[:60]}")

                # ── Analyse + Draft (single LLM call) ──
                try:
                    combined = analyze_and_draft(full_text)
                    analysis = combined
                    draft = combined.draft_response
                except Exception as ai_err:
                    err_str = str(ai_err)
                    # Detect quota / rate-limit errors and abort early
                    if "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower():
                        mail.close()
                        mail.logout()
                        elapsed = round(_time.time() - _start, 1)
                        print(f"  🚫 Groq API rate limit hit after {elapsed}s")
                        return {
                            "fetched": len(results),
                            "errors": len(errors) + 1,
                            "skipped_duplicates": skipped_dupes,
                            "tickets": results,
                            "error_details": [
                                "⚠️ Groq API rate limit reached (30 req/min). "
                                "Please wait 1-2 minutes and try again."
                            ],
                            "message": f"Processed {len(results)} email(s) before hitting rate limit.",
                            "quota_error": True,
                        }
                    raise

                # ── Save ticket ──
                ticket = Ticket(
                    customer_name=analysis.entities.customer_name or sender.split("<")[0].strip() or "Unknown",
                    email_body=full_text,
                    status="New",
                    priority=_PRIORITY_MAP.get(analysis.priority.value, TicketPriority.MEDIUM),
                    category=_CATEGORY_MAP.get(analysis.category.value, TicketCategory.GENERAL),
                    sentiment=analysis.sentiment.value,
                    intent=analysis.intent,
                    summary=analysis.summary,
                    transaction_id=analysis.entities.transaction_id,
                    amount=analysis.entities.amount,
                    draft_response=draft,
                )
                db.add(ticket)
                db.commit()
                db.refresh(ticket)

                mail.store(eid, "+FLAGS", "\\Seen")

                results.append({
                    "ticket_id": str(ticket.id),
                    "subject": subject,
                    "sender": sender,
                    "priority": analysis.priority.value,
                    "category": analysis.category.value,
                })
                print(f"    ✅ Ticket {str(ticket.id)[:8]} | {analysis.priority.value} | {analysis.category.value}  ({_time.time()-_start:.1f}s elapsed)")
            except Exception as e:
                errors.append(f"{subject if 'subject' in dir() else 'unknown'}: {str(e)}")
                print(f"    ❌ Error: {e}")
                continue

        mail.close()
        mail.logout()

        elapsed = round(_time.time() - _start, 1)
        msg = f"Fetched and processed {len(results)} email(s) in {elapsed}s."
        if skipped_dupes:
            msg += f" Skipped {skipped_dupes} duplicate(s)."
        print(f"  📊 Done: {msg}")

        return {
            "fetched": len(results),
            "errors": len(errors),
            "skipped_duplicates": skipped_dupes,
            "tickets": results,
            "error_details": errors[:10],
            "message": msg,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"  ❌ Email processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Email processing failed: {e}")
