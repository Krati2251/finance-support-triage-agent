"""
Finance Support Triage Agent — Streamlit Dashboard

Run:  streamlit run app.py
"""

import streamlit as st
import requests
from datetime import datetime

# -------------------- Configuration --------------------

API_BASE = "http://127.0.0.1:8000"

# -------------------- Page Setup --------------------

st.set_page_config(
    page_title="Finance Triage Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Custom CSS --------------------

st.markdown("""
<style>
    /* Priority badges */
    .priority-high {
        background-color: #FF4B4B;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .priority-medium {
        background-color: #FFA726;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .priority-low {
        background-color: #66BB6A;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Category badges */
    .category-fraud {
        background-color: #D32F2F;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .category-payment {
        background-color: #1976D2;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .category-general {
        background-color: #757575;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Sentiment badges */
    .sentiment-urgent {
        background-color: #FF1744;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .sentiment-negative {
        background-color: #E65100;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .sentiment-neutral {
        background-color: #78909C;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .sentiment-positive {
        background-color: #2E7D32;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Ticket card */
    .ticket-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 10px;
        transition: box-shadow 0.2s;
    }
    .ticket-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }

    /* Draft response box */
    .draft-box {
        background-color: #F5F5F5;
        border-left: 4px solid #1976D2;
        padding: 16px;
        border-radius: 6px;
        font-family: 'Segoe UI', sans-serif;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* Email body box */
    .email-box {
        background-color: #FFFDE7;
        border-left: 4px solid #FFA726;
        padding: 16px;
        border-radius: 6px;
        font-family: 'Segoe UI', sans-serif;
        white-space: pre-wrap;
        line-height: 1.6;
    }

    /* Stats cards */
    .stat-card {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- Helper Functions --------------------

def get_priority_badge(priority: str) -> str:
    css_class = f"priority-{priority.lower()}"
    return f'<span class="{css_class}">⚡ {priority}</span>'


def get_category_badge(category: str) -> str:
    if category == "Fraud":
        css_class = "category-fraud"
        icon = "🚨"
    elif category == "Payment Issue":
        css_class = "category-payment"
        icon = "💳"
    else:
        css_class = "category-general"
        icon = "📋"
    return f'<span class="{css_class}">{icon} {category}</span>'


def get_sentiment_badge(sentiment: str) -> str:
    css_class = f"sentiment-{sentiment.lower()}"
    icons = {"Urgent": "🔴", "Negative": "😠", "Neutral": "😐", "Positive": "😊"}
    icon = icons.get(sentiment, "❓")
    return f'<span class="{css_class}">{icon} {sentiment}</span>'


def format_timestamp(ts: str) -> str:
    if not ts:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%b %d, %Y  %I:%M %p")
    except Exception:
        return ts


def fetch_tickets(status_filter: str = None) -> list:
    """Fetch tickets from the backend API."""
    try:
        params = {}
        if status_filter and status_filter != "All":
            params["status"] = status_filter
        resp = requests.get(f"{API_BASE}/tickets", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Make sure FastAPI is running on port 8000.")
        return []
    except Exception as e:
        st.error(f"⚠️ Error fetching tickets: {e}")
        return []


def approve_ticket(ticket_id: str) -> bool:
    try:
        resp = requests.post(f"{API_BASE}/approve_ticket/{ticket_id}", timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to approve ticket: {e}")
        return False


def reject_ticket(ticket_id: str) -> bool:
    try:
        resp = requests.patch(f"{API_BASE}/tickets/{ticket_id}/reject", timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to reject ticket: {e}")
        return False


def process_new_ticket(email_body: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/process_ticket",
            json={"email_body": email_body},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error processing ticket: {e}")
        return None


def process_ticket_image(uploaded_file) -> dict | None:
    """Send an uploaded image to the OCR endpoint."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        resp = requests.post(
            f"{API_BASE}/process_ticket_image",
            files=files,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
        return None


# -------------------- Sidebar --------------------

with st.sidebar:
    st.markdown("## 🏦 Finance Triage")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📋 Ticket Dashboard", "✉️ Submit New Ticket"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "📋 Ticket Dashboard":
        status_filter = st.selectbox(
            "Filter by Status",
            ["New", "All", "In Progress", "Resolved", "Closed"],
        )
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "High", "Medium", "Low"],
        )
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")
    st.caption("Finance Support Triage Agent v1.0")


# ==================== PAGE: SUBMIT NEW TICKET ====================

if page == "✉️ Submit New Ticket":
    st.markdown("# ✉️ Submit New Ticket")
    st.markdown("Paste a customer email **or upload an image** of the email. The AI will analyse it, classify it, and draft a response automatically.")
    st.markdown("---")

    input_tab, image_tab = st.tabs(["✍️ Type / Paste Email", "🖼️ Upload Image (OCR)"])

    # ---------- Tab 1: Text input ----------
    with input_tab:
        email_input = st.text_area(
            "Customer Email Body",
            height=200,
            placeholder="Hi, my name is Rajesh Kumar. I noticed an unauthorized transaction of $500 on my account...",
        )

        if st.button("🚀 Process Ticket", type="primary", use_container_width=True, key="btn_text"):
            if not email_input or len(email_input.strip()) < 10:
                st.warning("Please enter at least 10 characters.")
            else:
                with st.spinner("🤖 AI is analysing the email..."):
                    result = process_new_ticket(email_input)

                if result:
                    st.success(f"✅ {result.get('message', 'Ticket processed!')}")
                    st.markdown("---")

                    analysis = result.get("analysis", {})

                    st.markdown("### 🔍 Analysis Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**Priority**")
                        st.markdown(get_priority_badge(analysis.get("priority", "Medium")), unsafe_allow_html=True)
                    with col2:
                        st.markdown("**Category**")
                        st.markdown(get_category_badge(analysis.get("category", "General")), unsafe_allow_html=True)
                    with col3:
                        st.markdown("**Sentiment**")
                        st.markdown(get_sentiment_badge(analysis.get("sentiment", "Neutral")), unsafe_allow_html=True)
                    with col4:
                        st.markdown("**Ticket ID**")
                        st.code(result.get("ticket_id", "N/A")[:8] + "...")

                    st.markdown("")
                    st.markdown(f"**Intent:** {analysis.get('intent', 'N/A')}")
                    st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")

                    entities = analysis.get("entities", {})
                    if any(entities.values()):
                        st.markdown("**Extracted Entities:**")
                        ent_cols = st.columns(3)
                        with ent_cols[0]:
                            st.metric("Customer Name", entities.get("customer_name") or "N/A")
                        with ent_cols[1]:
                            st.metric("Transaction ID", entities.get("transaction_id") or "N/A")
                        with ent_cols[2]:
                            st.metric("Amount", entities.get("amount") or "N/A")

                    st.markdown("---")
                    st.markdown("### 📝 AI Draft Response")
                    draft = result.get("draft_response", "No draft generated.")
                    st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)

    # ---------- Tab 2: Image upload ----------
    with image_tab:
        st.info("🖼️ Upload a screenshot or photo of a customer email. We'll extract the text using OCR and then run the full AI analysis.")

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP",
        )

        if uploaded_file:
            # Show preview
            st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=True)

            if st.button("🔍 Extract & Process", type="primary", use_container_width=True, key="btn_ocr"):
                with st.spinner("🔤 Running OCR and AI analysis..."):
                    result = process_ticket_image(uploaded_file)

                if result:
                    st.success(f"✅ {result.get('message', 'Image processed!')}")
                    st.markdown("---")

                    # ---- Show extracted text ----
                    extracted = result.get("extracted_text", "")
                    if extracted:
                        st.markdown("### 🔤 Extracted Text (OCR)")
                        st.markdown(f'<div class="email-box">{extracted}</div>', unsafe_allow_html=True)
                        st.markdown("")

                    analysis = result.get("analysis", {})

                    st.markdown("### 🔍 Analysis Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**Priority**")
                        st.markdown(get_priority_badge(analysis.get("priority", "Medium")), unsafe_allow_html=True)
                    with col2:
                        st.markdown("**Category**")
                        st.markdown(get_category_badge(analysis.get("category", "General")), unsafe_allow_html=True)
                    with col3:
                        st.markdown("**Sentiment**")
                        st.markdown(get_sentiment_badge(analysis.get("sentiment", "Neutral")), unsafe_allow_html=True)
                    with col4:
                        st.markdown("**Ticket ID**")
                        st.code(result.get("ticket_id", "N/A")[:8] + "...")

                    st.markdown("")
                    st.markdown(f"**Intent:** {analysis.get('intent', 'N/A')}")
                    st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")

                    entities = analysis.get("entities", {})
                    if any(entities.values()):
                        st.markdown("**Extracted Entities:**")
                        ent_cols = st.columns(3)
                        with ent_cols[0]:
                            st.metric("Customer Name", entities.get("customer_name") or "N/A")
                        with ent_cols[1]:
                            st.metric("Transaction ID", entities.get("transaction_id") or "N/A")
                        with ent_cols[2]:
                            st.metric("Amount", entities.get("amount") or "N/A")

                    st.markdown("---")
                    st.markdown("### 📝 AI Draft Response")
                    draft = result.get("draft_response", "No draft generated.")
                    st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)


# ==================== PAGE: TICKET DASHBOARD ====================

elif page == "📋 Ticket Dashboard":
    st.markdown("# 📋 Ticket Dashboard")
    st.markdown("View and manage support tickets processed by the AI triage agent.")
    st.markdown("---")

    # Fetch tickets
    tickets = fetch_tickets(status_filter)

    # Apply client-side priority filter
    if priority_filter and priority_filter != "All":
        tickets = [t for t in tickets if t.get("priority") == priority_filter]

    if not tickets:
        st.info("No tickets found. Submit a new ticket from the sidebar to get started!")
    else:
        # ---- Summary Stats ----
        total = len(tickets)
        high_count = sum(1 for t in tickets if t.get("priority") == "High")
        fraud_count = sum(1 for t in tickets if t.get("category") == "Fraud")

        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Tickets", total)
        with stat_cols[1]:
            st.metric("🔴 High Priority", high_count)
        with stat_cols[2]:
            st.metric("🚨 Fraud Cases", fraud_count)
        with stat_cols[3]:
            filter_label = status_filter
            if priority_filter and priority_filter != "All":
                filter_label += f" / {priority_filter}"
            st.metric("📊 Filter", filter_label)

        st.markdown("---")

        # ---- Ticket List ----
        for i, ticket in enumerate(tickets):
            ticket_id = ticket.get("id", "N/A")
            short_id = ticket_id[:8]
            customer = ticket.get("customer_name", "Unknown")
            priority = ticket.get("priority", "Medium")
            category = ticket.get("category", "General")
            status = ticket.get("status", "New")
            created = format_timestamp(ticket.get("created_at"))
            summary = ticket.get("summary", "No summary available.")

            # Ticket row
            with st.expander(
                f"🎫  {short_id}...  •  {customer}  •  {priority} Priority  •  {category}  •  {created}",
                expanded=False,
            ):
                # ---- Header badges ----
                badge_col1, badge_col2, badge_col3, badge_col4 = st.columns(4)
                with badge_col1:
                    st.markdown("**Priority**")
                    st.markdown(get_priority_badge(priority), unsafe_allow_html=True)
                with badge_col2:
                    st.markdown("**Category**")
                    st.markdown(get_category_badge(category), unsafe_allow_html=True)
                with badge_col3:
                    st.markdown("**Sentiment**")
                    sentiment = ticket.get("sentiment", "Neutral") or "Neutral"
                    st.markdown(get_sentiment_badge(sentiment), unsafe_allow_html=True)
                with badge_col4:
                    st.markdown("**Status**")
                    st.markdown(f"📌 **{status}**")

                st.markdown("")

                # ---- Summary & Intent ----
                st.markdown(f"**🎯 Intent:** {ticket.get('intent', 'N/A')}")
                st.markdown(f"**📝 Summary:** {summary}")

                # ---- Extracted Entities ----
                ent_cols = st.columns(3)
                with ent_cols[0]:
                    st.metric("Customer", customer)
                with ent_cols[1]:
                    st.metric("Transaction ID", ticket.get("transaction_id") or "N/A")
                with ent_cols[2]:
                    st.metric("Amount", ticket.get("amount") or "N/A")

                st.markdown("---")

                # ---- Email Body ----
                st.markdown("#### 📧 Original Email")
                email_body = ticket.get("email_body", "N/A")
                st.markdown(f'<div class="email-box">{email_body}</div>', unsafe_allow_html=True)

                st.markdown("")

                # ---- AI Draft Response ----
                st.markdown("#### 🤖 AI Draft Response")
                draft = ticket.get("draft_response", "No draft available.")
                if draft and draft != "No draft available.":
                    st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)
                else:
                    st.info("No AI draft response available for this ticket.")

                st.markdown("")

                # ---- Action Buttons ----
                if status == "New":
                    btn_col1, btn_col2, spacer = st.columns([1, 1, 2])

                    with btn_col1:
                        if st.button(
                            "✅ Approve & Send",
                            key=f"approve_{ticket_id}",
                            type="primary",
                            use_container_width=True,
                        ):
                            with st.spinner("Approving & sending..."):
                                if approve_ticket(ticket_id):
                                    st.success("✅ Ticket approved! Draft response sent to customer. Ticket closed.")
                                    st.rerun()

                    with btn_col2:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_{ticket_id}",
                            use_container_width=True,
                        ):
                            with st.spinner("Rejecting..."):
                                if reject_ticket(ticket_id):
                                    st.warning("❌ Ticket rejected and closed.")
                                    st.rerun()

                elif status == "Closed":
                    st.info("🔒 This ticket is closed.")
                elif status == "In Progress":
                    st.success("✅ This ticket is in progress.")
                elif status == "Resolved":
                    st.info("✔️ This ticket has been resolved.")
