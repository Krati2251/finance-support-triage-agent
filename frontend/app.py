"""
Finance Support Triage Agent — Blox-Style Modern Email Client UI
Clean white theme  ·  3-column inspired layout  ·  All features preserved
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import re
import time as _time
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Finance Triage",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════
for k, v in {
    "sel": None,
    "tickets": [],
    "fetch_res": None,
    "tab": "inbox",
    "page": "inbox",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════
#  GLOBAL CSS — Blox-inspired clean white theme
# ═══════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');

/* ─── Reset Streamlit chrome ─── */
header[data-testid="stHeader"]                               { background:#f8f9fb !important; }
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer                                            { display:none !important; }
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarNavCollapseButton"]             { display:none !important; }

/* ─── Lock sidebar open ─── */
section[data-testid="stSidebar"] {
    transform:none !important;
    min-width:248px !important; width:248px !important;
    visibility:visible !important; display:flex !important;
    background:#ffffff !important;
    border-right:1px solid #e5e7eb !important;
    box-shadow:2px 0 8px rgba(0,0,0,.03) !important;
}
section[data-testid="stSidebar"] > div { overflow-y:auto; }

/* ─── Global ─── */
html, body, .stApp {
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif !important;
    background:#f8f9fb !important;
    color:#1a1a2e !important;
}
.block-container { padding:1rem 1.5rem !important; max-width:1800px; }

/* ─── Sidebar internals ─── */
section[data-testid="stSidebar"] * { color:#6b7280 !important; }
section[data-testid="stSidebar"] .stMarkdown h3 {
    color:#1a1a2e !important; font-weight:700 !important; font-size:.72rem !important;
    text-transform:uppercase; letter-spacing:.8px; margin-top:16px !important;
}
section[data-testid="stSidebar"] hr { border-color:#e5e7eb !important; margin:6px 0 !important; }
section[data-testid="stSidebar"] .stButton>button {
    background:#fff !important; border:1px solid #e5e7eb !important;
    color:#6b7280 !important; border-radius:10px !important;
    font-weight:500 !important; font-size:.82rem !important;
    transition:all .15s ease !important; text-align:left !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background:#f3f4f6 !important; border-color:#d1d5db !important;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background:#f8f9fb; border:1px solid #e5e7eb;
    border-radius:10px; padding:10px 8px;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size:.58rem !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:.5px; color:#9ca3af !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-weight:700 !important; color:#1a1a2e !important; font-size:1.2rem !important;
}

/* ─── Top Bar ─── */
.top-bar {
    display:flex; align-items:center; gap:14px;
    padding:10px 0 14px;
}
.top-logo { display:flex; align-items:center; gap:9px; font-size:1.15rem; font-weight:800; color:#1a1a2e; }
.top-logo-icon {
    width:34px; height:34px; border-radius:9px;
    background:linear-gradient(135deg,#4f46e5,#7c3aed);
    display:flex; align-items:center; justify-content:center;
    color:#fff; font-size:.82rem; font-weight:800;
}
.search-box {
    flex:1; max-width:480px;
    background:#f3f4f6; border:1px solid #e5e7eb;
    border-radius:24px; padding:9px 18px;
    font-size:.82rem; color:#9ca3af;
    display:flex; align-items:center; gap:8px;
}
.search-box .material-icons-outlined { font-size:1.05rem; color:#9ca3af; }

/* ─── Email List ─── */
.email-list {
    background:#fff; border:1px solid #e5e7eb;
    border-radius:12px; overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,.04);
}
.date-group {
    font-size:.68rem; font-weight:700; color:#9ca3af;
    text-transform:uppercase; letter-spacing:.7px;
    padding:10px 18px 6px; background:#fafbfc;
    border-bottom:1px solid #f3f4f6;
}

/* ─── Email Row ─── */
.eml-row {
    display:grid; grid-template-columns:42px 1fr auto;
    align-items:center; gap:12px;
    padding:13px 18px; border-bottom:1px solid #f3f4f6;
    cursor:pointer; transition:background .1s ease;
}
.eml-row:hover { background:#f8f9fb; }
.eml-row.selected { background:#eef2ff; border-left:3px solid #4f46e5; }
.eml-row.read .eml-sender, .eml-row.read .eml-subject { font-weight:400; color:#6b7280; }

/* ─── Avatar ─── */
.avatar {
    width:38px; height:38px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:.78rem; color:#fff; flex-shrink:0;
}
.av-red    { background:#ef4444; }
.av-blue   { background:#3b82f6; }
.av-green  { background:#22c55e; }
.av-purple { background:#8b5cf6; }
.av-orange { background:#f97316; }
.av-pink   { background:#ec4899; }
.av-teal   { background:#14b8a6; }
.av-indigo { background:#6366f1; }

.eml-meta { min-width:0; overflow:hidden; }
.eml-sender {
    font-size:.84rem; font-weight:600; color:#1a1a2e;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.eml-subject {
    font-size:.82rem; font-weight:600; color:#1a1a2e;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:1px;
}
.eml-preview {
    font-size:.77rem; color:#9ca3af; font-weight:400;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:1px;
}
.eml-right {
    text-align:right; white-space:nowrap;
    display:flex; flex-direction:column; align-items:flex-end; gap:4px;
}
.eml-time { font-size:.68rem; color:#9ca3af; }
.eml-tag {
    display:inline-flex; padding:2px 8px; border-radius:4px;
    font-size:.6rem; font-weight:700; letter-spacing:.3px;
}
.tag-fraud   { background:#fef2f2; color:#dc2626; }
.tag-payment { background:#eff6ff; color:#2563eb; }
.tag-general { background:#f3f4f6; color:#6b7280; }

/* ─── Priority Dot ─── */
.p-dot { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; }
.pd-high   { background:#ef4444; }
.pd-medium { background:#f59e0b; }
.pd-low    { background:#22c55e; }

/* ─── Detail Panel ─── */
.detail-card {
    background:#fff; border:1px solid #e5e7eb;
    border-radius:12px; padding:24px 28px;
    box-shadow:0 1px 4px rgba(0,0,0,.04);
}
.detail-actions {
    display:flex; gap:14px; align-items:center;
    padding:10px 0; border-bottom:1px solid #e5e7eb; margin-bottom:18px;
}
.action-btn {
    display:inline-flex; align-items:center; gap:5px;
    font-size:.76rem; color:#6b7280; font-weight:500;
    cursor:pointer; padding:5px 10px; border-radius:6px;
    transition:all .1s ease; background:none; border:none;
}
.action-btn:hover { background:#f3f4f6; color:#1a1a2e; }
.action-icon { font-size:.92rem; }
.detail-from { display:flex; align-items:center; gap:14px; margin-bottom:18px; }
.detail-from-name { font-weight:700; color:#1a1a2e; font-size:.92rem; }
.detail-from-time { color:#9ca3af; font-size:.76rem; }
.detail-subject-line {
    font-size:1.2rem; font-weight:700; color:#1a1a2e;
    margin-bottom:18px; line-height:1.4;
}
.detail-body {
    color:#374151; font-size:.88rem; line-height:1.8;
    white-space:pre-wrap; padding:14px 0;
    border-top:1px solid #f3f4f6;
}

/* ─── Badges ─── */
.badge {
    display:inline-flex; align-items:center; gap:4px;
    padding:3px 10px; border-radius:14px;
    font-weight:600; font-size:.68rem; letter-spacing:.2px;
}
.b-high    { background:#fef2f2; color:#dc2626; }
.b-medium  { background:#fffbeb; color:#d97706; }
.b-low     { background:#f0fdf4; color:#16a34a; }
.b-fraud   { background:#fef2f2; color:#dc2626; }
.b-payment { background:#eff6ff; color:#2563eb; }
.b-general { background:#f3f4f6; color:#6b7280; }
.b-status  { background:#eef2ff; color:#4f46e5; }
.b-neg     { background:#fef2f2; color:#dc2626; }
.b-neu     { background:#f3f4f6; color:#6b7280; }
.b-pos     { background:#f0fdf4; color:#16a34a; }

/* ─── Insight Grid ─── */
.insight-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:14px 0; }
.insight-box {
    background:#f8f9fb; border:1px solid #e5e7eb;
    border-radius:10px; padding:12px;
}
.insight-label {
    font-size:.58rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.6px; color:#9ca3af; margin-bottom:5px;
}
.insight-value { font-size:.84rem; font-weight:600; color:#1a1a2e; }

/* ─── Section Header ─── */
.section-hdr {
    font-size:.72rem; font-weight:700; color:#4f46e5;
    text-transform:uppercase; letter-spacing:.7px;
    margin:18px 0 8px; padding-bottom:5px;
    border-bottom:2px solid #eef2ff; display:inline-block;
}

/* ─── Alert Bars ─── */
.alert-bar {
    display:flex; align-items:center; gap:10px;
    padding:11px 16px; border-radius:10px; margin-bottom:8px; font-size:.84rem;
}
.ab-red   { background:#fef2f2; border:1px solid #fecaca; color:#dc2626; }
.ab-amber { background:#fffbeb; border:1px solid #fde68a; color:#d97706; }
.ab-blue  { background:#eff6ff; border:1px solid #bfdbfe; color:#2563eb; }
.ab-green { background:#f0fdf4; border:1px solid #bbf7d0; color:#16a34a; }
.ab-icon  { font-size:1.15rem; }
.ab-text b { font-weight:700; }

/* ─── Stat Cards ─── */
.stat-row { display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap; }
.stat-card {
    flex:1; min-width:110px;
    background:#fff; border:1px solid #e5e7eb;
    border-radius:12px; padding:14px 12px;
    text-align:center; position:relative; overflow:hidden;
    transition:all .15s ease; box-shadow:0 1px 3px rgba(0,0,0,.03);
}
.stat-card:hover { transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.06); }
.stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.sc-total::before  { background:linear-gradient(90deg,#4f46e5,#7c3aed); }
.sc-high::before   { background:#ef4444; }
.sc-medium::before { background:#f59e0b; }
.sc-low::before    { background:#22c55e; }
.sc-fraud::before  { background:#dc2626; }
.stat-num { font-size:1.7rem; font-weight:800; color:#1a1a2e; line-height:1; }
.stat-lbl { font-size:.58rem; font-weight:700; color:#9ca3af; text-transform:uppercase; letter-spacing:.6px; margin-top:3px; }

/* ─── Queue Section ─── */
.queue-sec {
    background:#fff; border:1px solid #e5e7eb;
    border-radius:12px; padding:14px 16px;
    margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,.03);
}
.queue-hdr {
    display:flex; align-items:center; gap:8px;
    padding-bottom:8px; border-bottom:1px solid #f3f4f6; margin-bottom:8px;
}
.queue-title { font-size:.88rem; font-weight:700; color:#1a1a2e; }
.queue-cnt {
    margin-left:auto; font-size:.64rem; font-weight:700;
    padding:2px 9px; border-radius:10px;
}
.qc-r { background:#fef2f2; color:#dc2626; }
.qc-a { background:#fffbeb; color:#d97706; }
.qc-g { background:#f0fdf4; color:#16a34a; }

/* ─── Welcome ─── */
.welcome { text-align:center; padding:50px 20px; color:#9ca3af; }
.welcome-icon { font-size:2.8rem; margin-bottom:10px; display:block; }
.welcome-title { font-size:1.05rem; font-weight:700; color:#6b7280; margin-bottom:4px; }
.welcome-sub { font-size:.82rem; max-width:360px; margin:0 auto; line-height:1.6; }

/* ─── Fetch Page ─── */
.fetch-card {
    background:#fff; border:1px solid #e5e7eb;
    border-radius:12px; padding:22px 26px;
    box-shadow:0 1px 4px rgba(0,0,0,.04); margin-bottom:12px;
}

/* ─── Text Area ─── */
.stTextArea textarea {
    background:#f8f9fb !important; border:1px solid #e5e7eb !important;
    border-radius:10px !important; color:#1a1a2e !important;
    font-family:'Inter',sans-serif !important;
    font-size:.86rem !important; line-height:1.7 !important;
}
.stTextArea textarea:focus {
    border-color:#4f46e5 !important;
    box-shadow:0 0 0 3px rgba(79,70,229,.08) !important;
}

/* ─── Buttons ─── */
.stButton>button[kind="primary"] {
    background:#4f46e5 !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    padding:10px 22px !important; color:#fff !important;
    box-shadow:0 2px 8px rgba(79,70,229,.22) !important;
    transition:all .15s ease !important;
}
.stButton>button[kind="primary"]:hover {
    background:#4338ca !important;
    box-shadow:0 4px 14px rgba(79,70,229,.32) !important;
}
.stButton>button[kind="secondary"] {
    border-radius:10px !important; font-weight:500 !important;
    border-color:#e5e7eb !important; color:#6b7280 !important;
}
.stButton>button[kind="secondary"]:hover { background:#f3f4f6 !important; }

/* ─── Selectbox ─── */
.stSelectbox [data-baseweb="select"]>div {
    background:#f8f9fb !important; border-color:#e5e7eb !important;
    border-radius:8px !important; color:#1a1a2e !important;
}

/* ─── Metric ─── */
[data-testid="stMetric"] {
    background:#f8f9fb !important; border:1px solid #e5e7eb;
    border-radius:10px; padding:10px 8px;
}
[data-testid="stMetricLabel"]  { font-size:.58rem !important; color:#9ca3af !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:.5px; }
[data-testid="stMetricValue"]  { font-weight:700 !important; color:#1a1a2e !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar       { width:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#d1d5db; border-radius:3px; }

.stSpinner>div>div { border-top-color:#4f46e5 !important; }
.stAlert { border-radius:10px !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════
#  AUTO-REFRESH: poll for new tickets every 30 seconds
# ═══════════════════════════════════════════════════════
st_autorefresh(interval=30_000, limit=None, key="auto_refresh")

# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
_AV_COLORS = [
    "av-red", "av-blue", "av-green", "av-purple",
    "av-orange", "av-pink", "av-teal", "av-indigo",
]
_PRI_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "?"


def _av_color(name: str) -> str:
    return _AV_COLORS[hash(name) % len(_AV_COLORS)]


def _avatar(name: str) -> str:
    return f'<div class="avatar {_av_color(name)}">{_initials(name)}</div>'


def _pri_badge(p: str) -> str:
    c = {"High": "b-high", "Medium": "b-medium", "Low": "b-low"}.get(p, "b-low")
    ic = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    return f'<span class="badge {c}">{ic.get(p, "⚪")} {p}</span>'


def _cat_badge(c: str) -> str:
    cls = {"Fraud": "b-fraud", "Payment Issue": "b-payment"}.get(c, "b-general")
    ic = {"Fraud": "🚨", "Payment Issue": "💳", "General": "📋"}
    return f'<span class="badge {cls}">{ic.get(c, "📋")} {c}</span>'


def _sent_badge(s: str) -> str:
    cls = {"Negative": "b-neg", "Neutral": "b-neu", "Positive": "b-pos", "Urgent": "b-neg"}.get(s, "b-neu")
    ic = {"Negative": "😠", "Neutral": "😐", "Positive": "😊", "Urgent": "🔴"}
    return f'<span class="badge {cls}">{ic.get(s, "❓")} {s or "Neutral"}</span>'


def _tag_html(cat: str) -> str:
    cls = {"Fraud": "tag-fraud", "Payment Issue": "tag-payment"}.get(cat, "tag-general")
    return f'<span class="eml-tag {cls}">{cat}</span>'


# ── Time / Date ──────────────────────────────────────

def _fmt_time(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        diff = now - dt
        if dt.date() == now.date():
            mins = int(diff.total_seconds() / 60)
            if mins < 1:
                return "Just now"
            if mins < 60:
                return f"{mins} min ago"
            return dt.strftime("%I:%M %p")
        if dt.date() == (now - timedelta(days=1)).date():
            return "Yesterday"
        if diff.days < 7:
            return f"{diff.days} days ago"
        return dt.strftime("%b %d")
    except Exception:
        return str(ts)[:10]


def _fmt_full(ts: str) -> str:
    if not ts:
        return "N/A"
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M, %A, %b %d")
    except Exception:
        return ts


def _date_group(ts: str) -> str:
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(ts)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        if dt.date() == now.date():
            return "today"
        if dt.date() == (now - timedelta(days=1)).date():
            return "yesterday"
        if (now - dt).days < 7:
            return "this week"
        return "earlier"
    except Exception:
        return "earlier"


# ── Email parsing ────────────────────────────────────

def _extract_subject(body: str) -> str:
    if not body:
        return "No Subject"
    for line in body.split("\n"):
        if line.strip().lower().startswith("subject:"):
            return line.split(":", 1)[1].strip()[:80]
    return body.strip()[:55] + ("…" if len(body.strip()) > 55 else "")


def _extract_sender(body: str, customer_name: str) -> str:
    if customer_name and customer_name != "Unknown":
        return customer_name
    if body:
        for line in body.split("\n"):
            if line.strip().lower().startswith("from:"):
                s = line.split(":", 1)[1].strip()
                name = re.sub(r"<[^>]+>", "", s).strip()
                return name[:30] if name else s[:30]
    return "Unknown Sender"


def _get_preview(body: str) -> str:
    if not body:
        return ""
    parts = []
    for l in body.strip().split("\n"):
        if l.strip().lower().startswith(("from:", "to:", "subject:", "date:")):
            continue
        if l.strip():
            parts.append(l.strip())
    return " ".join(parts)[:100]


def _get_body(body: str) -> str:
    if not body:
        return ""
    lines = []
    for l in body.strip().split("\n"):
        if l.strip().lower().startswith(("from:", "to:", "subject:", "date:")):
            continue
        lines.append(l)
    return "\n".join(lines).strip()


# ═══════════════════════════════════════════════════════
#  API LAYER
# ═══════════════════════════════════════════════════════

def _fetch_tickets(status: str = "All"):
    try:
        params = {} if status == "All" else {"status": status}
        r = requests.get(f"{API}/tickets", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.toast("⚠️ Backend offline — start FastAPI on port 8000.", icon="🔌")
        return []
    except Exception as e:
        st.toast(f"Error: {e}", icon="❌")
        return []


def _api_approve(tid: str):
    try:
        r = requests.post(f"{API}/approve_ticket/{tid}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Approve failed: {e}")
        return None


def _api_close(tid: str):
    try:
        r = requests.patch(f"{API}/tickets/{tid}/reject", timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Close failed: {e}")
        return False


def _api_fetch_emails(include_read: bool = False, max_emails: int = 5):
    try:
        r = requests.post(
            f"{API}/fetch_emails",
            params={"include_read": include_read, "max_emails": max_emails},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API.")
        return None
    except Exception as e:
        st.error(f"⚠️ Email fetch failed: {e}")
        return None


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    # ── Profile ──
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:6px 0 2px;">'
        '<div class="avatar av-indigo" style="width:34px;height:34px;font-size:.72rem;">FT</div>'
        "<div>"
        '<div style="font-size:.72rem;color:#9ca3af !important;">Welcome</div>'
        '<div style="font-size:.88rem;font-weight:700;color:#1a1a2e !important;">Finance Triage</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Fetch button (primary action) ──
    if st.button("📥  Fetch New Emails", key="sb_fetch", use_container_width=True, type="primary"):
        st.session_state.page = "fetch"
        st.session_state.sel = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Navigation")

    # Load tickets for counts
    all_tickets = _fetch_tickets("All")
    st.session_state.tickets = all_tickets
    new_ct = sum(1 for t in all_tickets if t.get("status") == "New")
    high_ct = sum(1 for t in all_tickets if t.get("priority") == "High")
    fraud_ct = sum(1 for t in all_tickets if t.get("category") == "Fraud")

    nav_items = [
        ("inbox",    "📬", "Inbox",          new_ct),
        ("queue",    "⚡", "Priority Queue", high_ct),
        ("category", "🏷️", "By Category",    None),
        ("alerts",   "🚨", "Alerts",         fraud_ct),
    ]

    for key, icon, label, count in nav_items:
        suffix = f" ({count})" if count else ""
        btn_type = "primary" if st.session_state.tab == key and st.session_state.page == "inbox" else "secondary"
        if st.button(f"{icon}  {label}{suffix}", key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state.tab = key
            st.session_state.page = "inbox"
            st.session_state.sel = None
            st.rerun()

    st.markdown("---")
    st.markdown("### Filters")

    status_filter = st.selectbox(
        "Status", ["All", "New", "In Progress", "Resolved", "Closed"],
        label_visibility="collapsed",
    )
    priority_filter = st.radio(
        "Priority", ["All", "High", "Medium", "Low"],
        horizontal=True, label_visibility="collapsed",
    )

    # Apply filters
    filtered = all_tickets
    if status_filter != "All":
        filtered = [t for t in filtered if t.get("status") == status_filter]
    if priority_filter != "All":
        filtered = [t for t in filtered if t.get("priority") == priority_filter]
    st.session_state.tickets = filtered

    st.markdown("---")
    st.markdown("### Quick Stats")

    m1, m2 = st.columns(2)
    m1.metric("Total", len(filtered))
    m2.metric("High", sum(1 for t in filtered if t.get("priority") == "High"))
    m3, m4 = st.columns(2)
    m3.metric("Fraud", sum(1 for t in filtered if t.get("category") == "Fraud"))
    m4.metric("New", sum(1 for t in filtered if t.get("status") == "New"))

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.sel = None
        st.rerun()

    st.markdown("---")
    st.caption("v3.0 • Blox-Style AI Triage")


# ═══════════════════════════════════════════════════════
#  PAGE: FETCH EMAILS
# ═══════════════════════════════════════════════════════
if st.session_state.page == "fetch":
    st.markdown(
        '<div class="top-bar">'
        '<div class="top-logo"><div class="top-logo-icon">FT</div><span>Finance Triage</span></div>'
        '<div class="search-box"><span class="material-icons-outlined">search</span>'
        "<span>Fetch &amp; triage emails from your Gmail inbox</span></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="fetch-card">'
        '<div style="display:flex;gap:12px;align-items:flex-start;">'
        '<span style="font-size:1.2rem;">💡</span>'
        "<div>"
        '<div style="font-weight:700;color:#1a1a2e;margin-bottom:3px;font-size:.92rem;">How it works</div>'
        '<div style="color:#6b7280;font-size:.82rem;line-height:1.8;">'
        "1. Click <b>Fetch Emails</b> below<br/>"
        "2. Connects to your Gmail inbox via IMAP<br/>"
        "3. Fetches emails from the <b>last 2 days</b> (catches auto-read emails too)<br/>"
        "4. Emails are analysed by AI (Groq Llama 3.3)<br/>"
        "5. Tickets are created and sorted by urgency — duplicates auto-skipped<br/>"
        '6. Toggle <b>"Include already-read"</b> to go further back and re-process all emails'
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        include_read = st.checkbox("📖 Include already-read emails", value=False)
    with c2:
        max_emails = st.slider("Max emails", 1, 50, 5, 1)

    st.markdown("")
    bc, _ = st.columns([1, 2])
    with bc:
        if st.button("📥 Fetch Emails Now", type="primary", use_container_width=True):
            with st.spinner("📡 Connecting to Gmail and processing emails…"):
                result = _api_fetch_emails(include_read=include_read, max_emails=max_emails)
            st.session_state.fetch_res = result

    result = st.session_state.fetch_res
    if result:
        fetched = result.get("fetched", 0)
        errs = result.get("errors", 0)
        skipped = result.get("skipped_duplicates", 0)
        quota_error = result.get("quota_error", False)

        if quota_error:
            st.markdown(
                '<div class="alert-bar ab-red"><span class="ab-icon">🚫</span>'
                f'<div class="ab-text"><b>Rate Limit Hit</b> — Processed {fetched} email(s). '
                "Wait 1-2 min and retry.</div></div>",
                unsafe_allow_html=True,
            )

        if fetched > 0:
            parts = [f"{fetched} processed"]
            if skipped:
                parts.append(f"{skipped} duplicates skipped")
            if errs:
                parts.append(f"{errs} errors")
            st.markdown(
                f'<div class="alert-bar ab-green"><span class="ab-icon">✅</span>'
                f'<div class="ab-text"><b>{result.get("message","Done!")}</b> — '
                f'{"  •  ".join(parts)}</div></div>',
                unsafe_allow_html=True,
            )
            for i, t in enumerate(result.get("tickets", [])):
                tc1, tc2, tc3, tc4 = st.columns([3, 1, 1, 1])
                with tc1:
                    st.markdown(f"**{t.get('subject','N/A')}**")
                    st.caption(t.get("sender", ""))
                with tc2:
                    st.markdown(_pri_badge(t.get("priority", "Medium")), unsafe_allow_html=True)
                with tc3:
                    st.markdown(_cat_badge(t.get("category", "General")), unsafe_allow_html=True)
                with tc4:
                    st.code(t.get("ticket_id", "")[:8])
                if i < len(result.get("tickets", [])) - 1:
                    st.divider()
        else:
            st.markdown(
                '<div class="alert-bar ab-amber"><span class="ab-icon">📭</span>'
                '<div class="ab-text"><b>No new emails.</b> '
                'Try enabling "Include already-read emails".</div></div>',
                unsafe_allow_html=True,
            )

        if errs > 0:
            with st.expander(f"⚠️ {errs} error(s)"):
                for e in result.get("error_details", []):
                    st.code(e)

    st.stop()


# ═══════════════════════════════════════════════════════
#  PAGE: INBOX  (2-column: list + detail)
# ═══════════════════════════════════════════════════════
tickets = st.session_state.tickets

# Top bar
st.markdown(
    '<div class="top-bar">'
    '<div class="top-logo"><div class="top-logo-icon">FT</div><span>Finance Triage</span></div>'
    '<div class="search-box"><span class="material-icons-outlined">search</span>'
    "<span>Search emails by sender, subject, or category</span></div></div>",
    unsafe_allow_html=True,
)

# Counts
total = len(tickets)
high_c = sum(1 for t in tickets if t.get("priority") == "High")
med_c  = sum(1 for t in tickets if t.get("priority") == "Medium")
low_c  = sum(1 for t in tickets if t.get("priority") == "Low")
fraud_c = sum(1 for t in tickets if t.get("category") == "Fraud")

# Stat cards
st.markdown(
    f'<div class="stat-row">'
    f'<div class="stat-card sc-total"><div class="stat-num">{total}</div><div class="stat-lbl">Total</div></div>'
    f'<div class="stat-card sc-high"><div class="stat-num">{high_c}</div><div class="stat-lbl">High</div></div>'
    f'<div class="stat-card sc-medium"><div class="stat-num">{med_c}</div><div class="stat-lbl">Medium</div></div>'
    f'<div class="stat-card sc-low"><div class="stat-num">{low_c}</div><div class="stat-lbl">Low</div></div>'
    f'<div class="stat-card sc-fraud"><div class="stat-num">{fraud_c}</div><div class="stat-lbl">Fraud</div></div>'
    f"</div>",
    unsafe_allow_html=True,
)

# Alert bars
if fraud_c:
    st.markdown(
        f'<div class="alert-bar ab-red"><span class="ab-icon">🚨</span>'
        f'<div class="ab-text"><b>{fraud_c} Fraud Alert{"s" if fraud_c > 1 else ""}!</b> '
        f"Potential fraud — immediate review required.</div></div>",
        unsafe_allow_html=True,
    )
if high_c:
    st.markdown(
        f'<div class="alert-bar ab-amber"><span class="ab-icon">⚡</span>'
        f'<div class="ab-text"><b>{high_c} High Priority</b> email{"s" if high_c > 1 else ""} waiting.</div></div>',
        unsafe_allow_html=True,
    )

if not tickets:
    st.markdown(
        '<div class="welcome"><div class="welcome-icon">📭</div>'
        '<div class="welcome-title">Your inbox is empty</div>'
        '<div class="welcome-sub">Click <b>Fetch Emails</b> in the sidebar to pull emails from Gmail.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ──────────────────────────────────────────────────────
#  DETAIL VIEW HELPER
# ──────────────────────────────────────────────────────
def _render_detail(ticket: dict):
    """Render the right-hand detail panel for a ticket."""
    _p   = ticket.get("priority", "Medium")
    _c   = ticket.get("category", "General")
    _st  = ticket.get("status", "New")
    _sdr = _extract_sender(ticket.get("email_body", ""), ticket.get("customer_name", ""))
    _sub = _extract_subject(ticket.get("email_body", ""))
    _tm  = _fmt_full(ticket.get("created_at"))
    _bd  = _get_body(ticket.get("email_body", ""))
    tid  = ticket.get("id", "")

    st.markdown('<div class="detail-card">', unsafe_allow_html=True)

    # Actions row
    st.markdown(
        '<div class="detail-actions">'
        '<span class="action-btn"><span class="action-icon">↩️</span> Reply</span>'
        '<span class="action-btn"><span class="action-icon">↩️</span> Reply All</span>'
        '<span class="action-btn"><span class="action-icon">↪️</span> Forward</span>'
        '<span class="action-btn"><span class="action-icon">🗑️</span> Delete</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Subject
    st.markdown(f'<div class="detail-subject-line">{_sub}</div>', unsafe_allow_html=True)

    # From + badges
    st.markdown(
        f'<div class="detail-from">'
        f"{_avatar(_sdr)}"
        f"<div>"
        f'<div class="detail-from-name">{_sdr}</div>'
        f'<div class="detail-from-time">{_tm}</div>'
        f"</div>"
        f'<div style="margin-left:auto;display:flex;gap:5px;flex-wrap:wrap;">'
        f'{_pri_badge(_p)} {_cat_badge(_c)} <span class="badge b-status">📌 {_st}</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Body
    st.markdown(f'<div class="detail-body">{_bd}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── AI Analysis ──
    st.markdown('<div class="section-hdr">🧠 AI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insight-grid">'
        f'<div class="insight-box"><div class="insight-label">Category</div>{_cat_badge(_c)}</div>'
        f'<div class="insight-box"><div class="insight-label">Sentiment</div>{_sent_badge(ticket.get("sentiment", "Neutral"))}</div>'
        f'<div class="insight-box"><div class="insight-label">Intent</div><div class="insight-value">{ticket.get("intent", "N/A")}</div></div>'
        f'<div class="insight-box"><div class="insight-label">Amount</div><div class="insight-value">{ticket.get("amount") or "N/A"}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("Customer", ticket.get("customer_name") or "N/A")
    e2.metric("Transaction ID", ticket.get("transaction_id") or "N/A")
    e3.metric("Amount", ticket.get("amount") or "N/A")

    st.markdown(f"**📝 Summary:** {ticket.get('summary', 'N/A')}")

    # ── Draft Response ──
    st.markdown('<div class="section-hdr">✏️ Draft Response</div>', unsafe_allow_html=True)
    draft = st.text_area(
        "draft",
        value=ticket.get("draft_response", ""),
        height=170,
        key=f"draft_{tid}",
        label_visibility="collapsed",
    )

    if _st in ("New", "In Progress"):
        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("✅ Approve & Send", key=f"ap_{tid}", type="primary", use_container_width=True):
                with st.spinner("Sending email…"):
                    res = _api_approve(tid)
                if res:
                    if res.get("email_sent"):
                        st.success(f"✅ Sent to {res.get('recipient', 'customer')} & closed!")
                    else:
                        st.warning("⚠️ Closed but email could not be sent.")
                    _time.sleep(1.5)
                    st.session_state.sel = None
                    st.rerun()
        with b2:
            if st.button("🗑️ Close Ticket", key=f"cl_{tid}", use_container_width=True):
                with st.spinner("Closing…"):
                    if _api_close(tid):
                        st.warning("Ticket closed without reply.")
                        st.session_state.sel = None
                        st.rerun()
        with b3:
            if st.button("← Back to list", key=f"bk_{tid}", use_container_width=True):
                st.session_state.sel = None
                st.rerun()
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"🔒 This ticket is **{_st}**.")
        with c2:
            if st.button("← Back to list", key=f"bk2_{tid}", use_container_width=True):
                st.session_state.sel = None
                st.rerun()


# ──────────────────────────────────────────────────────
#  TAB: INBOX (All Mail)
# ──────────────────────────────────────────────────────
if st.session_state.tab == "inbox":
    sorted_tix = sorted(
        tickets,
        key=lambda t: (
            _PRI_ORDER.get(t.get("priority", "Low"), 2),
            -(datetime.fromisoformat(t["created_at"]).timestamp() if t.get("created_at") else 0),
        ),
    )

    sel_id = st.session_state.sel

    if sel_id:
        list_col, detail_col = st.columns([2, 3])
    else:
        list_col = st.container()
        detail_col = None

    # ── Email list ──
    with list_col:
        groups: dict[str, list] = {}
        for t in sorted_tix:
            g = _date_group(t.get("created_at"))
            groups.setdefault(g, []).append(t)

        st.markdown('<div class="email-list">', unsafe_allow_html=True)
        for grp_name, grp_tix in groups.items():
            st.markdown(f'<div class="date-group">{grp_name}</div>', unsafe_allow_html=True)
            for tkt in grp_tix:
                tid = tkt.get("id", "")
                sender = _extract_sender(tkt.get("email_body", ""), tkt.get("customer_name", ""))
                subject = _extract_subject(tkt.get("email_body", ""))
                preview = _get_preview(tkt.get("email_body", ""))
                pri = tkt.get("priority", "Medium")
                cat = tkt.get("category", "General")
                ts = _fmt_time(tkt.get("created_at"))
                is_new = tkt.get("status") == "New"
                selected = tid == sel_id
                row_cls = "selected" if selected else ("" if is_new else "read")
                pd_cls = {"High": "pd-high", "Medium": "pd-medium", "Low": "pd-low"}.get(pri, "pd-low")

                st.markdown(
                    f'<div class="eml-row {row_cls}">'
                    f"{_avatar(sender)}"
                    f'<div class="eml-meta">'
                    f'<div class="eml-sender"><span class="p-dot {pd_cls}"></span>{sender}</div>'
                    f'<div class="eml-subject">{subject}</div>'
                    f'<div class="eml-preview">{preview}</div></div>'
                    f'<div class="eml-right">'
                    f'<div class="eml-time">{ts}</div>'
                    f"{_tag_html(cat)}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"o_{tid}", use_container_width=True):
                    st.session_state.sel = tid
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Detail column ──
    if sel_id and detail_col:
        with detail_col:
            tkt = next((t for t in tickets if t.get("id") == sel_id), None)
            if not tkt:
                st.warning("Ticket not found.")
                if st.button("← Back"):
                    st.session_state.sel = None
                    st.rerun()
                st.stop()
            _render_detail(tkt)


# ──────────────────────────────────────────────────────
#  TAB: PRIORITY QUEUE
# ──────────────────────────────────────────────────────
elif st.session_state.tab == "queue":
    # If a ticket is selected, show detail
    if st.session_state.sel:
        tkt = next((t for t in tickets if t.get("id") == st.session_state.sel), None)
        if tkt:
            _render_detail(tkt)
            st.stop()

    for icon, title, pri, qc in [
        ("🔴", "High Priority — Immediate Attention", "High", "qc-r"),
        ("🟡", "Medium Priority — Review Soon", "Medium", "qc-a"),
        ("🟢", "Low Priority — When Available", "Low", "qc-g"),
    ]:
        grp = [t for t in tickets if t.get("priority") == pri]
        st.markdown(
            f'<div class="queue-sec"><div class="queue-hdr">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span class="queue-title">{title}</span>'
            f'<span class="queue-cnt {qc}">{len(grp)}</span>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if grp:
            for t in grp:
                sender = _extract_sender(t.get("email_body", ""), t.get("customer_name", ""))
                subject = _extract_subject(t.get("email_body", ""))
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"{_avatar(sender)} **{sender}** — {subject}", unsafe_allow_html=True)
                    st.caption(t.get("summary", "")[:100])
                with c2:
                    st.markdown(_cat_badge(t.get("category", "General")), unsafe_allow_html=True)
                with c3:
                    if st.button("Open →", key=f"q_{t['id']}", use_container_width=True):
                        st.session_state.sel = t["id"]
                        st.session_state.tab = "inbox"
                        st.rerun()
        else:
            st.caption(f"  No {pri.lower()} priority emails ✅")
        st.markdown("")


# ──────────────────────────────────────────────────────
#  TAB: BY CATEGORY
# ──────────────────────────────────────────────────────
elif st.session_state.tab == "category":
    if st.session_state.sel:
        tkt = next((t for t in tickets if t.get("id") == st.session_state.sel), None)
        if tkt:
            _render_detail(tkt)
            st.stop()

    for icon, title, cat, cc in [
        ("🚨", "Fraud", "Fraud", "qc-r"),
        ("💳", "Payment Issues", "Payment Issue", "qc-a"),
        ("📋", "General Inquiries", "General", "qc-g"),
    ]:
        grp = [t for t in tickets if t.get("category") == cat]
        st.markdown(
            f'<div class="queue-sec"><div class="queue-hdr">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span class="queue-title">{title}</span>'
            f'<span class="queue-cnt {cc}">{len(grp)}</span>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if grp:
            for t in grp:
                sender = _extract_sender(t.get("email_body", ""), t.get("customer_name", ""))
                subject = _extract_subject(t.get("email_body", ""))
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"{_avatar(sender)} **{sender}** — {subject}", unsafe_allow_html=True)
                    st.caption(t.get("summary", "")[:100])
                with c2:
                    st.markdown(_pri_badge(t.get("priority", "Medium")), unsafe_allow_html=True)
                with c3:
                    if st.button("Open →", key=f"c_{t['id']}", use_container_width=True):
                        st.session_state.sel = t["id"]
                        st.session_state.tab = "inbox"
                        st.rerun()
        else:
            st.caption(f"  No {title.lower()} ✅")
        st.markdown("")


# ──────────────────────────────────────────────────────
#  TAB: ALERTS
# ──────────────────────────────────────────────────────
elif st.session_state.tab == "alerts":
    if st.session_state.sel:
        tkt = next((t for t in tickets if t.get("id") == st.session_state.sel), None)
        if tkt:
            _render_detail(tkt)
            st.stop()

    alert_tix = []
    seen: set = set()
    for t in tickets:
        if t["id"] in seen:
            continue
        if (
            t.get("category") == "Fraud"
            or t.get("priority") == "High"
            or t.get("sentiment") in ("Urgent", "Negative")
        ):
            seen.add(t["id"])
            alert_tix.append(t)

    if not alert_tix:
        st.markdown(
            '<div class="alert-bar ab-green"><span class="ab-icon">✅</span>'
            "<div class=\"ab-text\"><b>All clear!</b> No alerts at this time.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-bar ab-red"><span class="ab-icon">🔔</span>'
            f'<div class="ab-text"><b>{len(alert_tix)} Active Alert{"s" if len(alert_tix) > 1 else ""}</b> — '
            f"Emails needing immediate attention.</div></div>",
            unsafe_allow_html=True,
        )

        for t in alert_tix:
            sender = _extract_sender(t.get("email_body", ""), t.get("customer_name", ""))
            subject = _extract_subject(t.get("email_body", ""))
            pri = t.get("priority", "Medium")
            cat = t.get("category", "General")
            snt = t.get("sentiment", "Neutral")

            reasons = []
            if cat == "Fraud":
                reasons.append("🚨 Fraud")
            if pri == "High":
                reasons.append("🔴 High Priority")
            if snt in ("Urgent", "Negative"):
                reasons.append(f"😠 {snt}")

            st.markdown(
                f'<div class="queue-sec" style="border-left:3px solid #ef4444;">'
                f'<div style="display:flex;gap:14px;align-items:center;">'
                f"{_avatar(sender)}"
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-weight:700;color:#1a1a2e;font-size:.88rem;">{sender}</div>'
                f'<div style="color:#6b7280;font-size:.82rem;margin-top:1px;">{subject}</div>'
                f'<div style="color:#9ca3af;font-size:.76rem;margin-top:2px;">{t.get("summary", "")[:100]}</div>'
                f'<div style="margin-top:6px;display:flex;gap:5px;flex-wrap:wrap;">'
                f'{_pri_badge(pri)} {_cat_badge(cat)} {_sent_badge(snt)}'
                f"</div></div>"
                f'<div style="text-align:right;">'
                f'<div style="font-size:.68rem;color:#9ca3af;">{_fmt_time(t.get("created_at"))}</div>'
                f'<div style="font-size:.64rem;color:#dc2626;font-weight:600;margin-top:3px;">{"  •  ".join(reasons)}</div>'
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
            if st.button("View Details →", key=f"al_{t['id']}", use_container_width=True):
                st.session_state.sel = t["id"]
                st.session_state.tab = "inbox"
                st.rerun()

