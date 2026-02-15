# 🏦 Finance Support Triage Agent

An AI-powered system that fetches customer support emails, classifies them by priority and category, and drafts professional replies — all from a single dashboard.

![AI Powered](https://img.shields.io/badge/AI-Powered-blueviolet?style=flat-square)
![Llama 3.3 70B](https://img.shields.io/badge/Llama_3.3-70B-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)

---

## ✨ Features

- **Auto-fetch emails** from Gmail via IMAP (catches auto-read emails using date-based search)
- **AI analysis** — sentiment, intent, priority, category, entity extraction in a single LLM call
- **Auto-generated draft replies** tailored to Fraud / Payment Issue / General categories
- **Approve & send** replies directly via Gmail SMTP from the dashboard
- **OCR support** — upload scanned documents/screenshots for AI processing
- **Modern dashboard** — Blox-style UI with email list, detail panel, priority queue, alerts
- **Auto-refresh** — dashboard polls for new tickets every 30 seconds
- **Duplicate detection** — skips already-processed emails automatically

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI Model** | Llama 3.3 70B Versatile (via [Groq](https://groq.com/)) |
| **AI Framework** | [LangChain](https://langchain.com/) — structured output + prompt engineering |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) (Python) |
| **Frontend** | [Streamlit](https://streamlit.io/) + streamlit-autorefresh |
| **Database** | [PostgreSQL](https://www.postgresql.org/) (Aiven Cloud) + SQLAlchemy ORM |
| **Email In** | Gmail IMAP (SSL, port 993) |
| **Email Out** | Gmail SMTP (TLS, port 587) |
| **OCR** | [EasyOCR](https://github.com/JaidedAI/EasyOCR) + Pillow |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) |

---

## 📂 Project Structure

```
finance-support-triage-agent/
├── backend/
│   ├── main.py             # FastAPI app — REST endpoints + SMTP sending
│   ├── agent.py            # AI analysis + draft generation (LangChain + Groq)
│   ├── models.py           # SQLAlchemy ORM models
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── database.py         # DB engine & session
│   ├── email_ingestion.py  # Standalone IMAP polling service
│   ├── ocr.py              # EasyOCR image-to-text
│   ├── schema.sql          # Raw SQL schema
│   ├── create_tables.py    # DB table creation script
│   ├── requirements.txt    # Backend dependencies
│   └── .env                # API keys, DB URL, email credentials
├── frontend/
│   ├── app.py              # Streamlit dashboard (Blox-style UI)
│   └── requirements.txt    # Frontend dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
python -m venv backend/venv
# Windows: backend\venv\Scripts\activate
# macOS/Linux: source backend/venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Configure environment

Create `backend/.env`:

```env
DATABASE_URL=postgresql://user:password@host:port/dbname?sslmode=require
GROQ_API_KEY=gsk_your_key_here
EMAIL_USER=yourname@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop   # Gmail App Password (not regular password)
```

### 3. Set up database

```bash
cd backend
python create_tables.py
```

### 4. Run

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py --server.port 8501
```

Open **http://localhost:8501** 🎉

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/analyze` | Analyse email → structured triage result |
| `POST` | `/process_ticket` | Analyse + save ticket + generate draft |
| `POST` | `/process_ticket_image` | OCR image → analyse → save ticket |
| `GET` | `/tickets` | List tickets (filter: `?status=New`) |
| `GET` | `/tickets/{id}` | Get single ticket |
| `POST` | `/approve_ticket/{id}` | Send reply via SMTP + close ticket |
| `PATCH` | `/tickets/{id}/reject` | Close without reply |
| `POST` | `/fetch_emails` | Fetch from Gmail + process with AI |

---

## 🧠 How the AI Works

1. Email text → **LangChain prompt** → **Groq Llama 3.3 70B**
2. Single LLM call returns **structured JSON** (analysis + draft reply)
3. Output is validated against a **Pydantic schema** — no regex parsing
4. **SHA-256 caching** skips the LLM for duplicate emails
5. **Template fallback** generates a basic draft if the LLM is unavailable

---

## � Deploy to Streamlit Cloud

The project includes a **self-contained** `streamlit_app.py` at the repo root that embeds all backend logic (AI agent, SQLite database, Gmail integration) — **no separate FastAPI server or PostgreSQL needed**.

### Steps

1. **Push to GitHub** — push this repo to a public or private GitHub repository.

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and click **"New app"**.

3. **Configure the app:**
   | Setting | Value |
   |---------|-------|
   | Repository | `your-username/finance-support-triage-agent` |
   | Branch | `main` |
   | Main file path | `streamlit_app.py` |

4. **Add Secrets** — in the Streamlit Cloud dashboard, go to **App Settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key"
   EMAIL_USER = "yourname@gmail.com"
   EMAIL_PASSWORD = "abcd efgh ijkl mnop"
   ```

5. **Deploy** — click **Deploy!** and your app will be live in ~2 minutes.

### What Changed for Cloud Deployment

| Feature | Local (FastAPI + Streamlit) | Streamlit Cloud |
|---------|---------------------------|-----------------|
| Database | PostgreSQL (Aiven) | SQLite (in-container) |
| Backend | FastAPI on port 8000 | Embedded in `streamlit_app.py` |
| Secrets | `.env` file | `st.secrets` (dashboard) |
| Requirements | `backend/requirements.txt` + `frontend/requirements.txt` | Single `requirements.txt` at root |

> **Note:** On Streamlit Cloud the SQLite database resets when the app restarts. For persistent storage, connect an external database.

---

## �📄 License

MIT