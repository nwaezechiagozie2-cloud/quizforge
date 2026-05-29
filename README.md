# ⚡ QuizForge

AI-powered quiz generator that turns any document into an interactive multiple-choice quiz.

##visit here: https://quizforge-five.vercel.app/

Built with **FastAPI**, **LangGraph**, and **NVIDIA NIM** (OpenAI-compatible API).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-orange)

---

## 🏗️ Architecture

```
[User Upload (PDF/TXT/Paste)]
        │
        ▼
┌──────────────────────────────────────┐
│  FastAPI Backend                     │
│                                      │
│  LangGraph State Machine:            │
│                                      │
│  START ──► [Text Processor Node]     │
│                    │                 │
│                    ▼                 │
│           [Quiz Generator Node]      │
│            (NVIDIA NIM / GPT)        │
│                    │                 │
│                    ▼                 │
│               END (JSON)             │
└──────────────────────────────────────┘
        │
        ▼
  [5 MCQ Quiz in Browser]
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/nwaezechiagozie2-cloud/quizforge.git
cd quizforge
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file:

```env
NVIDIA_API_KEY=your-nvidia-nim-api-key
```

Get a free key at [build.nvidia.com](https://build.nvidia.com/).

### 3. Run

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

## 📖 Usage

1. **Upload** a PDF or TXT file (or paste text directly)
2. Click **"Generate Quiz"**
3. Answer 5 AI-generated multiple-choice questions
4. See your score at the end

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| AI Orchestration | LangGraph (2-node state machine) |
| LLM | NVIDIA NIM (`openai/gpt-oss-20b`) |
| PDF Parsing | PyPDF2 |
| Frontend | Vanilla HTML/CSS/JS |

## 📁 Project Structure

```
quizforge/
├── app.py              # FastAPI backend + LangGraph agent
├── static/
│   └── index.html      # Frontend UI
├── .env                # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## 📄 API

### `POST /generate-quiz`

Accepts multipart form data:

- `file` — PDF or TXT file upload
- `text` — Raw text string (alternative to file)

**Response:**

```json
{
  "status": "success",
  "quiz": [
    {
      "id": 1,
      "question": "What is...",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "A"
    }
  ]
}
```

## 📝 License

MIT
