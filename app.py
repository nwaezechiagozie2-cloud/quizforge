from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import PyPDF2
import io
import os

app = FastAPI(title="QuizForge", description="AI-powered quiz generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Schemas ----------
class QuizQuestion(BaseModel):
    id: int
    question: str
    options: List[str] = Field(description="Exactly 4 multiple choice options")
    correct_answer: str = Field(description="The exact string matching the correct option")

class QuizOutput(BaseModel):
    quiz: List[QuizQuestion]

# ---------- LangGraph State ----------
class AgentState(TypedDict):
    raw_text: str
    quiz_json: List[dict]

# ---------- LangGraph Nodes ----------
def process_text_node(state: AgentState):
    """Node 1: Clean & truncate text for token limits."""
    text = state["raw_text"].strip()
    # Truncate to ~8000 chars to stay within token limits
    text = text[:8000]
    return {"raw_text": text}

def generate_quiz_node(state: AgentState):
    """Node 2: Use NVIDIA NIM with structured output to generate quiz."""
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        temperature=0.7,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    structured_llm = llm.with_structured_output(QuizOutput)

    prompt = (
        "You are an expert quiz master. Based on the following text, generate exactly 5 "
        "high-quality multiple choice questions that test understanding of the key concepts. "
        "Each question must have exactly 4 options with one correct answer. "
        "Make the questions challenging but fair.\n\n"
        f"TEXT:\n{state['raw_text']}"
    )

    response = structured_llm.invoke(prompt)
    return {"quiz_json": [q.model_dump() for q in response.quiz]}

# ---------- Build the Graph ----------
workflow = StateGraph(AgentState)
workflow.add_node("processor", process_text_node)
workflow.add_node("generator", generate_quiz_node)

workflow.add_edge(START, "processor")
workflow.add_edge("processor", "generator")
workflow.add_edge("generator", END)

quiz_agent = workflow.compile()

# ---------- API Endpoints ----------
@app.post("/generate-quiz")
async def generate_quiz(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """Accept a file upload (PDF/TXT) or raw text, return 5 quiz questions."""
    raw_text = ""

    if file:
        content = await file.read()
        if file.filename and file.filename.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                raw_text += page.extract_text() or ""
        else:
            raw_text = content.decode("utf-8")
    elif text:
        raw_text = text
    else:
        return {"status": "error", "message": "No file or text provided"}

    if not raw_text.strip():
        return {"status": "error", "message": "Extracted text is empty"}

    # Run the LangGraph agent
    initial_state: AgentState = {"raw_text": raw_text, "quiz_json": []}
    try:
        result = quiz_agent.invoke(initial_state)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"status": "error", "message": "Gemini API rate limit hit. Please wait 1 minute and try again."}
        return {"status": "error", "message": f"AI generation failed: {error_msg[:200]}"}

    return {"status": "success", "quiz": result["quiz_json"]}

# ---------- Serve Frontend ----------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
