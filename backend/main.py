import os, json, uuid, hashlib, math
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import sqlite3

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import google.generativeai as genai
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"
EMBED_MODEL = "models/text-embedding-004"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = "db.sqlite3"
CHROMA_PATH = "./chroma_db"

# ── DB Setup ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        name TEXT,
        type TEXT,
        path TEXT,
        created_at TEXT,
        chunks_count INTEGER DEFAULT 0,
        topics TEXT DEFAULT '[]'
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT,
        content TEXT,
        page_or_slide INTEGER,
        timestamp_sec REAL,
        topic TEXT,
        subtopic TEXT,
        chunk_index INTEGER
    );
    CREATE TABLE IF NOT EXISTS questions (
        id TEXT PRIMARY KEY,
        content TEXT,
        answer TEXT,
        options TEXT,
        qtype TEXT,
        topic TEXT,
        difficulty INTEGER,
        source_chunk_id TEXT,
        source_location TEXT,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS quiz_sessions (
        id TEXT PRIMARY KEY,
        student_id TEXT,
        questions TEXT,
        answers TEXT,
        score REAL,
        created_at TEXT,
        completed INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS learner_state (
        student_id TEXT PRIMARY KEY,
        mastery TEXT DEFAULT '{}',
        sessions_count INTEGER DEFAULT 0,
        created_at TEXT,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS chat_history (
        id TEXT PRIMARY KEY,
        student_id TEXT,
        role TEXT,
        content TEXT,
        citations TEXT DEFAULT '[]',
        created_at TEXT
    );
    """)
    conn.commit()
    conn.close()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = chroma_client.get_collection("study_chunks")
except Exception:
    collection = chroma_client.create_collection(
        "study_chunks", metadata={"hnsw:space": "cosine"}
    )

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="StudyCompanion AI", version="1.0.0",
              description="Multimodal AI Study Companion - Hackathon 2026 Track D")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
init_db()

# ── LLM / Embedding Helpers ───────────────────────────────────────────────────
def embed_text(text: str) -> List[float]:
    if not API_KEY:
        import random
        return [random.random() for _ in range(768)]
    result = genai.embed_content(
        model=EMBED_MODEL, content=text[:8000], task_type="retrieval_document"
    )
    return result['embedding']

def query_embed(text: str) -> List[float]:
    if not API_KEY:
        import random
        return [random.random() for _ in range(768)]
    result = genai.embed_content(
        model=EMBED_MODEL, content=text, task_type="retrieval_query"
    )
    return result['embedding']

def llm(prompt: str, system: str = "") -> str:
    if not API_KEY:
        return ("[Demo mode: GEMINI_API_KEY not set. "
                "Please set it via: flyctl secrets set GEMINI_API_KEY=your_key]")
    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system if system else None
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[LLM Error: {e}]"

# ── Document Extraction ───────────────────────────────────────────────────────
def extract_pdf(path: str) -> List[Dict]:
    try:
        import fitz
        doc = fitz.open(path)
        chunks = []
        for i, page in enumerate(doc):
            text = page.get_text()
            # Extract image descriptions if possible
            img_desc = ""
            if API_KEY:
                imgs = page.get_images()
                if imgs:
                    try:
                        pix = page.get_pixmap(dpi=72)
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                            pix.save(tf.name)
                            img_file = genai.upload_file(tf.name)
                            img_desc = llm(
                                "Briefly describe any diagrams, figures, or key visual "
                                "elements on this page (2-3 sentences)."
                            ).strip()
                            img_file.delete()
                            os.unlink(tf.name)
                    except Exception:
                        pass
            full = text.strip()
            if img_desc:
                full += f"\n[Visual: {img_desc}]"
            if full:
                chunks.append({"content": full, "page": i + 1, "timestamp": None})
        return chunks if chunks else [{"content": "Empty PDF.", "page": 1, "timestamp": None}]
    except Exception as e:
        return [{"content": f"PDF extraction error: {e}", "page": 1, "timestamp": None}]

def extract_pptx(path: str) -> List[Dict]:
    try:
        from pptx import Presentation
        prs = Presentation(path)
        chunks = []
        for i, slide in enumerate(prs.slides):
            parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    parts.append(shape.text.strip())
                # Extract table text
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            parts.append(row_text)
            content = "\n".join(parts)
            if content:
                chunks.append({"content": content, "page": i + 1, "timestamp": None})
        return chunks if chunks else [{"content": "Empty presentation.", "page": 1, "timestamp": None}]
    except Exception as e:
        return [{"content": f"PPTX extraction error: {e}", "page": 1, "timestamp": None}]

def extract_video(path: str) -> List[Dict]:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return [{
        "content": (
            f"[Video uploaded: {Path(path).name}, Size: {size_mb:.1f}MB. "
            "Transcription via Whisper/Google Speech-to-Text would run here in production. "
            "Please also upload accompanying slides or lecture notes for full functionality.]"
        ),
        "page": 1,
        "timestamp": 0.0
    }]

def extract_text(path: str) -> List[Dict]:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Split into ~1000 char chunks
        chunks = []
        step = 1000
        for i in range(0, len(content), step):
            chunk = content[i:i + step].strip()
            if chunk:
                chunks.append({"content": chunk, "page": (i // step) + 1, "timestamp": None})
        return chunks if chunks else [{"content": "Empty file.", "page": 1, "timestamp": None}]
    except Exception as e:
        return [{"content": f"Text extraction error: {e}", "page": 1, "timestamp": None}]

def extract_topics(chunks: List[Dict], doc_name: str) -> List[str]:
    if not API_KEY or not chunks:
        return ["General Content"]
    sample = "\n\n".join(c["content"][:500] for c in chunks[:6])
    prompt = (
        f"From the following content of '{doc_name}', identify the main topics. "
        "Return ONLY a JSON array of topic strings (max 8 topics), e.g. [\"Topic A\", \"Topic B\"].\n\n"
        f"{sample}"
    )
    try:
        raw = llm(prompt)
        import re
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            topics = json.loads(match.group())
            return [t.strip() for t in topics if isinstance(t, str)][:8]
    except Exception:
        pass
    return ["General Content"]

def tag_chunk_topic(content: str, topics: List[str]) -> str:
    if not topics:
        return "General"
    if not API_KEY or len(topics) <= 1:
        return topics[0]
    prompt = (
        f"Which of these topics best matches the content? Topics: {topics}\n\n"
        f"Content (first 300 chars): {content[:300]}\n\n"
        "Reply with just the topic name, nothing else."
    )
    try:
        result = llm(prompt).strip().strip('"').strip("'")
        if result in topics:
            return result
        # Fuzzy match
        for t in topics:
            if t.lower() in result.lower():
                return t
    except Exception:
        pass
    return topics[0]

# ── Bayesian Knowledge Tracing ────────────────────────────────────────────────
class BKT:
    """Bayesian Knowledge Tracing for learner modeling."""
    P_INIT  = 0.30   # Prior probability of knowing
    P_LEARN = 0.10   # Probability of transitioning from not-know to know
    P_SLIP  = 0.10   # Probability of slipping (know but answer wrong)
    P_GUESS = 0.25   # Probability of guessing (not-know but answer right)

    @classmethod
    def update(cls, p_know: float, correct: bool) -> float:
        if correct:
            likelihood_know     = 1.0 - cls.P_SLIP
            likelihood_not_know = cls.P_GUESS
        else:
            likelihood_know     = cls.P_SLIP
            likelihood_not_know = 1.0 - cls.P_GUESS

        p_not_know = 1.0 - p_know
        evidence   = likelihood_know * p_know + likelihood_not_know * p_not_know
        if evidence == 0:
            return p_know

        posterior_know = (likelihood_know * p_know) / evidence
        # Apply learning transition
        return posterior_know + (1.0 - posterior_know) * cls.P_LEARN

# ── Pydantic Models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    student_id: str = "default"
    session_id: Optional[str] = None

class QuizRequest(BaseModel):
    topics: List[str] = []
    num_questions: int = 5
    difficulty: int = 2          # 1=easy 2=medium 3=hard
    qtypes: List[str] = ["mcq", "short"]
    student_id: str = "default"

class QuizSubmission(BaseModel):
    session_id: str
    answers: Dict[str, str]     # question_id -> answer
    student_id: str = "default"

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "api_key_set": bool(API_KEY),
        "version": "1.0.0"
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    ext    = Path(file.filename).suffix.lower()
    save_path = UPLOAD_DIR / f"{doc_id}{ext}"
    content   = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Route to correct extractor
    if ext == ".pdf":
        raw_chunks, doc_type = extract_pdf(str(save_path)), "pdf"
    elif ext in (".pptx", ".ppt"):
        raw_chunks, doc_type = extract_pptx(str(save_path)), "slides"
    elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raw_chunks, doc_type = extract_video(str(save_path)), "video"
    elif ext in (".txt", ".md"):
        raw_chunks, doc_type = extract_text(str(save_path)), "text"
    else:
        raise HTTPException(400, f"Unsupported file type: {ext}. "
                            "Supported: .pdf, .pptx, .mp4, .txt, .md")

    topics = extract_topics(raw_chunks, file.filename)

    conn = get_db()
    conn.execute(
        "INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
        (doc_id, file.filename, doc_type, str(save_path),
         datetime.now().isoformat(), len(raw_chunks), json.dumps(topics))
    )

    # Embed and store each chunk
    embed_errors = 0
    for i, chunk_data in enumerate(raw_chunks):
        chunk_id = f"{doc_id}_{i}"
        topic = (
            tag_chunk_topic(chunk_data["content"], topics)
            if API_KEY and len(topics) > 1
            else (topics[0] if topics else "General")
        )
        page = chunk_data.get("page") or 0
        ts   = chunk_data.get("timestamp")

        conn.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
            (chunk_id, doc_id, chunk_data["content"], page, ts, topic, "", i)
        )
        try:
            emb = embed_text(chunk_data["content"][:2000])
            collection.add(
                ids=[chunk_id],
                embeddings=[emb],
                documents=[chunk_data["content"][:2000]],
                metadatas=[{
                    "doc_id": doc_id, "doc_name": file.filename,
                    "doc_type": doc_type, "page": page or 0,
                    "timestamp": ts or 0, "topic": topic
                }]
            )
        except Exception as e:
            embed_errors += 1
            print(f"Embed error chunk {chunk_id}: {e}")

    conn.commit()
    conn.close()

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "type": doc_type,
        "chunks": len(raw_chunks),
        "topics": topics,
        "embed_errors": embed_errors
    }

@app.get("/api/documents")
def list_documents():
    conn = get_db()
    docs = conn.execute(
        "SELECT * FROM documents ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "id": d["id"], "name": d["name"], "type": d["type"],
            "chunks": d["chunks_count"], "topics": json.loads(d["topics"]),
            "created_at": d["created_at"]
        }
        for d in docs
    ]

@app.post("/api/chat")
def chat(req: ChatRequest):
    # Retrieve relevant chunks from ChromaDB
    docs_found, metas = [], []
    try:
        q_emb   = query_embed(req.message)
        results = collection.query(query_embeddings=[q_emb], n_results=5)
        docs_found = results["documents"][0] if results["documents"] else []
        metas      = results["metadatas"][0]  if results["metadatas"]  else []
    except Exception as e:
        print(f"Retrieval error: {e}")

    # Build source context with citations
    citations     = []
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs_found, metas)):
        cite_num = i + 1
        loc      = ""
        if meta.get("doc_type") == "video" and meta.get("timestamp"):
            ts  = int(meta["timestamp"])
            loc = f"timestamp {ts//60}:{ts%60:02d}"
        elif meta.get("page"):
            prefix = "Slide" if meta.get("doc_type") == "slides" else "Page"
            loc    = f"{prefix} {meta['page']}"

        context_parts.append(
            f"[Source {cite_num}: {meta.get('doc_name','Unknown')}"
            + (f", {loc}" if loc else "") + f"]\n{doc}"
        )
        citations.append({
            "id": cite_num, "doc_name": meta.get("doc_name"),
            "doc_type": meta.get("doc_type"), "page": meta.get("page"),
            "timestamp": meta.get("timestamp"), "location": loc,
            "topic": meta.get("topic")
        })

    context = "\n\n---\n\n".join(context_parts)

    # Fetch recent chat history for context
    conn = get_db()
    history_rows = conn.execute(
        "SELECT role, content FROM chat_history "
        "WHERE student_id=? ORDER BY created_at DESC LIMIT 8",
        (req.student_id,)
    ).fetchall()
    hist_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in reversed(history_rows)
    )

    system_prompt = """You are StudyCompanion AI, a precise and helpful academic tutor.
RULES:
1. Answer ONLY using the provided SOURCE MATERIAL.
2. Cite every claim with [Source N] where N matches the source number.
3. If the question cannot be answered from the sources, say exactly:
   "⚠️ This topic is not covered in your uploaded material. I can only answer questions based on your study documents."
4. Be clear, educational, and structured in your responses.
5. Never make up information not present in the sources."""

    if context:
        prompt = (
            f"Previous conversation:\n{hist_text}\n\n"
            f"SOURCE MATERIAL:\n{context}\n\n"
            f"STUDENT QUESTION: {req.message}\n\n"
            "Answer based ONLY on the SOURCE MATERIAL above. Cite sources as [Source N]."
        )
    else:
        prompt = (
            f"Previous conversation:\n{hist_text}\n\n"
            f"STUDENT QUESTION: {req.message}\n\n"
            "No documents have been uploaded yet. Tell the student to upload study materials first."
        )

    answer = llm(prompt, system_prompt)

    # Persist to history
    conn.execute(
        "INSERT INTO chat_history VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), req.student_id, "user",
         req.message, "[]", datetime.now().isoformat())
    )
    conn.execute(
        "INSERT INTO chat_history VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()), req.student_id, "assistant",
         answer, json.dumps(citations), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    return {
        "answer": answer,
        "citations": citations,
        "has_sources": bool(context)
    }

@app.get("/api/chat/history")
def chat_history(student_id: str = "default", limit: int = 50):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM chat_history WHERE student_id=? ORDER BY created_at ASC LIMIT ?",
        (student_id, limit)
    ).fetchall()
    conn.close()
    return [
        {
            "role": r["role"], "content": r["content"],
            "citations": json.loads(r["citations"]),
            "created_at": r["created_at"]
        }
        for r in rows
    ]

@app.post("/api/quiz/generate")
def generate_quiz(req: QuizRequest):
    conn = get_db()

    # Select relevant chunks
    if req.topics:
        placeholders = ",".join("?" for _ in req.topics)
        chunks_rows = conn.execute(
            f"SELECT * FROM chunks WHERE topic IN ({placeholders}) ORDER BY RANDOM() LIMIT 25",
            req.topics
        ).fetchall()
    else:
        chunks_rows = conn.execute(
            "SELECT * FROM chunks ORDER BY RANDOM() LIMIT 25"
        ).fetchall()

    if not chunks_rows:
        conn.close()
        raise HTTPException(
            400, "No study material found. Please upload documents first."
        )

    # Get previously used question hashes to avoid repetition
    prev_sessions = conn.execute(
        "SELECT questions FROM quiz_sessions WHERE student_id=?",
        (req.student_id,)
    ).fetchall()
    used_q_hashes = set()
    for s in prev_sessions:
        q_ids = json.loads(s["questions"])
        for qid in q_ids:
            q_row = conn.execute(
                "SELECT content FROM questions WHERE id=?", (qid,)
            ).fetchone()
            if q_row:
                used_q_hashes.add(hashlib.md5(q_row["content"].encode()).hexdigest())

    difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
    diff_label     = difficulty_map.get(req.difficulty, "medium")
    generated_qs   = []

    for chunk_row in chunks_rows:
        if len(generated_qs) >= req.num_questions:
            break

        content = chunk_row["content"]
        qtype   = req.qtypes[len(generated_qs) % len(req.qtypes)] if req.qtypes else "mcq"

        if qtype == "mcq":
            prompt = (
                f"Create one {diff_label} multiple-choice question from this study content.\n"
                "Return ONLY valid JSON (no markdown) in this exact format:\n"
                '{"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],'
                '"answer":"A","explanation":"..."}\n\n'
                f"Content:\n{content[:800]}"
            )
        elif qtype == "numerical":
            prompt = (
                f"Create one {diff_label} numerical or calculation question from this content. "
                "If no numbers/formulas exist, create a short answer question instead.\n"
                "Return ONLY valid JSON:\n"
                '{"question":"...","answer":"...","explanation":"..."}\n\n'
                f"Content:\n{content[:800]}"
            )
        else:  # short answer
            prompt = (
                f"Create one {diff_label} short-answer question from this study content.\n"
                "Return ONLY valid JSON:\n"
                '{"question":"...","answer":"...","explanation":"..."}\n\n'
                f"Content:\n{content[:800]}"
            )

        try:
            raw = llm(prompt)
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                continue
            q_data  = json.loads(match.group())
            q_text  = q_data.get("question", "")
            q_hash  = hashlib.md5(q_text.encode()).hexdigest()

            if q_hash in used_q_hashes or not q_text:
                continue

            used_q_hashes.add(q_hash)
            q_id = str(uuid.uuid4())

            # Source location string
            loc_str = ""
            if chunk_row["timestamp_sec"]:
                ts      = int(chunk_row["timestamp_sec"])
                loc_str = f"Video {ts//60}:{ts%60:02d}"
            elif chunk_row["page_or_slide"]:
                prefix  = "Slide" if chunk_row["doc_id"] else "Page"
                loc_str = f"{prefix} {chunk_row['page_or_slide']}"

            conn.execute(
                "INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    q_id, q_text, q_data.get("answer", ""),
                    json.dumps(q_data.get("options", [])),
                    qtype, chunk_row["topic"], req.difficulty,
                    chunk_row["id"], loc_str, datetime.now().isoformat()
                )
            )

            generated_qs.append({
                "id": q_id,
                "question": q_text,
                "type": qtype,
                "options": q_data.get("options", []),
                "topic": chunk_row["topic"],
                "difficulty": req.difficulty,
                "source_location": loc_str,
                "explanation": q_data.get("explanation", "")
            })
        except Exception as e:
            print(f"Question generation error: {e}")
            continue

    if not generated_qs:
        conn.close()
        raise HTTPException(
            500,
            "Could not generate questions. Ensure GEMINI_API_KEY is set and material is uploaded."
        )

    session_id = str(uuid.uuid4())
    q_ids      = [q["id"] for q in generated_qs]
    conn.execute(
        "INSERT INTO quiz_sessions VALUES (?,?,?,?,?,?,?)",
        (session_id, req.student_id, json.dumps(q_ids),
         "{}", 0.0, datetime.now().isoformat(), 0)
    )
    conn.commit()
    conn.close()

    return {"session_id": session_id, "questions": generated_qs}

@app.post("/api/quiz/submit")
def submit_quiz(submission: QuizSubmission):
    conn = get_db()

    session = conn.execute(
        "SELECT * FROM quiz_sessions WHERE id=?", (submission.session_id,)
    ).fetchone()
    if not session:
        conn.close()
        raise HTTPException(404, "Session not found")

    q_ids         = json.loads(session["questions"])
    results       = []
    correct_count = 0
    topic_results: Dict[str, Dict] = {}

    for q_id in q_ids:
        q = conn.execute("SELECT * FROM questions WHERE id=?", (q_id,)).fetchone()
        if not q:
            continue

        student_ans = submission.answers.get(q_id, "").strip()
        correct_ans = q["answer"].strip()

        # Answer verification
        if q["qtype"] == "mcq":
            is_correct = (
                student_ans[:1].upper() == correct_ans[:1].upper()
            )
            feedback = ""
        else:
            verify_prompt = (
                f"Is this student answer correct?\n"
                f"Question: {q['content']}\n"
                f"Correct answer: {correct_ans}\n"
                f"Student answer: {student_ans}\n\n"
                "Reply with ONLY JSON: {\"correct\":true,\"feedback\":\"brief explanation\"}"
            )
            try:
                raw = llm(verify_prompt)
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                v    = json.loads(match.group()) if match else {}
                is_correct = bool(v.get("correct", False))
                feedback   = v.get("feedback", "")
            except Exception:
                is_correct = student_ans.lower()[:20] in correct_ans.lower()
                feedback   = ""

        if is_correct:
            correct_count += 1

        topic = q["topic"] or "General"
        if topic not in topic_results:
            topic_results[topic] = {"correct": 0, "total": 0}
        topic_results[topic]["total"] += 1
        if is_correct:
            topic_results[topic]["correct"] += 1

        results.append({
            "question_id": q_id,
            "question": q["content"],
            "student_answer": student_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct,
            "feedback": feedback,
            "topic": topic,
            "source_location": q["source_location"]
        })

    score = correct_count / len(q_ids) if q_ids else 0.0

    # Update BKT mastery
    learner = conn.execute(
        "SELECT * FROM learner_state WHERE student_id=?", (submission.student_id,)
    ).fetchone()
    mastery = json.loads(learner["mastery"]) if learner else {}

    for topic, tr in topic_results.items():
        p = mastery.get(topic, BKT.P_INIT)
        # Update BKT once per question answered
        for _ in range(tr["total"]):
            is_corr = (tr["correct"] / tr["total"]) >= 0.5
            p = BKT.update(p, is_corr)
        mastery[topic] = round(p, 4)

    now = datetime.now().isoformat()
    if learner:
        conn.execute(
            "UPDATE learner_state "
            "SET mastery=?, sessions_count=sessions_count+1, updated_at=? "
            "WHERE student_id=?",
            (json.dumps(mastery), now, submission.student_id)
        )
    else:
        conn.execute(
            "INSERT INTO learner_state VALUES (?,?,?,?,?)",
            (submission.student_id, json.dumps(mastery), 1, now, now)
        )

    conn.execute(
        "UPDATE quiz_sessions SET score=?, completed=1, answers=? WHERE id=?",
        (score, json.dumps(submission.answers), submission.session_id)
    )
    conn.commit()

    weak_topics = [t for t, v in mastery.items() if v < 0.6]
    conn.close()

    return {
        "session_id": submission.session_id,
        "score": round(score * 100, 1),
        "correct": correct_count,
        "total": len(q_ids),
        "results": results,
        "topic_performance": topic_results,
        "mastery": mastery,
        "weak_topics": weak_topics
    }

@app.get("/api/learner/profile")
def learner_profile(student_id: str = "default"):
    conn = get_db()
    learner  = conn.execute(
        "SELECT * FROM learner_state WHERE student_id=?", (student_id,)
    ).fetchone()
    sessions = conn.execute(
        "SELECT score, created_at FROM quiz_sessions "
        "WHERE student_id=? AND completed=1 ORDER BY created_at ASC",
        (student_id,)
    ).fetchall()
    conn.close()

    if not learner:
        return {
            "student_id": student_id, "mastery": {},
            "sessions_count": 0, "progress_history": [],
            "weak_topics": [], "strong_topics": []
        }

    mastery = json.loads(learner["mastery"])
    return {
        "student_id": student_id,
        "mastery": mastery,
        "sessions_count": learner["sessions_count"],
        "progress_history": [
            {"score": round(s["score"] * 100, 1), "date": s["created_at"]}
            for s in sessions
        ],
        "weak_topics":   [t for t, v in mastery.items() if v < 0.6],
        "strong_topics": [t for t, v in mastery.items() if v >= 0.8],
        "updated_at": learner["updated_at"]
    }

@app.get("/api/topics")
def get_topics():
    conn = get_db()
    docs = conn.execute("SELECT topics FROM documents").fetchall()
    conn.close()
    all_topics: set = set()
    for d in docs:
        all_topics.update(json.loads(d["topics"]))
    return {"topics": sorted(list(all_topics))}

@app.get("/api/dashboard")
def dashboard(student_id: str = "default"):
    conn = get_db()
    docs_count    = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()["c"]
    chunks_count  = conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
    sessions_done = conn.execute(
        "SELECT COUNT(*) as c FROM quiz_sessions WHERE student_id=? AND completed=1",
        (student_id,)
    ).fetchone()["c"]
    learner = conn.execute(
        "SELECT mastery FROM learner_state WHERE student_id=?", (student_id,)
    ).fetchone()
    conn.close()

    mastery     = json.loads(learner["mastery"]) if learner else {}
    avg_mastery = sum(mastery.values()) / len(mastery) if mastery else 0.0

    return {
        "documents_count": docs_count,
        "chunks_count":    chunks_count,
        "quiz_sessions":   sessions_done,
        "avg_mastery":     round(avg_mastery * 100, 1),
        "mastery_by_topic": {k: round(v * 100, 1) for k, v in mastery.items()},
        "weak_topics": [t for t, v in mastery.items() if v < 0.6]
    }

# ── Static File Serving ───────────────────────────────────────────────────────
frontend_html = Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/")
def read_root():
    if frontend_html.exists():
        return FileResponse(str(frontend_html))
    return {"message": "StudyCompanion AI API is running. Frontend not found."}

@app.get("/{path:path}")
def catch_all(path: str):
    if frontend_html.exists():
        return FileResponse(str(frontend_html))
    raise HTTPException(404, "Not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
