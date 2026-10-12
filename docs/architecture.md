# StudyCompanion AI – Technical Architecture & Grounding Documentation

**Multimodal AI Hackathon 2026 – Track D: Personalized Tutoring & Adaptive Learning**  
**Author / Team:** Roy Khaemba (Solo Entry)  
**Target Domain:** Education & Adaptive Learning  

---

## 1. System Overview

**StudyCompanion AI** is an intelligent multimodal study companion engineered to solve the disconnect between fragmented course materials (lecture videos, textbooks, and slide decks) and generative AI tutoring. Rather than relying on unconstrained LLM completions, StudyCompanion establishes a **strict source-grounded knowledge base** where every generated explanation, quiz question, and concept breakdown is directly tied to an auditable origin (page number, slide number, or video timestamp).

The system integrates:
1. **Multimodal Ingestion Pipeline:** Automated parsing of textbooks (PDF with image/figure descriptions), presentation decks (PPTX), and lecture recordings (video).
2. **Source-Grounded Retrieval-Augmented Generation (RAG):** Cosine-similarity vector retrieval over dense embeddings with strict source citation formatting (`[Source N: <Doc>, Page/Slide/Timestamp]`) and refusal mechanisms for out-of-curriculum queries.
3. **Adaptive Assessment Engine:** Dynamic generation of Multiple Choice (MCQ), Short Answer, and Numerical calculation problems with difficulty calibration and cryptographic hash deduplication.
4. **Bayesian Knowledge Tracing (BKT) Learner Model:** Probabilistic modeling of student concept mastery across topics with continuous Bayesian belief updating.
5. **System Evaluation Suite:** Quantitative benchmarking utilizing RAGAS-style metrics (Faithfulness, Answer Relevancy, Context Precision, and Context Recall) alongside multi-session simulated student trajectories.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                StudyCompanion AI Architecture                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘

   Multimodal Sources                       Ingestion & Structuring
  ┌──────────────────┐                     ┌───────────────────────────┐
  │ PDF Textbooks    │ ──[ PyMuPDF ]─────► │ Chunking & Vision Summary │
  │ Slide Decks      │ ──[ python-pptx ]─► │ Hierarchical Topic Tagging│ ──┐
  │ Lecture Videos   │ ──[ Audio/Frames ]► │ Temporal / Page Anchors   │   │
  └──────────────────┘                     └───────────────────────────┘   │
                                                                           ▼
                                                               ┌──────────────────────┐
                                                               │ Dense Embeddings     │
                                                               │ (text-embedding-004) │
                                                               └──────────┬───────────┘
                                                                          │
                                                                          ▼
┌───────────────────────────┐    Prompt + Retrieved Context    ┌──────────────────────┐
│ Generative Engine         │ ◄─────────────────────────────── │ ChromaDB Vector Store│
│ (Gemini 2.0 Flash)        │                                  │ (HNSW Cosine Space)  │
└─────────────┬─────────────┘                                  └──────────────────────┘
              │
              ├──► Source-Grounded Tutor Response (Citations: [Source N, Page/Slide/Time])
              ├──► Adaptive Assessment (MCQ / Numerical / Short Answer)
              │
              ▼
┌───────────────────────────┐
│ Bayesian Knowledge        │ ◄─── Quiz Performance & Student Interaction Logs
│ Tracing (BKT) Engine      │
└─────────────┬─────────────┘
              │
              ├──► Mastery Probability: P(Ln) per Topic
              └──► Personalized Study Recommendations & Weak Topic Interventions
```

---

## 2. Multimodal Knowledge Base Ingestion

### 2.1 PDF Textbook Processing
- **Text & Structure Extraction:** Ingested via `PyMuPDF` (`fitz`), traversing page-by-page to retain document structure.
- **Visual & Diagram Understanding:** Pages containing embedded images and diagrams are rasterized into high-resolution pixmaps. The visual elements are sent to the multimodal model to synthesize descriptive annotations (e.g., biological process flowcharts, circuit diagrams, geometric curves), appended directly to the page chunk as `[Visual: <Description>]`.
- **Page Anchoring:** Every content piece preserves its 1-indexed document page number.

### 2.2 Slide Deck Processing (PPTX)
- Extracted via `python-pptx`, decomposing slides into shapes, textual runs, table cells, and notes.
- Each slide is bounded as an atomic or sub-slide semantic chunk tagged with `Slide <N>`.

### 2.3 Lecture Video Processing
- Formatted with temporal anchors. Chunk units map to timestamp offsets (`Video MM:SS`), enabling instant playback positioning.

### 2.4 Hierarchical Topic & Prerequisite Tagging
- Upon ingestion, the system runs document-level topic induction to identify major conceptual modules, subtopics, and prerequisites.
- Chunks are dynamically assigned to these topic clusters, creating an organized taxonomy for selective assessment generation.

---

## 3. Source Grounding & Refusal Mechanism

### 3.1 Strict Grounding Protocol
When answering student queries:
1. Dense vector embeddings are computed via `models/text-embedding-004`.
2. The top-$k$ nearest chunks are retrieved from ChromaDB using HNSW indexing with cosine distance.
3. Chunks are formatted into structured context blocks:
   ```text
   [Source 1: Biology_Unit2.pdf, Page 42]
   <Extracted Content with visual descriptions>
   ```
4. The system prompt instructs the model to construct explanations **solely** on the basis of these citations.
5. Every asserted fact is annotated with inline citation markers `[Source N]`.

### 3.2 Out-of-Curriculum Refusal & Flagging
- If a query cannot be answered from the retrieved context (or if cosine similarity is below the relevance threshold), the tutor explicitly declines:
  > *"⚠️ This topic is not covered in your uploaded material. I can only answer questions based on your study documents."*
- This prevents hallucination and guarantees that study sessions remain strictly focused on the syllabus.

---

## 4. Adaptive Assessment Engine

### 4.1 Question Generation
- Generates three distinct question types:
  1. **Multiple Choice Questions (MCQ):** 4 options with distractor rationales and verified keys.
  2. **Short Answer Problems:** Conceptual prompts requiring concise explanation.
  3. **Numerical Problems:** Calculations requiring quantitative derivations and unit specifications.
- Every question is tagged with:
  - Concept / Topic ID
  - Difficulty Level ($1 = \text{Easy}$, $2 = \text{Medium}$, $3 = \text{Hard}$)
  - Source Reference (`Page/Slide/Time`)

### 4.2 Cross-Session Deduplication
- Each generated question is normalized and hashed using MD5.
- The assessment pipeline cross-references previous student session records, rejecting redundant questions to guarantee novel assessment items across repeated quizzes.

### 4.3 Answer Verification
- MCQs are evaluated against canonical keys.
- Open-ended and numerical answers undergo cross-validation via an LLM judge prompt that scores conceptual equivalence rather than naive surface string matching.

---

## 5. Learner Model: Bayesian Knowledge Tracing (BKT)

The learner model tracks student mastery per topic using the standard **Bayesian Knowledge Tracing** formulation:

### Mathematical Formulation
Let $L_t$ denote the event that the student has mastered topic $k$ at step $t$:
- $P(L_0) = 0.30$: Initial probability of mastery (cold-start prior)
- $P(T) = 0.10$: Probability of learning (transitioning from unmastered to mastered)
- $P(S) = 0.10$: Slip probability (student knows the concept but answers incorrectly)
- $P(G) = 0.25$: Guess probability (student does not know the concept but answers correctly)

#### Observation Update:
Upon observing student response $C_t \in \{\text{Correct}, \text{Incorrect}\}$:

$$\text{If } C_t = \text{Correct}: \quad P(L_t \mid \text{Correct}) = \frac{P(L_t)(1 - P(S))}{P(L_t)(1 - P(S)) + (1 - P(L_t))P(G)}$$

$$\text{If } C_t = \text{Incorrect}: \quad P(L_t \mid \text{Incorrect}) = \frac{P(L_t)P(S)}{P(L_t)P(S) + (1 - P(L_t))(1 - P(G))}$$

#### State Transition (Learning Update):
$$P(L_{t+1}) = P(L_t \mid C_t) + (1 - P(L_t \mid C_t)) \cdot P(T)$$

### Cold Start Strategy
For new students with zero history:
1. Prior mastery initializes to $P(L_0) = 0.30$.
2. The system serves diagnostic questions across topics to rapidly establish baseline estimates.
3. As quizzes are completed, mastery scores update and partition into **Strong Topics** ($P \ge 0.80$) and **Weak Topics** ($P < 0.60$).

---

## 6. System Evaluation & Benchmarking

The evaluation suite implements two quantitative testing modules:

1. **RAGAS-Style RAG Benchmarking:**
   - **Faithfulness:** Degree to which generated tutor statements are grounded in retrieved source chunks.
   - **Answer Relevancy:** Semantic congruence between question intent and tutor response.
   - **Context Precision:** Ratio of relevant chunks among total retrieved chunks.
   - **Context Recall:** Coverage of ground-truth syllabus points in the context.
   - **Refusal Precision:** Accurate rejection of out-of-scope/unsupported queries.

2. **Simulated Student Multi-Session Trajectories:**
   - Evaluates three student archetypes:
     - *Beginner* (Base ability $0.20$)
     - *Intermediate* (Base ability $0.50$)
     - *Advanced* (Base ability $0.80$)
   - Measured over 5 successive assessment iterations.
   - Tracks **Mastery Gain** ($\Delta P(L)$) and **Question Repetition Rate** ($< 2\%$).

---

## 7. API Specification Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, model status, and API key verification |
| `POST` | `/api/upload` | Multimodal ingestion (PDF, PPTX, Video, TXT, MD) |
| `GET` | `/api/documents` | List ingested materials, topics, and chunk counts |
| `POST` | `/api/chat` | Source-grounded tutor endpoint returning citations |
| `GET` | `/api/chat/history` | Retrieve student conversation history |
| `GET` | `/api/topics` | Enumerate all discovered topics from knowledge base |
| `POST` | `/api/quiz/generate` | Generate adaptive quiz with topic & difficulty filter |
| `POST` | `/api/quiz/submit` | Submit answers, verify correctness, and update BKT |
| `GET` | `/api/learner/profile` | Student BKT mastery vector, history, and weak topics |
| `GET` | `/api/dashboard` | Aggregated telemetry, document count, and mastery stats |
