# 🎥 YouTube RAG Assistant — Production - Style  GenAI Engineer Architecture

### LangChain · OpenAI · ChromaDB · Streamlit · Modular Architecture

An end-to-end **Retrieval-Augmented Generation (RAG)** system that allows users to ask grounded questions about any YouTube video using transcript-based semantic search.

This repository demonstrates **production-style GenAI engineering practices**, including modular design, metadata-aware retrieval, vector databases, and context-grounded LLM responses.

---

# 🚀 Features

✅ Accepts both **YouTube URL** and **Video ID**
✅ Automatic transcript extraction
✅ Language detection support (optional extension)
✅ Semantic chunking using RecursiveCharacterTextSplitter
✅ OpenAI embeddings + vector search (ChromaDB)
✅ Context-aware answers using RAG pipeline
✅ Clean architecture (UI + services + prompts separation)
✅ Streamlit interactive UI

---

# 🧠 Architecture Overview

User Question
↓
Streamlit UI
↓
YouTube Transcript Service
↓
Chunking + Metadata
↓
Embeddings (OpenAI)
↓
Chroma Vector Store
↓
Retriever
↓
LLM (Context-Grounded Prompt)
↓
Answer

---

# 🧱 Project Structure

youtube_rag_app/

app.py                 → Streamlit UI
services/
 youtube_service.py   → Fetch transcript logic
 rag_service.py       → Chunking + embedding + retrieval pipeline
 utils.py             → Helpers (video id extraction, formatting)

prompts/
 prompts.py           → Prompt templates

requirements.txt
README.md

---

# ⚙️ Installation

## 1️⃣ Clone Repository

git clone https://github.com/YOUR_USERNAME/youtube-rag-app.git
cd youtube-rag-app

---

## 2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate

---

## 3️⃣ Install Dependencies

pip install -r requirements.txt

---

## 4️⃣ Configure API Key

Create a `.env` file:

OPENAI_API_KEY=your_key_here

---

# ▶️ Run Application

streamlit run app.py

Open browser:

http://localhost:8501

---

# 🔎 How the RAG Pipeline Works

### Step 1 — Transcript Ingestion

The system extracts YouTube transcripts via youtube-transcript-api.

### Step 2 — Semantic Chunking

Transcript text is split into smaller contextual chunks using RecursiveCharacterTextSplitter.

### Step 3 — Embeddings

Each chunk is converted into semantic vectors using:

OpenAI → text-embedding-3-small

### Step 4 — Vector Indexing

Chunks are stored in ChromaDB for fast similarity search.

### Step 5 — Retrieval

User queries retrieve relevant transcript chunks using semantic similarity.

### Step 6 — LLM Answer Generation

The LLM receives:

Context + Question

Prompt rules enforce:

* Answer only from transcript context
* Avoid hallucination
* Structured output

---

# 🧪 Example Questions

Summarize this video
What is DeepMind?
Explain Medallion Architecture mentioned in the video
What topics were discussed around minute 20?

---

# 🛠 Tech Stack

Python
LangChain
OpenAI API
ChromaDB
Streamlit
YouTube Transcript API

---

# 🧩 Engineering Decisions

### Why RAG instead of fine-tuning?

* No retraining required
* Lower cost
* Dynamic knowledge updates

### Why ChromaDB?

* Local persistence
* Lightweight vector storage
* Fast setup for GenAI apps

### Why Modular Structure?

* Easier testing
* Cleaner architecture
* Future backend scalability (FastAPI / Agents)

---

# 📈 Future Improvements

Multi-language translation pipeline
Hybrid retrieval (BM25 + Vector)
Multi-query retrieval
Agent workflows (LangGraph)
Token usage observability
Docker deployment

---

# 👨‍💻 Author

Built as part of a **Senior GenAI Engineering / Architect learning roadmap** focused on:

RAG Systems
LLM Architecture
Production AI Design Patterns

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub!
