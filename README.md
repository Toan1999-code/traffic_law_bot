# Traffic Law AI Assistant (RAG-based)

This project implements an intelligent domain-specific AI assistant designed to answer questions about Vietnam’s:

- **2024 Road Traffic Law** (Luật 36/2024/QH15)
- **Decree 168/2024/NĐ-CP** (administrative penalties & driving-license point system)

It uses **Retrieval-Augmented Generation (RAG)** to ensure responses are grounded strictly in the original legal texts.

## ⭐ Features

### ✅ 1. Document Processing & Chunking
- Parse DOCX legal documents.
- Split into structured semantic chunks.
- Extract metadata (law name, article, clause, chapter, source file).
- Use overlapping chunks for better semantic continuity.
- Export to `.jsonl` for ingestion.

### ✅ 2. Vector Database (ChromaDB)
- Store embeddings created via **OpenAI text-embedding-3-small**.
- Persistent on-disk ChromaDB collection.
- Fast semantic search.
- Supports multi-document ingestion (Law + Decree).

### ✅ 3. RAG Pipeline
- Question → Embedding → Retrieval → Prompt Construction → Response.
- LLM answers **only using retrieved context**.
- Cosine-distance threshold to detect low-confidence matches.
- Auto-citation (Điều/Khoản + document source).

### ✅ 4. Web Interface (Flask + HTML/JS)
- Modern ChatGPT-style UI.
- Scrollable chat history.
- Rename conversation.
- Delete conversation.
- Real-time backend API integration.

### ✅ 5. Evaluation Dataset (50 Test Queries)
- Includes direct questions, ambiguous queries, misleading questions, and domain traps.
- Used for retrieval & hallucination benchmarking.

## 🛠 Tech Stack
- Python 3.12
- Flask
- OpenAI API
- ChromaDB
- python-docx
- dotenv
- HTML/CSS/JS

## 📁 Project Structure

```
project/
│── documents/
│   ├── Law-36-2024-QH15.docx
│   └── 168-2024-NĐ-CP.docx
│
│── Step1_chunk_documents.py
│── Step2_ingest_to_chroma.py
│── Step3_rag_traffic_law_bot.py
│── app.py
│── templates/          
│── static/            
│── chroma_db/         
│── README.md
│── .env
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Create `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Chunk documents
```bash
python Step1_chunk_documents.py
```

### 4. Ingest to ChromaDB
```bash
python Step2_ingest_to_chroma.py
```

### 5. Run the chatbot UI
```bash
python app.py
```

App will auto-open:

👉 http://127.0.0.1:8000

## ⚠️ Limitations
- Bot only answers based on ingested content.
- Not official legal advice.
- Retrieval accuracy depends on chunk quality & embedding model.

## 🔧 Future Improvements
- Reranking with **BGE / ColBERT**.
- Multi-turn memory with compression.
- Docker deployment (Railway/Vercel).
- Admin UI for uploading new legal documents.
