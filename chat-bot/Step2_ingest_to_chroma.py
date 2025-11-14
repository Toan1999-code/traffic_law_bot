import json
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from openai import OpenAI
from dotenv import load_dotenv
import os


# ===========================================
#            LOAD ENVIRONMENT VARIABLES
# ===========================================

# Load variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client using env variable
oa_client = OpenAI(api_key=OPENAI_API_KEY)


# ===========================================
#            CONFIGURATION
# ===========================================

# Path to your JSONL chunk file
# (output của Step1_ingest_traffic_docs.py)
CHUNKS_FILE = Path("traffic_corpus_chunks.jsonl")

# Persistent ChromaDB directory
CHROMA_DB_DIR = Path("chroma_db")

# Collection name
COLLECTION_NAME = "traffic_law_2024"     # có thể đổi nếu muốn tách collection mới

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"


# ===========================================
#            HELPER FUNCTIONS
# ===========================================

def load_chunks(jsonl_path: Path) -> List[Dict]:
    """Load all chunk JSON objects from a JSONL file."""
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_metadata(chunk: Dict) -> Dict:
    """
    Extract metadata fields from chunk.

    Schema hiện tại của chunk (Step1):
    {
        "source": "law_36_2024" | "nd_168_2024",
        "source_file": "...docx",
        "article_number": int,
        "article_title": str | null,
        "clause_number": int,
        "content": "...",
        "id": "..."
    }

    Ta chỉ cần đảm bảo:
    - Không có None
    - Value là bool/int/float/str
    """
    raw_meta = {
        "source": chunk.get("source"),               # phân biệt Luật / Nghị định
        "source_file": chunk.get("source_file"),
        "article_number": chunk.get("article_number"),
        "article_title": chunk.get("article_title"),
        "clause_number": chunk.get("clause_number"),
    }

    meta: Dict[str, object] = {}

    for k, v in raw_meta.items():
        if v is None:
            # Skip keys with None value (Chroma does not allow None)
            continue

        # If it's not a primitive, cast to string
        if isinstance(v, (bool, int, float, str)):
            meta[k] = v
        else:
            meta[k] = str(v)

    return meta


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    response = oa_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


# ===========================================
#              INGEST FUNCTION
# ===========================================

def ingest_to_chroma():
    """Load chunks, embed them, and store into ChromaDB."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        ids.append(c["id"])
        documents.append(c["content"])
        metadatas.append(build_metadata(c))

    print("Starting ingestion into ChromaDB...")

    # Batch processing
    batch_size = 100
    total = len(documents)

    for start in range(0, total, batch_size):
        end = start + batch_size

        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        embeddings = create_embeddings(batch_docs)

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

        print(f"✔ Ingested batch {start}–{min(end, total) - 1}")

    print("\n==============================")
    print("Ingestion completed!")
    print(f"ChromaDB Directory: {CHROMA_DB_DIR}")
    print(f"Collection Name: {COLLECTION_NAME}")
    print("==============================\n")


# ===========================================
#              TEST QUERY (optional)
# ===========================================

def test_query():
    """Test retrieval from ChromaDB after ingestion, using the SAME OpenAI embeddings."""

    # Reconnect to Chroma
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query = "Giấy phép lái xe hạng B1 được sử dụng để lái những loại xe nào?"
    print(f"\n=== TEST QUERY ===\n{query}\n")

    # Use the SAME embedding model as ingestion
    query_embedding = create_embeddings([query])[0]  # 1 vector of dimension 1536

    # Use query_embeddings instead
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        print(f"--- Result #{i + 1} ---")
        print("Source:", meta.get("source"))
        print("Article:", meta.get("article_number"), "-", meta.get("article_title"))
        print("Clause:", meta.get("clause_number"))
        print("Text:", doc)
        print()

    print("=== END TEST ===")


# ===========================================
#                  MAIN
# ===========================================

if __name__ == "__main__":
    ingest_to_chroma()
    # Optional:
    test_query()
