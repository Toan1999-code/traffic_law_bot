import os
import re
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


# ==============================
#       LOAD ENV + CLIENTS
# ==============================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma config (must match ingest script)
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "traffic_law_2024"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # hoặc model bạn muốn dùng


# ==============================
#       TEXT UTILS (LEXICAL)
# ==============================

# Một danh sách stopwords tiếng Việt đơn giản để làm sạch,
# đủ dùng cho overlap tổng quát (không lệ thuộc từng điều luật).
VI_STOPWORDS = {
    "là", "và", "hoặc", "những", "các", "với", "cho", "khi", "được",
    "trên", "dưới", "từ", "đến", "theo", "tại", "này", "đó", "nào",
    "người", "xe", "việc", "hành", "vi", "tham", "gia", "giao", "thông",
    "trật", "tự", "an", "toàn", "đường", "bộ"
}


def tokenize(text: str) -> List[str]:
    """Tách từ đơn giản: lower, bỏ ký tự không phải chữ/số, split theo whitespace."""
    text = text.lower()
    # thay mọi thứ không phải chữ cái/ số thành khoảng trắng
    text = re.sub(
        r"[^0-9a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
        r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự"
        r"ýỳỷỹỵđ\s]",
        " ",
        text,
    )
    tokens = text.split()
    return [t for t in tokens if t not in VI_STOPWORDS]


def lexical_overlap_score(question: str, doc: str) -> float:
    """
    Tính điểm overlap từ vựng giữa câu hỏi và doc: |giao| / |q_tokens|.
    Tổng quát, không phụ thuộc domain cụ thể.
    """
    q_tokens = set(tokenize(question))
    d_tokens = set(tokenize(doc))

    if not q_tokens:
        return 0.0

    inter = q_tokens.intersection(d_tokens)
    return len(inter) / len(q_tokens)


# ==============================
#       HELPER FUNCTIONS
# ==============================

def create_embedding(text: str) -> List[float]:
    """Create a single embedding vector for the given text."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def get_collection():
    """Reconnect to Chroma and get the collection."""
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return chroma_client.get_collection(COLLECTION_NAME)


def infer_source_filter(question: str) -> Optional[str]:
    """
    Heuristic tổng quát:
    - Nếu hỏi về xử phạt, phạt tiền, trừ điểm, tước GPLX → ưu tiên Nghị định 168/2024/NĐ-CP.
    - Nếu hỏi về khái niệm, quy tắc, quyền & nghĩa vụ → ưu tiên Luật 36/2024/QH15.
    - Nếu không đoán ra → trả về None (không filter).
    """
    q = question.lower()

    penalty_keywords = [
        "xử phạt", "phạt tiền", "mức phạt", "xử lý vi phạm",
        "trừ điểm", "phục hồi điểm", "tước quyền sử dụng giấy phép",
        "tước giấy phép", "tước bằng", "xử lý hành chính"
    ]
    if any(k in q for k in penalty_keywords):
        return "nd_168_2024"

    law_keywords = [
        "là gì", "định nghĩa", "khái niệm",
        "quy tắc", "nguyên tắc", "trách nhiệm", "quyền", "nghĩa vụ"
    ]
    if any(k in q for k in law_keywords):
        return "law_36_2024"

    return None


def retrieve_context(question: str, top_k: int = 5) -> Dict:
    """
    Retrieve top_k relevant chunks từ Chroma cho một câu hỏi.

    Kết hợp:
    - Vector search (cosine distance) để lấy candidate.
    - Lexical overlap (question vs doc) để rerank tổng quát.

    Có hỗ trợ filter theo metadata 'source' (Luật / Nghị định),
    nhưng không gắn với keyword cụ thể nào ngoài heuristic chung.
    """
    collection = get_collection()
    query_emb = create_embedding(question)

    # Heuristic filter theo nguồn (nếu đoán được)
    source_filter = infer_source_filter(question)
    where_clause = {"source": source_filter} if source_filter else None

    # 1) Query vector rộng hơn top_k để có đủ candidate
    N_CANDIDATES = max(top_k * 3, 20)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=N_CANDIDATES,
        include=["documents", "metadatas", "distances"],
        where=where_clause,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    # Nếu filter quá chặt, không ra gì → bỏ filter, query lại
    if (not docs) and source_filter is not None:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=N_CANDIDATES,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

    # Nếu vẫn không có gì thì trả về kết quả rỗng
    if not docs:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # 2) Rerank bằng score tổng hợp: sim_embedding + lexical_overlap
    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        # distance (0 = giống, lớn = khác) → similarity ~ [0,1]
        sim_emb = 1.0 - min(max(dist, 0.0), 2.0) / 2.0
        lex_score = lexical_overlap_score(question, doc)

        # Trọng số có thể tinh chỉnh, đây là ví dụ:
        alpha = 0.7  # embedding similarity
        beta = 0.3   # lexical overlap

        final_score = alpha * sim_emb + beta * lex_score
        scored.append((doc, meta, dist, final_score))

    # sort theo final_score giảm dần
    scored.sort(key=lambda x: x[3], reverse=True)

    # 3) Chọn lại top_k sau rerank
    top_docs = [s[0] for s in scored[:top_k]]
    top_metas = [s[1] for s in scored[:top_k]]
    top_dists = [s[2] for s in scored[:top_k]]

    return {
        "documents": [top_docs],
        "metadatas": [top_metas],
        "distances": [top_dists],
    }


def format_source_label(meta: Dict) -> str:
    """Đổi metadata 'source' thành tên văn bản dễ hiểu."""
    src = meta.get("source")
    if src == "law_36_2024":
        return "Luật Trật tự, an toàn giao thông đường bộ 2024 (Luật 36/2024/QH15)"
    if src == "nd_168_2024":
        return "Nghị định 168/2024/NĐ-CP (xử phạt, trừ/khôi phục điểm GPLX)"
    return "Văn bản pháp luật khác"


def build_system_prompt() -> str:
    """System prompt: define role & constraints of legal assistant."""
    return (
        "Bạn là trợ lý pháp lý tiếng Việt, chuyên về các quy định trong:\n"
        "- Luật Trật tự, an toàn giao thông đường bộ 2024 (Luật 36/2024/QH15), và\n"
        "- Nghị định 168/2024/NĐ-CP về xử phạt, trừ điểm, phục hồi điểm giấy phép lái xe.\n\n"
        "Bạn CHỈ được trả lời dựa trên NGỮ CẢNH (các điều, khoản luật, nghị định) được cung cấp.\n\n"
        "Quy tắc trả lời:\n"
        "- Giải thích bằng tiếng Việt đơn giản, dễ hiểu.\n"
        "- Luôn cố gắng nhắc lại nguồn (Luật / Nghị định), kèm Điều / Khoản tương ứng nếu có.\n"
        "- Không được bịa ra quy định, hình phạt hoặc điều luật không xuất hiện rõ ràng trong ngữ cảnh.\n"
        "- Nếu ngữ cảnh không chứa thông tin đủ rõ để trả lời câu hỏi, hãy nói rằng bạn "
        "\"không tìm thấy quy định rõ ràng trong các trích đoạn luật/nghị định được cung cấp\" "
        "và khuyên người dùng tra cứu trực tiếp văn bản hoặc hỏi ý kiến cơ quan có thẩm quyền / luật sư.\n"
        "- Nhấn mạnh rằng đây chỉ là hỗ trợ tra cứu thông tin, KHÔNG phải tư vấn pháp lý chính thức."
    )


def build_user_prompt(question: str, results: Dict) -> str:
    """Build the user-facing prompt including retrieved context."""
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(docs, metas):
        src_label = format_source_label(meta)
        art = meta.get("article_number")
        art_title = meta.get("article_title")
        clause = meta.get("clause_number")

        header_parts = [f"Nguồn: {src_label}"]
        if art is not None:
            header_parts.append(f"Điều {art}")
        if clause is not None:
            header_parts.append(f"Khoản {clause}")
        if art_title:
            header_parts.append(f"({art_title})")

        header = " - ".join(header_parts)

        block = f"{header}:\n{doc}"
        context_blocks.append(block)

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"Ngữ cảnh (các trích đoạn từ Luật & Nghị định giao thông):\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Câu hỏi của người dùng:\n{question}\n\n"
        f"Hãy trả lời dựa HOÀN TOÀN trên ngữ cảnh trên. "
        f"Nếu ngữ cảnh không đủ để trả lời chắc chắn, hãy nói rõ là bạn không chắc chắn."
    )

    return prompt


def build_reference_block(results: Dict) -> str:
    """
    Tự động sinh phần 'Nguồn tham khảo' ở cuối câu trả lời,
    liệt kê rõ Luật/Nghị định + Điều + Khoản.
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs or not metas:
        return ""

    seen = set()
    lines = []

    for meta in metas:
        src = meta.get("source")
        art = meta.get("article_number")
        clause = meta.get("clause_number")

        key = (src, art, clause)
        if key in seen:
            continue
        seen.add(key)

        src_label = format_source_label(meta)

        parts = [src_label]
        if art is not None:
            parts.append(f"Điều {art}")
        if clause is not None:
            parts.append(f"Khoản {clause}")

        lines.append(" - ".join(parts))

    if not lines:
        return ""

    ref_text = "Nguồn tham khảo:\n" + "\n".join(f"- {line}" for line in lines)
    return ref_text


def ask_traffic_law_bot(question: str, top_k: int = 5) -> str:
    """High-level function: retrieve context from Chroma + call LLM."""

    # 1. Retrieve relevant chunks (đã rerank)
    results = retrieve_context(question, top_k=top_k)

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]  # cosine distance: càng thấp càng giống

    # 1.a. Nếu không có doc nào -> fallback
    if not docs or not dists:
        return (
            "Hiện tại tôi không tìm thấy trích đoạn luật/nghị định nào phù hợp với câu hỏi này "
            "trong kho dữ liệu đã được nạp, nên không thể trả lời chắc chắn. "
            "Bạn nên tra cứu trực tiếp văn bản pháp luật gốc hoặc hỏi ý kiến cơ quan chức năng / luật sư."
        )

    # 1.b. Đánh giá độ tin cậy cho doc tốt nhất bằng cả distance + lexical overlap (tổng quát)
    best_doc = docs[0]
    best_dist = dists[0]

    sim_emb = 1.0 - min(max(best_dist, 0.0), 2.0) / 2.0
    lex_score = lexical_overlap_score(question, best_doc)

    alpha = 0.7
    beta = 0.3
    confidence = alpha * sim_emb + beta * lex_score  # ~ [0,1]

    # Ngưỡng tin cậy, có thể chỉnh (0.4–0.5 tuỳ bạn)
    if confidence < 0.4:
        return (
            "Các trích đoạn luật/nghị định tôi tìm được có vẻ không liên quan chặt chẽ đến câu hỏi này, "
            "nên tôi không thể trả lời một cách chắc chắn dựa trên dữ liệu hiện có. "
            "Bạn nên tra cứu thêm văn bản pháp luật gốc hoặc hỏi ý kiến chuyên gia pháp lý."
        )

    # 2. Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(question, results)

    # 3. Call chat model
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = resp.choices[0].message.content or ""

    return answer


# ==============================
#       SIMPLE CLI DEMO
# ==============================

if __name__ == "__main__":
    print("🚦 Traffic Law Legal Assistant (Luật 36/2024 + NĐ 168/2024)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("❓ Câu hỏi: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            reply = ask_traffic_law_bot(q, top_k=8)  # tăng top_k để có thêm context
            print("\n💬 Trả lời:")
            print(reply)
            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print("Error:", e)
            break
