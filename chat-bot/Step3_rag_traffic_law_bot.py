import os
import re
from typing import List, Dict, Optional, TypedDict, Literal

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

# LangGraph
from langgraph.graph import StateGraph, START, END


# ==============================
#       LOAD ENV + CLIENTS
# ==============================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Chroma config (path & collection phải trùng với Step2_ingest_to_chroma.py)
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "traffic_law_2024"  # có thể đổi tên nếu dùng cho corpus khác

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # hoặc model bạn muốn


# ==============================
#       TEXT UTILS (LEXICAL)
# ==============================

VI_STOPWORDS = {
    "là", "và", "hoặc", "những", "các", "với", "cho", "khi", "được",
    "trên", "dưới", "từ", "đến", "theo", "tại", "này", "đó", "nào",
    "người", "xe", "việc", "hành", "vi", "tham", "gia", "giao", "thông",
    "trật", "tự", "an", "toàn", "đường", "bộ"
}


def tokenize(text: str) -> List[str]:
    """Tách từ đơn giản: lower, bỏ ký tự không phải chữ/số, split theo whitespace."""
    text = text.lower()
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
    Dùng được cho mọi domain.
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
    """Tạo embedding cho 1 đoạn text."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def get_collection():
    """Reconnect tới Chroma và lấy collection."""
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return chroma_client.get_collection(COLLECTION_NAME)


def retrieve_context(query_text: str, top_k: int = 5) -> Dict:
    """
    Retrieve top_k chunks từ Chroma.

    Pipeline:
    - Vector search (cosine distance) để lấy candidate.
    - Rerank lại bằng lexical overlap + embedding similarity.
    - Không gắn với bất kỳ luật/định danh cố định nào → dùng được cho nhiều corpus.
    """
    collection = get_collection()
    query_emb = create_embedding(query_text)

    # Lấy rộng hơn để rerank
    N_CANDIDATES = max(top_k * 3, 20)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=N_CANDIDATES,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    # Không có doc nào
    if not docs:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        # distance (0 = giống) → similarity ~ [0,1]
        sim_emb = 1.0 - min(max(dist, 0.0), 2.0) / 2.0
        lex_score = lexical_overlap_score(query_text, doc)

        alpha = 0.7  # embedding similarity
        beta = 0.3   # lexical overlap

        final_score = alpha * sim_emb + beta * lex_score
        scored.append((doc, meta, dist, final_score))

    scored.sort(key=lambda x: x[3], reverse=True)

    top_docs = [s[0] for s in scored[:top_k]]
    top_metas = [s[1] for s in scored[:top_k]]
    top_dists = [s[2] for s in scored[:top_k]]

    return {
        "documents": [top_docs],
        "metadatas": [top_metas],
        "distances": [top_dists],
    }


def format_source_label(meta: Dict) -> str:
    """
    Đọc metadata để in nguồn tài liệu.
    - Nếu có 'source' thì dùng.
    - Nếu không, dùng 'document_id' / 'file_name' nếu có.
    - Nếu cũng không, trả về 'Tài liệu tham khảo'.
    """
    if "source" in meta:
        return str(meta["source"])
    if "file_name" in meta:
        return str(meta["file_name"])
    if "document_id" in meta:
        return f"Document {meta['document_id']}"
    return "Tài liệu tham khảo"


def build_system_prompt() -> str:
    """
    System prompt tổng quát:
    - Trợ lý chỉ trả lời dựa trên context.
    - Phù hợp cho mọi loại tài liệu (luật, hướng dẫn, sổ tay...).
    """
    return (
        "Bạn là trợ lý tra cứu tài liệu tiếng Việt.\n\n"
        "Nguyên tắc:\n"
        "- Chỉ được sử dụng thông tin có trong NGỮ CẢNH được cung cấp.\n"
        "- Không được bịa ra dữ kiện, số liệu, quy định không xuất hiện trong ngữ cảnh.\n"
        "- Nếu ngữ cảnh không đủ để trả lời chắc chắn, hãy nói rõ điều đó và gợi ý người dùng "
        "xem thêm tài liệu gốc hoặc hỏi chuyên gia.\n"
        "- Cố gắng trích dẫn lại tên tài liệu / nguồn / điều khoản nếu metadata có cho phép.\n"
        "- Trả lời bằng tiếng Việt rõ ràng, dễ hiểu."
    )


def build_user_prompt(question: str, rewritten_question: str, results: Dict) -> str:
    """Tạo prompt user kèm context."""
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

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "Không có trích đoạn nào."

    display_question = rewritten_question or question

    prompt = (
        f"Ngữ cảnh (các trích đoạn từ tài liệu):\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Câu hỏi của người dùng (đã chuẩn hóa nếu cần):\n{display_question}\n\n"
        f"Hãy trả lời DỰA HOÀN TOÀN trên ngữ cảnh trên. "
        f"Nếu ngữ cảnh không đủ thông tin thì nói rõ là bạn không chắc chắn."
    )
    return prompt


def build_reference_block(results: Dict) -> str:
    """
    Sinh phần 'Nguồn tham khảo' tổng quát từ metadata.
    Hỗ trợ các key: source, file_name, article_number, clause_number, article_title.
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs or not metas:
        return ""

    seen = set()
    lines = []

    for meta in metas:
        src = format_source_label(meta)
        art = meta.get("article_number")
        clause = meta.get("clause_number")
        title = meta.get("article_title")

        key = (src, art, clause, title)
        if key in seen:
            continue
        seen.add(key)

        parts = [src]
        if art is not None:
            parts.append(f"Điều {art}")
        if clause is not None:
            parts.append(f"Khoản {clause}")
        if title:
            parts.append(f"({title})")

        lines.append(" - ".join(parts))

    if not lines:
        return ""

    ref_text = "Nguồn tham khảo:\n" + "\n".join(f"- {line}" for line in lines)
    return ref_text


def compute_confidence(query_text: str, results: Dict) -> float:
    """
    Độ tin cậy dựa trên doc tốt nhất:
    - embedding similarity
    - lexical overlap
    """
    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs or not dists:
        return 0.0

    best_doc = docs[0]
    best_dist = dists[0]

    sim_emb = 1.0 - min(max(best_dist, 0.0), 2.0) / 2.0
    lex_score = lexical_overlap_score(query_text, best_doc)

    alpha = 0.7
    beta = 0.3
    confidence = alpha * sim_emb + beta * lex_score

    return confidence


# ==============================
#        LANGGRAPH STATE
# ==============================

class RAGState(TypedDict, total=False):
    """
    Trạng thái cho LangGraph.

    - question: câu hỏi mới nhất từ user.
    - chat_history: lịch sử hội thoại (list[{"role": "...", "content": "..."}]),
      giống format OpenAI, để hiểu các câu follow-up kiểu "Còn ô tô thì sao".
    - rewritten_question: câu hỏi đã được viết lại thành câu độc lập.
    - top_k: số context muốn lấy.
    - results: kết quả retrieve từ Chroma.
    - answer: câu trả lời cuối cùng.
    """
    question: str
    chat_history: List[Dict[str, str]]
    rewritten_question: str
    top_k: int
    results: Dict
    answer: str


# ==============================
#       LANGGRAPH NODES
# ==============================

def rewrite_question_node(state: RAGState) -> RAGState:
    """
    Node 1: Viết lại câu hỏi dựa trên lịch sử hội thoại.

    Nếu không có history → giữ nguyên.
    Nếu có history → gọi LLM để biến "Còn ô tô thì sao" thành
    "Mức xử phạt vi phạm nồng độ cồn đối với ô tô thì sao?".
    """
    question = state["question"]
    history = state.get("chat_history", [])

    if not history:
        # Không có lịch sử → không cần rewrite
        state["rewritten_question"] = question
        return state

    system_msg = {
        "role": "system",
        "content": (
            "Bạn là bộ máy chuẩn hóa câu hỏi. "
            "Nhiệm vụ của bạn là biến câu hỏi cuối cùng của người dùng thành "
            "một câu hỏi đầy đủ, độc lập, có thể hiểu được mà không cần lịch sử hội thoại.\n"
            "- Giữ nguyên ngôn ngữ gốc (Việt/Anh/khác).\n"
            "- Không trả lời câu hỏi.\n"
            "- Chỉ xuất ra DUY NHẤT câu hỏi đã được viết lại."
        ),
    }

    messages = [system_msg]
    # Đưa lịch sử vào để model hiểu ngữ cảnh
    for msg in history:
        if msg.get("role") in {"user", "assistant"}:
            messages.append({"role": msg["role"], "content": msg["content"]})
    # Câu hỏi mới
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=messages,
    )

    rewritten = resp.choices[0].message.content.strip()
    if not rewritten:
        rewritten = question

    state["rewritten_question"] = rewritten
    return state


def retrieve_node(state: RAGState) -> RAGState:
    """Node 2: lấy context từ Chroma + rerank."""
    rewritten = state.get("rewritten_question") or state["question"]
    top_k = state.get("top_k", 5)

    results = retrieve_context(rewritten, top_k=top_k)
    state["results"] = results
    return state


def route_after_retrieval(state: RAGState) -> Literal["generate", "fallback"]:
    """
    Điều hướng sau retrieve:
    - Nếu độ tin cậy < 0.4 → fallback.
    - Ngược lại → generate.
    """
    rewritten = state.get("rewritten_question") or state["question"]
    results = state.get("results", {"documents": [[]], "distances": [[]]})
    confidence = compute_confidence(rewritten, results)

    # Bạn có thể in ra để debug nếu muốn
    # print("Confidence:", confidence)

    if confidence < 0.4:
        return "fallback"
    return "generate"


def generate_answer_node(state: RAGState) -> RAGState:
    """Node 3: Gọi LLM để sinh câu trả lời dựa trên context."""
    results = state["results"]
    question = state["question"]
    rewritten = state.get("rewritten_question") or question

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(question, rewritten, results)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = resp.choices[0].message.content or ""
    refs = build_reference_block(results)

    if refs:
        answer = f"{answer}\n\n{refs}"

    state["answer"] = answer
    return state


def fallback_node(state: RAGState) -> RAGState:
    """Node 4: trả lời khi context không đủ / quá mơ hồ."""
    state["answer"] = (
        "Các trích đoạn tài liệu tôi tìm được không đủ rõ hoặc không liên quan chặt chẽ "
        "để trả lời chắc chắn câu hỏi này. "
        "Bạn nên xem trực tiếp tài liệu gốc hoặc hỏi ý kiến chuyên gia để có tư vấn chính xác hơn."
    )
    return state


# ==============================
#        BUILD LANGGRAPH
# ==============================

def create_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite", rewrite_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("fallback", fallback_node)

    # START → rewrite → retrieve
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")

    # retrieve → generate / fallback (conditional)
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    # generate / fallback → END
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


rag_app = create_rag_graph()


# ==============================
#  PUBLIC API: ask_traffic_law_bot
# ==============================

def ask_traffic_law_bot(
    question: str,
    top_k: int = 8,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Hàm public cho app khác (Flask, UI, CLI).

    - question: câu hỏi mới nhất từ người dùng.
    - top_k: số lượng context dùng để RAG.
    - chat_history: lịch sử hội thoại theo format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    """
    if chat_history is None:
        chat_history = []

    result_state = rag_app.invoke({
        "question": question,
        "top_k": top_k,
        "chat_history": chat_history,
    })
    return result_state["answer"]


# ==============================
#       SIMPLE CLI DEMO
# ==============================

if __name__ == "__main__":
    print("🤖 RAG Assistant (LangGraph, conversational)")
    print("Gõ 'exit' để thoát.\n")

    history: List[Dict[str, str]] = []

    while True:
        q = input("❓ Bạn: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            answer = ask_traffic_law_bot(q, top_k=8, chat_history=history)
            print("\n💬 Bot:")
            print(answer)
            print("\n" + "=" * 60 + "\n")

            # Cập nhật history cho lần hỏi sau
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print("Error:", e)
            break
