import os
import json
import re
from typing import List, Dict, Optional

from docx import Document


# ==============================
#   PATTERN CONFIG
# ==============================

# Regex to detect article heading, e.g. "Điều 57. Giấy phép lái xe..."
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\.?\s*(.*)", re.IGNORECASE)

# Regex to detect clause heading, e.g. "1. Nội dung..."
CLAUSE_RE = re.compile(r"^(\d+)\.\s*(.*)")


# ==============================
#   CORE CHUNKING LOGIC
# ==============================

def chunk_docx_by_article_clause(
    file_path: str,
    source_tag: str,
) -> List[Dict]:
    """
    Load a DOCX file and chunk it by article (Điều) and clause (Khoản).
    Each chunk corresponds to one clause inside one article.

    :param file_path: Path to the .docx file.
    :param source_tag: Short string to identify the source
                       (e.g. "law_36_2024" or "nd_168_2024").
    :return: List of chunk dicts.
    """
    doc = Document(file_path)

    chunks: List[Dict] = []

    current_article_number: Optional[int] = None
    current_article_title: str = ""
    current_clause_number: Optional[int] = None
    current_clause_lines: List[str] = []

    def flush_clause():
        """Save the current clause as a chunk if it has content."""
        if (
            current_article_number is not None
            and current_clause_number is not None
            and current_clause_lines
        ):
            content = "\n".join(current_clause_lines).strip()
            if not content:
                return

            chunk = {
                "source": source_tag,
                "source_file": os.path.basename(file_path),
                "article_number": current_article_number,
                "article_title": current_article_title.strip() or None,
                "clause_number": current_clause_number,
                "content": content,
            }
            chunks.append(chunk)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue  # skip empty paragraphs

        # 1) Check if this is an article heading
        art_match = ARTICLE_RE.match(text)
        if art_match:
            # Flush previous clause before starting a new article
            flush_clause()
            # Reset clause buffer
            current_clause_lines.clear()

            current_article_number = int(art_match.group(1))
            current_article_title = art_match.group(2) or ""
            current_clause_number = None
            continue

        # 2) Check if this is a clause heading (Khoản)
        clause_match = CLAUSE_RE.match(text)
        if clause_match and current_article_number is not None:
            # Flush previous clause before starting new one
            flush_clause()
            current_clause_lines.clear()

            current_clause_number = int(clause_match.group(1))
            # Remaining text of this paragraph (after "1.")
            remainder = clause_match.group(2).strip()
            if remainder:
                current_clause_lines.append(remainder)
            continue

        # 3) Normal text: belongs to the current clause (if any)
        if current_article_number is not None:
            # If no clause has started yet, you can choose:
            # - either attach it to a "0" pseudo-clause
            # - or just ignore / handle differently.
            # Here we attach to a pseudo-clause 0.
            if current_clause_number is None:
                current_clause_number = 0  # header / intro of article
                current_clause_lines.clear()
            current_clause_lines.append(text)

    # Flush the last clause at the end of document
    flush_clause()

    return chunks


# ==============================
#   MAIN INGEST PIPELINE
# ==============================

def ingest_traffic_docs():
    """
    Ingest 2 Word documents:
    - Law 36/2024/QH15 on road traffic order & safety
    - Decree 168/2024/NĐ-CP on administrative penalties, point deduction/restoration
    and output a single JSONL file with all chunks.
    """
    # Adjust these paths according to your project structure
    base_dir = "../documents"

    law_file = os.path.join(base_dir, "Law-36-2024-QH15.docx")
    decree_file = os.path.join(base_dir, "1682024NĐ-CP.docx")

    output_path = "traffic_corpus_chunks.jsonl"

    # 1) Chunk the Law
    print(f"Chunking law file: {law_file}")
    law_chunks = chunk_docx_by_article_clause(
        file_path=law_file,
        source_tag="law_36_2024",
    )
    print(f"   → {len(law_chunks)} chunks from law.")

    # 2) Chunk the Decree
    print(f"Chunking decree file: {decree_file}")
    decree_chunks = chunk_docx_by_article_clause(
        file_path=decree_file,
        source_tag="nd_168_2024",
    )
    print(f"   → {len(decree_chunks)} chunks from decree.")

    all_chunks = []
    all_chunks.extend(law_chunks)
    all_chunks.extend(decree_chunks)

    # 3) Add unique IDs for each chunk
    print("Assigning IDs...")
    for idx, chunk in enumerate(all_chunks, start=1):
        art = chunk.get("article_number")
        clause = chunk.get("clause_number")
        src = chunk.get("source")
        chunk_id = f"{src}-d{art}-k{clause}-{idx}"
        chunk["id"] = chunk_id

    # 4) Write to JSONL
    print(f"Writing {len(all_chunks)} chunks to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    ingest_traffic_docs()
