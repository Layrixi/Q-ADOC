"""

Responsible for parsing the data from documents, dividing them into chunks and saving them into a json
"""
import json
import os
import uuid
from typing import List

from pypdf import PdfReader         
from docx import Document as DocxDocument  


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def read_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_document_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".txt":
        return read_txt(path)
    else:
        raise ValueError(f"Wrong file format: {ext}")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Divides the text into chunk_size chunks of words with overlap between them to keep the context
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # go back a # overlap words

    return chunks


def ingest_document(path: str) -> List[dict]:
    """
    Returns a list of chunk in this format
        {"id": "<uuid>", "source": "<file name>", "text": "<chunk data>"}
    """
    text = load_document_text(path)
    raw_chunks = chunk_text(text)

    filename = os.path.basename(path)
    result = []
    for chunk in raw_chunks:
        result.append({
            "id": str(uuid.uuid4()),
            "source": filename,
            "text": chunk,
        })
    return result


def ingest_directory(directory: str) -> List[dict]:
    """Processes all coduments in a directory."""
    all_chunks = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".pdf", ".docx", ".txt"):
            continue
        print(f"Processing: {filename}")
        try:
            chunks = ingest_document(path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Error] processing file {filename}: {e}")
    return all_chunks


def save_chunks(chunks: List[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    
    input_dir = os.path.join(os.path.dirname(__file__), "..", "training", "data", "raw_docs")
    output_path = os.path.join(os.path.dirname(__file__), "..", "training", "data", "chunks.jsonl")

    if not os.path.isdir(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"{input_dir} does not exist. Creating {input_dir}")

    chunks = ingest_directory(input_dir)
    save_chunks(chunks, output_path)
