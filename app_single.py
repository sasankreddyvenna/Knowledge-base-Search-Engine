# app_single.py
# Dependencies:
#   pip install flask numpy faiss-cpu sentence-transformers google-genai pypdf
# Usage:
#   PowerShell (one-time session):  $Env:GEMINI_API_KEY="YOUR_API_KEY"
#   Run:  python app_single.py

import os
import json
import re
from typing import List, Tuple, Optional

from flask import Flask, request, jsonify, send_from_directory
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai  # SDK: pip install -U google-genai


# ---------------- Configuration ----------------
UPLOAD_FOLDER = "uploads"
FAISS_DIR = "faiss_index"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
CORPUS_PATH = os.path.join(FAISS_DIR, "corpus.jsonl")

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
RELEVANCE_THRESHOLD = 0.20  # cosine similarity threshold for determining relevance

GEMINI_MODEL = "gemini-2.5-flash"  # change to "gemini-2.5-pro" if needed
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = None  # e.g., 2048

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
# ------------------------------------------------


# ---------------- Utilities ----------------
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    text = "\n".join(texts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks
# ------------------------------------------------


# ---------------- Embeddings & Index ----------------
_embedder = SentenceTransformer(EMBED_MODEL)

def _iter_pdf_chunks(upload_folder: str):
    for fname in sorted(os.listdir(upload_folder)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(upload_folder, fname)
        try:
            text = extract_text_from_pdf(fpath)
            for i, chunk in enumerate(chunk_text(text)):
                yield chunk, fname, i
        except Exception as e:
            print(f"[warn] Skipping {fname}: {e}")

def build_faiss_index(
    upload_folder: str = UPLOAD_FOLDER,
    index_path: str = INDEX_PATH,
    corpus_path: str = CORPUS_PATH,
    batch_size: int = 64,
) -> Tuple[List[str], List[dict], str]:
    corpus_meta = []
    corpus_texts = []
    for chunk, src, idx in _iter_pdf_chunks(upload_folder):
        corpus_texts.append(chunk)
        corpus_meta.append({"text": chunk, "source": src, "chunk_id": idx})

    if not corpus_texts:
        raise ValueError(f"No PDF chunks found in '{upload_folder}'.")

    # Encode in batches, cast to float32
    vecs = []
    for start in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[start:start + batch_size]
        emb = _embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        vecs.append(emb)
    vectors = np.vstack(vecs)

    # Cosine similarity: normalize then use inner-product index
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Persist
    faiss.write_index(index, index_path)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in corpus_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return corpus_texts, corpus_meta, index_path

def load_corpus(corpus_path: str = CORPUS_PATH):
    texts = []
    meta = []
    if not os.path.exists(corpus_path):
        return texts, meta
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            texts.append(obj.get("text", ""))
            meta.append({"source": obj.get("source"), "chunk_id": obj.get("chunk_id")})
    return texts, meta

def retrieve_context(query: str, corpus: List[str], index_path: str = INDEX_PATH, top_k: int = TOP_K) -> Tuple[str, List[dict], List[float]]:
    """
    Returns: (context_text, list_of_metadatas, list_of_scores)
    If index or corpus missing, returns warnings in context_text.
    """
    if not corpus:
        return "⚠️ No documents found in corpus. Please upload and index content first.", [], []

    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        return f"⚠️ Could not read FAISS index: {str(e)}", [], []

    q_vec = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    scores = D[0].tolist()
    indices = I[0].tolist()

    valid_pairs = [(i, s) for i, s in zip(indices, scores) if 0 <= i < len(corpus)]
    if not valid_pairs:
        return "⚠️ No relevant document chunks found for this query.", [], []

    # Build context and metadata list
    context_pieces = []
    metadatas = []
    for idx, score in valid_pairs:
        context_pieces.append(corpus[idx])
        # metadata read from CORPUS_PATH file (on-disk)
        # We'll load metadata separately and include it in response
        try:
            # load metadata file lazily
            _, metas = load_corpus(index_path.replace('index.faiss','corpus.jsonl')) if False else load_corpus()
            # fallback: we won't rely on that; return index and score
        except Exception:
            metas = []

        metadatas.append({"index": idx, "score": float(score)})
    context = "\n\n".join(context_pieces)
    return context, metadatas, scores

# ----------------------------------------------------


# ---------------- Gemini Client & Call ----------------
def get_gemini_client() -> "genai.Client":
    # Reads GEMINI_API_KEY when no explicit api_key passed
    api_key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()

def query_gemini(
    context: str,
    question: str,
    model_name: str = GEMINI_MODEL,
    temperature: float = TEMPERATURE,
    max_output_tokens: Optional[int] = MAX_OUTPUT_TOKENS
) -> str:
    if context and context.strip() and not context.startswith("⚠️"):
        prompt = (
            "You are an intelligent research assistant.\n"
            "Use only the given context to answer the user's question accurately and concisely.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer succinctly and cite sources where applicable."
        )
    else:
        prompt = f"You are an expert assistant. Answer the following question using your own knowledge:\n\n{question}"

    config = {"temperature": temperature}
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens

    try:
        client = get_gemini_client()
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"
# -----------------------------------------------------


# ---------------- Flask App ----------------
app = Flask(__name__, static_folder='.', static_url_path='')

# Globals holding corpus in memory for fast retrieval
corpus_texts: List[str] = []
corpus_meta: List[dict] = []

@app.route("/")
def index():
    return send_from_directory(".", "frontend.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    global corpus_texts, corpus_meta
    try:
        corpus_texts, corpus_meta, _ = build_faiss_index(UPLOAD_FOLDER, index_path=INDEX_PATH, corpus_path=CORPUS_PATH)
    except Exception as e:
        return jsonify({"error": f"Indexing failed: {str(e)}"}), 500
    return jsonify({"message": f"{file.filename} uploaded and indexed successfully!", "chunks": len(corpus_texts)}), 200

@app.route("/query", methods=["POST"])
def query_docs():
    data = request.json or {}
    question = data.get("query", "").strip()
    if not question:
        return jsonify({"error": "Missing 'query'"}), 400

    # Attempt retrieval
    context, metadatas, scores = retrieve_context(question, corpus_texts, index_path=INDEX_PATH, top_k=TOP_K)

    # Determine if retrieval is relevant
    if metadatas and len(scores) > 0 and max(scores) >= RELEVANCE_THRESHOLD and not context.startswith("⚠️"):
        # RAG mode
        answer = query_gemini(context, question, model_name=GEMINI_MODEL, temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS)
        return jsonify({"answer": answer, "mode": "rag", "sources": metadatas}), 200
    else:
        # Free / general LLM mode
        answer = query_gemini("", question, model_name=GEMINI_MODEL, temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS)
        return jsonify({"answer": answer, "mode": "free", "sources": []}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render injects PORT env var automatically
    app.run(host="0.0.0.0", port=port)

