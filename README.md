# RAG Project - Simple RAG + Gemini Demo

This project is a minimal Retrieval-Augmented Generation (RAG) demo:
- Flask backend (`app_single.py`) that indexes uploaded PDFs using **SentenceTransformers** + **FAISS**
- Uses **Google Gemini** (via `google-genai` SDK) as the LLM to synthesize answers
- Simple single-file frontend (`frontend.html`) for uploading PDFs and asking questions

Features:
- If a question is relevant to indexed PDFs, the system replies using retrieved document context (RAG mode).
- Otherwise, it falls back to a general Gemini response (free mode).

## Requirements

```bash
pip install flask numpy faiss-cpu sentence-transformers google-genai pypdf
```

Note: On some platforms, `faiss-cpu` installation may require wheels or build tools. If you have trouble, consider using a Docker image or switch to `chromadb` as an alternative vector store.

## Setup

1. Set your Gemini API key in PowerShell:
```powershell
$Env:GEMINI_API_KEY="YOUR_API_KEY"
```

2. Run the app:
```bash
python app_single.py
```

3. Open your browser to `http://127.0.0.1:5000` to access the frontend.

## Files
- `app_single.py` - Flask backend + FAISS indexer + Gemini integration
- `frontend.html` - Plain HTML/JS frontend
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Notes & Improvements
- The project keeps the FAISS index and corpus in `faiss_index/` and uploaded PDFs in `uploads/`.
- Retrieval relevance threshold is configurable in `app_single.py` (`RELEVANCE_THRESHOLD`).
- Consider adding source filenames & chunk ids to corpus metadata for better provenance display.
- For production, add auth, rate limits, background workers for indexing, and persistent vector DB (e.g., Chroma/Pinecone).

