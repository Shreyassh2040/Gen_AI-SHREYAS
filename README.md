# NoteBot - RAG chatbot for PDF notes

Upload a PDF, ask questions, get answers grounded in the document (with the
retrieved chunks shown under **Sources used**).

**Stack:** Streamlit · LangChain · Google Gemini (free tier) · FAISS · pypdf

## Setup

```bash
cd server/GEN_AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get a free key at <https://aistudio.google.com/apikey> (no credit card), then:

```bash
cp .env.example .env
# edit .env -> GOOGLE_API_KEY=AIza...
```

## Run

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>. This does not clash with the Node server on
port 5000.

## How it works

```
PDF  ->  pypdf  ->  1000-char chunks (150 overlap)  ->  Gemini embeddings
     ->  FAISS index  ->  top-4 chunks for the question  ->  Gemini answer
```

The index is cached per file, so the document is embedded once, not on every
keystroke.

## Switching models

Edit the constants at the top of `app.py`:

```python
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
```

Other free-tier chat options: `gemini-2.5-flash-lite`, `gemini-2.5-pro`.

Notes:

- The embedding model **must** be one your key can see. Check with:
  `curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY"`
  `gemini-embedding-001` / `gemini-embedding-2` are the current ones;
  `text-embedding-004` is retired for newer projects.
- `thinking_budget=0` is set on the chat model. Gemini 2.5 reasons before
  answering, and that reasoning consumes `max_output_tokens` - without this the
  reply can come back empty or truncated.
- `transport="rest"` forces HTTP instead of gRPC. It is more reliable behind
  corporate proxies, VPNs and DNS-restricted networks.

