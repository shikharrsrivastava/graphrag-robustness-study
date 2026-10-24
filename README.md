# Local PDF RAG Starter (Ollama + ChromaDB)

A minimal local RAG pipeline for learning. It extracts text from PDFs, splits it into overlapping chunks, creates embeddings with Google's EmbeddingGemma through Ollama, stores them in local ChromaDB, retrieves relevant chunks for a question, and asks Gemma 3 to answer using that evidence.

## Prerequisites

- macOS with Ollama installed and running
- Python 3.12 recommended for a conservative package environment
- Internet for the initial model and Python package downloads

## 1. Download the models

In Terminal, run:

```bash
ollama pull embeddinggemma
ollama pull gemma3:4b
```

Check they are present:

```bash
ollama list
```

## 2. Create and activate a virtual environment

Open Terminal and `cd` into this project folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3` is not installed, install Python from https://www.python.org/downloads/macos/ first. If `python3 --version` shows a version older than 3.10, install Python 3.12 or newer and use that version to create the virtual environment.

## 3. Add a PDF

Copy a normal, text-readable PDF into the `data/` folder. A course syllabus is a good first test. Scanned/image-only PDFs are not supported by this starter because OCR is not included.

## 4. Run

```bash
python rag.py
```

- Type `I` to extract PDF text, split it, create embeddings and index the chunks in ChromaDB.
- Type `Q` to ask a question. The script prints the retrieved chunks before the generated answer so you can inspect the evidence.
- Type `X` to exit.

Run `I` again after adding or changing PDFs. The local vector database is saved under `chroma_db/` and excluded from Git by `.gitignore`.

## What this starter intentionally does not include

- OCR for scanned PDFs
- Web/HTML ingestion
- A user interface
- Graph construction or GraphRAG
- Formal retrieval/answer evaluation

Those can be added after the baseline RAG pipeline works.
