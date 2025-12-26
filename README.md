# MultiModal RAG System

Production-ready Multi-Modal Retrieval-Augmented Generation (RAG) system for academic documents, with PDF + image + video processing capabilities.

This repository also includes a separate Streamlit PDF-chat app (QueryCore) under `querycore-pdf-chat/`.

---

## What’s inside

- **Main multimodal RAG pipeline**
	- Core logic in `src/`
	- Config in `config/config.yaml`
	- Local vector DB artifacts in `data/` (if present)

- **QueryCore (PDF Chat with Ollama + FAISS)**
	- Location: `querycore-pdf-chat/`
	- A Streamlit UI to upload PDFs and ask questions using local Ollama models + FAISS retrieval

---
## Screenshots
<img width="1882" height="943" alt="Screenshot 2025-08-09 144428" src="https://github.com/user-attachments/assets/42f907a9-75e6-44db-ae3d-39661ebe3abd" />
<img width="1906" height="975" alt="Screenshot 2025-08-09 144318" src="https://github.com/user-attachments/assets/5a0792d1-b56f-4f61-ac5a-2eadb31da861" />
## Quickstart (Main project)

### 1) Clone
```bash
git clone https://github.com/vedantnavadiya111/multimodal-rag-system.git
cd multimodal-rag-system
```

### 2) Create virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run (choose one entry)

Depending on which entrypoint you are using in this repo:

- Streamlit UI (if applicable):
	```bash
	streamlit run frontend/streamlit_app.py
	```
- Or Python app:
	```bash
	python app.py
	```

If you’re unsure which one your setup uses, check the README sections in `frontend/` and `querycore-pdf-chat/`, or tell me which UI you want (Streamlit vs CLI) and I’ll point to the exact command.

---

## QueryCore (PDF Chat with Ollama + FAISS)

QueryCore is a Streamlit app for asking natural-language questions about PDF files using a **local** Ollama model + embeddings + FAISS retrieval.

### Setup (Windows PowerShell)
```powershell
cd querycore-pdf-chat
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Install / pull Ollama models

Install Ollama: https://ollama.com/

Then pull models:
```bash
ollama pull nomic-embed-text
ollama pull mistral
ollama pull tinyllama
```

### Run QueryCore
```bash
streamlit run app.py
```

Open the Streamlit URL (usually `http://localhost:8501`).

---

## Notes on files & persistence

- `faiss_index/` is created locally when you process PDFs in QueryCore and should not be committed.
- Local virtual environments like `.venv/` should not be committed.
- If you see `data/chroma_db/` committed, it may contain local vector DB artifacts; treat it as optional and environment-dependent.

---

## Troubleshooting

- **No models in dropdown / Ollama not responding**
	Ensure Ollama is running and models exist:
	```bash
	ollama list
	```

- **FAISS install issues**
	This repo uses `faiss-cpu` for widest compatibility.

- **PDF image preview issues**
	`pdfplumber` relies on system PDF rendering dependencies on some platforms; if previews fail, text extraction + QA can still work.

---

## License

MIT (see `LICENSE`).
