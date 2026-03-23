# 📄 DocuMind — RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about its content in natural language — entirely free, using Hugging Face models.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-free-yellow)

---

## 🧠 How It Works

```
PDF → Chunking → Embeddings (local) → FAISS Index
                                            ↓
Question → Embed query → Retrieve top-k chunks → LLM → Answer
```

1. **Load & Split** — PDF is loaded and split into overlapping chunks
2. **Embed** — Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)
3. **Index** — Vectors stored in a FAISS index for fast similarity search
4. **Retrieve** — Top-k most relevant chunks are retrieved for each query
5. **Generate** — A Hugging Face LLM synthesizes the final answer from retrieved context

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Framework | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (in-memory) |
| LLM | `Qwen/Qwen2.5-72B-Instruct` |
| UI | Streamlit |
| PDF Parsing | PyPDF |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/ntaliaa/documind-rag.git
cd documind-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Hugging Face token
Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Read access

### 4. Run the app
```bash
streamlit run app.py
```

---

## 💡 Usage

1. Paste your Hugging Face token in the sidebar
2. Upload any PDF (reports, papers, contracts, notes...)
3. Ask questions in natural language
4. The app highlights which parts of the document it used to answer

---

## 🔧 Configuration

In `rag_pipeline.py` you can tune:

```python
CHUNK_SIZE = 500        # characters per chunk — increase for longer context
CHUNK_OVERLAP = 80      # overlap to preserve context across chunks
TOP_K_CHUNKS = 4        # how many chunks to retrieve per query

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"

```

---

## 📁 Project Structure

```
documind-rag/
│
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG logic (load, embed, retrieve, generate)
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Support for multiple PDFs simultaneously
- [ ] Persistent vector store (save/load FAISS index)
- [ ] Swap LLM for local model (Ollama integration)
- [ ] Add document summarization feature
- [ ] Deploy to Hugging Face Spaces

---

## 👩‍💻 Author

**Aikaterina Aslanidou** — AI/ML Engineer  
[GitHub](https://github.com/ntaliaa) · [LinkedIn](https://linkedin.com/in/katerina-aslanidou-6101b4230)
