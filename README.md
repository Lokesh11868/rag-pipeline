# RAG Pipeline: Grounded Question-Answering System

## Overview

The **RAG Pipeline** is an end-to-end Retrieval-Augmented Generation system that enables a large language model to answer domain-specific questions grounded strictly in local reference documentation. 

In Week 1, we learned to invoke LLMs via the Groq API. In Week 2, we implemented semantic search using dense vector embeddings and manual cosine similarity. In Week 3, we synthesize these concepts into a production-grade RAG architecture backed by **ChromaDB**, an embedded local vector database, and the **Groq API** for high-speed inference with strict hallucination controls.


## Features

- **Sliding-Window Document Chunking:** Splits long-form technical texts into configurable word chunks (default: 250 words) with parameterizable overlap (default: 50 words) to prevent information loss across segment boundaries.
- **Local Sentence-Transformers:** Uses `all-MiniLM-L6-v2` locally to produce 384-dimensional dense semantic embeddings without external API latency or costs.
- **Persistent ChromaDB Storage:** Automatically creates, indexes, and persists vectors and structured metadata on local disk in `./chroma_db/` with zero server setup.
- **Semantic Vector Retrieval:** Retrieves the top-$k$ most relevant context chunks using cosine similarity ($1 - \text{distance}$) and HNSW approximate nearest neighbors.
- **Strict Grounding Guardrails:** Employs prompt isolation and strict negative constraints: if the requested facts are absent from the retrieved context, the LLM unconditionally responds with:
  > `I don't know based on the provided context.`
- **Interactive Command-Line Interface (CLI):** Full-featured terminal interface supporting live query input, dynamic top-$k$ tuning (`k <num>`), grounding mode toggling (`grounding on/off`), and real-time retrieved chunk inspection (`debug`).
- **Hallucination & Chunk-Size Benchmarking:** Repeatable evaluation suite comparing small (100w), default (250w), and large (500w) chunks alongside supported vs. out-of-domain queries.

---

## Tech Stack

- **Language:** Python 3.10+
- **Vector Database:** `chromadb` (v1.5.9)
- **Embedding Model:** `sentence-transformers` (v2.3.1) with `all-MiniLM-L6-v2`
- **LLM Inference:** `groq` (v0.33.0) via `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`
- **Environment & Utilities:** `python-dotenv`, `numpy`

---

## Architecture Flow

```text
[ source_document.txt (4,028 words) ]
                 │
                 ▼
    [ chunk_document.py ]
    (250-word chunks with 50-word overlap)
                 │
                 ▼
    [ sentence-transformers: all-MiniLM-L6-v2 ]
    (384-dim dense vector generation)
                 │
                 ▼
    [ ChromaDB Persistent Collection ]
    (Local storage in ./chroma_db)
                 │
  ┌──────────────┴──────────────┐
  │ User Question via CLI       │
  ▼                             ▼
[ Embed Query ] ──► [ Top-k Cosine Search ] ──► [ Retrieved Chunks + Metadata ]
                                                           │
                                                           ▼
                                            [ Strict Grounding Prompt Template ]
                                                           │
                                                           ▼
                                                [ Groq LLM Inference ]
                                                           │
                                                           ▼
                                                [ Final Grounded Answer ]
                                              (or Exact Fallback if Absent)
```

---

## Project Structure

```text
rag-pipeline/
├── data/
│   └── source_document.txt     # Reference document on Computer Networking (4,028 words)
├── chroma_db/                  # Persistent local ChromaDB database (auto-generated)
├── chunk_document.py           # Document chunking module with sliding window overlap
├── build_vector_store.py       # Embedding generation & ChromaDB persistent indexing
├── rag_query.py                # Core RAG pipeline & interactive CLI interface
├── evaluate_experiments.py     # Evaluation harness for chunk size & hallucination tests
├── experiment_results.json     # Saved metrics and benchmark outputs
├── reflections.md              # Detailed reflection write-up answering Week 3 questions
├── requirements.txt            # Minimal verified Python dependencies
├── .env.example                # Example configuration template (placeholder only)
├── .gitignore                  # Git ignore rules for secrets, virtual environments, caches
└── README.md                   # Comprehensive project documentation
```

---

## Prerequisites

1. **Python 3.10 or higher** installed on your system.
2. A **Groq API Key** (obtainable free of charge at [https://console.groq.com](https://console.groq.com)).

---

## Installation & Setup

### 1. Navigate to the project directory
```bash
cd rag-pipeline
```

### 2. Create and activate a virtual environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install required dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Configure your Groq API Key
Copy `.env.example` to a new `.env` file:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On Linux / macOS:**
```bash
cp .env.example .env
```

Open `.env` in any text editor and supply your real Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

> **Security Note:** The `.env` file is excluded in `.gitignore` and must never be committed to public repositories.

---

## Step-by-Step Execution

### Step 1: Verify Document Chunking
Run `chunk_document.py` to inspect the chunking breakdown, word count distribution, and overlap verification:
```bash
python chunk_document.py
```
*Output Summary:* Splits the 4,028-word reference into 20 overlapping chunks (250 words per chunk, 50-word overlap) with exact boundary continuity.

### Step 2: Build the Persistent ChromaDB Vector Store
Run `build_vector_store.py` to compute local embeddings and persist the vector collection:
```bash
python build_vector_store.py
```
*Output Summary:* Computes 384-dimensional embeddings using `all-MiniLM-L6-v2`, populates the `networking_knowledge_base` collection in `./chroma_db/`, and executes a direct verification query demonstrating successful cosine retrieval.

### Step 3: Run the Interactive RAG CLI
Launch the interactive command-line interface:
```bash
python rag_query.py
```

You can also run a direct one-off query with debug logging:
```bash
python rag_query.py --query "What are the four phases of TCP congestion control?" --debug
```

---

## CLI Commands & Usage

When running `python rag_query.py`, the interactive terminal supports several helper commands:

| Command | Action |
| :--- | :--- |
| `<any question>` | Executes semantic retrieval, constructs the prompt, and outputs the grounded answer. |
| `debug` | Toggles retrieved-chunk inspection mode on/off (displays chunk IDs, cosine similarities, and text snippets). |
| `k <number>` | Adjusts retrieval depth (e.g., `k 4` retrieves the top-4 chunks). |
| `grounding on` | Enforces strict grounding (model must decline out-of-scope queries). |
| `grounding off` | Relaxes grounding (allows LLM to use prior training knowledge for comparison). |
| `exit` or `quit` | Terminates the CLI session cleanly. |

---

## Evaluation & Benchmarks

The pipeline was benchmarked using an automated evaluation harness (`evaluate_experiments.py`) on 5 in-domain answerable questions and 3 out-of-domain unanswerable questions.

### 1. In-Domain Answerable Questions (Document-Supported)

| ID | Test Question | Top Retrieved Chunk | Cosine Sim | Grounded Verification |
| :---: | :--- | :---: | :---: | :---: |
| **A1** | *What are the four phases of TCP congestion control?* | `chunk_007` | **0.5944** | Supported: Identifies *Slow Start, Congestion Avoidance, Fast Retransmit, and Fast Recovery*. |
| **A2** | *What are the three private IPv4 address ranges reserved by RFC 1918?* | `chunk_011` | **0.7452** | Supported: Identifies `10.0.0.0/8`, `172.16.0.0/12`, and `192.168.0.0/16`. |
| **A3** | *What algorithm does OSPF use to calculate shortest paths, and how is its metric determined?* | `chunk_013` | **0.5189** | Supported: Identifies *Dijkstra's SPF algorithm* and *Cost = Reference Bandwidth / Interface Bandwidth*. |
| **A4** | *Explain the four steps of the DHCP client configuration lifecycle (the DORA process).* | `chunk_015` | **0.5410** | Supported: Identifies *Discover, Offer, Request, Acknowledge*. |
| **A5** | *What are the two sublayers of the Data Link Layer, and what does each manage?* | `chunk_003` | **0.5150** | Supported: Identifies *MAC (Medium Access Control)* and *LLC (Logical Link Control)* with respective roles. |

### 2. Out-of-Domain Unanswerable Questions (Hallucination Test)

| ID | Out-of-Domain Query | Retrieval Sim | Strict Grounded Response | Fallback Adhered? |
| :---: | :--- | :---: | :--- | :---: |
| **U1** | *What is the quantum key distribution protocol BB84 and how does it prevent eavesdropping?* | 0.2949 (Orthogonal) | `I don't know based on the provided context.` | **YES** |
| **U2** | *What were the quarterly revenue figures and stock market ticker symbol for Cisco Systems in 2024?* | 0.1593 (Orthogonal) | `I don't know based on the provided context.` | **YES** |
| **U3** | *How does the Bitcoin blockchain proof-of-work algorithm adjust mining difficulty?* | 0.1886 (Orthogonal) | `I don't know based on the provided context.` | **YES** |

---

## Chunk Size Experiment Summary

We compared three chunking configurations on the same 4,028-word document:

1. **Small Chunks (100 words, 20-word overlap | 51 chunks):**
   - *Advantage:* High retrieval precision and strong top similarity scores (0.62–0.70).
   - *Disadvantage:* Severely fragmented multi-sentence and multi-step concepts (e.g., separating TCP handshake steps across chunk borders).
2. **Default Chunks (250 words, 50-word overlap | 20 chunks):**
   - *Advantage:* Optimal engineering balance. Retains complete protocol lifecycles in single contexts while maintaining high semantic similarity (0.55–0.69).
3. **Large Chunks (500 words, 100-word overlap | 10 chunks):**
   - *Advantage:* Broad context preserved.
   - *Disadvantage:* Significant relevance dilution (OSPF similarity fell to **0.3575**) because unrelated routing protocols were merged into the same embedding vector.

---

## Limitations

1. **Semantic Search Mismatch:** Dense vector embeddings rely on semantic similarity in latent space. If a user asks a query using obscure terminology not present in the vector space, retrieval may score low despite the document containing the answer.
2. **Context Window Limitations:** Retrieving high numbers of chunks ($k \ge 10$) increases token consumption, latency, and susceptibility to the "lost-in-the-middle" attention phenomenon.
3. **Imperfect Grounding Guarantees:** Prompt instructions drastically reduce hallucinations, but no prompt can provide 100% mathematical immunity against subtle extrapolations. Temperature must be kept at `0.0`.
4. **Static Local Index:** Changes to the source document require re-running `build_vector_store.py` to refresh the ChromaDB embeddings.

---

## Security Best Practices

- **Zero Hardcoded Secrets:** No API keys are embedded in Python files or committed to Git.
- **Local Embedding Execution:** Embeddings are generated entirely locally via PyTorch and `sentence-transformers`; source document text is never transmitted to third parties during indexing.
- **Environment Isolation:** Keys are loaded at runtime from `.env` via `python-dotenv`.
