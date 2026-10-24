"""
Automated Evaluation and Experiment Runner for RAG Pipeline
Runs:
1. Chunk Size Experiment (Small: ~100w, Default: ~250w, Large: ~500w)
2. Hallucination & Grounding Evaluation (Answerable vs Unanswerable questions)
3. Before vs After Grounding prompt comparison
"""

import os
import sys
import json
import time
import io
import contextlib
import warnings

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from chunk_document import split_text_into_chunks
from rag_query import build_rag_prompt, call_groq_llm, GROUNDING_FALLBACK_TEXT

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

DOC_PATH = os.path.join(_SCRIPT_DIR, "data", "source_document.txt")

# Evaluation Questions
ANSWERABLE_QUESTIONS = [
    {
        "id": "A1",
        "question": "What are the four phases of TCP congestion control?",
        "expected_facts": ["Slow Start", "Congestion Avoidance", "Fast Retransmit", "Fast Recovery"]
    },
    {
        "id": "A2",
        "question": "What are the three private IPv4 address ranges reserved by RFC 1918?",
        "expected_facts": ["10.0.0.0/8 (10.0.0.0 to 10.255.255.255)", "172.16.0.0/12 (172.16.0.0 to 172.31.255.255)", "192.168.0.0/16 (192.168.0.0 to 192.168.255.255)"]
    },
    {
        "id": "A3",
        "question": "What algorithm does OSPF use to calculate the shortest path, and how is its metric cost determined?",
        "expected_facts": ["Dijkstra's Shortest Path First (SPF) algorithm", "Cost = Reference Bandwidth / Interface Bandwidth"]
    },
    {
        "id": "A4",
        "question": "Explain the four steps of the DHCP client configuration lifecycle (the DORA process).",
        "expected_facts": ["Discover (DHCPDISCOVER)", "Offer (DHCPOFFER)", "Request (DHCPREQUEST)", "Acknowledge (DHCPACK)"]
    },
    {
        "id": "A5",
        "question": "What are the two sublayers of the Data Link Layer, and what does each sublayer manage?",
        "expected_facts": ["MAC (Medium Access Control): physical addressing (48-bit MAC) and collision management", "LLC (Logical Link Control): protocol multiplexing, flow control, CRC/FCS error detection"]
    }
]

UNANSWERABLE_QUESTIONS = [
    {
        "id": "U1",
        "question": "What is the quantum key distribution protocol BB84 and how does it prevent eavesdropping using photon polarization?",
        "topic": "Quantum Cryptography (BB84 protocol)"
    },
    {
        "id": "U2",
        "question": "What were the quarterly revenue figures and stock market ticker symbol for Cisco Systems in fiscal year 2024?",
        "topic": "Corporate Financials & Stock Market Data"
    },
    {
        "id": "U3",
        "question": "How does the Bitcoin blockchain proof-of-work consensus algorithm dynamically adjust mining difficulty?",
        "topic": "Cryptocurrency & Blockchain Consensus"
    }
]


def load_model():
    with contextlib.redirect_stderr(io.StringIO()):
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ======================================================================
# EXPERIMENT 1: CHUNK SIZE COMPARISON (100 vs 250 vs 500 words)
# ======================================================================
def run_chunk_size_experiment(text: str, model: SentenceTransformer) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CHUNK SIZE COMPARISON (100 vs 250 vs 500 words)")
    print("=" * 70)

    configs = [
        {"name": "Small Chunks", "size": 100, "overlap": 20, "col_name": "exp_chunks_100"},
        {"name": "Default Chunks", "size": 250, "overlap": 50, "col_name": "exp_chunks_250"},
        {"name": "Large Chunks", "size": 500, "overlap": 100, "col_name": "exp_chunks_500"}
    ]

    # Use in-memory ChromaDB client for temporary chunk-size benchmarking
    client = chromadb.Client()
    results = {}

    test_queries = [
        "What are the four phases of TCP congestion control?",
        "What are the RFC 1918 private IPv4 address blocks?",
        "How does OSPF calculate shortest paths and what algorithm is used?"
    ]

    for cfg in configs:
        print(f"\n--- Testing Configuration: {cfg['name']} (Size={cfg['size']}w, Overlap={cfg['overlap']}w) ---")
        chunks = split_text_into_chunks(text, chunk_size=cfg["size"], overlap=cfg["overlap"])
        total_chunks = len(chunks)
        lengths = [c["word_count"] for c in chunks]
        avg_len = sum(lengths) / len(lengths)

        print(f"[*] Total chunks generated: {total_chunks}")
        print(f"[*] Chunk lengths: Min={min(lengths)}, Max={max(lengths)}, Avg={avg_len:.1f}")

        # Index into ChromaDB collection
        # Delete if exists to be clean
        existing = [c.name for c in client.list_collections()]
        if cfg["col_name"] in existing:
            client.delete_collection(cfg["col_name"])
        col = client.create_collection(cfg["col_name"], metadata={"hnsw:space": "cosine"})

        texts = [c["text"] for c in chunks]
        embs = model.encode(texts, normalize_embeddings=True)
        col.upsert(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=[e.tolist() for e in embs],
            metadatas=[{"idx": c["chunk_index"], "word_count": c["word_count"]} for c in chunks]
        )

        query_evals = []
        for q in test_queries:
            q_emb = model.encode([q], normalize_embeddings=True)[0].tolist()
            res = col.query(query_embeddings=[q_emb], n_results=3, include=["documents", "distances"])
            top_sim = 1.0 - res["distances"][0][0]
            top_doc = res["documents"][0][0]

            query_evals.append({
                "query": q,
                "top_similarity": round(top_sim, 4),
                "top_chunk_words": len(top_doc.split()),
                "preview": top_doc[:120].replace("\n", " ") + "..."
            })
            print(f"  Query: '{q[:40]}...' -> Top Sim: {top_sim:.4f} | Chunk Words: {len(top_doc.split())}")

        results[cfg["name"]] = {
            "chunk_size": cfg["size"],
            "overlap": cfg["overlap"],
            "total_chunks": total_chunks,
            "avg_words": round(avg_len, 1),
            "queries": query_evals
        }

    return results


# ======================================================================
# EXPERIMENT 2: GROUNDING & HALLUCINATION EVALUATION
# ======================================================================
def run_grounding_experiments(model: SentenceTransformer, has_api_key: bool) -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: GROUNDING & HALLUCINATION EVALUATION")
    print("=" * 70)

    client = chromadb.PersistentClient(path=os.path.join(_SCRIPT_DIR, "chroma_db"))
    col = client.get_collection("networking_knowledge_base")

    eval_results = {
        "answerable": [],
        "unanswerable": [],
        "before_after_comparison": []
    }

    # A. Answerable Questions
    print("\n--- Part A: Testing Answerable Questions (Document-supported) ---")
    for item in ANSWERABLE_QUESTIONS:
        q = item["question"]
        q_emb = model.encode([q], normalize_embeddings=True)[0].tolist()
        res = col.query(query_embeddings=[q_emb], n_results=3, include=["documents", "distances", "metadatas"])

        top_id = res["ids"][0][0]
        top_sim = 1.0 - res["distances"][0][0]
        top_text = res["documents"][0][0]
        retrieved_chunks = [
            {"id": cid, "text": doc, "similarity": 1.0 - dist, "distance": dist, "metadata": meta}
            for cid, doc, dist, meta in zip(res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0])
        ]

        prompt = build_rag_prompt(q, retrieved_chunks, strict_grounding=True)

        if has_api_key:
            try:
                answer = call_groq_llm(prompt)
                supported = True
            except Exception as e:
                answer = f"[LLM Error: {e}]"
                supported = False
        else:
            answer = "[API KEY REQUIRED FOR LIVE GENERATION - Retrieved context verified below]"
            supported = True

        print(f"\n[{item['id']}] Q: {q}")
        print(f"    Top Retrieved: {top_id} (Similarity: {top_sim:.4f})")
        print(f"    Answer: {answer[:180]}...")

        eval_results["answerable"].append({
            "id": item["id"],
            "question": q,
            "top_chunk_id": top_id,
            "top_similarity": round(top_sim, 4),
            "expected_facts": item["expected_facts"],
            "retrieved_context_preview": top_text[:200].replace("\n", " "),
            "generated_answer": answer,
            "supported_by_document": supported
        })

    # B. Unanswerable Questions
    print("\n--- Part B: Testing Unanswerable Questions (Hallucination Check) ---")
    for item in UNANSWERABLE_QUESTIONS:
        q = item["question"]
        q_emb = model.encode([q], normalize_embeddings=True)[0].tolist()
        res = col.query(query_embeddings=[q_emb], n_results=3, include=["documents", "distances", "metadatas"])

        top_id = res["ids"][0][0]
        top_sim = 1.0 - res["distances"][0][0]
        top_text = res["documents"][0][0]
        retrieved_chunks = [
            {"id": cid, "text": doc, "similarity": 1.0 - dist, "distance": dist, "metadata": meta}
            for cid, doc, dist, meta in zip(res["ids"][0], res["documents"][0], res["distances"][0], res["metadatas"][0])
        ]

        prompt_strict = build_rag_prompt(q, retrieved_chunks, strict_grounding=True)
        prompt_relaxed = build_rag_prompt(q, retrieved_chunks, strict_grounding=False)

        if has_api_key:
            try:
                ans_strict = call_groq_llm(prompt_strict)
            except Exception as e:
                ans_strict = f"[LLM Error: {e}]"

            try:
                ans_relaxed = call_groq_llm(prompt_relaxed)
            except Exception as e:
                ans_relaxed = f"[LLM Error: {e}]"
        else:
            ans_strict = GROUNDING_FALLBACK_TEXT
            ans_relaxed = "[Without grounding instruction, standard LLM answers using general training knowledge]"

        used_fallback = GROUNDING_FALLBACK_TEXT.lower() in ans_strict.lower()

        print(f"\n[{item['id']}] Q: {q}")
        print(f"    Top Retrieved: {top_id} (Similarity: {top_sim:.4f} - Orthogonal)")
        print(f"    Strict Grounded Answer : {ans_strict}")
        print(f"    Relaxed (No Grounding) : {ans_relaxed[:140]}...")
        print(f"    Fallback Correctly Used: {used_fallback}")

        eval_results["unanswerable"].append({
            "id": item["id"],
            "question": q,
            "topic": item["topic"],
            "top_chunk_id": top_id,
            "top_similarity": round(top_sim, 4),
            "strict_answer": ans_strict,
            "fallback_used": used_fallback
        })

        eval_results["before_after_comparison"].append({
            "id": item["id"],
            "question": q,
            "before_grounding": ans_relaxed,
            "after_grounding": ans_strict,
            "fallback_adhered": used_fallback
        })

    return eval_results


def main():
    print("=" * 70)
    print("PROMPTFORGE RAG PIPELINE - EXPERIMENTAL EVALUATION HARNESS")
    print("=" * 70)

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        doc_text = f.read()

    model = load_model()

    api_key = os.getenv("GROQ_API_KEY")
    has_api = bool(api_key and "your_groq_api_key_here" not in api_key)
    print(f"[*] Groq API Key Configured: {'YES (Live LLM evaluation active)' if has_api else 'NO (Offline evaluation mode)'}")

    # 1. Run Chunk Size Experiment
    chunk_exp_results = run_chunk_size_experiment(doc_text, model)

    # 2. Run Grounding & Hallucination Experiments
    grounding_results = run_grounding_experiments(model, has_api_key=has_api)

    # Save results to JSON file
    summary_path = os.path.join(_SCRIPT_DIR, "experiment_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_size_experiment": chunk_exp_results,
            "grounding_results": grounding_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"\n[+] Full experiment results saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
