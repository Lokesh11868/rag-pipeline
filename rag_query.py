"""
RAG Query Module & Interactive CLI
Retrieves relevant document chunks from local ChromaDB and generates strictly grounded answers using Groq LLM.
"""

import os
import sys
import argparse
import io
import contextlib
import warnings
from typing import List, Dict, Any, Optional

# Suppress backend environment warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import chromadb

# Find and load .env file from script directory or current working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POSSIBLE_ENV_PATHS = [
    os.path.join(_SCRIPT_DIR, ".env"),
    os.path.join(os.getcwd(), ".env"),
    os.path.join(os.getcwd(), "rag-pipeline", ".env")
]
for env_p in _POSSIBLE_ENV_PATHS:
    if os.path.exists(env_p):
        load_dotenv(dotenv_path=env_p)
        break
else:
    load_dotenv()

DEFAULT_DB_PATH = os.path.join(_SCRIPT_DIR, "chroma_db")
DEFAULT_COLLECTION_NAME = "networking_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"
GROUNDING_FALLBACK_TEXT = "I don't know based on the provided context."

# Global cached embedding model and ChromaDB collection for fast interactive response
_CACHED_MODEL = None
_CACHED_COLLECTION = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Loads and caches the local SentenceTransformer embedding model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        with contextlib.redirect_stderr(io.StringIO()):
            from sentence_transformers import SentenceTransformer
            _CACHED_MODEL = SentenceTransformer(model_name)
    return _CACHED_MODEL


def get_collection(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> chromadb.Collection:
    """Connects to local persistent ChromaDB and retrieves the specified collection."""
    global _CACHED_COLLECTION
    if _CACHED_COLLECTION is None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"ChromaDB persistent directory not found at: '{db_path}'.\n"
                f"Please run 'python build_vector_store.py' first to build and index the collection."
            )
        client = chromadb.PersistentClient(path=db_path)
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            raise KeyError(
                f"Collection '{collection_name}' not found in ChromaDB.\n"
                f"Available collections: {existing}\n"
                f"Please run 'python build_vector_store.py' to generate the collection."
            )
        _CACHED_COLLECTION = client.get_collection(name=collection_name)
    return _CACHED_COLLECTION


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 3,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> List[Dict[str, Any]]:
    """
    Embeds the user's question and retrieves the top-k most similar chunks from ChromaDB.

    Returns:
        List of dicts containing chunk id, text, similarity score, distance, and metadata.
    """
    model = get_embedding_model(model_name)
    collection = get_collection(db_path, collection_name)

    # Generate query embedding using the same model as the indexed documents
    query_emb = model.encode([question], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results and results.get("ids") and len(results["ids"][0]) > 0:
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # In cosine space with normalized vectors, similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)
            chunks.append({
                "id": chunk_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity
            })

    return chunks


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    strict_grounding: bool = True
) -> Dict[str, str]:
    """
    Constructs the system prompt and user prompt for the Groq LLM.
    Strictly isolates the retrieved context from the question to prevent injection.
    """
    if strict_grounding:
        system_instruction = (
            "You are a strict, grounded question-answering assistant.\n"
            "Your task is to answer the user's question ONLY using the factual statements directly present "
            "in the provided context below.\n"
            "Rules you must obey unconditionally:\n"
            "1. Answer ONLY using the facts stated in the provided context.\n"
            "2. If the answer cannot be found in the provided context, respond EXACTLY with:\n"
            "   I don't know based on the provided context.\n"
            "3. Do NOT extrapolate, speculate, assume, or introduce outside knowledge from prior training.\n"
            "4. Do NOT attempt to answer questions about people, events, dates, or topics not mentioned in the context.\n"
            "5. If only partial information is present, answer only the part that is directly supported, or use the required fallback if the core question is unanswered."
        )
    else:
        system_instruction = (
            "You are an AI assistant. Answer the user's question. If the provided context is helpful, "
            "use it; otherwise, feel free to use your general knowledge."
        )

    # Format the retrieved context cleanly
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"--- Context Chunk {idx} (ID: {c['id']}, Similarity: {c['similarity']:.3f}) ---\n"
            f"{c['text'].strip()}"
        )

    joined_context = "\n\n".join(context_blocks) if context_blocks else "No relevant context retrieved."

    user_content = (
        f"=== BEGIN RETRIEVED CONTEXT ===\n"
        f"{joined_context}\n"
        f"=== END RETRIEVED CONTEXT ===\n\n"
        f"USER QUESTION: {question}\n\n"
        f"INSTRUCTION: Answer strictly according to the system rules above."
    )

    return {
        "system": system_instruction,
        "user": user_content
    }


def call_groq_llm(
    prompt: Dict[str, str],
    model_name: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512
) -> str:
    """
    Sends the constructed RAG prompt to the Groq API.
    Handles API keys, fallback models, and provides clear actionable errors.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or "your_groq_api_key_here" in api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured.\n"
            "Please create a .env file in the project directory with your valid key:\n"
            "    GROQ_API_KEY=gsk_your_actual_groq_key_here\n"
            "See .env.example for reference."
        )

    from groq import Groq
    client = Groq(api_key=api_key.strip())

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]}
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        err_msg = str(e)
        # Attempt fallback model if the primary model is unavailable or deprecated
        if "model" in err_msg.lower() or "not found" in err_msg.lower():
            try:
                print(f"[!] Primary model '{model_name}' failed. Retrying with fallback '{FALLBACK_GROQ_MODEL}'...")
                completion = client.chat.completions.create(
                    model=FALLBACK_GROQ_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return completion.choices[0].message.content.strip()
            except Exception as e_inner:
                raise RuntimeError(f"Groq API call failed with fallback model: {e_inner}") from e_inner
        raise RuntimeError(f"Groq API call failed: {err_msg}") from e


def rag_query(
    question: str,
    top_k: int = 3,
    debug: bool = False,
    strict_grounding: bool = True,
    model_name: str = DEFAULT_GROQ_MODEL,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> Dict[str, Any]:
    """
    Full end-to-end RAG pipeline function:
    1. Embeds question.
    2. Queries ChromaDB for top-k chunks.
    3. Builds prompt with strict grounding.
    4. Calls Groq LLM.
    5. Returns answer and retrieval metadata.

    Args:
        question: User's query.
        top_k: Number of chunks to retrieve.
        debug: If True, prints retrieval debug info.
        strict_grounding: Whether to enforce strict grounding in prompt.
        model_name: Groq LLM model name.
        db_path: Persistent ChromaDB storage path.
        collection_name: ChromaDB collection name.
        embedding_model: Local sentence-transformer model name.

    Returns:
        Dict containing answer, retrieved_chunks, prompt, and execution stats.
    """
    question = question.strip()
    if not question:
        return {
            "question": "",
            "answer": "Please provide a non-empty question.",
            "retrieved_chunks": [],
            "prompt": {},
            "model": model_name
        }

    # 1. Retrieve chunks
    chunks = retrieve_relevant_chunks(
        question=question,
        top_k=top_k,
        db_path=db_path,
        collection_name=collection_name,
        model_name=embedding_model
    )

    # 2. Build prompt
    prompt = build_rag_prompt(question, chunks, strict_grounding=strict_grounding)

    # 3. Call LLM
    try:
        answer = call_groq_llm(prompt, model_name=model_name)
    except Exception as e:
        answer = f"[ERROR Calling Groq LLM: {e}]"

    # 4. Optional debug output
    if debug:
        print("\n" + "-" * 60)
        print(f"DEBUG: Retrieved {len(chunks)} chunks for query: '{question}'")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] ID: {c['id']} | Cosine Similarity: {c['similarity']:.4f} | Distance: {c['distance']:.4f}")
            print(f"      Source: {c['metadata'].get('source')} | Chunk #{c['metadata'].get('chunk_index')}")
            snippet = c["text"][:160].replace("\n", " ")
            print(f"      Text: \"{snippet}...\"")
        print("-" * 60 + "\n")

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": chunks,
        "prompt": prompt,
        "model": model_name,
        "strict_grounding": strict_grounding
    }


def run_interactive_cli():
    """Runs the interactive command-line interface for the RAG pipeline."""
    print("=" * 70)
    print("      PROMPTFORGE RAG ASSISTANT - INTERACTIVE CLI (WEEK 3)")
    print("=" * 70)
    print("  Knowledge Base : Computer Networking Fundamentals")
    print(f"  Embedding Model: {DEFAULT_EMBEDDING_MODEL} (Local sentence-transformers)")
    print(f"  Vector Store   : ChromaDB (Local persistent storage)")
    print(f"  LLM Provider   : Groq ({DEFAULT_GROQ_MODEL})")
    print("-" * 70)
    print("Commands:")
    print("  - Type any question to get a grounded answer.")
    print("  - 'debug'           : Toggle retrieved-chunk inspection mode on/off.")
    print("  - 'k <number>'      : Set number of chunks to retrieve (e.g. 'k 4').")
    print("  - 'grounding on/off': Toggle strict grounding mode.")
    print("  - 'exit' or 'quit'  : Exit the CLI.")
    print("=" * 70)

    # Check vector store readiness before prompting
    try:
        col = get_collection()
        total_chunks = col.count()
        print(f"[*] ChromaDB collection loaded successfully. Indexed chunks: {total_chunks}")
    except Exception as e:
        print(f"\n[!] Warning: Could not open vector store: {e}")
        print("[!] Please run 'python build_vector_store.py' first.\n")

    # Check API key status
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "your_groq_api_key_here" in api_key:
        print("[!] Note: GROQ_API_KEY is not yet configured in .env.")
        print("[!] Chunks can still be retrieved in debug mode, but LLM generation requires a valid key.\n")

    debug_mode = False
    top_k = 3
    strict_grounding = True

    while True:
        try:
            prompt_label = f"\n[k={top_k}|debug={'ON' if debug_mode else 'OFF'}|grounding={'STRICT' if strict_grounding else 'RELAXED'}]"
            user_input = input(f"{prompt_label}\nEnter question (or command): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            print("[!] Please enter a valid question or command.")
            continue

        cmd = user_input.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Session terminated. Goodbye!")
            break

        if cmd == "debug":
            debug_mode = not debug_mode
            print(f"[+] Debug mode is now: {'ON (retrieved chunks will be displayed)' if debug_mode else 'OFF'}")
            continue

        if cmd.startswith("k "):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                val = int(parts[1])
                if 1 <= val <= 10:
                    top_k = val
                    print(f"[+] Retrieval count set to top-{top_k} chunks.")
                else:
                    print("[!] Please specify a value between 1 and 10.")
            else:
                print("[!] Usage: k <number> (e.g. 'k 4')")
            continue

        if cmd == "grounding on":
            strict_grounding = True
            print("[+] Strict grounding ENABLED: Model must decline out-of-scope questions.")
            continue
        elif cmd == "grounding off":
            strict_grounding = False
            print("[+] Strict grounding DISABLED: Model may use general knowledge.")
            continue

        # Execute RAG query
        print(f"\n[*] Searching vector store for top-{top_k} relevant chunks...")
        try:
            result = rag_query(
                question=user_input,
                top_k=top_k,
                debug=debug_mode,
                strict_grounding=strict_grounding
            )

            print("\n" + "=" * 50 + " GROUNDED ANSWER " + "=" * 50)
            print(result["answer"])
            print("=" * 117)

        except Exception as e:
            print(f"\n[-] Query processing error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline or launch the interactive CLI.")
    parser.add_argument("--query", "-q", type=str, default=None, help="Direct question to answer.")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument("--debug", "-d", action="store_true", help="Print retrieved context and similarity scores.")
    parser.add_argument("--no-strict", action="store_true", help="Disable strict grounding (allow general knowledge).")
    args = parser.parse_args()

    if args.query:
        # Non-interactive single query mode
        res = rag_query(
            question=args.query,
            top_k=args.top_k,
            debug=args.debug,
            strict_grounding=not args.no_strict
        )
        print("\n=== ANSWER ===")
        print(res["answer"])
        return 0
    else:
        # Interactive CLI
        run_interactive_cli()
        return 0


if __name__ == "__main__":
    sys.exit(main())
