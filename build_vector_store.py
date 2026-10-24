"""
Vector Store Builder for RAG Pipeline
Generates sentence-transformer embeddings and stores document chunks in a local persistent ChromaDB collection.
"""

import os
import sys
import argparse
import io
import contextlib
import warnings

# Suppress backend environment warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import chromadb
from chunk_document import chunk_document


DEFAULT_DB_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "networking_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DOC_PATH = "data/source_document.txt"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Loads and returns the local sentence-transformer embedding model with clean output."""
    print(f"[*] Loading embedding model: '{model_name}'...")
    with contextlib.redirect_stderr(io.StringIO()):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    return model


def get_chroma_client(db_path: str = DEFAULT_DB_PATH) -> chromadb.PersistentClient:
    """
    Initializes and returns a persistent ChromaDB client pointing to local storage.
    """
    resolved_path = os.path.abspath(db_path)
    os.makedirs(resolved_path, exist_ok=True)
    return chromadb.PersistentClient(path=resolved_path)


def build_vector_store(
    doc_path: str = DEFAULT_DOC_PATH,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = 250,
    overlap: int = 50,
    rebuild: bool = True
) -> chromadb.Collection:
    """
    Splits the source document, embeds chunks locally, and stores them in persistent ChromaDB.

    Args:
        doc_path: Path to source document.
        db_path: Path to ChromaDB persistent storage directory.
        collection_name: Name of ChromaDB collection.
        model_name: Sentence-transformer model identifier.
        chunk_size: Word count target per chunk.
        overlap: Overlap in words between consecutive chunks.
        rebuild: If True, purges existing collection and rebuilds from scratch to prevent stale/duplicate chunks.

    Returns:
        The populated ChromaDB Collection object.
    """
    print("=" * 70)
    print("BUILDING LOCAL CHROMADB VECTOR STORE")
    print("=" * 70)

    # 1. Chunk document
    print(f"[*] Step 1: Chunking source document '{doc_path}'...")
    chunks = chunk_document(file_path=doc_path, chunk_size=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    print(f"[+] Document split into {total_chunks} overlapping chunks (size={chunk_size}, overlap={overlap}).")

    if total_chunks == 0:
        raise ValueError("No chunks were generated from the source document.")

    # 2. Generate embeddings locally
    print(f"\n[*] Step 2: Generating embeddings using {model_name}...")
    model = get_embedding_model(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embedding_dim = len(embeddings[0])
    print(f"[+] Computed {len(embeddings)} chunk embeddings. Dimension: {embedding_dim}")

    # 3. Connect to Persistent ChromaDB
    print(f"\n[*] Step 3: Initializing persistent ChromaDB at '{os.path.abspath(db_path)}'...")
    client = get_chroma_client(db_path)

    # Rebuild handling: Delete existing collection if requested to avoid duplicate or orphaned chunks
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        if rebuild:
            print(f"[*] Collection '{collection_name}' already exists. Deleting for clean rebuild...")
            client.delete_collection(name=collection_name)
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[+] Created fresh collection '{collection_name}' with cosine distance.")
        else:
            print(f"[*] Collection '{collection_name}' exists. Will upsert chunks...")
            collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[+] Created new collection '{collection_name}' with cosine distance.")

    # 4. Prepare data for ChromaDB
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "source": str(c["source"]),
            "chunk_index": int(c["chunk_index"]),
            "word_count": int(c["word_count"]),
            "char_count": int(c["char_count"]),
            "start_word_idx": int(c["start_word_idx"]),
            "end_word_idx": int(c["end_word_idx"])
        }
        for c in chunks
    ]
    # Convert numpy embeddings to float lists for ChromaDB storage
    embeddings_list = [emb.tolist() for emb in embeddings]

    # 5. Upsert chunks into ChromaDB
    print(f"\n[*] Step 4: Storing {len(ids)} chunks and vectors into ChromaDB...")
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings_list,
        metadatas=metadatas
    )

    stored_count = collection.count()
    print(f"[+] Successfully stored and persisted {stored_count} chunks in ChromaDB!")

    return collection


def verify_vector_store(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    sample_query: str = "How does the TCP three-way handshake establish a connection?"
):
    """
    Reopens the persistent ChromaDB collection from disk and executes a test query
    to verify that storage, indexing, and retrieval operate properly.
    """
    print("\n" + "=" * 70)
    print("VERIFYING PERSISTENT VECTOR STORE VIA DIRECT QUERY")
    print("=" * 70)

    client = get_chroma_client(db_path)
    collection = client.get_collection(name=collection_name)
    count = collection.count()
    print(f"[*] Reopened collection: '{collection_name}'")
    print(f"[*] Total indexed chunks in collection: {count}")

    if count == 0:
        print("[-] Collection is empty! Verification failed.", file=sys.stderr)
        return False

    print(f"[*] Running verification query: \"{sample_query}\"")
    model = get_embedding_model(model_name)
    query_emb = model.encode([sample_query], normalize_embeddings=True)[0].tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print("\n[+] Verification Query Results:")
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    retrieved_dists = results["distances"][0]

    for rank, (chunk_id, doc, meta, dist) in enumerate(zip(retrieved_ids, retrieved_docs, retrieved_metas, retrieved_dists), start=1):
        # In cosine distance: similarity = 1 - distance
        similarity = 1.0 - dist
        snippet = doc[:180].replace("\n", " ")
        print(f"\n  Rank {rank}: [{chunk_id}] (Cosine Similarity: {similarity:.4f}, Distance: {dist:.4f})")
        print(f"  Metadata: Chunk #{meta['chunk_index']} | {meta['word_count']} words | Source: {meta['source']}")
        print(f"  Content Preview: \"{snippet}...\"")

    print("\n[+] Vector store verification succeeded! All chunks persisted and retrievable.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build and verify persistent ChromaDB vector store.")
    parser.add_argument("--doc", default=DEFAULT_DOC_PATH, help="Path to source document text file.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Directory for persistent ChromaDB storage.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Collection name in ChromaDB.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer model name.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Target chunk word count.")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in words.")
    parser.add_argument("--no-rebuild", action="store_true", help="Do not delete existing collection before building.")
    args = parser.parse_args()

    # Resolve document path relative to script directory if needed
    doc_path = args.doc
    if not os.path.exists(doc_path):
        alt_doc = os.path.join(os.path.dirname(__file__), doc_path)
        if os.path.exists(alt_doc):
            doc_path = alt_doc

    # Resolve db path relative to script directory if relative
    db_path = args.db_path
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(__file__), db_path)

    try:
        build_vector_store(
            doc_path=doc_path,
            db_path=db_path,
            collection_name=args.collection,
            model_name=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            rebuild=not args.no_rebuild
        )

        verify_vector_store(
            db_path=db_path,
            collection_name=args.collection,
            model_name=args.model
        )
        return 0
    except Exception as e:
        print(f"\n[-] Fatal error building vector store: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
