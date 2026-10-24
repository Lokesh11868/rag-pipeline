"""
End-to-End Test Suite for RAG Pipeline
Verifies:
1. Document handling & word count constraints (2,000–5,000 words)
2. Chunking logic, boundary handling, and overlap continuity
3. Local sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dims)
4. ChromaDB persistence, collection loading, and direct querying
5. Semantic retrieval quality on in-scope questions
6. Hallucination checks & orthogonal retrieval on out-of-scope questions
7. Prompt isolation & strict grounding rules
8. Security audits (.env.example placeholder, .gitignore exclusions)
"""

import os
import sys
import io
import contextlib
import unittest
import warnings

# Suppress backend environment warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import chromadb
from sentence_transformers import SentenceTransformer
from chunk_document import chunk_document, split_text_into_chunks
from rag_query import (
    retrieve_relevant_chunks,
    build_rag_prompt,
    GROUNDING_FALLBACK_TEXT,
    DEFAULT_DB_PATH,
    DEFAULT_COLLECTION_NAME
)


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.doc_path = os.path.join(cls.script_dir, "data", "source_document.txt")
        cls.db_path = os.path.join(cls.script_dir, "chroma_db")
        with contextlib.redirect_stderr(io.StringIO()):
            cls.model = SentenceTransformer("all-MiniLM-L6-v2")

    def test_01_source_document_word_count(self):
        """Verifies source document exists and is strictly within 2,000–5,000 words."""
        self.assertTrue(os.path.exists(self.doc_path), f"Document not found at {self.doc_path}")
        with open(self.doc_path, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        word_count = len(words)
        print(f"\n[TEST 01] Source Document Word Count: {word_count} words.")
        self.assertGreaterEqual(word_count, 2000, "Source document has fewer than 2,000 words.")
        self.assertLessEqual(word_count, 5000, "Source document exceeds 5,000 words.")

    def test_02_chunking_and_overlap(self):
        """Verifies chunking produces non-empty overlapping chunks with exact overlap."""
        chunks = chunk_document(file_path=self.doc_path, chunk_size=250, overlap=50)
        self.assertGreater(len(chunks), 0, "No chunks generated.")
        print(f"\n[TEST 02] Generated {len(chunks)} chunks.")

        for i in range(len(chunks) - 1):
            c1 = chunks[i]
            c2 = chunks[i+1]
            self.assertGreater(c1["word_count"], 0, f"Chunk {c1['id']} is empty.")
            # Verify overlap between consecutive chunks
            w1 = c1["text"].split()
            w2 = c2["text"].split()
            tail_w1 = w1[-50:]
            head_w2 = w2[:50]
            self.assertEqual(tail_w1, head_w2, f"Overlap mismatch between chunk {i} and {i+1}")

    def test_03_embedding_dimension(self):
        """Verifies that all-MiniLM-L6-v2 generates 384-dimensional embeddings."""
        emb = self.model.encode("Computer networking test sentence", normalize_embeddings=True)
        self.assertEqual(len(emb), 384, f"Expected 384 dimensions, got {len(emb)}")
        print(f"\n[TEST 03] Verified all-MiniLM-L6-v2 embedding dimension: {len(emb)}")

    def test_04_chromadb_persistence(self):
        """Verifies ChromaDB collection exists on disk and can be reopened with stored chunks."""
        client = chromadb.PersistentClient(path=self.db_path)
        col = client.get_collection(DEFAULT_COLLECTION_NAME)
        count = col.count()
        print(f"\n[TEST 04] ChromaDB persistent chunks count: {count}")
        self.assertGreaterEqual(count, 15, "Fewer chunks than expected in vector store.")

    def test_05_semantic_retrieval_answerable(self):
        """Verifies semantic search retrieves relevant chunks for factual questions."""
        q = "What are the four phases of TCP congestion control?"
        chunks = retrieve_relevant_chunks(q, top_k=3, db_path=self.db_path)
        self.assertGreaterEqual(len(chunks), 1)
        combined_text = " ".join(c["text"] for c in chunks)
        print(f"\n[TEST 05] Query: '{q}' -> Top Chunk: {chunks[0]['id']} (Sim: {chunks[0]['similarity']:.4f})")
        self.assertIn("Slow Start", combined_text)
        self.assertIn("Congestion Avoidance", combined_text)
        self.assertGreater(chunks[0]["similarity"], 0.45)

    def test_06_unanswerable_query_orthogonality(self):
        """Verifies out-of-domain questions have low similarity and clear distinction."""
        unanswerable_q = "What were Cisco Systems quarterly stock earnings and fiscal revenue in 2024?"
        chunks = retrieve_relevant_chunks(unanswerable_q, top_k=3, db_path=self.db_path)
        self.assertGreaterEqual(len(chunks), 1)
        top_sim = chunks[0]["similarity"]
        print(f"\n[TEST 06] Out-of-scope query: '{unanswerable_q[:35]}...' -> Top Sim: {top_sim:.4f}")
        # Cosine similarity for completely unrelated financial topics should be very low (< 0.35)
        self.assertLess(top_sim, 0.35, "Out-of-scope question retrieved unexpectedly high similarity.")

    def test_07_strict_grounding_prompt_structure(self):
        """Verifies system and user prompt maintain strict context isolation and negative constraint."""
        chunks = [{"id": "c1", "text": "TCP uses a three-way handshake.", "similarity": 0.85}]
        prompt = build_rag_prompt("How does TCP connect?", chunks, strict_grounding=True)
        self.assertIn(GROUNDING_FALLBACK_TEXT, prompt["system"])
        self.assertIn("=== BEGIN RETRIEVED CONTEXT ===", prompt["user"])
        self.assertIn("=== END RETRIEVED CONTEXT ===", prompt["user"])
        self.assertIn("USER QUESTION: How does TCP connect?", prompt["user"])

    def test_08_security_and_secrets(self):
        """Verifies .env.example contains only placeholder and .gitignore protects secrets."""
        env_ex_path = os.path.join(self.script_dir, ".env.example")
        gitignore_path = os.path.join(self.script_dir, ".gitignore")

        self.assertTrue(os.path.exists(env_ex_path))
        with open(env_ex_path, "r", encoding="utf-8") as f:
            env_content = f.read()
        self.assertIn("your_groq_api_key_here", env_content)
        self.assertNotIn("gsk_", env_content)

        self.assertTrue(os.path.exists(gitignore_path))
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
        self.assertIn(".env", gi_content)
        self.assertIn("__pycache__", gi_content)


if __name__ == "__main__":
    unittest.main()
