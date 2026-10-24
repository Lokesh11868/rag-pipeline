"""
Document Chunking Module for RAG Pipeline
Splits a source document into overlapping chunks with configurable size and overlap.
"""

import os
import sys
import argparse
from typing import List, Dict, Any


def split_text_into_chunks(
    text: str,
    chunk_size: int = 250,
    overlap: int = 50,
    source_name: str = "source_document.txt"
) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping word chunks.

    Args:
        text: The raw input document string.
        chunk_size: Target number of words per chunk (default: 250).
        overlap: Number of words shared between consecutive chunks (default: 50).
        source_name: Name or path of the source document for metadata.

    Returns:
        List of chunk dictionaries containing text, id, word_count, and metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size.")

    # Split text into whitespace-delimited words
    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        # Handle edge case: if remainder is tiny (less than half of overlap) and we already have chunks,
        # merge with previous chunk to avoid a dangling, fragmented chunk
        if len(chunk_words) < max(15, overlap // 2) and chunks:
            chunks[-1]["text"] += " " + chunk_text
            chunks[-1]["word_count"] = len(chunks[-1]["text"].split())
            chunks[-1]["char_count"] = len(chunks[-1]["text"])
            chunks[-1]["end_word_idx"] = end
            break

        chunk_id = f"chunk_{len(chunks):03d}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "chunk_index": len(chunks),
            "word_count": len(chunk_words),
            "char_count": len(chunk_text),
            "start_word_idx": start,
            "end_word_idx": end,
            "source": source_name
        })

        if end >= total_words:
            break

        start += step

    return chunks


def chunk_document(
    file_path: str = "data/source_document.txt",
    chunk_size: int = 250,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Reads a text file and chunks it using sliding word window with overlap.

    Args:
        file_path: Path to the source text document.
        chunk_size: Target words per chunk.
        overlap: Overlap in words between consecutive chunks.

    Returns:
        List of chunk dictionaries.
    """
    if not os.path.isabs(file_path):
        # Resolve relative to current script directory or working directory
        possible_paths = [
            file_path,
            os.path.join(os.path.dirname(__file__), file_path),
            os.path.join(os.path.dirname(__file__), "data", os.path.basename(file_path))
        ]
        resolved = None
        for p in possible_paths:
            if os.path.exists(p):
                resolved = p
                break
        if resolved is None:
            raise FileNotFoundError(f"Source document not found at: {file_path}")
        file_path = resolved

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    source_name = os.path.basename(file_path)
    return split_text_into_chunks(content, chunk_size=chunk_size, overlap=overlap, source_name=source_name)


def main():
    parser = argparse.ArgumentParser(description="Split source document into overlapping chunks.")
    parser.add_argument("--file", default="data/source_document.txt", help="Path to source document text file.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Target words per chunk (default: 250).")
    parser.add_argument("--overlap", type=int, default=50, help="Word overlap between consecutive chunks (default: 50).")
    args = parser.parse_args()

    print("=" * 70)
    print("PROMPTFORGE RAG PIPELINE - DOCUMENT CHUNKING MODULE")
    print("=" * 70)

    try:
        # Determine actual file path
        file_path = args.file
        if not os.path.exists(file_path):
            alt_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(alt_path):
                file_path = alt_path

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        total_words = len(raw_text.split())
        print(f"[*] Source Document Path : {file_path}")
        print(f"[*] Total Source Words   : {total_words}")
        print(f"[*] Configured Chunk Size: {args.chunk_size} words")
        print(f"[*] Configured Overlap   : {args.overlap} words (effective step: {args.chunk_size - args.overlap} words)")

        chunks = chunk_document(file_path=file_path, chunk_size=args.chunk_size, overlap=args.overlap)

        print(f"\n[+] Total Chunks Generated: {len(chunks)}")

        # Calculate statistics
        lengths = [c["word_count"] for c in chunks]
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        avg_len = sum(lengths) / len(lengths) if lengths else 0

        print(f"[*] Chunk Word Counts    : Min={min_len}, Max={max_len}, Avg={avg_len:.1f}")

        # Summary of each chunk
        print("\n--- Chunk Overview ---")
        for c in chunks:
            print(f"  [{c['id']}] Words: {c['word_count']:3d} | Range: {c['start_word_idx']:4d} to {c['end_word_idx']:4d}")

        # Overlap verification
        print("\n--- Overlap Verification Between Consecutive Chunks ---")
        if len(chunks) > 1:
            for i in range(min(3, len(chunks) - 1)):
                c1 = chunks[i]
                c2 = chunks[i+1]
                words1 = c1["text"].split()
                words2 = c2["text"].split()
                # Last N words of c1 and first N words of c2
                overlap_c1_tail = " ".join(words1[-args.overlap:])
                overlap_c2_head = " ".join(words2[:args.overlap])
                is_exact = (overlap_c1_tail == overlap_c2_head)
                print(f"  Chunk {c1['id']} -> Chunk {c2['id']} overlap match: {'EXACT MATCH (Confirmed)' if is_exact else 'Partial/Bound'}")
                print(f"    Tail of {c1['id']}: '...{overlap_c1_tail[:60]}...'")
                print(f"    Head of {c2['id']}: '...{overlap_c2_head[:60]}...'")

        # Preview representative chunks
        print("\n--- Representative Chunk Previews ---")
        preview_indices = [0, len(chunks) // 2, len(chunks) - 1]
        for idx in preview_indices:
            c = chunks[idx]
            print(f"\n>> PREVIEW: {c['id']} (Index {c['chunk_index']}, {c['word_count']} words):")
            snippet = c["text"][:220].replace("\n", " ")
            print(f"   \"{snippet}...\"")

        print("\n[+] Document chunking completed successfully.")
        return 0

    except Exception as e:
        print(f"\n[-] Error during document chunking: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
