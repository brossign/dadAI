#!/usr/bin/env python3
"""
DadAI v3 — Step 2: Build the RAG vector database

Reads the chunked book data (from chunk_book.py) and builds a ChromaDB
vector database with sentence-transformer embeddings.

At query time, DadAI will search this database to find the most relevant
book passages for a dad's question, then include them in the prompt.

Usage:
    python scripts/build_rag_db.py
    python scripts/build_rag_db.py --input data/rag_chunks.jsonl --db-path data/rag_db
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


# Use a small, fast embedding model that runs locally on CPU
# all-MiniLM-L6-v2: 384 dimensions, ~80 MB, great quality/speed tradeoff
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector database")
    parser.add_argument("--input", default="data/rag_chunks.jsonl",
                        help="Input JSONL file with chunks")
    parser.add_argument("--db-path", default="data/rag_db",
                        help="Output directory for ChromaDB")
    parser.add_argument("--collection", default="dadai_books",
                        help="Collection name in ChromaDB")
    args = parser.parse_args()

    input_path = Path(args.input)
    db_path = Path(args.db_path)

    if not input_path.exists():
        print(f"Error: Chunks file not found: {input_path}")
        print("Run scripts/chunk_book.py first.")
        raise SystemExit(1)

    # Load chunks
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks from {input_path}")

    # Initialize embedding function
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~80 MB model — subsequent runs are instant)")
    t0 = time.time()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create ChromaDB
    print(f"\nBuilding vector database at {db_path}/")

    # Remove existing DB to rebuild cleanly
    if db_path.exists():
        import shutil
        shutil.rmtree(db_path)
        print("  (removed existing database)")

    client = chromadb.PersistentClient(path=str(db_path))

    collection = client.create_collection(
        name=args.collection,
        embedding_function=ef,
        metadata={"description": "DadAI book knowledge base for RAG"},
    )

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i:04d}")
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk.get("source", "unknown"),
            "chapter": chunk.get("chapter", "unknown"),
            "chunk_type": chunk.get("chunk_type", "narrative"),
        })

    # Add to collection in batches (ChromaDB handles batching internally)
    print(f"\nEmbedding and indexing {len(documents)} chunks...")
    t0 = time.time()

    # ChromaDB has a batch size limit, so add in groups of 100
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {end}/{len(documents)} chunks...")

    elapsed = time.time() - t0
    print(f"\nDone! {len(documents)} chunks indexed in {elapsed:.1f}s")

    # Verify with a test query
    print("\n" + "=" * 60)
    print("Test queries:")
    print("=" * 60)

    test_queries = [
        "I had a terrible father and I'm scared I'll be like him",
        "I don't feel connected to my newborn baby",
        "My wife says I'm emotionally unavailable",
        "How do I break the cycle of absent fatherhood",
    ]

    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=2,
        )

        print(f"\nQ: {query}")
        for j, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"  [{j+1}] ({meta['chapter'][:30]})")
            print(f"      {doc[:150]}...")

    # Show DB stats
    print(f"\n{'=' * 60}")
    print(f"Database stats:")
    print(f"  Location:    {db_path}/")
    print(f"  Collection:  {args.collection}")
    print(f"  Chunks:      {collection.count()}")
    print(f"  Model:       {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
