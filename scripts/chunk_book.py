#!/usr/bin/env python3
"""
DadAI v3 — Step 1: Extract and chunk books for RAG

Reads a book (EPUB or PDF), extracts text, and splits it into meaningful
chunks for the vector database.

Chunking strategy:
  - Split text by chapter (using EPUB structure or headings)
  - Within each chapter, group paragraphs into ~300-word chunks
  - Overlap between chunks to preserve context at boundaries

Output: JSONL file with chunks, each containing:
  - text: the chunk content
  - source: book title
  - chapter: chapter name
  - chunk_type: "narrative"

Usage:
    python scripts/chunk_book.py
    python scripts/chunk_book.py --input books/my_book.epub
    python scripts/chunk_book.py --input books/my_book.pdf
"""

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_chapters_from_epub(epub_path: str) -> list[dict]:
    """Extract chapter text from an EPUB file, preserving paragraph structure."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = item.get_name()
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Get plain text length to skip short items
        plain = soup.get_text(strip=True)
        if len(plain) < 500:
            continue

        # Skip bibliography, table of contents, copyright
        skip_patterns = ['biblio', 'toc', 'copy', 'dedication', 'halftitle']
        if any(pat in name.lower() for pat in skip_patterns):
            continue

        # Extract paragraph-separated text using HTML <p> tags
        paragraphs = []
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'blockquote']):
            text = tag.get_text(separator=' ', strip=True)
            if text:
                paragraphs.append(text)

        text = '\n\n'.join(paragraphs)

        # Extract chapter title from first heading
        heading = soup.find(['h1', 'h2', 'h3'])
        if heading:
            chapter_title = heading.get_text(strip=True)
        else:
            chapter_title = name.replace('.html', '').replace('_', ' ').title()

        chapters.append({
            "title": chapter_title,
            "text": text,
        })

    return chapters


def extract_chapters_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF as a single chapter (less structured)."""
    import fitz

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Clean Bookey artifacts if present
    full_text = re.sub(r'--\s*\d+\s+of\s+\d+\s*--', '', full_text)
    full_text = re.sub(r'Install Bookey App to Unlock Full Text and\s*Audio', '', full_text)
    full_text = re.sub(r'Written by Bookey', '', full_text)

    return [{"title": "Full Text", "text": full_text}]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def clean_chapter_text(text: str) -> str:
    """Clean a chapter's text for chunking."""
    # Normalize whitespace within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_chapter(chapter_title: str, text: str,
                  max_chunk_words: int = 300,
                  overlap_words: int = 50) -> list[dict]:
    """Split a chapter into overlapping word-based chunks."""
    text = clean_chapter_text(text)

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph exceeds the limit, save current chunk
        if len(current_words) + len(para_words) > max_chunk_words and current_words:
            chunk_text = ' '.join(current_words)
            if len(chunk_text) > 100:  # Skip tiny chunks
                chunks.append({
                    "text": chunk_text,
                    "chapter": chapter_title,
                    "chunk_type": "narrative",
                })
            # Keep overlap for context continuity
            current_words = current_words[-overlap_words:] + para_words
        else:
            current_words.extend(para_words)

    # Don't forget the last chunk
    if current_words:
        chunk_text = ' '.join(current_words)
        if len(chunk_text) > 100:
            chunks.append({
                "text": chunk_text,
                "chapter": chapter_title,
                "chunk_type": "narrative",
            })

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chunk a book for RAG")
    parser.add_argument(
        "--input",
        default="books/Absent Fathers, Lost Sons. The Search for Masculine Identity - Guy Corneau.epub",
        help="Path to the book file (EPUB or PDF)"
    )
    parser.add_argument("--output", default="data/rag_chunks.jsonl",
                        help="Output JSONL file for chunks")
    parser.add_argument("--max-chunk-words", type=int, default=300,
                        help="Max words per chunk (default: 300)")
    parser.add_argument("--overlap-words", type=int, default=50,
                        help="Overlap words between chunks (default: 50)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Book not found: {input_path}")
        raise SystemExit(1)

    # Extract chapters based on file type
    ext = input_path.suffix.lower()
    if ext == '.epub':
        print(f"Reading EPUB: {input_path.name}")
        chapters = extract_chapters_from_epub(str(input_path))
    elif ext == '.pdf':
        print(f"Reading PDF: {input_path.name}")
        chapters = extract_chapters_from_pdf(str(input_path))
    else:
        print(f"Error: Unsupported format: {ext} (use .epub or .pdf)")
        raise SystemExit(1)

    total_chars = sum(len(c["text"]) for c in chapters)
    print(f"Extracted {total_chars:,} characters from {len(chapters)} chapters")

    # Derive book title from filename
    book_title = input_path.stem.split(' - ')[0].strip()
    if len(book_title) > 60:
        book_title = book_title[:60]
    print(f"Book: {book_title}")

    # Chunk each chapter
    print(f"\nChunking (max {args.max_chunk_words} words/chunk, {args.overlap_words} overlap)...")
    all_chunks = []

    for chapter in chapters:
        chapter_chunks = chunk_chapter(
            chapter["title"],
            chapter["text"],
            max_chunk_words=args.max_chunk_words,
            overlap_words=args.overlap_words,
        )
        for chunk in chapter_chunks:
            chunk["source"] = book_title
        all_chunks.extend(chapter_chunks)
        print(f"  {chapter['title'][:50]:50s} → {len(chapter_chunks):3d} chunks")

    print(f"\n  Total chunks: {len(all_chunks)}")

    # Stats
    word_counts = [len(c["text"].split()) for c in all_chunks]
    char_counts = [len(c["text"]) for c in all_chunks]
    print(f"  Avg chunk:    {sum(word_counts) // len(word_counts)} words / {sum(char_counts) // len(char_counts)} chars")
    print(f"  Min chunk:    {min(word_counts)} words")
    print(f"  Max chunk:    {max(word_counts)} words")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_chunks)} chunks to {output_path}")

    # Show samples from different chapters
    print("\n" + "=" * 60)
    print("Sample chunks (first 300 chars each):")
    print("=" * 60)

    shown_chapters = set()
    for chunk in all_chunks:
        ch = chunk["chapter"]
        if ch not in shown_chapters and len(shown_chapters) < 4:
            shown_chapters.add(ch)
            print(f"\n--- [{ch}] ---")
            print(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
