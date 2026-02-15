#!/usr/bin/env python3
"""
DadAI v3 — Step 1: Extract and chunk book PDFs for RAG

Reads a PDF book, extracts text, and splits it into meaningful chunks
for the vector database. Handles three types of content:

1. Narrative passages — split by section headers + size limits
2. Q&A pairs — each question/answer becomes its own chunk
3. Key quotes — grouped by chapter

Output: JSONL file with chunks, each containing:
  - text: the chunk content
  - source: book title
  - chapter: chapter number/name
  - chunk_type: "narrative", "qa", or "quote"
  - page: approximate page number

Usage:
    python scripts/chunk_book.py
    python scripts/chunk_book.py --input books/my_book.pdf --output data/rag_chunks.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Clean extracted text: remove page markers, fix whitespace."""
    # Remove Bookey page markers like "-- 1 of 103 --"
    text = re.sub(r'--\s*\d+\s+of\s+\d+\s*--', '', text)
    # Remove Bookey promo lines
    text = re.sub(r'Install Bookey App to Unlock Full Text and\s*Audio', '', text)
    text = re.sub(r'Check more about .+ Summary', '', text)
    text = re.sub(r'Listen .+ Audiobook', '', text)
    text = re.sub(r'Written by Bookey', '', text)
    text = re.sub(r'View on Bookey Website.*', '', text)
    text = re.sub(r'Check the Correct Answer on Bookey Website', '', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_qa_pairs(text: str) -> list[dict]:
    """Extract Q&A pairs from the book's Q&A sections."""
    chunks = []

    # Match patterns like "1.Question\nWhat does..." followed by "Answer:..."
    qa_pattern = re.compile(
        r'(\d+)\.\s*Question\s*\n(.+?)\n\s*Answer:\s*(.+?)(?=\n\d+\.\s*Question|\nChapter \d|\Z)',
        re.DOTALL
    )

    for match in qa_pattern.finditer(text):
        question = match.group(2).strip()
        answer = match.group(3).strip()

        # Clean up the answer
        answer = re.sub(r'\s+', ' ', answer)
        question = re.sub(r'\s+', ' ', question)

        if len(answer) > 50:  # Skip very short/broken answers
            chunks.append({
                "text": f"Q: {question}\n\nA: {answer}",
                "chunk_type": "qa",
            })

    return chunks


def extract_quotes(text: str) -> list[dict]:
    """Extract key quotes from the quotes sections."""
    chunks = []

    # Match numbered quotes like "1.these father signifiers are empty."
    quote_pattern = re.compile(
        r'(?:^|\n)\d+\.\s*(.+?)(?=\n\d+\.|\nChapter \d|\Z)',
        re.DOTALL
    )

    # Find quotes sections
    quotes_sections = re.finditer(
        r'Quotes From Pages \d+-\d+\s*\n(.+?)(?=Chapter \d|Best Quotes|Quiz and Test|\Z)',
        text, re.DOTALL
    )

    for section in quotes_sections:
        section_text = section.group(1)
        for match in quote_pattern.finditer(section_text):
            quote = match.group(1).strip()
            quote = re.sub(r'\s+', ' ', quote)
            if len(quote) > 30 and len(quote) < 500:
                chunks.append({
                    "text": f'"{quote}"',
                    "chunk_type": "quote",
                })

    return chunks


def extract_narrative_chunks(text: str, max_chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Extract narrative passages from chapter summaries."""
    chunks = []

    # Find chapter summary sections
    chapter_pattern = re.compile(
        r'(Chapter \d+ Summary\s*:\s*\d+\.\s*(.+?)\n)(.*?)(?=Chapter \d+ Summary|Best Quotes|Absent Fathers, Lost Sons Questions|\Z)',
        re.DOTALL
    )

    for match in chapter_pattern.finditer(text):
        chapter_name = match.group(2).strip()
        content = match.group(3).strip()

        # Remove Q&A sections from narrative (we handle those separately)
        content = re.sub(r'\d+\.\s*Question\s*\n.+?Answer:.+?(?=\n\d+\.\s*Question|\Z)', '', content, flags=re.DOTALL)

        # Remove "Example" and "Critical Thinking" headers but keep content
        content = re.sub(r'\n(Example|Critical Thinking)\n', '\n', content)

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        # Group paragraphs into chunks of ~max_chunk_size characters
        current_chunk = ""
        for para in paragraphs:
            para_clean = re.sub(r'\s+', ' ', para)
            if len(para_clean) < 30:
                continue

            if len(current_chunk) + len(para_clean) > max_chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_type": "narrative",
                    "chapter": chapter_name,
                })
                # Keep overlap
                words = current_chunk.split()
                overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else []
                current_chunk = " ".join(overlap_words) + " " + para_clean
            else:
                current_chunk += (" " if current_chunk else "") + para_clean

        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_type": "narrative",
                "chapter": chapter_name,
            })

    return chunks


def extract_section_summaries(text: str) -> list[dict]:
    """Extract the section summary tables (compact concept descriptions)."""
    chunks = []

    # Match section summary rows: "Section Name | Description"
    # These appear as table-like structures in the text
    section_pattern = re.compile(
        r'Section Summary\n(.+?)(?=\n(?:Chapter \d|The \w))',
        re.DOTALL
    )

    for match in section_pattern.finditer(text):
        table_text = match.group(1).strip()
        # Each row is a concept + description
        rows = [r.strip() for r in table_text.split('\n') if r.strip() and len(r.strip()) > 30]
        for row in rows:
            chunks.append({
                "text": row,
                "chunk_type": "section_summary",
            })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk a book PDF for RAG")
    parser.add_argument("--input", default="books/Absent Fathers, Lost Sons PDF.pdf",
                        help="Path to the PDF file")
    parser.add_argument("--output", default="data/rag_chunks.jsonl",
                        help="Output JSONL file for chunks")
    parser.add_argument("--max-chunk-size", type=int, default=800,
                        help="Max characters per narrative chunk")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: PDF not found: {input_path}")
        raise SystemExit(1)

    # Extract and clean text
    print(f"Reading PDF: {input_path}")
    raw_text = extract_text_from_pdf(str(input_path))
    text = clean_text(raw_text)

    book_title = "Absent Fathers, Lost Sons"
    print(f"Extracted {len(text):,} characters of text")

    # Extract different chunk types
    print("\nChunking...")
    qa_chunks = extract_qa_pairs(text)
    print(f"  Q&A pairs:         {len(qa_chunks)}")

    quote_chunks = extract_quotes(text)
    print(f"  Key quotes:        {len(quote_chunks)}")

    narrative_chunks = extract_narrative_chunks(text, max_chunk_size=args.max_chunk_size)
    print(f"  Narrative passages: {len(narrative_chunks)}")

    section_chunks = extract_section_summaries(text)
    print(f"  Section summaries: {len(section_chunks)}")

    # Combine all chunks
    all_chunks = []
    for chunk in qa_chunks + quote_chunks + narrative_chunks + section_chunks:
        chunk["source"] = book_title
        all_chunks.append(chunk)

    print(f"\n  Total chunks:      {len(all_chunks)}")

    # Show stats
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"  Avg chunk size:    {sum(lengths) // len(lengths)} chars")
    print(f"  Min chunk size:    {min(lengths)} chars")
    print(f"  Max chunk size:    {max(lengths)} chars")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_chunks)} chunks to {output_path}")

    # Show samples
    print("\n" + "=" * 60)
    print("Sample chunks:")
    print("=" * 60)

    for chunk_type in ["qa", "quote", "narrative", "section_summary"]:
        samples = [c for c in all_chunks if c["chunk_type"] == chunk_type]
        if samples:
            sample = samples[0]
            print(f"\n--- {chunk_type.upper()} ---")
            print(sample["text"][:300] + ("..." if len(sample["text"]) > 300 else ""))


if __name__ == "__main__":
    main()
