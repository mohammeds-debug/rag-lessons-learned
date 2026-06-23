"""
Document chunking for arXiv AI/ML papers.

Extracts text page-by-page via PyMuPDF, detects standard academic section
headers (Abstract, Introduction, Methodology, Results, etc.), and attempts to
extract the paper title from the first page. Every chunk carries enriched
metadata — paper_title, section_title, page_number — so retrieval results can
display provenance without secondary lookups.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# Ordered list of canonical academic paper section names
KNOWN_SECTIONS = [
    "abstract",
    "introduction",
    "related work",
    "literature review",
    "background",
    "preliminaries",
    "problem formulation",
    "problem statement",
    "methodology",
    "method",
    "methods",
    "approach",
    "proposed method",
    "proposed approach",
    "model",
    "architecture",
    "framework",
    "system design",
    "training",
    "training details",
    "implementation",
    "implementation details",
    "experiments",
    "experimental setup",
    "experimental results",
    "evaluation",
    "results",
    "results and discussion",
    "analysis",
    "ablation study",
    "ablation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "conclusion and future work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "appendix",
]


def _normalize_header(text: str) -> Optional[str]:
    """
    If text looks like a section header, return its canonical form.
    Handles patterns like "1. Introduction", "3.2 Methodology", "A. Background".
    Returns None if text is not a recognizable section header.
    """
    cleaned = text.strip().lower()
    # Strip leading numbering: "1.", "1.2", "3.2.1", "A.", "B.1" etc.
    cleaned = re.sub(r"^[a-z\d]+(?:\.\d+)*\.?\s+", "", cleaned)
    # Strip trailing colon
    cleaned = cleaned.rstrip(":").strip()

    for section in KNOWN_SECTIONS:
        if cleaned == section or cleaned.startswith(section + " "):
            return section.title()
    return None


def _extract_title_candidate(blocks: list) -> Optional[str]:
    """
    Heuristic: the paper title is usually the first non-trivial text block on
    page 1 that is short enough to be a title and not an author/affiliation line.
    """
    for block in blocks[:15]:
        if len(block) < 5:
            continue
        text = block[4].strip()
        if len(text) < 15 or len(text) > 250:
            continue
        # Skip lines that look like author lines (emails, many commas)
        if "@" in text or text.count(",") > 4:
            continue
        # Skip lines that are all numbers or very short words
        words = text.split()
        if len(words) < 3:
            continue
        return text
    return None


def _extract_pages(pdf_path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Open a PDF and return structured page data plus the detected paper title.

    Returns:
        pages: list of {page_num, text_blocks: [{text, is_header, section}]}
        paper_title: best-guess title string or None
    """
    doc = fitz.open(pdf_path)
    pages = []
    paper_title = None

    for page_num, page in enumerate(doc):
        raw_blocks = page.get_text("blocks")

        if page_num == 0 and paper_title is None:
            paper_title = _extract_title_candidate(raw_blocks)

        text_blocks = []
        for block in raw_blocks:
            if len(block) < 5:
                continue
            text = block[4].strip()
            if not text or len(text) < 3:
                continue
            section = _normalize_header(text) if len(text) < 120 else None
            text_blocks.append({
                "text": text,
                "is_header": section is not None,
                "section": section,
            })

        pages.append({
            "page_num": page_num + 1,
            "text_blocks": text_blocks,
        })

    doc.close()
    return pages, paper_title


def _build_chunks(
    pages: List[Dict[str, Any]],
    paper_title: Optional[str],
    source_filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """
    Build a flat list of text segments from pages, then split with
    SentenceSplitter while preserving page / section context per chunk.
    """
    # Build (text, page_num, current_section) triplets
    segments: List[Tuple[str, int, str]] = []
    current_section = "Unknown"

    for page in pages:
        for block in page["text_blocks"]:
            if block["is_header"] and block["section"]:
                current_section = block["section"]
            segments.append((block["text"], page["page_num"], current_section))

    if not segments:
        return []

    # Concatenate with separator tokens we can use to recover context
    # Format: <<P{page}|S:{section}>>\n{text}
    tagged_parts = []
    for text, pnum, section in segments:
        tagged_parts.append(f"<<P{pnum}|S:{section}>>\n{text}")

    full_tagged = "\n\n".join(tagged_parts)

    # Split into chunks
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = Document(text=full_tagged)
    nodes = splitter.get_nodes_from_documents([doc])

    # For each chunk, extract the last page/section tag seen in that chunk
    page_tag_re = re.compile(r"<<P(\d+)\|S:([^>]+)>>")

    chunks = []
    for i, node in enumerate(nodes):
        chunk_text = node.text

        # Find all P/S tags in the chunk, take the first for provenance
        matches = page_tag_re.findall(chunk_text)
        if matches:
            page_num = int(matches[0][0])
            section = matches[0][1]
        else:
            page_num = 1
            section = "Unknown"

        # Strip out the tag markers from the final content
        clean_text = page_tag_re.sub("", chunk_text).strip()
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

        if not clean_text:
            continue

        chunks.append({
            "content": clean_text,
            "metadata": {
                "source": source_filename,
                "paper_title": paper_title,
                "section_title": section,
                "page_number": page_num,
                "chunk_id": i,
                "total_chunks": len(nodes),
                "node_id": node.node_id,
                "language": "en",
            },
        })

    return chunks


def process_documents(
    data_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Process all PDFs in data_dir and return a flat list of enriched chunks.

    Each chunk dict has:
        content  : str   — the text of the chunk
        metadata : dict  — source, paper_title, section_title, page_number,
                           chunk_id, total_chunks, node_id, language
    """
    if not os.path.exists(data_dir):
        print(f"Warning: data directory '{data_dir}' does not exist.")
        return []

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Warning: no PDF files found in '{data_dir}'.")
        return []

    all_chunks: List[Dict[str, Any]] = []

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        try:
            print(f"  Processing {filename}...")
            pages, paper_title = _extract_pages(path)
            if not pages:
                print(f"    Warning: no text extracted from {filename}")
                continue

            total_blocks = sum(len(p["text_blocks"]) for p in pages)
            print(f"    Extracted {len(pages)} pages, {total_blocks} text blocks")
            if paper_title:
                print(f"    Detected title: {paper_title[:80]}")

            chunks = _build_chunks(pages, paper_title, filename, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print(f"    Created {len(chunks)} chunks")

        except Exception as exc:
            print(f"    Error processing {filename}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks
