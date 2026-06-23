#!/usr/bin/env python3
"""
Download arXiv papers for indexing.

Fetches PDFs from specified arXiv categories and saves them to the data/ directory.
Useful for quickly building a demo corpus of recent AI/ML papers.

Usage:
    python scripts/fetch_papers.py                          # Default: 20 papers, cs.AI + cs.LG + cs.CL
    python scripts/fetch_papers.py --count 50              # Download 50 papers
    python scripts/fetch_papers.py --query "attention transformers"
    python scripts/fetch_papers.py --categories cs.CV cs.RO
    python scripts/fetch_papers.py --output ./my-data
"""

import argparse
import os
import sys
import time


def fetch_papers(
    categories: list = None,
    query: str = None,
    max_results: int = 20,
    output_dir: str = "./data",
    sort_by: str = "relevance",
):
    """
    Download PDFs from arXiv to output_dir.

    Args:
        categories: arXiv category list, e.g. ['cs.AI', 'cs.LG']
        query: Free-text search query (overrides categories if provided)
        max_results: Number of papers to download
        output_dir: Directory to save PDFs
        sort_by: "relevance" or "lastUpdatedDate" or "submittedDate"
    """
    try:
        import arxiv
    except ImportError:
        print("Error: arxiv package not installed. Run: pip install arxiv>=2.1.0")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if query:
        search_query = query
    else:
        cats = categories or ["cs.AI", "cs.LG", "cs.CL"]
        cat_query = " OR ".join(f"cat:{c}" for c in cats)
        search_query = cat_query

    sort_criterion = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }.get(sort_by, arxiv.SortCriterion.Relevance)

    print(f"Searching arXiv: {search_query}")
    print(f"Target: {max_results} papers → {output_dir}")
    print("-" * 60)

    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=3.0)
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    downloaded = 0
    skipped = 0
    failed = 0

    for paper in client.results(search):
        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)
        safe_title = safe_title[:80].strip()
        arxiv_id = paper.entry_id.split("/")[-1].replace("/", "_")
        filename = f"{arxiv_id}_{safe_title}.pdf"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"  [skip] {filename} (already exists)")
            skipped += 1
            continue

        try:
            paper.download_pdf(dirpath=output_dir, filename=filename)
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            print(f"  [ok]   {filename}")
            print(f"         {authors} ({paper.published.year if paper.published else 'n/a'})")
            downloaded += 1
            time.sleep(1)  # Be polite to arXiv
        except Exception as e:
            print(f"  [fail] {filename}: {e}")
            failed += 1

    print("-" * 60)
    print(f"Downloaded: {downloaded}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"Papers saved to: {os.path.abspath(output_dir)}")

    if downloaded > 0 or skipped > 0:
        print("\nNext steps:")
        print("  1. Index the papers: python backend/index_documents.py")
        print("  2. Start the backend: uvicorn backend.server:app --host 0.0.0.0 --port 5000")


def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv AI/ML papers for indexing"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        metavar="CAT",
        help="arXiv categories (default: cs.AI cs.LG cs.CL). "
             "Examples: cs.CV cs.RO cs.NE stat.ML"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Free-text search query (overrides --categories). "
             "Example: 'attention mechanism transformers'"
    )
    parser.add_argument(
        "--count", type=int, default=20, metavar="N",
        help="Number of papers to download (default: 20)"
    )
    parser.add_argument(
        "--output", type=str, default="./data", metavar="DIR",
        help="Output directory for PDFs (default: ./data)"
    )
    parser.add_argument(
        "--sort", choices=["relevance", "lastUpdatedDate", "submittedDate"],
        default="relevance",
        help="Sort order (default: relevance)"
    )

    args = parser.parse_args()

    fetch_papers(
        categories=args.categories,
        query=args.query,
        max_results=args.count,
        output_dir=args.output,
        sort_by=args.sort,
    )


if __name__ == "__main__":
    main()
