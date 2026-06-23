#!/usr/bin/env python3
"""
Index PDF documents into Pinecone for arXiv RAG.

Usage:
    python index_documents.py              # Index (skip unchanged files)
    python index_documents.py --force      # Force full reindex
    python index_documents.py --help       # Show help
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retriever import DocumentRetriever


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into Pinecone for arXiv RAG")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with PDF files (default: ./data)")
    parser.add_argument("--force", action="store_true",
                        help="Force reindex even if no changes detected")
    parser.add_argument("--index-name", type=str, default=None,
                        help="Pinecone index name (default: from PINECONE_INDEX_NAME env var)")
    args = parser.parse_args()

    # Validate environment
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set")
        sys.exit(1)
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_KEY"):
        print("Error: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY required for embeddings")
        sys.exit(1)

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"Error: no PDF files found in {data_dir}")
        print("Tip: run 'python scripts/fetch_papers.py' to download sample papers")
        sys.exit(1)

    print("=" * 60)
    print("arXiv RAG — Document Indexing")
    print("=" * 60)
    print(f"Data directory : {data_dir}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Force reindex  : {'yes' if args.force else 'no (skips unchanged files)'}")
    print(f"Index name     : {args.index_name or os.getenv('PINECONE_INDEX_NAME', 'arxiv-rag')}")
    print("=" * 60)

    try:
        retriever = DocumentRetriever(index_name=args.index_name)
        count = retriever.index_documents(data_dir=data_dir, force_reindex=args.force)

        if count > 0:
            stats = retriever.get_collection_stats()
            print(f"\nIndexed {count} chunks")
            print(f"Total vectors in index: {stats.get('total_documents', 0)}")
        else:
            print("\nNo changes — documents already indexed")

        print("\nDone. Start the backend with:")
        print("  uvicorn backend.server:app --host 0.0.0.0 --port 5000")

    except Exception as e:
        print(f"\nIndexing error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
