#!/usr/bin/env python3
"""
fetch_paper.py - Fetches a paper by arXiv ID, URL, or DOI, downloads the PDF, extracts metadata, and creates a note skeleton. Integrates with DocInsight if available.
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fetch a paper by arXiv ID, URL, or DOI.")
    parser.add_argument('--arxiv_id', type=str, help='arXiv ID of the paper')
    parser.add_argument('--url', type=str, help='Direct URL to the paper PDF')
    parser.add_argument('--doi', type=str, help='DOI of the paper')
    args = parser.parse_args()

    # Placeholder: implement logic to fetch paper, metadata, and create note
    print(f"[Stub] Would fetch paper with arXiv_id={args.arxiv_id}, url={args.url}, doi={args.doi}")

if __name__ == '__main__':
    main()
