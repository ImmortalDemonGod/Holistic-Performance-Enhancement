#!/usr/bin/env python3
"""
fetch_arxiv_batch.py - Batch fetch arXiv papers based on queries and track state.
"""
import argparse
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import requests

from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper

ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query?search_query="


def load_state(path: Path) -> dict:
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_state(path: Path, state: dict) -> None:
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def get_new_ids_for_query(query: str, since: datetime) -> list[str]:
    # query and date filter for submissions since 'since'
    date_str = since.strftime('%Y%m%d')
    url = f"{ARXIV_API_BASE_URL}{query}+AND+submittedDate:[{date_str}0000+TO+*]"
    resp = requests.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    ids: list[str] = []
    for entry in root.findall('atom:entry', ns):
        id_url = entry.find('atom:id', ns).text or ''
        arxiv_id = id_url.rsplit('/', 1)[-1]
        ids.append(arxiv_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch fetch arXiv papers.")
    parser.add_argument(
        '--queries', nargs='+', required=True,
        help='ArXiv search queries'
    )
    parser.add_argument(
        '--state-file', type=Path,
        default=Path('.fetch_batch_state.json'),
        help='Path to state file storing last run dates'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    state = load_state(args.state_file)
    now = datetime.utcnow()

    for query in args.queries:
        last = state.get(query)
        if last:
            last_run = datetime.fromisoformat(last)
        else:
            last_run = now - timedelta(days=7)
        logging.info(f"Fetching new IDs for '{query}' since {last_run}")
        try:
            new_ids = get_new_ids_for_query(query, last_run)
        except Exception as e:
            logging.error(f"Error fetching IDs for '{query}': {e}")
            continue
        logging.info(f"Found {len(new_ids)} new papers for '{query}'")
        for arxiv_id in new_ids:
            success = fetch_arxiv_paper(arxiv_id)
            logging.info(f"Processed {arxiv_id}: {success}")
        state[query] = now.isoformat()

    save_state(args.state_file, state)


if __name__ == '__main__':
    main()
