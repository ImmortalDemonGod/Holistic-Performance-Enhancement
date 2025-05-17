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

from cultivation.scripts.literature.fetch_paper import fetch_arxiv_paper, robust_get

ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query?search_query="


def load_state(path: Path) -> dict:
    """
    Loads and returns the state dictionary from a JSON file if it exists.
    
    If the specified file does not exist, returns an empty dictionary.
    
    Args:
        path: Path to the JSON state file.
    
    Returns:
        The loaded state as a dictionary, or an empty dictionary if the file is missing.
    """
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_state(path: Path, state: dict) -> None:
    """
    Saves the given state dictionary as a JSON file at the specified path.
    
    Args:
        path: The file path where the state should be saved.
        state: The dictionary representing the state to persist.
    """
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def get_new_ids_for_query(query: str, since: datetime) -> list[str]:
    # query and date filter for submissions since 'since'
    """
    Fetches new arXiv paper IDs matching a query submitted since a given date.
    
    Args:
        query: The arXiv search query string.
        since: Only papers submitted on or after this date are considered.
    
    Returns:
        A list of arXiv paper IDs matching the query and date filter. Returns an empty list if the API request fails.
    """
    from urllib.parse import quote_plus
    date_str = since.strftime('%Y%m%d')
    encoded_query = quote_plus(f"{query}+AND+submittedDate:[{date_str}0000+TO+*]")
    url = f"{ARXIV_API_BASE_URL}{encoded_query}"
    resp = robust_get(url)
    if resp is None:
        logging.error(f"Failed to fetch arXiv API for query '{query}' after retries.")
        return []
    # status checks and retries handled in robust_get
    root = ET.fromstring(resp.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    ids: list[str] = []
    for entry in root.findall('atom:entry', ns):
        id_url = entry.find('atom:id', ns).text or ''
        arxiv_id = id_url.rsplit('/', 1)[-1]
        ids.append(arxiv_id)
    return ids


def main() -> None:
    """
    Parses command-line arguments and batch fetches new arXiv papers for each query.
    
    For each provided search query, retrieves arXiv paper IDs submitted since the last recorded fetch, processes each new paper, and updates the state file to track the latest fetch time. Handles errors per query to ensure continued processing.
    """
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
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Maximum number of papers to process in a single run'
    )
    parser.add_argument(
        '--rate-limit', type=float, default=3.0,
        help='Minimum seconds between API requests'
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
