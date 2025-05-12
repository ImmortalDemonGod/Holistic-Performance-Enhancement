#!/usr/bin/env python3
"""
fetch_paper.py - Fetches a paper by arXiv ID, URL, or DOI, downloads the PDF, extracts metadata, and creates a note skeleton. Integrates with DocInsight if available.
"""
import argparse
import requests
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import datetime
import logging
import re # For sanitizing filenames
import time
import os
import sys
import jsonschema
from jsonschema import validate, ValidationError
from typing import Optional, Any, Dict

SCHEMA_PATH = Path(__file__).parent.parent.parent / 'schemas' / 'paper.schema.json'

# Attempt relative import for DocInsightClient
try:
    from .docinsight_client import DocInsightClient, DocInsightAPIError, DocInsightTimeoutError
except ImportError:
    from docinsight_client import DocInsightClient, DocInsightAPIError, DocInsightTimeoutError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = Path(__file__).resolve().parent
LIT_DIR = SCRIPT_DIR.parent.parent / "literature"
PDF_DIR = LIT_DIR / "pdf"
METADATA_DIR = LIT_DIR / "metadata"
NOTES_DIR = LIT_DIR / "notes"

ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query?id_list="
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf/"
ARXIV_ABS_BASE_URL = "https://arxiv.org/abs/"

def sanitize_filename_component(text, max_length=50):
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[-\s]+', '-', text)
    return text[:max_length]

def robust_get(url: str, max_retries: int = 5, backoff_factor: float = 1.0, timeout: int = 30, allowed_statuses=(500, 502, 503, 504)) -> Optional[requests.Response]:
    """
    Robust GET with retries and exponential backoff for transient errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            # Retry on all exceptions (including StopIteration from mocks)
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response is not None and e.response.status_code not in allowed_statuses:
                logging.error(f"Non-retryable error {e.response.status_code} on {url}: {e}")
                break
            wait = min(backoff_factor * (2 ** (attempt - 1)), 10)
            logging.warning(f"Attempt {attempt} failed for {url}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    logging.error(f"All retries failed for {url}")
    return None

def fetch_arxiv_paper(arxiv_id: str, force_redownload: bool = False):
    cleaned_arxiv_id = arxiv_id.replace("arxiv:", "", 1).lower()
    logging.info(f"Processing arXiv ID: {cleaned_arxiv_id}")

    # Override output directories via env var
    override_base = os.getenv('LIT_DIR_OVERRIDE')
    if override_base:
        pdf_dir = Path(override_base) / 'pdf'
        metadata_dir = Path(override_base) / 'metadata'
        notes_dir = Path(override_base) / 'notes'
    else:
        pdf_dir = PDF_DIR
        metadata_dir = METADATA_DIR
        notes_dir = NOTES_DIR

    pdf_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    pdf_url = f"{ARXIV_PDF_BASE_URL}{cleaned_arxiv_id}.pdf"
    abs_url = f"{ARXIV_ABS_BASE_URL}{cleaned_arxiv_id}"
    api_url = f"{ARXIV_API_BASE_URL}{cleaned_arxiv_id}"

    pdf_path = pdf_dir / f"{cleaned_arxiv_id}.pdf"
    metadata_path = metadata_dir / f"{cleaned_arxiv_id}.json"
    note_path = notes_dir / f"{cleaned_arxiv_id}.md"

    # 1. Download PDF
    if force_redownload or not pdf_path.exists():
        logging.info(f"Downloading PDF from {pdf_url}...")
        response = robust_get(pdf_url)
        if response is not None:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"PDF saved to {pdf_path}")
        else:
            logging.error(f"Failed to download PDF for {cleaned_arxiv_id} after retries.")
            return False
    else:
        logging.info(f"PDF already exists: {pdf_path}")

    # 2. Fetch and Parse Metadata from arXiv API
    metadata_payload = {}
    if force_redownload or not metadata_path.exists():
        logging.info(f"Fetching metadata from {api_url}...")
        try:
            response = robust_get(api_url)
            if response is None:
                logging.error(f"Failed to fetch metadata for {cleaned_arxiv_id} after retries.")
                return False
        except Exception:
            logging.error(f"Failed to fetch metadata for {cleaned_arxiv_id}")
            return False
        # proceed to parse XML if GET succeeded
        try:
            xml_root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

            entry = xml_root.find('atom:entry', ns)
            if entry is None:
                logging.error(f"No entry found in arXiv API response for {cleaned_arxiv_id}")
                return False

            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ').replace('  ', ' ')
            authors_xml = entry.findall('atom:author', ns)
            authors = [author.find('atom:name', ns).text for author in authors_xml]
            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            published_date_str = entry.find('atom:published', ns).text
            updated_date_str = entry.find('atom:updated', ns).text
            date_to_use_str = updated_date_str if updated_date_str > published_date_str else published_date_str
            paper_date = datetime.datetime.strptime(date_to_use_str, "%Y-%m-%dT%H:%M:%SZ")

            primary_category_xml = entry.find('arxiv:primary_category', ns)
            primary_category = primary_category_xml.attrib.get('term') if primary_category_xml is not None else "unknown"

            metadata_payload = {
                "arxiv_id": cleaned_arxiv_id,
                "title": title,
                "authors": authors,
                "year": paper_date.year,
                "month": paper_date.month,
                "day": paper_date.day,
                "source": "arXiv",
                "abstract": abstract,
                "pdf_path": str(pdf_path),
                "note_path": str(note_path),
                "pdf_link": pdf_url,
                "abs_link": abs_url,
                "primary_category": primary_category,
                "tags": [],
                "imported_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

            # JSON Schema validation
            try:
                with open(SCHEMA_PATH, 'r') as schema_file:
                    schema = json.load(schema_file)
                validate(instance=metadata_payload, schema=schema)
            except ValidationError as ve:
                logging.error(f"Metadata schema validation failed for {cleaned_arxiv_id}: {ve.message} (field: {list(ve.path)})")
                logging.error(f"Marking {cleaned_arxiv_id} as processing_error_metadata.")
                return False
            except Exception as ve:
                logging.warning(f"Skipping metadata schema validation for {cleaned_arxiv_id}: {ve}")
            with open(metadata_path, 'w') as f:
                json.dump(metadata_payload, f, indent=2)
            logging.info(f"Metadata saved to {metadata_path}")
        except ET.ParseError as e:
            logging.error(f"Failed to parse XML metadata for {cleaned_arxiv_id}: {e}")
            return False
    else:
        logging.info(f"Metadata file already exists: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata_payload = json.load(f)

    # 3. Create Note Skeleton
    if not note_path.exists() and metadata_payload:
        note_template = f"""# {metadata_payload.get('title', 'N/A')}
*ArXiv {metadata_payload.get('id', cleaned_arxiv_id)} · {metadata_payload.get('year', 'N/A')}*

> **Abstract (autofilled):**
> {metadata_payload.get('abstract', 'N/A')}

## TL;DR <!-- mark complete when filled -->
- [ ]

## Key Points
- [ ]

## Relevance to Cultivation
- [ ]

## TODOs
- [ ] implement ___ in `scripts/`
"""
        with open(note_path, 'w') as f:
            f.write(note_template)
        logging.info(f"Note skeleton created: {note_path}")
    elif note_path.exists():
        logging.info(f"Note file already exists: {note_path}")
    elif not metadata_payload:
        logging.warning(f"Cannot create note for {cleaned_arxiv_id} as metadata is missing.")

    # 4. DocInsight Client Integration with polling
    api_url = os.getenv('DOCINSIGHT_API_URL')
    client = DocInsightClient(base_url=api_url) if api_url else DocInsightClient()
    logging.info(f"Initiating DocInsight processing for {cleaned_arxiv_id}")
    query_text = (
        f"Summarize the abstract and key contributions of the paper titled: '{metadata_payload.get('title', 'N/A')}' (arXiv:{cleaned_arxiv_id})"
    )
    try:
        job_id = client.start_research(query=query_text, force_index=[str(pdf_path.resolve())])
        logging.info(f"DocInsight job started with ID: {job_id} for {cleaned_arxiv_id}")
        # Poll until done or timeout
        result = client.wait_for_result(job_id)
        summary = result.get('answer') or result.get('docinsight_summary') or ''
        novelty = result.get('novelty')
        metadata_payload['docinsight_summary'] = summary
        metadata_payload['docinsight_novelty'] = novelty
        # persist updated metadata
        if metadata_path.exists():
            with open(metadata_path, 'w') as f_meta:
                json.dump(metadata_payload, f_meta, indent=2)
            logging.info(f"Updated metadata for {cleaned_arxiv_id} with DocInsight results.")
        # append to note
        if note_path.exists() and '## DocInsight Summary' not in note_path.read_text():
            with open(note_path, 'a') as f_note:
                f_note.write("\n\n## DocInsight Summary\n")
                f_note.write(f"{summary}\n")
                f_note.write(f"\n*Novelty Score (DocInsight): {novelty}*\n")
            logging.info(f"Appended DocInsight summary to note for {cleaned_arxiv_id}")
    except DocInsightTimeoutError as te:
        logging.error(f"DocInsight job {job_id} timed out for {cleaned_arxiv_id}: {te}")
    except DocInsightAPIError as ae:
        logging.error(f"DocInsight API error for {cleaned_arxiv_id}: {ae}")
    except Exception as e:
        logging.error(f"Unexpected DocInsight error for {cleaned_arxiv_id}: {e}")

    return True

def main():
    parser = argparse.ArgumentParser(description="Fetch a paper by arXiv ID, URL, or DOI.")
    parser.add_argument('--force-redownload', action='store_true', help='Re-download PDF and metadata even if files exist')
    parser.add_argument('--arxiv_id', type=str, help='arXiv ID of the paper (e.g., "2310.01234")')
    parser.add_argument('--url', type=str, help='Direct URL to the paper PDF (Not yet implemented)')
    parser.add_argument('--doi', type=str, help='DOI of the paper (Not yet implemented)')
    args = parser.parse_args()

    if not any([args.arxiv_id, args.url, args.doi]):
        parser.error("At least one of (--arxiv_id, --url, --doi) must be provided.")
        return

    if args.arxiv_id:
        success = fetch_arxiv_paper(args.arxiv_id, force_redownload=args.force_redownload)
        if success:
            logging.info(f"Successfully processed arXiv ID: {args.arxiv_id}")
        else:
            logging.error(f"Failed to process arXiv ID: {args.arxiv_id}")
    elif args.url:
        logging.warning("--url processing is not yet implemented.")
    elif args.doi:
        logging.warning("--doi processing is not yet implemented.")

if __name__ == '__main__':
    main()
