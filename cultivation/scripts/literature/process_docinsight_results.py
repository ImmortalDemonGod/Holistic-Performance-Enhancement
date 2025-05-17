"""
process_docinsight_results.py - Background worker to fetch DocInsight results for pending jobs.
"""
import sys
from pathlib import Path
# Ensure project root is in sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import json
import logging
from cultivation.scripts.literature.docinsight_client import DocInsightClient, DocInsightAPIError, DocInsightTimeoutError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory paths (same as fetch_paper)
SCRIPT_DIR = Path(__file__).resolve().parent
LIT_DIR = SCRIPT_DIR.parent.parent / "literature"
METADATA_DIR = LIT_DIR / "metadata"
NOTES_DIR = LIT_DIR / "notes"


def process_pending():
    """Scan metadata files for pending DocInsight jobs and fetch results."""
    api_url = None  # picks up default BASE_SERVER_URL
    client = DocInsightClient(base_url=api_url)
    for meta_file in METADATA_DIR.glob("*.json"):
        try:
            data = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to read metadata {meta_file}: {e}")
            continue
        job_id = data.get('docinsight_job_id')
        # skip if no job or already processed
        if not job_id or data.get('docinsight_summary'):
            continue
        logging.info(f"Fetching results for job {job_id}")
        try:
            result = client.wait_for_result(job_id)
            summary = result.get('answer') or result.get('docinsight_summary', '')
            novelty = result.get('novelty')
            if not summary:
                logging.warning(f"Empty summary received for job {job_id}")
            if novelty is None:
                logging.warning(f"Missing novelty score for job {job_id}")
                novelty = 0.0
            data['docinsight_summary'] = summary
            data['docinsight_novelty'] = novelty
            meta_file.write_text(json.dumps(data, indent=2))
            logging.info(f"Updated metadata {meta_file.name} with DocInsight results.")
            # append to note
            note_file = NOTES_DIR / f"{meta_file.stem}.md"
            if note_file.exists():
                try:
                    with open(note_file, 'a') as nf:
                        nf.write("\n\n## DocInsight Summary\n")
                        nf.write(summary + "\n")
                        nf.write(f"\n*Novelty Score (DocInsight): {novelty}*\n")
                    logging.info(f"Appended summary to note {note_file.name}")
                except (IOError, OSError) as e:
                    logging.error(f"Failed to update note file {note_file}: {e}")
        except DocInsightTimeoutError as te:
            logging.warning(f"Job {job_id} not ready: {te}")
        except DocInsightAPIError as ae:
            logging.error(f"Error for job {job_id}: {ae}")
        except Exception as e:
            logging.error(f"Unexpected error for job {job_id}: {e}")


if __name__ == '__main__':
    process_pending()
