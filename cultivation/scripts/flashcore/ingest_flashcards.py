#!/usr/bin/env python3
"""
Ingest flashcard YAML decks into the DuckDB database.
Supports CLI arguments for source dir, assets dir, DB path, and strict mode.
Defaults align with flashcore conventions and environment variables.
"""
import os
import sys
import argparse
from pathlib import Path

from cultivation.scripts.flashcore.yaml_processor import load_and_process_flashcard_yamls
from cultivation.scripts.flashcore.database import FlashcardDatabase, DEFAULT_FLASHCORE_DATA_DIR, DEFAULT_DATABASE_FILENAME

def main():
    parser = argparse.ArgumentParser(description="Ingest flashcard YAML decks into DuckDB database.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=os.environ.get("FLASHCARD_YAML_SOURCE_DIR", "cultivation/outputs/flashcards/yaml"),
        help="Directory containing YAML deck files (default: cultivation/outputs/flashcards/yaml or $FLASHCARD_YAML_SOURCE_DIR)"
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=os.environ.get("FLASHCARD_ASSETS_DIR", "cultivation/outputs/flashcards/yaml/assets"),
        help="Directory containing flashcard media assets (default: cultivation/outputs/flashcards/yaml/assets or $FLASHCARD_ASSETS_DIR)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.environ.get(
            "FLASHCORE_DB_PATH",
            str(DEFAULT_FLASHCORE_DATA_DIR / DEFAULT_DATABASE_FILENAME)
        ),
        help="Path to DuckDB database file (default: ~/.cultivation/flashcore_data/flash.db or $FLASHCORE_DB_PATH)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any YAML errors are encountered (default: False)"
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    assets_dir = Path(args.assets_dir)
    db_path = Path(args.db_path)

    print(f"[INFO] Loading YAML decks from: {source_dir.resolve()}")
    print(f"[INFO] Using assets directory: {assets_dir.resolve()}")
    print(f"[INFO] Target DB: {db_path.resolve()}")

    cards, errors = load_and_process_flashcard_yamls(
        source_directory=source_dir,
        assets_root_directory=assets_dir
    )

    if errors:
        print(f"[ERROR] Encountered {len(errors)} errors during YAML processing:")
        for err in errors:
            print(f"  - {err}")
        if args.strict:
            print("[FATAL] Strict mode is enabled. Aborting ingestion.")
            sys.exit(1)
        else:
            print("[WARN] Proceeding to ingest only valid cards.")

    if not cards:
        print("[WARN] No valid cards to ingest. Exiting.")
        sys.exit(0)

    db = FlashcardDatabase(db_path)
    db.initialize_schema()
    db.upsert_cards_batch(cards)
    print(f"[SUCCESS] Ingested {len(cards)} cards into {db_path.resolve()}.")

if __name__ == "__main__":
    main()
