"""
CLI entry point for flashcard system (cultivation-flashcards).
"""

import uuid
from pathlib import Path

import typer

from cultivation.scripts.flashcore.cli.review_ui import start_review_flow
from cultivation.scripts.flashcore.database import FlashcardDatabase
from cultivation.scripts.flashcore.review_manager import ReviewSessionManager
from cultivation.scripts.flashcore.scheduler import FSRS_Scheduler

app = typer.Typer()

# Define a constant for the project root to make path resolution robust.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "cultivation" / "data" / "flash.db"

# A default, hardcoded user UUID. In a real system, this would come from a config file.
DEFAULT_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")


@app.command()
def review(
    deck_name: str = typer.Argument(..., help="The name of the deck to review."),
    db_path: Path = typer.Option(
        DB_PATH, "--db-path", help="Path to the flashcard database file."
    ),
):
    """
    Start a command-line interface session to review flashcards for a given deck.
    """
    if not db_path.parent.exists():
        print(f"Database directory {db_path.parent} does not exist. Creating it.")
        db_path.parent.mkdir(parents=True, exist_ok=True)

    db_manager = FlashcardDatabase(db_path=str(db_path))
    db_manager.initialize_schema()

    scheduler = FSRS_Scheduler()

    manager = ReviewSessionManager(
        db_manager=db_manager,
        scheduler=scheduler,
        user_uuid=DEFAULT_USER_UUID,
        deck_name=deck_name,
    )

    start_review_flow(manager)


if __name__ == "__main__":
    app()

