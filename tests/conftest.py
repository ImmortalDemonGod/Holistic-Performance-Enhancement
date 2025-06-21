"""
This file contains shared fixtures for the entire test suite.

Fixtures defined here are automatically available to all tests without needing
to be imported. The `scope` parameter controls how often a fixture is set up
and torn down.
"""
import sys
from pathlib import Path

# Add the project root to the Python path before any other imports
# This allows tests to import modules from the 'cultivation' package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from uuid import uuid4

import pytest

from cultivation.scripts.flashcore.card import Card, CardState
from cultivation.scripts.flashcore.database import FlashcardDatabase
from cultivation.scripts.flashcore.review_manager import ReviewSessionManager
from cultivation.scripts.flashcore.scheduler import FSRS_Scheduler as FSRS


@pytest.fixture(scope="function")
def db_manager(tmp_path):
    """
    Fixture to provide a DatabaseManager instance with a temporary, isolated database.
    This fixture is function-scoped, so each test function gets a fresh database.
    """
    # Use a temporary file for the database to ensure test isolation
    db_file = tmp_path / "test_flashcards.db"
    db = FlashcardDatabase(db_path=str(db_file))
    db.init_db()  # Ensure tables are created
    yield db
    db.close()


@pytest.fixture(scope="session")
def fsrs_scheduler():
    """
    Fixture to provide a single, session-scoped FSRS scheduler instance.
    The scheduler is stateless, so it can be shared across all tests in a session.
    """
    return FSRS()


@pytest.fixture(scope="function")
def review_manager(db_manager, fsrs_scheduler):
    """Fixture to provide a ReviewSessionManager instance."""
    # Provide dummy UUIDs for user and deck, as required by the new constructor.
    # This is a common practice in tests where the specific UUID value isn't critical.
    user_uuid = uuid4()
    deck_uuid = uuid4()
    return ReviewSessionManager(
        db_manager=db_manager,
        scheduler=fsrs_scheduler,
        user_uuid=user_uuid,
        deck_uuid=deck_uuid,
    )


@pytest.fixture(scope="function")
def new_card_factory(db_manager):
    """
    Factory fixture to create and add new cards to the database.
    This allows tests to easily create card data as needed.
    """

    def _create_card(deck_uuid, content="Test Content"):
        card = Card(deck_uuid=deck_uuid, content=content)
        db_manager.add_card(card)
        return db_manager.get_card_by_uuid(card.uuid)

    return _create_card


@pytest.fixture(scope="function")
def reviewed_card_factory(db_manager, new_card_factory):
    """
    Factory fixture to create a card that has already been reviewed once.
    This is useful for testing cards in states other than 'New'.
    """

    def _create_card(deck_uuid, content="Reviewed Card Content"):
        card = new_card_factory(deck_uuid, content)
        # Simulate a review to move it out of the 'New' state
        card.state = CardState.LEARNING
        card.stability = 1.0
        card.difficulty = 5.0
        db_manager.update_card(card)
        return db_manager.get_card_by_uuid(card.uuid)

    return _create_card
