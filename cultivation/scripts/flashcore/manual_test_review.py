"""
Script for manually testing the CLI review flow.
"""

import os
import sys
from datetime import datetime, timezone, date
from typing import List, Optional
from uuid import UUID, uuid4

# Add the project root to the Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cultivation.scripts.flashcore.card import Card, Review, CardState
from cultivation.scripts.flashcore.database import FlashcardDatabase
from cultivation.scripts.flashcore.review_manager import ReviewSessionManager
from cultivation.scripts.flashcore.scheduler import FSRS_Scheduler
from cultivation.scripts.flashcore.cli.review_ui import start_review_flow


class MockDatabase(FlashcardDatabase):
    """An in-memory mock of the FlashcardDatabase for testing."""

    def __init__(self, cards: List[Card]):
        self._cards = {c.uuid: c for c in cards}
        self._reviews: dict[UUID, List[Review]] = {}

    def get_due_cards(self, on_date: date, limit: int = 50) -> List[Card]:
        # For this test, all cards are new, so they are all due.
        return list(self._cards.values())[:limit]

    def get_card_by_uuid(self, card_uuid: UUID) -> Optional[Card]:
        return self._cards.get(card_uuid)

    def get_reviews_for_card(self, card_uuid: UUID, order_by_ts_desc: bool = True) -> List[Review]:
        reviews = self._reviews.get(card_uuid, [])
        # The mock doesn't need to implement sorting for this test, but it accepts the arg.
        return reviews

    def add_review(self, review: Review) -> None:
        if review.card_uuid not in self._reviews:
            self._reviews[review.card_uuid] = []
        self._reviews[review.card_uuid].append(review)

    def update_card_state(self, card_uuid: UUID, state: CardState, due: date) -> None:
        card = self._cards.get(card_uuid)
        if card:
            # Pydantic models can be immutable, so we create a new instance
            card_dict = card.model_dump()
            card_dict['state'] = state
            card_dict['next_due_date'] = due
            self._cards[card_uuid] = Card(**card_dict)

    def get_all_cards(self) -> List[Card]:
        return list(self._cards.values())


def main():
    """Sets up and runs a manual review session."""
    # 1. Create a sample deck with a few cards
    cards = [
        Card(
            uuid=uuid4(),
            deck_name="Geography",
            front="What is the capital of Japan?",
            back="Tokyo",
            tags={"capitals", "asia"},
            added_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        ),
        Card(
            uuid=uuid4(),
            deck_name="Geography",
            front="What is the highest mountain in the world?",
            back="Mount Everest",
            tags={"mountains", "geography-facts"},
            added_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        ),
        Card(
            uuid=uuid4(),
            deck_name="Geography",
            front="Which river is the longest in Africa?",
            back="The Nile",
            tags={"rivers", "africa"},
            added_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        ),
    ]

    # 2. Initialize the mock database, scheduler, and manager
    mock_db = MockDatabase(cards=cards)
    scheduler = FSRS_Scheduler()
    manager = ReviewSessionManager(db=mock_db, scheduler=scheduler)

    # 3. Start the review flow
    print("--- Starting Manual CLI Review Test ---")
    print("This will simulate a real review session with 3 sample cards.")
    print("All cards are new, so they will be presented for review.")
    print("Follow the prompts to complete the session.")
    print("-----------------------------------------")

    start_review_flow(manager)


if __name__ == "__main__":
    main()
