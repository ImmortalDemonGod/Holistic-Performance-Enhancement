"""
Review session management logic for flashcore.
Handles starting review sessions, fetching cards, and processing reviews.
"""

import logging
from typing import List, Optional
from uuid import UUID
import datetime

from .database import FlashcardDatabase
from .scheduler import FSRS_Scheduler # FSRSSchedulerConfig if needed for custom config
from .card import Card, Review

logger = logging.getLogger(__name__)

class ReviewSessionManager:
    """
    Manages the lifecycle of a flashcard review session.

    Responsibilities:
    - Starting a new review session (fetching due cards).
    - Providing the next card for review.
    - Submitting a review for a card (calculating new state, persisting review).
    - Getting the count of currently due cards.
    """

    def __init__(self, db: FlashcardDatabase):
        """
        Initializes the ReviewSessionManager.

        Args:
            db: An instance of FlashcardDatabase to interact with the card store.
        """
        if not isinstance(db, FlashcardDatabase):
            raise TypeError("db must be an instance of FlashcardDatabase")
        
        self.db: FlashcardDatabase = db
        self.scheduler: FSRS_Scheduler = FSRS_Scheduler() # Uses default config
        self.review_queue: List[Card] = []
        self.current_session_card_uuids: set[UUID] = set()

        logger.info("ReviewSessionManager initialized.")

    def start_session(self, limit: int = 20) -> None:
        """
        Starts a new review session or refreshes the current one.

        Fetches due cards from the database and populates the internal review queue.
        Clears any existing queue before populating.

        Args:
            limit: The maximum number of cards to fetch for the session.
        """
        today = datetime.datetime.now(datetime.timezone.utc).date()
        logger.info(f"Starting new review session for date {today} with limit {limit}.")
        
        due_cards = self.db.get_due_cards(on_date=today, limit=limit)
        self.review_queue = due_cards
        self.current_session_card_uuids = {card.uuid for card in due_cards}
        
        logger.info(f"Review session started with {len(self.review_queue)} cards.")

    def get_next_card(self) -> Optional[Card]:
        """
        Retrieves the next card from the review queue.

        Returns:
            The next Card to review, or None if the queue is empty.
        """
        if not self.review_queue:
            logger.info("Review queue is empty. No next card.")
            return None
        
        next_card = self.review_queue.pop(0) # FIFO
        # self.current_session_card_uuids.remove(next_card.uuid) # No, keep track of all cards loaded in this session
        logger.info(f"Retrieved card {next_card.uuid} from queue. {len(self.review_queue)} cards remaining.")
        return next_card

    # --- Placeholder for future methods ---
    # def submit_review(self, card_uuid: UUID, rating: int, resp_ms: Optional[int] = None) -> Optional[Review]:
    #     pass

    # def get_due_card_count(self) -> int:
    #     pass

    # def _calculate_elapsed_days(self, card: Card, history: List[Review], review_ts: datetime.datetime) -> int:
    #     pass

