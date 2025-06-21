"""
This module defines the ReviewSessionManager class, which is responsible for
managing a flashcard review session. It interacts with the database to fetch
cards, uses a scheduler to determine review timings, and records review outcomes.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Set
from uuid import UUID, uuid4

from .card import Card, Review, Rating
from .database import FlashcardDatabase
from .scheduler import FSRS_Scheduler as FSRS, SchedulerOutput

# Initialize logger
logger = logging.getLogger(__name__)


class ReviewSessionManager:
    """
    Manages a review session for flashcards.

    This class is responsible for:
    - Initializing a review session with a specific set of cards.
    - Providing cards one by one for review.
    - Processing user reviews and updating card states.
    - Interacting with the database to persist changes.
    """

    def __init__(
        self,
        db_manager: FlashcardDatabase,
        scheduler: FSRS,
        user_uuid: UUID,
        deck_name: str,
    ):
        """
        Initializes the ReviewSessionManager.

        Args:
            db_manager: An instance of DatabaseManager to interact with the database.
            scheduler: An instance of a scheduling algorithm (e.g., FSRS).
            user_uuid: The UUID of the user conducting the review.
            deck_name: The name of the deck being reviewed.
        """
        self.db = db_manager
        self.scheduler = scheduler
        self.user_uuid = user_uuid
        self.deck_name = deck_name
        self.session_uuid = uuid4()
        self.review_queue: list[Card] = []
        self.current_session_card_uuids: Set[UUID] = set()
        self.session_start_time = datetime.now(timezone.utc)

    def initialize_session(self, limit: int = 20) -> None:
        """
        Initializes the review session by fetching due cards from the database.

        Args:
            limit: The maximum number of cards to fetch for the session.
        """
        logger.info(
            f"Initializing review session {self.session_uuid} for user {self.user_uuid} and deck '{self.deck_name}'"
        )
        today = datetime.now(timezone.utc).date()
        due_cards = self.db.get_due_cards(self.deck_name, on_date=today, limit=limit)
        self.review_queue = sorted(due_cards, key=lambda c: c.modified_at)
        self.current_session_card_uuids = {card.uuid for card in self.review_queue}
        logger.info(f"Initialized session with {len(self.review_queue)} cards.")

    def get_next_card(self) -> Optional[Card]:
        """
        Retrieves the next card to be reviewed.

        Returns:
            The next Card object to be reviewed, or None if the queue is empty.
        """
        if not self.review_queue:
            logger.info("Review queue is empty. Session may be complete.")
            return None
        return self.review_queue[0]

    def _get_card_from_queue(self, card_uuid: UUID) -> Optional[Card]:
        """
        Finds a card in the current review queue by its UUID.

        Args:
            card_uuid: The UUID of the card to find.

        Returns:
            The Card object if found, otherwise None.
        """
        for card in self.review_queue:
            if card.uuid == card_uuid:
                return card
        return None

    def _remove_card_from_queue(self, card_uuid: UUID) -> None:
        """
        Removes a card from the review queue.

        Args:
            card_uuid: The UUID of the card to remove.
        """
        self.review_queue = [card for card in self.review_queue if card.uuid != card_uuid]

    def submit_review(
        self,
        card_uuid: UUID,
        rating: Rating,
        reviewed_at: Optional[datetime] = None,
        resp_ms: int = 0,
    ) -> Card:
        """
        Submits a review for a card, updates its state, and schedules the next review.

        Args:
            card_uuid: The UUID of the card being reviewed.
            rating: The user's rating of the card (e.g., Again, Hard, Good, Easy).
            reviewed_at: The timestamp of the review. Defaults to now.
            resp_ms: The response time in milliseconds.

        Returns:
            The updated Card object.

        Raises:
            ValueError: If the card is not in the current session.
            CardOperationError: If there's an issue with the database update.
        """
        ts = reviewed_at or datetime.now(timezone.utc)

        card = self._get_card_from_queue(card_uuid)
        if not card:
            raise ValueError(f"Card {card_uuid} not found in the current review session.")

        # Fetch the full review history for the scheduler, in chronological order.
        review_history = self.db.get_reviews_for_card(card.uuid, order_by_ts_desc=False)

        # The scheduler needs the full history to compute the next state correctly.
        scheduler_output: SchedulerOutput = self.scheduler.compute_next_state(
            history=review_history,
            new_rating=rating.value,  # Pass the integer value of the enum
            review_ts=ts
        )

        new_review = Review(
            card_uuid=card_uuid,
            ts=ts,
            rating=rating.value,
            resp_ms=resp_ms,
            stab_before=card.stability,  # Comes from the card's last state
            stab_after=scheduler_output.stab,
            diff=scheduler_output.diff,
            next_due=scheduler_output.next_due,
            elapsed_days_at_review=scheduler_output.elapsed_days,
            scheduled_days_interval=scheduler_output.scheduled_days,
            review_type=scheduler_output.review_type,
        )

        try:
            # The database function handles updating the card's state, stability, etc.
            updated_card = self.db.add_review_and_update_card(
                review=new_review,
                new_card_state=scheduler_output.state
            )

            self._remove_card_from_queue(card_uuid)

            return updated_card
        except self.db.CardOperationError as e:
            logger.error(f"Failed to submit review for card {card_uuid}: {e}")
            raise

    def get_session_stats(self) -> Dict[str, int]:
        """
        Returns statistics for the current review session.

        Returns:
            A dictionary with session statistics.
        """
        total_cards = len(self.current_session_card_uuids)
        reviewed_cards = total_cards - len(self.review_queue)
        return {"total_cards": total_cards, "reviewed_cards": reviewed_cards}

    def get_due_card_count(self) -> int:
        """
        Gets the total number of cards currently due for review using the efficient database count method.

        Returns:
            The count of due cards.
        """
        today = datetime.now(timezone.utc).date()
        return self.db.get_due_card_count(deck_name=self.deck_name, on_date=today)
