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

    def _calculate_elapsed_days(self, card: Card, history: List[Review], review_ts: datetime.datetime) -> int:
        """
        Calculates the number of days elapsed since the card was last scheduled or added.

        Args:
            card: The card being reviewed.
            history: The review history for the card, sorted chronologically (oldest first).
            review_ts: The timestamp of the current review.

        Returns:
            The number of elapsed days.
        """
        if not history:
            # First review for this card
            # Ensure card.added_at is timezone-aware (should be UTC from DB/model)
            # Ensure review_ts is timezone-aware (should be UTC)
            elapsed = (review_ts.date() - card.added_at.date()).days
        else:
            # Subsequent review
            last_review = history[-1]
            # last_review.next_due is a date object, review_ts.date() is also a date object
            elapsed = (review_ts.date() - last_review.next_due).days
        
        # Elapsed days cannot be negative (e.g. reviewing earlier than scheduled)
        # FSRS expects non-negative elapsed_days.
        return max(0, elapsed)

    def submit_review(self, card_uuid: UUID, rating: int, review_ts: Optional[datetime.datetime] = None, resp_ms: Optional[int] = None) -> Optional[Review]:
        """
        Submits a review for a given card.

        Args:
            card_uuid: The UUID of the card being reviewed.
            rating: The user's rating (0=Again, 1=Hard, 2=Good, 3=Easy).
            review_ts: The UTC timestamp of the review. Defaults to now if None.
            resp_ms: Optional response time in milliseconds.

        Returns:
            The newly created Review object if successful, else None.
        """
        if review_ts is None:
            review_ts = datetime.datetime.now(datetime.timezone.utc)
        elif review_ts.tzinfo is None or review_ts.tzinfo.utcoffset(review_ts) is None:
            # If a naive datetime is passed, assume it's UTC, then localize
            review_ts = review_ts.replace(tzinfo=datetime.timezone.utc)
        elif review_ts.tzinfo != datetime.timezone.utc:
            review_ts = review_ts.astimezone(datetime.timezone.utc)

        card = self.db.get_card_by_uuid(card_uuid)
        if not card:
            logger.error(f"submit_review: Card with UUID {card_uuid} not found.")
            return None

        # Fetch history sorted chronologically (oldest first) for FSRS
        history = self.db.get_reviews_for_card(card_uuid, order_desc=False)

        elapsed_days = self._calculate_elapsed_days(card, history, review_ts)

        try:
            scheduler_output = self.scheduler.compute_next_state(history, rating, review_ts)
        except ValueError as e:
            logger.error(f"Error computing next state for card {card_uuid}: {e}")
            return None

        stab_before = history[-1].stab_after if history else None # FSRS handles initial stability if None

        new_review = Review(
            card_uuid=card_uuid,
            ts=review_ts,
            rating=rating,
            resp_ms=resp_ms,
            stab_before=stab_before, # Will be None for the first review
            stab_after=scheduler_output["stability"],
            diff=scheduler_output["difficulty"],
            next_due=scheduler_output["next_review_due"],
            elapsed_days_at_review=elapsed_days,
            scheduled_days_interval=scheduler_output["scheduled_days"],
            review_type=scheduler_output["state"].lower() # e.g. 'review', 'learn'
        )

        try:
            review_id = self.db.add_review(new_review)
            new_review.review_id = review_id # Update the model with the DB-generated ID
            logger.info(f"Successfully submitted review ID {review_id} for card {card_uuid}.")
            
            # If card was in the current session's queue, remove it
            # This prevents re-reviewing in the same session after submission
            if card_uuid in self.current_session_card_uuids:
                self.review_queue = [c for c in self.review_queue if c.uuid != card_uuid]
                # self.current_session_card_uuids.remove(card_uuid) # No, keep for session stats if needed

            return new_review
        except Exception as e: # Catch broader exceptions from DB layer
            logger.error(f"Database error submitting review for card {card_uuid}: {e}")
            return None

    def get_due_card_count(self) -> int:
        """
        Gets the total number of cards currently due for review.

        Returns:
            The count of due cards.
        """
        today = datetime.datetime.now(datetime.timezone.utc).date()
        count = self.db.get_due_card_count(on_date=today)
        logger.info(f"There are {count} cards due as of {today}.")
        return count


