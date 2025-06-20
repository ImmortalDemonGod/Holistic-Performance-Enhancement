"""
Review session management logic for flashcore.
Handles starting review sessions, fetching cards, and processing reviews.
"""

import logging
from typing import List, Optional
from uuid import UUID
import datetime
from collections import deque

from .database import FlashcardDatabase
from .scheduler import FSRS_Scheduler
from .card import Card, Review, CardState

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

    def __init__(self, db: FlashcardDatabase, scheduler: FSRS_Scheduler):
        """
        Initializes the ReviewSessionManager.

        Args:
            db: An instance of FlashcardDatabase to interact with the card store.
            scheduler: An instance of FSRS_Scheduler for review scheduling.
        """
        if not isinstance(db, FlashcardDatabase):
            raise TypeError("db must be an instance of FlashcardDatabase")
        if not isinstance(scheduler, FSRS_Scheduler):
            raise TypeError("scheduler must be an instance of FSRS_Scheduler")
        
        self.db: FlashcardDatabase = db
        self.scheduler: FSRS_Scheduler = scheduler
        self.review_queue: deque[Card] = deque()
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
        self.review_queue = deque(due_cards)
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
        
        next_card = self.review_queue.popleft() # O(1) operation
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

    def _normalize_review_timestamp(self, review_ts: Optional[datetime.datetime]) -> datetime.datetime:
        """Ensures the review timestamp is a timezone-aware UTC datetime."""
        if review_ts is None:
            return datetime.datetime.now(datetime.timezone.utc)
        
        if review_ts.tzinfo is None or review_ts.tzinfo.utcoffset(review_ts) is None:
            # If a naive datetime is passed, assume it's UTC
            return review_ts.replace(tzinfo=datetime.timezone.utc)
        
        if review_ts.tzinfo != datetime.timezone.utc:
            return review_ts.astimezone(datetime.timezone.utc)
        
        return review_ts

    def _build_review_object(
        self,
        *,
        card_uuid: UUID,
        rating: int,
        review_ts: datetime.datetime,
        elapsed_days: int,
        scheduler_output: dict,
        history: List[Review],
        resp_ms: Optional[int] = None,
    ) -> Review:
        """Constructs the Review object from computed and provided data."""
        stab_before = history[-1].stab_after if history else None
        
        return Review(
            card_uuid=card_uuid,
            ts=review_ts,
            rating=rating,
            resp_ms=resp_ms,
            stab_before=stab_before,
            stab_after=scheduler_output["stability"],
            diff=scheduler_output["difficulty"],
            next_due=scheduler_output["next_review_due"],
            elapsed_days_at_review=elapsed_days,
            scheduled_days_interval=scheduler_output["scheduled_days"],
            review_type=scheduler_output["state"].lower(),
        )

    def _persist_review(self, review: Review) -> Optional[Review]:
        """Saves the review to the database."""
        try:
            review_id = self.db.add_review(review)
            review.review_id = review_id  # Update model with DB-generated ID
            logger.info(f"Successfully submitted review ID {review_id} for card {review.card_uuid}.")
            return review
        except Exception as e:
            logger.error(f"Database error submitting review for card {review.card_uuid}: {e}")
            return None

    def submit_review(self, card_uuid: UUID, rating: int, resp_ms: int) -> Optional[CardState]:
        """
        Processes a review for a given card.

        - Fetches the card and its history.
        - Calculates the next review state using the scheduler.
        - Creates a new Review object.
        - Persists the new review and updates the card's state in the database.

        Args:
            card_uuid: The UUID of the card being reviewed.
            rating: The user's rating for the card (e.g., 1-4).
            resp_ms: The user's response time in milliseconds.

        Returns:
            The new state of the card after the review, or None if the review failed.
        """
        if card_uuid not in self.current_session_card_uuids:
            logger.warning(f"Attempted to review card {card_uuid} not in the current session. Ignoring.")
            return None

        ts = self._normalize_review_timestamp(None)
        card = self.db.get_card_by_uuid(card_uuid)
        if not card:
            logger.error(f"Card with UUID {card_uuid} not found for review submission.")
            return None

        history = self.db.get_reviews_for_card(card_uuid, order_by_ts_desc=False)
        elapsed_days = self._calculate_elapsed_days(card, history, ts)

        try:
            scheduler_output = self.scheduler.compute_next_state(history, rating, ts)
        except ValueError as e:
            logger.error(f"Error computing next state for card {card_uuid}: {e}")
            return None

        new_review = self._build_review_object(
            card_uuid=card_uuid,
            rating=rating,
            review_ts=ts,
            elapsed_days=elapsed_days,
            scheduler_output=scheduler_output,
            history=history,
            resp_ms=resp_ms,
        )

        persisted_review = self._persist_review(new_review)
        if not persisted_review:
            return None

        # Update the card's state and due date in the database
        card.next_due_date = scheduler_output["next_review_due"]
        self.db.update_card_state(
            card_uuid=card_uuid,
            state=scheduler_output["next_state"],
            due=scheduler_output["next_review_due"]
        )
        card.reps += 1
        # FSRS 'state' can be 'New', 'Learn', 'Review', 'Relearn'. 'Relearn' indicates a lapse.
        if scheduler_output["state"] == "Relearn":
            card.lapses += 1
        
        # Persist the updated card state. Assuming this method exists in the DB layer.
        self.db.update_card_state(
            card_uuid=card.uuid,
            due=card.due,
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps,
            lapses=card.lapses,
        )
        logger.info(f"Updated card {card_uuid} with new state. New due date: {card.due.date()}.")

        # Return the new state
        new_card_state = CardState(
            due=card.due,
            stability=card.stability,
            difficulty=card.difficulty,
            reps=card.reps,
            lapses=card.lapses,
        )
        return new_card_state

    def get_due_card_count(self) -> int:
        """
        Gets the total number of cards currently due for review using the efficient database count method.

        Returns:
            The count of due cards.
        """
        today = datetime.datetime.now(datetime.timezone.utc).date()
        return self.db.get_due_card_count(on_date=today)
