"""
Unit and integration tests for ReviewSessionManager in flashcore.review_manager.
"""

import pytest
import uuid
from collections import deque
from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock, patch

from cultivation.scripts.flashcore.card import Card, Review
from cultivation.scripts.flashcore.database import FlashcardDatabase
from cultivation.scripts.flashcore.scheduler import FSRS_Scheduler
from cultivation.scripts.flashcore.review_manager import ReviewSessionManager

# --- Fixtures --- 

@pytest.fixture
def mock_db() -> MagicMock:
    """Provides a MagicMock for FlashcardDatabase."""
    db = MagicMock(spec=FlashcardDatabase)
    db.get_due_cards = MagicMock(return_value=[])
    db.get_card_by_uuid = MagicMock(return_value=None)
    db.get_reviews_for_card = MagicMock(return_value=[])
    db.add_review = MagicMock(return_value=123) # Mock review_id
    db.get_due_card_count = MagicMock(return_value=0)
    return db

@pytest.fixture
def mock_scheduler() -> MagicMock:
    """Provides a MagicMock for FSRS_Scheduler."""
    scheduler = MagicMock(spec=FSRS_Scheduler)
    scheduler.compute_next_state = MagicMock(
        return_value={
            "stability": 10.0,
            "difficulty": 5.0,
            "next_review_due": date.today() + timedelta(days=10),
            "scheduled_days": 10,
            "state": "REVIEW"
        }
    )
    return scheduler

@pytest.fixture
def sample_card_data() -> dict:
    return {
        "uuid": uuid.uuid4(),
        "deck_name": "Test Deck",
        "front": "Test Front",
        "back": "Test Back",
        "added_at": datetime.now(timezone.utc) - timedelta(days=30)
    }

@pytest.fixture
def sample_card(sample_card_data: dict) -> Card:
    return Card(**sample_card_data)

@pytest.fixture
def review_manager(mock_db: MagicMock, mock_scheduler: MagicMock) -> ReviewSessionManager:
    """Provides a ReviewSessionManager instance with mocked DB and scheduler."""
    return ReviewSessionManager(db=mock_db, scheduler=mock_scheduler)


@pytest.fixture
def in_memory_db() -> FlashcardDatabase:
    """Provides an in-memory FlashcardDatabase instance for integration tests."""
    db = FlashcardDatabase(db_path=':memory:')
    db.initialize_schema()
    return db

# --- Test Cases ---

class TestReviewSessionManagerInit:
    def test_init_successful(self, mock_db: MagicMock, mock_scheduler: MagicMock):
        """Test successful initialization of ReviewSessionManager."""
        manager = ReviewSessionManager(db=mock_db, scheduler=mock_scheduler)
        assert manager.db == mock_db
        assert manager.scheduler == mock_scheduler
        assert isinstance(manager.review_queue, deque)
        assert manager.current_session_card_uuids == set()

    def test_init_type_error_for_db(self, mock_scheduler: MagicMock):
        """Test that TypeError is raised if db is not FlashcardDatabase."""
        with pytest.raises(TypeError, match="db must be an instance of FlashcardDatabase"):
            ReviewSessionManager(db="not_a_db_instance", scheduler=mock_scheduler)

    def test_init_type_error_for_scheduler(self, mock_db: MagicMock):
        """Test that TypeError is raised if scheduler is not FSRS_Scheduler."""
        with pytest.raises(TypeError, match="scheduler must be an instance of FSRS_Scheduler"):
            ReviewSessionManager(db=mock_db, scheduler="not_a_scheduler")

class TestStartSessionAndGetNextCard:
    def test_start_session_populates_queue(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test start_session fetches due cards and populates the review queue."""
        mock_cards = [sample_card, Card(**{**sample_card.model_dump(), "uuid": uuid.uuid4()})]
        mock_db.get_due_cards.return_value = mock_cards
        
        review_manager.start_session(limit=10)
        
        mock_db.get_due_cards.assert_called_once_with(on_date=date.today(), limit=10)
        assert list(review_manager.review_queue) == mock_cards
        assert review_manager.current_session_card_uuids == {c.uuid for c in mock_cards}

    def test_start_session_clears_existing_queue(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test start_session clears any pre-existing queue before populating."""
        # Pre-populate queue
        initial_card = Card(**{**sample_card.model_dump(), "uuid": uuid.uuid4(), "front": "Old Card"})
        review_manager.review_queue = deque([initial_card])
        review_manager.current_session_card_uuids = {initial_card.uuid}
        
        new_mock_cards = [sample_card]
        mock_db.get_due_cards.return_value = new_mock_cards
        
        review_manager.start_session(limit=5)
        
        mock_db.get_due_cards.assert_called_with(on_date=date.today(), limit=5)
        assert list(review_manager.review_queue) == new_mock_cards
        assert initial_card not in review_manager.review_queue
        assert review_manager.current_session_card_uuids == {c.uuid for c in new_mock_cards}

    def test_get_next_card_returns_card_from_queue(self, review_manager: ReviewSessionManager, sample_card: Card):
        """Test get_next_card returns the next card and removes it from queue (FIFO)."""
        card1 = sample_card
        card2_uuid = uuid.uuid4()
        card2 = Card(**{**sample_card.model_dump(), "uuid": card2_uuid, "front": "Second Card"})
        review_manager.review_queue = deque([card1, card2])
        review_manager.current_session_card_uuids = {card1.uuid, card2.uuid}
        
        next_c = review_manager.get_next_card()
        assert next_c == card1
        assert list(review_manager.review_queue) == [card2]
        # current_session_card_uuids should remain unchanged by get_next_card
        assert review_manager.current_session_card_uuids == {card1.uuid, card2.uuid} 

        next_c_2 = review_manager.get_next_card()
        assert next_c_2 == card2
        assert list(review_manager.review_queue) == []
        assert review_manager.current_session_card_uuids == {card1.uuid, card2.uuid}

    def test_get_next_card_empty_queue_returns_none(self, review_manager: ReviewSessionManager):
        """Test get_next_card returns None when the queue is empty."""
        review_manager.review_queue = []
        review_manager.current_session_card_uuids = set()
        
        next_c = review_manager.get_next_card()
        assert next_c is None

class TestSubmitReviewAndHelpers:
    def test_calculate_elapsed_days_new_card(self, review_manager: ReviewSessionManager, sample_card: Card):
        """Test _calculate_elapsed_days for a card with no prior reviews."""
        review_ts = sample_card.added_at + timedelta(days=5, hours=2) # Reviewed 5 days after adding
        elapsed_days = review_manager._calculate_elapsed_days(sample_card, [], review_ts)
        assert elapsed_days == 5

    def test_calculate_elapsed_days_with_history(self, review_manager: ReviewSessionManager, sample_card: Card):
        """Test _calculate_elapsed_days for a card with prior reviews."""
        last_review_ts = datetime.now(timezone.utc) - timedelta(days=10)
        history = [
            Review(
                card_uuid=sample_card.uuid,
                ts=last_review_ts - timedelta(days=20),
                rating=2, stab_after=5.0, diff=5.0, 
                next_due=last_review_ts.date() - timedelta(days=10), # Due 10 days before last review
                elapsed_days_at_review=0, scheduled_days_interval=10
            ),
            Review(
                card_uuid=sample_card.uuid,
                ts=last_review_ts, 
                rating=2, stab_after=10.0, diff=5.0, 
                next_due=last_review_ts.date() + timedelta(days=7), # Due 7 days after last review
                elapsed_days_at_review=10, scheduled_days_interval=7
            )
        ]
        # Current review is 3 days after it was due from the last review
        current_review_ts = last_review_ts + timedelta(days=10, hours=1)
        expected_elapsed_days = (current_review_ts.date() - history[-1].next_due).days
        
        elapsed_days = review_manager._calculate_elapsed_days(sample_card, history, current_review_ts)
        assert elapsed_days == expected_elapsed_days
        assert elapsed_days == 3 # Explicitly (10 days from last review - 7 days scheduled interval)

    def test_calculate_elapsed_days_review_early(self, review_manager: ReviewSessionManager, sample_card: Card):
        """Test _calculate_elapsed_days when reviewing earlier than scheduled (should be 0)."""
        last_review_ts = datetime.now(timezone.utc) - timedelta(days=10)
        history = [
            Review(
                card_uuid=sample_card.uuid,
                ts=last_review_ts,
                rating=2, stab_after=10.0, diff=5.0,
                next_due=last_review_ts.date() + timedelta(days=5), # Due in 5 days
                elapsed_days_at_review=0, scheduled_days_interval=5
            )
        ]
        # Current review is 2 days before it was due
        current_review_ts = last_review_ts + timedelta(days=3)
        elapsed_days = review_manager._calculate_elapsed_days(sample_card, history, current_review_ts)
        assert elapsed_days == 0 # Max(0, -2) is 0

    def test_submit_review_successful_new_card(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test submit_review for a new card (no history)."""
        mock_db.get_card_by_uuid.return_value = sample_card
        mock_db.get_reviews_for_card.return_value = [] # No history
        
        rating = 2 # Good
        review_ts = sample_card.added_at + timedelta(days=1) # Reviewed 1 day after adding
        resp_ms = 5000

        # Expected elapsed_days for _calculate_elapsed_days
        expected_elapsed_days = (review_ts.date() - sample_card.added_at.date()).days
        assert expected_elapsed_days == 1

        # Scheduler's expected output (already configured in review_manager fixture)
        scheduler_output = review_manager.scheduler.compute_next_state.return_value

        returned_review = review_manager.submit_review(sample_card.uuid, rating, review_ts, resp_ms)

        mock_db.get_card_by_uuid.assert_called_once_with(sample_card.uuid)
        mock_db.get_reviews_for_card.assert_called_once_with(sample_card.uuid, order_by_ts_desc=False)
        review_manager.scheduler.compute_next_state.assert_called_once_with([], rating, review_ts)
        
        expected_review_args = Review(
            card_uuid=sample_card.uuid, ts=review_ts, rating=rating, resp_ms=resp_ms,
            stab_before=None, # New card
            stab_after=scheduler_output["stability"], diff=scheduler_output["difficulty"],
            next_due=scheduler_output["next_review_due"], 
            elapsed_days_at_review=expected_elapsed_days, 
            scheduled_days_interval=scheduler_output["scheduled_days"],
            review_type=scheduler_output["state"].lower()
        )
        mock_db.add_review.assert_called_once()
        # Check the Review object passed to add_review
        actual_review_arg = mock_db.add_review.call_args[0][0]
        assert actual_review_arg.card_uuid == expected_review_args.card_uuid
        assert actual_review_arg.ts == expected_review_args.ts
        assert actual_review_arg.rating == expected_review_args.rating
        # ... (assert other fields similarly or use model_dump for full comparison if Pydantic supports it well with mocks)
        assert actual_review_arg.model_dump(exclude={'review_id'}) == expected_review_args.model_dump(exclude={'review_id'})

        assert returned_review is not None
        assert returned_review.review_id == 123 # From mock_db.add_review
        assert returned_review.stab_after == scheduler_output["stability"]

    def test_submit_review_successful_with_history(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test submit_review for a card with existing review history."""
        mock_db.get_card_by_uuid.return_value = sample_card
        
        prev_review_ts = sample_card.added_at + timedelta(days=5)
        prev_next_due = prev_review_ts.date() + timedelta(days=2)
        history = [
            Review(
                review_id=100, card_uuid=sample_card.uuid, ts=prev_review_ts, rating=1, resp_ms=6000,
                stab_before=1.0, stab_after=2.5, diff=6.0, next_due=prev_next_due,
                elapsed_days_at_review=5, scheduled_days_interval=2, review_type="review"
            )
        ]
        mock_db.get_reviews_for_card.return_value = history
        
        rating = 3 # Easy
        review_ts = datetime.combine(prev_next_due + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) # Reviewed 1 day after it was due
        resp_ms = 3000

        expected_elapsed_days = (review_ts.date() - prev_next_due).days
        assert expected_elapsed_days == 1

        scheduler_output = review_manager.scheduler.compute_next_state.return_value

        returned_review = review_manager.submit_review(sample_card.uuid, rating, review_ts, resp_ms)

        mock_db.get_card_by_uuid.assert_called_once_with(sample_card.uuid)
        mock_db.get_reviews_for_card.assert_called_once_with(sample_card.uuid, order_by_ts_desc=False)
        review_manager.scheduler.compute_next_state.assert_called_once_with(history, rating, review_ts)
        
        expected_review_args = Review(
            card_uuid=sample_card.uuid, ts=review_ts, rating=rating, resp_ms=resp_ms,
            stab_before=history[-1].stab_after,
            stab_after=scheduler_output["stability"], diff=scheduler_output["difficulty"],
            next_due=scheduler_output["next_review_due"], 
            elapsed_days_at_review=expected_elapsed_days, 
            scheduled_days_interval=scheduler_output["scheduled_days"],
            review_type=scheduler_output["state"].lower()
        )
        mock_db.add_review.assert_called_once()
        actual_review_arg = mock_db.add_review.call_args[0][0]
        assert actual_review_arg.model_dump(exclude={'review_id'}) == expected_review_args.model_dump(exclude={'review_id'})

        assert returned_review is not None
        assert returned_review.review_id == 123 # From mock_db.add_review

    def test_submit_review_card_not_found(self, review_manager: ReviewSessionManager, mock_db: MagicMock):
        """Test submit_review when the card UUID does not exist."""
        unknown_uuid = uuid.uuid4()
        mock_db.get_card_by_uuid.return_value = None
        
        returned_review = review_manager.submit_review(unknown_uuid, 2, resp_ms=1000)
        
        assert returned_review is None
        mock_db.get_card_by_uuid.assert_called_once_with(unknown_uuid)
        mock_db.get_reviews_for_card.assert_not_called()
        review_manager.scheduler.compute_next_state.assert_not_called()
        mock_db.add_review.assert_not_called()

    def test_submit_review_scheduler_error(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test submit_review when the scheduler raises a ValueError."""
        mock_db.get_card_by_uuid.return_value = sample_card
        mock_db.get_reviews_for_card.return_value = []
        review_manager.scheduler.compute_next_state.side_effect = ValueError("Invalid rating for FSRS")
        
        returned_review = review_manager.submit_review(sample_card.uuid, 99, resp_ms=1000) # Invalid rating to trigger error
        
        assert returned_review is None
        review_manager.scheduler.compute_next_state.assert_called_once()
        mock_db.add_review.assert_not_called()

    def test_submit_review_db_add_error(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test submit_review when db.add_review raises an exception."""
        mock_db.get_card_by_uuid.return_value = sample_card
        mock_db.get_reviews_for_card.return_value = []
        mock_db.add_review.side_effect = Exception("DB connection failed")
        
        returned_review = review_manager.submit_review(sample_card.uuid, 2, resp_ms=1000)
        
        assert returned_review is None
        mock_db.add_review.assert_called_once()

    def test_submit_review_removes_card_from_active_queue(self, review_manager: ReviewSessionManager, mock_db: MagicMock, sample_card: Card):
        """Test that a successfully reviewed card is removed from the active review_queue if it was part of the current session."""
        # Setup: card is in the current session queue
        review_manager.review_queue = [sample_card, Card(**{**sample_card.model_dump(), "uuid": uuid.uuid4()})]
        review_manager.current_session_card_uuids = {c.uuid for c in review_manager.review_queue}
        assert sample_card in review_manager.review_queue

        mock_db.get_card_by_uuid.return_value = sample_card
        mock_db.get_reviews_for_card.return_value = []

        review_manager.submit_review(sample_card.uuid, rating=2, resp_ms=1000)

        assert sample_card not in review_manager.review_queue
        # current_session_card_uuids should still contain it, as it tracks all cards loaded in session
        assert sample_card.uuid in review_manager.current_session_card_uuids 

class TestGetDueCardCount:
    def test_get_due_card_count_calls_db(self, review_manager: ReviewSessionManager, mock_db: MagicMock):
        """Test get_due_card_count calls the database method and returns its result."""
        expected_count = 42
        # The method should call the efficient get_due_card_count method on the db.
        mock_db.get_due_card_count.return_value = expected_count

        count = review_manager.get_due_card_count()

        assert count == expected_count
        # Verify the underlying DB method is called correctly
        mock_db.get_due_card_count.assert_called_once_with(on_date=date.today())
        mock_db.get_due_cards.assert_not_called() # Ensure the less efficient method is not called.

class TestReviewSessionManagerIntegration:
    def test_e2e_session_flow(self, in_memory_db: FlashcardDatabase, sample_card_data: dict):
        """Test the end-to-end flow of a review session using an in-memory DB."""
        # 1. Setup: Initialize schema and add cards to the in-memory DB
        in_memory_db.initialize_schema()
        today = date.today()
        now_utc = datetime.now(timezone.utc)

        # Card 1: Due today (based on sample_card_data's default next_due_date)
        card1_uuid = uuid.uuid4()
        card1_data = {**sample_card_data, "uuid": card1_uuid, "front": "Card 1 Due Today", "added_at": now_utc - timedelta(days=2)}
        card1 = Card(**card1_data)
        in_memory_db.upsert_cards_batch([card1])
        # Add a review to make the card due tomorrow
        review1 = Review(
            card_uuid=card1.uuid,
            ts=now_utc - timedelta(days=10),
            rating=3,
            stab_before=1.0, stab_after=2.5, diff=6.0, 
            next_due=today - timedelta(days=1),
            elapsed_days_at_review=0, scheduled_days_interval=1, review_type="learn"
        )
        in_memory_db.add_review(review1)

        # Card 2: Due in the future
        card2_uuid = uuid.uuid4()
        card2_data = {**sample_card_data, "uuid": card2_uuid, "front": "Card 2 Due Future", "added_at": now_utc - timedelta(days=10)}
        card2 = Card(**card2_data)
        in_memory_db.upsert_cards_batch([card2])
        
        # Add a review to make card2 due in the future
        review_for_card2 = Review(
            card_uuid=card2_uuid,
            ts=now_utc - timedelta(days=5),
            rating=2, stab_after=5.0, diff=5.0, # Example FSRS values
            next_due=today + timedelta(days=3), # Explicitly due in 3 days
            elapsed_days_at_review=0, scheduled_days_interval=8, review_type="review"
        )
        in_memory_db.add_review(review_for_card2) # This will also update card2's next_due_date in DB

        # 2. Initialize ReviewSessionManager with the real DB and a real scheduler
        scheduler = FSRS_Scheduler()
        manager = ReviewSessionManager(db=in_memory_db, scheduler=scheduler)

        # 3. Start session & verify due counts
        # get_due_cards_count uses `next_due_date <= on_date`
        assert manager.get_due_card_count() == 1, f"Expected 1 due card, found {manager.get_due_card_count()}. Card1 due: {card1.next_due_date}, Card2 due: {in_memory_db.get_card_by_uuid(card2_uuid).next_due_date}"
        
        manager.start_session(limit=10)
        assert len(manager.review_queue) == 1
        assert manager.review_queue[0].uuid == card1_uuid

        # 4. Get next card
        next_card_to_review = manager.get_next_card()
        assert next_card_to_review is not None
        assert next_card_to_review.uuid == card1_uuid

        # 5. Submit review for the card
        rating = 2 # Good
        review_ts = datetime.now(timezone.utc) # Use current time for review
        resp_ms = 4000
        submitted_review = manager.submit_review(card_uuid=card1_uuid, rating=rating, review_ts=review_ts, resp_ms=resp_ms)
        
        assert submitted_review is not None
        assert submitted_review.card_uuid == card1_uuid
        assert submitted_review.rating == rating
        assert submitted_review.review_type == "review" # New/Lapsed cards rated Good graduate
        assert submitted_review.next_due > today # Should be scheduled for the future

        # 6. Verify DB state
        card1_reviews = in_memory_db.get_reviews_for_card(card1_uuid)
        assert len(card1_reviews) == 2
        assert card1_reviews[0].review_id == submitted_review.review_id # DB generates review_id
        assert card1_reviews[0].rating == rating

        updated_card1_from_db = in_memory_db.get_card_by_uuid(card1_uuid)
        assert updated_card1_from_db is not None
        assert updated_card1_from_db.last_review_id == submitted_review.review_id
        assert updated_card1_from_db.next_due_date == submitted_review.next_due

        # 7. Verify manager state after review
        assert card1_uuid not in [c.uuid for c in manager.review_queue] # Card removed from active queue
        assert manager.get_next_card() is None # Queue should be empty now

        # 8. Verify due card count again (card1 should no longer be due today)
        assert manager.get_due_card_count() == 0

# More test classes and methods will follow for other functionalities...

