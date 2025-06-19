import pytest
import datetime
from uuid import uuid4, UUID

from cultivation.scripts.flashcore.scheduler import FSRS_Scheduler, FSRSSchedulerConfig
from cultivation.scripts.flashcore.card import Review
from cultivation.scripts.flashcore.config import DEFAULT_FSRS_PARAMETERS, DEFAULT_DESIRED_RETENTION

# Helper to create datetime objects easily
UTC = datetime.timezone.utc

@pytest.fixture
def scheduler() -> FSRS_Scheduler:
    """Provides an FSRS_Scheduler instance with default parameters."""
    config = FSRSSchedulerConfig(
        parameters=tuple(DEFAULT_FSRS_PARAMETERS), # Ensure it's a tuple as per Pydantic model
        desired_retention=DEFAULT_DESIRED_RETENTION,
        # Assuming other FSRSSchedulerConfig fields have defaults or are not needed for these tests
    )
    return FSRS_Scheduler(config=config)

@pytest.fixture
def sample_card_uuid() -> UUID:
    return uuid4()


def test_first_review_new_card(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """Test scheduling for the first review of a new card."""
    history: list[Review] = []
    review_ts = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    
    # Rating: Good (2)
    rating = 2 
    result_good = scheduler.compute_next_state(history, rating, review_ts)

    assert isinstance(result_good, dict)
    assert "stability" in result_good
    assert "difficulty" in result_good
    assert "next_review_due" in result_good
    assert "scheduled_days" in result_good

    assert isinstance(result_good["stability"], float)
    assert isinstance(result_good["difficulty"], float)
    assert isinstance(result_good["next_review_due"], datetime.date)
    assert isinstance(result_good["scheduled_days"], int)

    assert result_good["scheduled_days"] > 0
    assert result_good["next_review_due"] > review_ts.date()

    # Rating: Again (0)
    # For a new card, 'Again' should result in a shorter interval than 'Good'
    rating_again = 0
    result_again = scheduler.compute_next_state(history, rating_again, review_ts)
    
    assert result_again["scheduled_days"] < result_good["scheduled_days"]
    assert result_again["next_review_due"] < result_good["next_review_due"]


def test_invalid_rating_input(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """Test that invalid rating inputs raise ValueError."""
    history: list[Review] = []
    review_ts = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="Invalid rating: 4. Must be between 0 and 3."):
        scheduler.compute_next_state(history, 4, review_ts)

    with pytest.raises(ValueError, match="Invalid rating: -1. Must be between 0 and 3."):
        scheduler.compute_next_state(history, -1, review_ts)


def test_rating_impact_on_interval(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """Test that lower ratings generally result in shorter intervals."""
    # Simulate a card that has been reviewed once and was 'Good'
    # These values are illustrative; actual FSRS outputs would be used in a real scenario
    # For this test, we mainly care about the *relative* intervals from the *next* review.
    review1_ts = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    
    # First, get a baseline state after one 'Good' review
    initial_good_result = scheduler.compute_next_state([], 2, review1_ts)

    history: list[Review] = [
        Review(
            card_uuid=sample_card_uuid,
            ts=review1_ts,
            rating=2,
            stab_before=0, # For first review, stab_before is not critical for this test setup
            stab_after=initial_good_result["stability"],
            diff=initial_good_result["difficulty"],
            next_due=initial_good_result["next_review_due"],
            elapsed_days_at_review=0, # First review
            scheduled_days_interval=initial_good_result["scheduled_days"]
        )
    ]

    # Next review occurs exactly on the scheduled date
    review2_ts = datetime.datetime.combine(initial_good_result["next_review_due"], datetime.time(10,0,0), tzinfo=UTC)

    result_again = scheduler.compute_next_state(history, 0, review2_ts) # Rating: Again
    result_hard = scheduler.compute_next_state(history, 1, review2_ts)  # Rating: Hard
    result_good = scheduler.compute_next_state(history, 2, review2_ts)  # Rating: Good
    result_easy = scheduler.compute_next_state(history, 3, review2_ts)  # Rating: Easy

    assert result_again["scheduled_days"] < result_hard["scheduled_days"]
    assert result_hard["scheduled_days"] < result_good["scheduled_days"]
    assert result_good["scheduled_days"] < result_easy["scheduled_days"]

    assert result_again["next_review_due"] < result_hard["next_review_due"]
    assert result_hard["next_review_due"] < result_good["next_review_due"]
    assert result_good["next_review_due"] < result_easy["next_review_due"]



def test_multiple_reviews_stability_increase(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """Test that stability and scheduled_days generally increase with multiple successful (Good) reviews."""
    history: list[Review] = []
    review_ts_base = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    current_card_uuid = sample_card_uuid

    # Review 1: New card, rated Good (2)
    rating1 = 2
    result1 = scheduler.compute_next_state(history, rating1, review_ts_base)
    
    assert result1["scheduled_days"] > 0
    stability1 = result1["stability"]
    scheduled_days1 = result1["scheduled_days"]
    next_due1 = result1["next_review_due"]

    history.append(Review(
        card_uuid=current_card_uuid,
        ts=review_ts_base,
        rating=rating1,
        stab_before=0, # Approx for new card
        stab_after=stability1,
        diff=result1["difficulty"],
        next_due=next_due1,
        elapsed_days_at_review=0,
        scheduled_days_interval=scheduled_days1
    ))

    # Review 2: Reviewed on its due date, rated Good (2)
    review_ts2 = datetime.datetime.combine(next_due1, datetime.time(10, 0, 0), tzinfo=UTC)
    rating2 = 2
    result2 = scheduler.compute_next_state(history, rating2, review_ts2)

    stability2 = result2["stability"]
    scheduled_days2 = result2["scheduled_days"]
    next_due2 = result2["next_review_due"]

    assert stability2 > stability1
    assert scheduled_days2 > scheduled_days1
    assert next_due2 > next_due1

    history.append(Review(
        card_uuid=current_card_uuid,
        ts=review_ts2,
        rating=rating2,
        stab_before=stability1,
        stab_after=stability2,
        diff=result2["difficulty"],
        next_due=next_due2,
        elapsed_days_at_review=(review_ts2.date() - review_ts_base.date()).days, # More accurate elapsed_days
        scheduled_days_interval=scheduled_days2
    ))

    # Review 3: Reviewed on its due date, rated Good (2)
    review_ts3 = datetime.datetime.combine(next_due2, datetime.time(10, 0, 0), tzinfo=UTC)
    rating3 = 2
    result3 = scheduler.compute_next_state(history, rating3, review_ts3)

    stability3 = result3["stability"]
    scheduled_days3 = result3["scheduled_days"]
    next_due3 = result3["next_review_due"]

    assert stability3 > stability2
    assert scheduled_days3 > scheduled_days2
    assert next_due3 > next_due2

# --- Placeholder for more tests --- 

# def test_review_lapsed_card():
#     """Test scheduling for a card reviewed after its due date."""
#     pass

# def test_multiple_reviews_stability_increase():
#     """Test that stability generally increases with multiple successful (Good/Easy) reviews."""
#     pass

# def test_difficulty_adjustment():
#     """Test how difficulty changes based on different rating sequences."""
#     pass

# def test_timezone_insensitivity_of_inputs():
#     """Ensure that providing naive datetimes (if ever allowed by mistake) or different timezones
#        are handled consistently, assuming scheduler forces UTC internally.
#        (Current FSRS_Scheduler._ensure_utc should handle this)
#     """
#     pass
