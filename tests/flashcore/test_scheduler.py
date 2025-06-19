import pytest
import datetime
from uuid import uuid4, UUID

from cultivation.scripts.flashcore.scheduler import FSRS_Scheduler, FSRSSchedulerConfig
from cultivation.scripts.flashcore.card import Review
from cultivation.scripts.flashcore.config import DEFAULT_PARAMETERS, DEFAULT_DESIRED_RETENTION

# Helper to create datetime objects easily
UTC = datetime.timezone.utc

@pytest.fixture
def scheduler() -> FSRS_Scheduler:
    """
    Pytest fixture that returns an FSRS_Scheduler instance configured with default parameters and desired retention.
    """
    config = FSRSSchedulerConfig(
        parameters=tuple(DEFAULT_PARAMETERS),
        desired_retention=DEFAULT_DESIRED_RETENTION,
        # Assuming other FSRSSchedulerConfig fields have defaults or are not needed for these tests
    )
    return FSRS_Scheduler(config=config)

@pytest.fixture
def sample_card_uuid() -> UUID:
    """
    Generate and return a random UUID for use as a sample card identifier in tests.
    
    Returns:
        UUID: A randomly generated UUID.
    """
    return uuid4()


def test_first_review_new_card(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """
    Test that the scheduler computes correct intervals and state for the first review of a new card.
    
    Verifies that a "Good" rating produces a valid next review state with positive interval, and that an "Again" rating results in a shorter interval and earlier due date than "Good".
    """
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
    """
    Verify that compute_next_state raises ValueError for ratings outside the valid range (0-3).
    """
    history: list[Review] = []
    review_ts = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="Invalid rating: 4. Must be 0-3."):
        scheduler.compute_next_state(history, 4, review_ts)

    with pytest.raises(ValueError, match="Invalid rating: -1. Must be 0-3."):
        scheduler.compute_next_state(history, -1, review_ts)


def test_rating_impact_on_interval(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """
    Test that lower review ratings produce shorter intervals and earlier next review dates.
    
    Simulates a card with one prior "Good" review, then verifies that subsequent ratings of "Again", "Hard", "Good", and "Easy" on the scheduled review date result in monotonically increasing scheduled intervals and next review due dates.
    """
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
    """
    Test that repeated successful ("Good") reviews increase a card's stability, scheduled interval, and next review due date.
    
    Simulates three consecutive "Good" reviews on their respective due dates and asserts that stability, scheduled_days, and next_review_due increase with each review.
    """
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

def test_review_lapsed_card(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """
    Test that reviewing a card after its due date (lapsed review) results in greater increases in stability, scheduled interval, and next review date compared to an on-time review.
    """
    history: list[Review] = []
    review_ts_base = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    current_card_uuid = sample_card_uuid

    # Review 1: New card, rated Good (2), scheduled for a future date.
    rating1 = 2
    result1 = scheduler.compute_next_state(history, rating1, review_ts_base)
    next_due1 = result1["next_review_due"]

    history.append(Review(
        card_uuid=current_card_uuid,
        ts=review_ts_base,
        rating=rating1,
        stab_before=0,
        stab_after=result1["stability"],
        diff=result1["difficulty"],
        next_due=next_due1,
        elapsed_days_at_review=0,
        scheduled_days_interval=result1["scheduled_days"]
    ))

    # Scenario 1 (Control): Review on the exact due date.
    review_ts_on_time = datetime.datetime.combine(next_due1, datetime.time(10, 0, 0), tzinfo=UTC)
    result_on_time = scheduler.compute_next_state(history, 2, review_ts_on_time) # Rated Good

    # Scenario 2 (Lapsed): Review 10 days AFTER the due date.
    review_ts_lapsed = review_ts_on_time + datetime.timedelta(days=10)
    result_lapsed = scheduler.compute_next_state(history, 2, review_ts_lapsed) # Rated Good

    # FSRS theory: A successful review after a longer-than-scheduled delay indicates
    # stronger memory retention, thus stability should increase more.
    assert result_lapsed["stability"] > result_on_time["stability"]
    assert result_lapsed["scheduled_days"] > result_on_time["scheduled_days"]
    assert result_lapsed["next_review_due"] > result_on_time["next_review_due"]


def test_review_early_card(scheduler: FSRS_Scheduler, sample_card_uuid: UUID):
    """
    Test that reviewing a card before its due date results in smaller stability and interval increases compared to an on-time review.
    
    Simulates a card's first review, then compares the scheduler's output for a second review performed on the due date versus two days early. Asserts that early reviews yield lower stability and shorter scheduled intervals.
    """
    history: list[Review] = []
    review_ts_base = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    current_card_uuid = sample_card_uuid

    # Review 1: New card, rated Good (2), scheduled for a future date.
    rating1 = 2
    result1 = scheduler.compute_next_state(history, rating1, review_ts_base)
    next_due1 = result1["next_review_due"]

    history.append(Review(
        card_uuid=current_card_uuid,
        ts=review_ts_base,
        rating=rating1,
        stab_before=0,
        stab_after=result1["stability"],
        diff=result1["difficulty"],
        next_due=next_due1,
        elapsed_days_at_review=0,
        scheduled_days_interval=result1["scheduled_days"]
    ))

    # Scenario 1 (Control): Review on the exact due date.
    review_ts_on_time = datetime.datetime.combine(next_due1, datetime.time(10, 0, 0), tzinfo=UTC)
    result_on_time = scheduler.compute_next_state(history, 2, review_ts_on_time) # Rated Good

    # Scenario 2 (Early): Review 2 days BEFORE the due date.
    # Ensure we don't go back in time if the interval is very short.
    review_ts_early = review_ts_on_time - datetime.timedelta(days=min(2, result1["scheduled_days"] - 1))
    if review_ts_early <= review_ts_base:
        pytest.skip("Scheduled interval is too short to test an early review.")

    result_early = scheduler.compute_next_state(history, 2, review_ts_early) # Rated Good

    # FSRS theory: A successful early review provides less information about memory
    # strength, so the stability increase should be smaller.
    assert result_early["stability"] < result_on_time["stability"]
    assert result_early["scheduled_days"] < result_on_time["scheduled_days"]


def test_mature_card_lapse(sample_card_uuid: UUID):
    """
    Test that forgetting a mature card (rating 'Again') resets stability, increases difficulty, and sets the scheduled interval to the first relearning step.
    
    Simulates a card reaching maturity through multiple successful reviews, then verifies that a lapse review causes the expected changes in scheduling state.
    """
    # Use a dedicated scheduler with explicit relearning steps to isolate the test
    config = FSRSSchedulerConfig(
        relearning_steps=(datetime.timedelta(days=1),)
    )
    scheduler = FSRS_Scheduler(config=config)
    history: list[Review] = []
    review_ts = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    current_card_uuid = sample_card_uuid

    # Build up a mature card with high stability through several 'Good' reviews
    last_result = scheduler.compute_next_state(history, 2, review_ts)
    history.append(Review(
        card_uuid=current_card_uuid, ts=review_ts, rating=2, stab_before=0,
        stab_after=last_result["stability"], diff=last_result["difficulty"],
        next_due=last_result["next_review_due"], elapsed_days_at_review=0,
        scheduled_days_interval=last_result["scheduled_days"]
    ))

    for _ in range(4): # 4 more successful reviews
        review_ts = datetime.datetime.combine(last_result["next_review_due"], datetime.time(10,0,0), tzinfo=UTC)
        
        stab_before = last_result["stability"]
        
        last_result = scheduler.compute_next_state(history, 2, review_ts)
        history.append(Review(
            card_uuid=current_card_uuid, ts=review_ts, rating=2, stab_before=stab_before,
            stab_after=last_result["stability"], diff=last_result["difficulty"],
            next_due=last_result["next_review_due"],
            elapsed_days_at_review=(review_ts.date() - history[-1].ts.date()).days,
            scheduled_days_interval=last_result["scheduled_days"]
        ))
    
    mature_stability = last_result["stability"]
    mature_difficulty = last_result["difficulty"]
    assert mature_stability > 20 # Arbitrary check for maturity

    # Now, the user forgets the card (rates 'Again')
    lapse_review_ts = datetime.datetime.combine(last_result["next_review_due"], datetime.time(10,0,0), tzinfo=UTC)
    lapse_result = scheduler.compute_next_state(history, 0, lapse_review_ts) # Rating: Again

    # After a lapse, stability should be significantly reduced.
    # Difficulty should increase.
    assert lapse_result["stability"] < mature_stability
    assert lapse_result["difficulty"] > mature_difficulty
    # The new interval should be short, typical for relearning.
    # After a lapse, the card enters relearning. The first interval is short.
    # With default relearning steps, this should be 1 day.
    assert lapse_result["scheduled_days"] == 1


def test_config_impact_on_scheduling():
    """
    Test that changing the scheduler's desired_retention parameter affects the scheduled review interval.
    
    Creates two FSRS_Scheduler instances with different desired_retention values and verifies that a higher desired_retention results in a shorter scheduled_days interval for the same review history and rating.
    """
    # Initial review to create some history, as retention has no effect on the first review's stability
    base_scheduler = FSRS_Scheduler()
    initial_history: list[Review] = []
    review_ts1 = datetime.datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    initial_result = base_scheduler.compute_next_state(initial_history, 2, review_ts1) # Good

    history: list[Review] = [
        Review(
            card_uuid=UUID("a3f4b1d0-c2e8-4a6a-8f9a-3b1c5d7a9e0f"),
            ts=review_ts1,
            rating=2,
            stab_before=0,
            stab_after=initial_result["stability"],
            diff=initial_result["difficulty"],
            next_due=initial_result["next_review_due"],
            elapsed_days_at_review=0,
            scheduled_days_interval=initial_result["scheduled_days"]
        )
    ]
    review_ts2 = datetime.datetime.combine(initial_result["next_review_due"], datetime.time(10,0,0), tzinfo=UTC)

    # Scheduler 1: Default retention (e.g., 0.9)
    config1 = FSRSSchedulerConfig(
        parameters=tuple(DEFAULT_PARAMETERS),
        desired_retention=0.9,
    )
    scheduler1 = FSRS_Scheduler(config=config1)
    result1 = scheduler1.compute_next_state(history, 2, review_ts2) # Good

    # Scheduler 2: Higher retention (e.g., 0.95) - should result in shorter intervals
    config2 = FSRSSchedulerConfig(
        parameters=tuple(DEFAULT_PARAMETERS),
        desired_retention=0.95,
    )
    scheduler2 = FSRS_Scheduler(config=config2)
    result2 = scheduler2.compute_next_state(history, 2, review_ts2) # Good

    # Higher desired retention means we need to review more often to achieve it.
    # Higher desired retention means we need to review more often to achieve it.
    # The stability calculation itself is not directly affected by desired_retention,
    # but the resulting interval is.
    assert result2["scheduled_days"] < result1["scheduled_days"]
