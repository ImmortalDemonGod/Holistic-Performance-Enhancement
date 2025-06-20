# cultivation/scripts/flashcore/scheduler.py

"""
Defines the BaseScheduler abstract class and the FSRS_Scheduler for flashcore,
integrating py-fsrs for scheduling.
"""

import logging
from abc import ABC, abstractmethod
import datetime 
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field

from cultivation.scripts.flashcore.config import (
    DEFAULT_PARAMETERS,
    DEFAULT_DESIRED_RETENTION,
)

from fsrs import Card as FSRSCard
from fsrs import Rating as FSRSRating
from fsrs import Scheduler as PyFSRSScheduler

from .card import Review

logger = logging.getLogger(__name__)

class BaseScheduler(ABC):
    """
    Abstract base class for all schedulers in flashcore.
    """

    @abstractmethod
    def compute_next_state(
        self, history: List[Review], new_rating: int, review_ts: datetime.datetime
    ) -> Dict[str, Any]:
        """
        Computes the next state of a card based on its review history and a new rating.

        Args:
            history: A list of past Review objects for the card, sorted chronologically.
            new_rating: The rating given for the current review (0=Again, 1=Hard, 2=Good, 3=Easy).
            review_ts: The UTC timestamp of the current review.

        Returns:
            A dictionary containing the new state:
            {
                "stability": float,
                "difficulty": float,
                "next_review_date": datetime.date, 
                "scheduled_days": int
            }
        
        Raises:
            ValueError: If the new_rating is invalid.
        """
        pass




class FSRSSchedulerConfig(BaseModel):
    """Configuration for the FSRS Scheduler."""

    parameters: Tuple[float, ...] = Field(default_factory=lambda: tuple(DEFAULT_PARAMETERS))
    desired_retention: float = DEFAULT_DESIRED_RETENTION
    learning_steps: Tuple[datetime.timedelta, ...] = ()
    relearning_steps: Tuple[datetime.timedelta, ...] = Field(
        default_factory=lambda: (datetime.timedelta(days=1),)
    )
    max_interval: int = 36500


class FSRS_Scheduler(BaseScheduler):
    """
    FSRS (Free Spaced Repetition Scheduler) implementation for flashcore.
    This scheduler uses the py-fsrs library to determine card states and next review dates.
    """

    RATING_MAP = {
        0: FSRSRating.Again,
        1: FSRSRating.Hard,
        2: FSRSRating.Good,
        3: FSRSRating.Easy,
    }

    def __init__(self, config: Optional[FSRSSchedulerConfig] = None):
        if config is None:
            config = FSRSSchedulerConfig()
        self.config = config

        # The py-fsrs library expects steps as lists of integers.
        learning_steps = list(self.config.learning_steps)
        relearning_steps = list(self.config.relearning_steps)

        scheduler_args = {
            "parameters": list(self.config.parameters),
            "desired_retention": self.config.desired_retention,
            "learning_steps": learning_steps,
            "relearning_steps": relearning_steps,
            "maximum_interval": self.config.max_interval
        }
        
        self.fsrs_scheduler = PyFSRSScheduler(**scheduler_args)

    def _ensure_utc(self, ts: datetime.datetime) -> datetime.datetime:
        """Ensures the given datetime is UTC. Assumes UTC if naive."""
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            return ts.replace(tzinfo=datetime.timezone.utc)
        if ts.tzinfo != datetime.timezone.utc:
            return ts.astimezone(datetime.timezone.utc)
        return ts

    def _map_flashcore_rating_to_fsrs(self, flashcore_rating: int) -> FSRSRating:
        """Maps flashcore rating (0-3) to FSRSRating and validates."""
        if not (0 <= flashcore_rating <= 3):
            raise ValueError(f"Invalid rating: {flashcore_rating}. Must be 0-3.")
        return self.RATING_MAP[flashcore_rating]

    def compute_next_state(
        self, history: List[Review], new_rating: int, review_ts: datetime.datetime
    ) -> Dict[str, Any]:
        """
        Computes the next state of a card by replaying its entire history.
        """
        # Start with a fresh card object.
        fsrs_card = FSRSCard()

        # Replay the entire review history to build the correct current state.
        for review in history:
            rating = self._map_flashcore_rating_to_fsrs(review.rating)
            ts = self._ensure_utc(review.ts)
            fsrs_card, _ = self.fsrs_scheduler.review_card(fsrs_card, rating, review_datetime=ts)

        # Now, apply the new review to the final state.
        current_fsrs_rating = self._map_flashcore_rating_to_fsrs(new_rating)
        utc_review_ts = self._ensure_utc(review_ts)
        updated_fsrs_card, _ = self.fsrs_scheduler.review_card(
            fsrs_card, current_fsrs_rating, review_datetime=utc_review_ts
        )

        # Calculate scheduled days based on the new due date.
        scheduled_days = (updated_fsrs_card.due.date() - utc_review_ts.date()).days

        return {
            "stability": round(updated_fsrs_card.stability, 4),
            "difficulty": round(updated_fsrs_card.difficulty, 4),
            "next_review_due": updated_fsrs_card.due.date(),
            "scheduled_days": scheduled_days,
            "state": updated_fsrs_card.state.name,
        }
