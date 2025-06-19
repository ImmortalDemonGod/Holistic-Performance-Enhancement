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

from .config import DEFAULT_FSRS_PARAMETERS, DEFAULT_DESIRED_RETENTION

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
    parameters: Tuple[float, ...] = Field(default_factory=lambda: tuple(DEFAULT_FSRS_PARAMETERS))
    desired_retention: float = DEFAULT_DESIRED_RETENTION
    learning_steps: List[datetime.timedelta] = Field(default_factory=list)  
    relearning_steps: Optional[List[datetime.timedelta]] = None
    max_interval: Optional[int] = None


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

        # Ensure parameters is a tuple if provided as a list in a custom config
        # The default_factory in FSRSSchedulerConfig already handles this for default params.
        effective_params = tuple(config.parameters) if isinstance(config.parameters, list) else config.parameters

        scheduler_args = {
            "parameters": effective_params,
            "desired_retention": config.desired_retention,
            "learning_steps": config.learning_steps, 
        }
        
        if config.relearning_steps is not None:
            scheduler_args["relearning_steps"] = config.relearning_steps
        if config.max_interval is not None:
            scheduler_args["max_interval"] = config.max_interval
        
        self.fsrs_scheduler = PyFSRSScheduler(**scheduler_args)

        # Store the configuration for potential logging/debugging
        self.config_params = effective_params
        self.config_retention = config.desired_retention
        self.config_learning_steps = config.learning_steps
        self.config_relearning_steps = config.relearning_steps
        self.config_max_interval = config.max_interval

    def _ensure_utc(self, ts: datetime.datetime) -> datetime.datetime:
        """Ensures the given datetime is UTC. Assumes UTC if naive."""
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            logger.warning(f"Timestamp {ts} is naive. Assuming UTC.")
            return ts.replace(tzinfo=datetime.timezone.utc)
        if ts.tzinfo != datetime.timezone.utc:
            logger.warning(f"Timestamp {ts} is not UTC. Converting to UTC.")
            return ts.astimezone(datetime.timezone.utc)
        return ts

    def _prepare_fsrs_card_from_history(self, history: List[Review]) -> FSRSCard:
        """Prepares an FSRSCard object from the review history."""
        fsrs_card = FSRSCard() # Initializes card in State.New, S=0, D=0
        if history:
            last_review = history[-1]
            fsrs_card.stability = last_review.stab_after
            fsrs_card.difficulty = last_review.diff
            
            # prev_due_dt: previous due date, must be datetime for FSRSCard
            prev_due_dt = datetime.datetime.combine(last_review.next_due, datetime.datetime.min.time(), tzinfo=datetime.timezone.utc)
            fsrs_card.due = prev_due_dt
            
            fsrs_card.last_review = self._ensure_utc(last_review.ts)
            
            logger.debug(
                f"Card state from history: S={fsrs_card.stability:.4f}, D={fsrs_card.difficulty:.4f}, "
                f"PrevDue={fsrs_card.due.date()}, LastReview={fsrs_card.last_review.date() if fsrs_card.last_review else 'None'}"
            )
        else:
            logger.debug("New card (no history). Initializing with default S=0, D=0.")
        return fsrs_card

    def _map_flashcore_rating_to_fsrs(self, flashcore_rating: int) -> FSRSRating:
        """Maps flashcore rating (0-3) to FSRSRating and validates."""
        if not (0 <= flashcore_rating <= 3):
            logger.error(f"Invalid rating received: {flashcore_rating}")
            raise ValueError(f"Invalid rating: {flashcore_rating}. Must be between 0 and 3.")
        
        fsrs_rating = self.RATING_MAP.get(flashcore_rating)
        # This check is technically redundant due to the initial validation, but good for safety.
        if fsrs_rating is None: # Should not happen if RATING_MAP is correct and validation passes
            raise ValueError(f"Internal error: Rating {flashcore_rating} could not be mapped.")
        return fsrs_rating

    def compute_next_state(
        self, history: List[Review], new_rating: int, review_ts: datetime.datetime
    ) -> Dict[str, Any]:
        
        current_fsrs_rating = self._map_flashcore_rating_to_fsrs(new_rating)
        utc_review_ts = self._ensure_utc(review_ts)
        fsrs_card = self._prepare_fsrs_card_from_history(history)

        logger.debug(f"Reviewing card at {utc_review_ts.isoformat()} with flashcore_rating={new_rating} (py-fsrs.Rating.{current_fsrs_rating.name})")

        updated_fsrs_card, review_log_entry = self.fsrs_scheduler.review_card(fsrs_card, current_fsrs_rating, review_datetime=utc_review_ts)

        logger.info(
            f"Card state after review: S={updated_fsrs_card.stability:.4f}, "
            f"D={updated_fsrs_card.difficulty:.4f}, NewDueFull={updated_fsrs_card.due}, "
            f"ReviewTSFull={utc_review_ts}, NewDueDateOnly={updated_fsrs_card.due.date()}, "
            f"State={updated_fsrs_card.state.name}"
        )
        logger.debug(f"FSRS Learning Steps: {self.fsrs_scheduler.learning_steps}")

        return {
            "stability": round(updated_fsrs_card.stability, 4),
            "difficulty": round(updated_fsrs_card.difficulty, 4),
            "next_review_due": updated_fsrs_card.due.date(),
            "scheduled_days": (updated_fsrs_card.due.date() - utc_review_ts.date()).days,
        }
