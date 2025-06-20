"""
_summary_
"""

from __future__ import annotations

import uuid
import re
from enum import IntEnum
from uuid import UUID
from datetime import datetime, date, timezone
from typing import List, Optional, Set
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Regex for Kebab-case validation (e.g., "my-cool-tag", "learning-python-3")
KEBAB_CASE_REGEX_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class CardState(IntEnum):
    """
    Represents the FSRS-defined state of a card's memory trace.
    """
    New = 0
    Learning = 1
    Review = 2
    Relearning = 3


class Card(BaseModel):
    """
    Represents a single flashcard after parsing and processing from YAML.
    This is the canonical internal representation of a card's content and metadata.

    Media asset paths are always relative to 'cultivation/outputs/flashcards/yaml/assets/'.
    """
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    uuid: UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique UUIDv4 identifier for the card. Auto-generated if not provided in YAML 'id'."
    )
    last_review_id: Optional[int] = Field(default=None, description="The ID of the last review record associated with this card.")
    next_due_date: Optional[date] = Field(default=None, description="The next date the card is scheduled for review.")
    state: CardState = Field(default=CardState.New, description="The current FSRS state of the card.")

    deck_name: str = Field(
        ...,
        min_length=1,
        description="Fully qualified name of the deck the card belongs to (e.g., 'Backend::Auth'). Derived from YAML 'deck'."
    )
    front: str = Field(
        ...,
        max_length=1024,
        description="The question or prompt text. Supports Markdown and KaTeX. Maps from YAML 'q'."
    )
    back: str = Field(
        ...,
        max_length=1024,
        description="The answer text. Supports Markdown and KaTeX. Maps from YAML 'a'."
    )
    tags: Set[str] = Field(
        default_factory=set,
        description="Set of unique, kebab-case tags. Result of merging deck-level global tags and card-specific tags from YAML."
    )
    added_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp indicating when the card was first added/ingested into the system. This timestamp persists even if the card content is updated later."
    )
    modified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp indicating when the card was last modified. It is updated upon any change to the card's content."
    )
    origin_task: Optional[str] = Field(
        default=None,
        description="Optional reference to an originating task ID (e.g., from Task Master)."
    )
    media: List[Path] = Field(
        default_factory=list,
        description="Optional list of paths to media files (images, audio, etc.) associated with the card. Paths should be relative to a defined assets root directory (e.g., 'outputs/flashcards/assets/')."
    )
    source_yaml_file: Optional[Path] = Field(
        default=None,
        description="The path to the YAML file from which this card was loaded. Essential for traceability, debugging, and tools that might update YAML files (like 'tm-fc vet' sorting)."
    )
    internal_note: Optional[str] = Field(
        default=None,
        description="A field for internal system notes or flags about the card, not typically exposed to the user (e.g., 'needs_review_for_xss_risk_if_sanitizer_fails', 'generated_by_task_hook')."
    )

    @field_validator("tags")
    @classmethod
    def validate_tags_kebab_case(cls, tags: Set[str]) -> Set[str]:
        """Ensure each tag matches the kebab-case pattern."""
        for tag in tags:
            if not re.match(KEBAB_CASE_REGEX_PATTERN, tag):
                raise ValueError(f"Tag '{tag}' is not in kebab-case.")
        return tags

class Review(BaseModel):
    """
    Represents a single review event for a flashcard, including user feedback
    and FSRS scheduling parameters.
    """
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    review_id: Optional[int] = Field(
        default=None,
        description="The auto-incrementing primary key from the 'reviews' database table. Will be None for new Review objects before they are persisted."
    )
    card_uuid: UUID = Field(
        ...,
        description="The UUID of the card that was reviewed, linking to Card.uuid."
    )
    ts: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="The UTC timestamp when the review occurred."
    )
    rating: int = Field(
        ...,
        ge=0,
        le=3,
        description="The user's rating of their recall performance (0=Again, 1=Hard, 2=Good, 3=Easy)."
    )
    resp_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="The response time in milliseconds taken by the user to recall the answer before revealing it. Nullable if not captured."
    )
    stab_before: Optional[float] = Field(
        default=None,  # Handled by FSRS logic for first reviews
        description="The card's memory stability (in days) *before* this review was incorporated by FSRS. For the very first review of a card, the FSRS scheduler will use a default initial stability."
    )
    stab_after: float = Field(
        ...,
        ge=0.1,  # Stability should generally be positive and non-zero after review
        description="The card's new memory stability (in days) *after* this review and FSRS calculation."
    )
    diff: float = Field(
        ...,
        description="The card's new difficulty rating *after* this review and FSRS calculation."
    )
    next_due: date = Field(
        ...,
        description="The date when this card is next scheduled for review, calculated by FSRS."
    )
    elapsed_days_at_review: int = Field(
        ...,
        ge=0,
        description="The number of days that had actually elapsed between the *previous* review's 'next_due' date (or card's 'added_at' for a new card) and the current review's 'ts'. This is a crucial input for FSRS."
    )
    scheduled_days_interval: int = Field(
        ...,
        ge=1,  # The interval calculated by FSRS must be at least 1 day.
        description="The interval in days (e.g., 'nxt' from fsrs_once) that FSRS calculated for this review. next_due would be 'ts.date() + timedelta(days=scheduled_days_interval)'."
    )
    review_type: Optional[str] = Field(
        default="review",
        description="Type of review, e.g., 'learn', 'review', 'relearn', 'manual'. Useful for advanced FSRS variants or analytics."
    )

    @field_validator("review_type")
    @classmethod
    def check_review_type_is_allowed(cls, v: str | None) -> str | None:
        """Ensures review_type is one of the predefined allowed values or None."""
        ALLOWED_REVIEW_TYPES = {"learn", "review", "relearn", "manual"}
        if v is not None and v not in ALLOWED_REVIEW_TYPES:
            raise ValueError(
                f"Invalid review_type: '{v}'. Allowed types are: {ALLOWED_REVIEW_TYPES} or None."
            )
        return v
