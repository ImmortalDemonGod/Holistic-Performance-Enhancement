"""
Houses all validation-related functions for the YAML processing pipeline.
"""
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import bleach
from pydantic import ValidationError

from .yaml_models import (
    DEFAULT_ALLOWED_HTML_ATTRIBUTES,
    DEFAULT_ALLOWED_HTML_TAGS,
    DEFAULT_CSS_SANITIZER,
    DEFAULT_SECRET_PATTERNS,
    _CardProcessingContext,
    _RawYAMLCardEntry,
    YAMLProcessingError,
)


def validate_card_uuid(
    raw_card: _RawYAMLCardEntry, context: _CardProcessingContext
) -> Union[uuid.UUID, YAMLProcessingError]:
    """Validates the raw UUID, returning a UUID object or a YAMLProcessingError."""
    if raw_card.id is None:
        return uuid.uuid4()  # Assign a new UUID if none is provided
    try:
        return uuid.UUID(raw_card.id)
    except ValueError:
        return YAMLProcessingError(
            file_path=context.source_file_path,
            card_index=context.card_index,
            card_question_snippet=context.card_q_preview,
            field_name="id",
            message=f"Invalid UUID format for 'id': '{raw_card.id}'.",
        )


def sanitize_card_text(raw_card: _RawYAMLCardEntry) -> Tuple[str, str]:
    """Normalizes and sanitizes card front and back text."""
    front_normalized = raw_card.q.strip()
    back_normalized = raw_card.a.strip()
    front_sanitized = bleach.clean(
        front_normalized,
        tags=DEFAULT_ALLOWED_HTML_TAGS,
        attributes=DEFAULT_ALLOWED_HTML_ATTRIBUTES,
        css_sanitizer=DEFAULT_CSS_SANITIZER,
        strip=True,
    )
    back_sanitized = bleach.clean(
        back_normalized,
        tags=DEFAULT_ALLOWED_HTML_TAGS,
        attributes=DEFAULT_ALLOWED_HTML_ATTRIBUTES,
        css_sanitizer=DEFAULT_CSS_SANITIZER,
        strip=True,
    )
    return front_sanitized, back_sanitized


def check_for_secrets(
    front: str, back: str, context: _CardProcessingContext
) -> Optional[YAMLProcessingError]:
    """Scans text for secrets, returning an error if a secret is found."""
    if context.skip_secrets_detection:
        return None
    for pattern in DEFAULT_SECRET_PATTERNS:
        if pattern.search(back):
            return YAMLProcessingError(
                file_path=context.source_file_path,
                card_index=context.card_index,
                card_question_snippet=context.card_q_preview,
                field_name="a",
                message=f"Potential secret detected in card answer. Matched pattern: '{pattern.pattern[:50]}...'.",
            )
        if pattern.search(front):
            return YAMLProcessingError(
                file_path=context.source_file_path,
                card_index=context.card_index,
                card_question_snippet=context.card_q_preview,
                field_name="q",
                message=f"Potential secret detected in card question. Matched pattern: '{pattern.pattern[:50]}...'.",
            )
    return None


def compile_card_tags(deck_tags: Set[str], card_tags: Optional[List[str]]) -> Set[str]:
    """Combines deck-level and card-level tags into a single set."""
    final_tags = deck_tags.copy()
    if card_tags:
        final_tags.update(tag.strip().lower() for tag in card_tags)
    return final_tags


def validate_media_paths(
    raw_media: List[str], context: _CardProcessingContext
) -> Union[List[Path], YAMLProcessingError]:
    """Validates all media paths for a card, returning a list of Paths or an error."""
    processed_media_paths = []
    for media_item_str in raw_media:
        result = validate_single_media_path(media_item_str, context)
        if isinstance(result, YAMLProcessingError):
            return result
        processed_media_paths.append(result)
    return processed_media_paths


def validate_single_media_path(
    media_item_str: str, context: _CardProcessingContext
) -> Union[Path, YAMLProcessingError]:
    """Validates a single media path, returning a Path object or a YAMLProcessingError."""
    media_path = Path(media_item_str.strip())
    if media_path.is_absolute():
        return YAMLProcessingError(
            file_path=context.source_file_path,
            card_index=context.card_index,
            card_question_snippet=context.card_q_preview,
            field_name="media",
            message=f"Media path must be relative: '{media_path}'.",
        )

    if not context.skip_media_validation:
        try:
            full_media_path = (context.assets_root_directory / media_path).resolve(
                strict=False
            )
            abs_assets_root = context.assets_root_directory.resolve(strict=True)
            if not str(full_media_path).startswith(str(abs_assets_root)):
                return YAMLProcessingError(
                    file_path=context.source_file_path,
                    card_index=context.card_index,
                    card_question_snippet=context.card_q_preview,
                    field_name="media",
                    message=f"Media path '{media_path}' resolves outside the assets root directory.",
                )
            if not full_media_path.exists():
                return YAMLProcessingError(
                    file_path=context.source_file_path,
                    card_index=context.card_index,
                    card_question_snippet=context.card_q_preview,
                    field_name="media",
                    message=f"Media file not found at expected path: '{full_media_path}'.",
                )
        except Exception as e:
            return YAMLProcessingError(
                file_path=context.source_file_path,
                card_index=context.card_index,
                card_question_snippet=context.card_q_preview,
                field_name="media",
                message=f"Error validating media path '{media_path}': {e}.",
            )
    return media_path


def run_card_validation_pipeline(
    raw_card: _RawYAMLCardEntry, context: _CardProcessingContext, deck_tags: Set[str]
) -> Union[Tuple[uuid.UUID, str, str, Set[str], List[Path]], YAMLProcessingError]:
    """
    Runs the validation pipeline for a raw card, returning processed data or an error.
    Orchestrates UUID validation, text sanitization, secret scanning, tag compilation,
    and media path validation.
    """
    uuid_or_error = validate_card_uuid(raw_card, context)
    if isinstance(uuid_or_error, YAMLProcessingError):
        return uuid_or_error

    front, back = sanitize_card_text(raw_card)

    secret_error = check_for_secrets(front, back, context)
    if secret_error:
        return secret_error

    final_tags = compile_card_tags(deck_tags, raw_card.tags)

    media_paths: List[Path] = []
    if raw_card.media:
        media_paths_or_error = validate_media_paths(raw_card.media, context)
        if isinstance(media_paths_or_error, YAMLProcessingError):
            return media_paths_or_error
        media_paths = media_paths_or_error

    return uuid_or_error, front, back, final_tags, media_paths


def validate_directories(
    source_directory: Path,
    assets_root_directory: Path,
    skip_media_validation: bool,
) -> Optional[YAMLProcessingError]:
    """Validates that the source and asset directories exist."""
    if not source_directory.is_dir():
        return YAMLProcessingError(
            source_directory, "Source directory does not exist or is not a directory."
        )

    if not assets_root_directory.is_dir():
        # This error is only critical if we are validating media
        if not skip_media_validation:
            return YAMLProcessingError(
                assets_root_directory,
                "Assets root directory does not exist or is not a directory.",
            )

    return None


def extract_deck_name(raw_yaml_content: Dict, file_path: Path) -> str:
    """Extracts and validates the deck name from the raw YAML content."""
    deck_value = raw_yaml_content.get("deck")
    if deck_value is None:
        raise YAMLProcessingError(file_path, "Missing 'deck' field at top level.")
    if not isinstance(deck_value, str):
        raise YAMLProcessingError(file_path, "'deck' field must be a string.")
    if not deck_value.strip():
        raise YAMLProcessingError(
            file_path, "'deck' field cannot be empty or just whitespace."
        )
    return deck_value.strip()


def extract_deck_tags(raw_yaml_content: Dict, file_path: Path) -> Set[str]:
    """Extracts and validates deck-level tags from the raw YAML content."""
    tags = raw_yaml_content.get("tags", [])
    if tags is not None and not isinstance(tags, list):
        raise YAMLProcessingError(file_path, "'tags' field must be a list if present.")
    return {t.strip().lower() for t in tags if isinstance(t, str)} if tags else set()


def extract_cards_list(raw_yaml_content: Dict, file_path: Path) -> List[Dict]:
    """Extracts and validates the list of cards from the raw YAML content."""
    if "cards" not in raw_yaml_content or not isinstance(
        raw_yaml_content["cards"], list
    ):
        raise YAMLProcessingError(
            file_path, "Missing or invalid 'cards' list at top level."
        )
    cards_list = raw_yaml_content["cards"]
    if not cards_list:
        raise YAMLProcessingError(file_path, "No cards found in 'cards' list.")
    return cards_list


def validate_deck_and_extract_metadata(
    raw_yaml_content: Dict, file_path: Path
) -> Tuple[str, Set[str], List[Dict]]:
    """Validates the deck structure and extracts metadata by calling specialized helpers."""
    deck_name = extract_deck_name(raw_yaml_content, file_path)
    deck_tags = extract_deck_tags(raw_yaml_content, file_path)
    cards_list = extract_cards_list(raw_yaml_content, file_path)
    return deck_name, deck_tags, cards_list


def validate_raw_card_structure(
    card_dict: Dict, idx: int, file_path: Path
) -> Union[_RawYAMLCardEntry, YAMLProcessingError]:
    """Validates the structure of a raw card dict using Pydantic."""
    try:
        return _RawYAMLCardEntry.model_validate(card_dict)
    except ValidationError as e:
        error_details = "; ".join(
            [f"{''.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
        )
        q_preview = card_dict.get("q", "N/A")
        return YAMLProcessingError(
            file_path=file_path,
            message=f"Card validation failed. Details: {error_details}",
            card_index=idx,
            card_question_snippet=q_preview,
        )
