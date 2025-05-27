"""
cultivation/scripts/flashcore/yaml_processor.py

Handles loading, parsing, validation, and transformation of flashcard definitions
from source YAML files into canonical `flashcore.card.Card` Pydantic model instances.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple, Union
import re
import logging

import yaml       # PyYAML for YAML parsing
import bleach     # For HTML content sanitization
from pydantic import BaseModel as PydanticBaseModel, Field, validator, constr, ValidationError
from dataclasses import dataclass

# Local project imports
try:
    from cultivation.scripts.flashcore.card import Card, KEBAB_CASE_REGEX_PATTERN
except ImportError:
    from card import Card, KEBAB_CASE_REGEX_PATTERN

# --- Configuration Constants ---
RAW_KEBAB_CASE_PATTERN = KEBAB_CASE_REGEX_PATTERN

DEFAULT_ALLOWED_HTML_TAGS = [
    "p", "br", "strong", "em", "b", "i", "u", "s", "strike", "del", "sub", "sup",
    "ul", "ol", "li", "dl", "dt", "dd",
    "blockquote", "pre", "code", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption",
    "img", "a", "figure", "figcaption",
    "span", "math", "semantics", "mrow", "mi", "mo", "mn", "ms",
    "mtable", "mtr", "mtd", "msup", "msub", "msubsup",
    "mfrac", "msqrt", "mroot", "mstyle", "merror", "mpadded",
    "mphantom", "mfenced", "menclose", "annotation"
]
DEFAULT_ALLOWED_HTML_ATTRIBUTES = {
    "*": ["class", "id", "style"],
    "a": ["href", "title", "target", "rel"],
    "img": ["src", "alt", "title", "width", "height", "style"],
    "table": ["summary", "align", "border", "cellpadding", "cellspacing", "width"],
    "td": ["colspan", "rowspan", "align", "valign", "width", "height"],
    "th": ["colspan", "rowspan", "align", "valign", "scope", "width", "height"],
    "span": ["style", "class", "aria-hidden"],
    "math": ["display", "xmlns"],
    "annotation": ["encoding"],
}
DEFAULT_SECRET_PATTERNS = [
    re.compile(r"""
        (?:key|token|secret|password|passwd|pwd|auth|credential|cred|api_key|apikey|access_key|secret_key)
        \s*[:=]\s*
        (['"]?)
        (?!\s*ENC\[GPG\])(?!\s*ENC\[AES256\])(?!\s*<placeholder>)(?!\s*<\w+>)(?!\s*\{\{\s*\w+\s*\}\})
        ([A-Za-z0-9_/\-\.+]{20,})
        \1
    """, re.IGNORECASE | re.VERBOSE),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|PGP|EC|DSA) PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"(?:(?:sk|pk)_(?:live|test)_|rk_live_)[0-9a-zA-Z]{20,}", re.IGNORECASE),
    re.compile(r"xox[pbar]-[0-9a-zA-Z]{10,}-[0-9a-zA-Z]{10,}-[0-9a-zA-Z]{10,}-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"ghp_[0-9a-zA-Z]{36}", re.IGNORECASE),
]

logger = logging.getLogger(__name__)

# --- Internal Pydantic Models for Raw YAML Validation ---
class _RawYAMLCardEntry(PydanticBaseModel):
    id: Optional[str] = Field(default=None)
    q: str = Field(..., min_length=1)
    a: str = Field(..., min_length=1)
    tags: Optional[List[constr(pattern=RAW_KEBAB_CASE_PATTERN)]] = Field(default_factory=list) # type: ignore
    origin_task: Optional[str] = Field(default=None)
    media: Optional[List[str]] = Field(default_factory=list)

    class Config:
        extra = "forbid"

class _RawYAMLDeckFile(PydanticBaseModel):
    deck: constr(min_length=1) # type: ignore
    tags: Optional[List[constr(pattern=RAW_KEBAB_CASE_PATTERN)]] = Field(default_factory=list) # type: ignore
    cards: List[_RawYAMLCardEntry] = Field(..., min_items=1)

    class Config:
        extra = "forbid"

# --- Custom Error Reporting Dataclass ---
@dataclass
class YAMLProcessingError:
    file_path: Path
    message: str
    card_index: Optional[int] = None
    card_question_snippet: Optional[str] = None
    field_name: Optional[str] = None
    yaml_path_segment: Optional[str] = None # e.g., "cards[2].q"

    def __str__(self) -> str:
        context_parts = [f"File: {self.file_path.name}"]
        if self.card_index is not None:
            context_parts.append(f"Card Index: {self.card_index}")
        if self.card_question_snippet:
            snippet = (self.card_question_snippet[:47] + '...') if len(self.card_question_snippet) > 50 else self.card_question_snippet
            context_parts.append(f"Q: '{snippet}'")
        if self.field_name:
            context_parts.append(f"Field: '{self.field_name}'")
        if self.yaml_path_segment:
            context_parts.append(f"YAML Path: '{self.yaml_path_segment}'")
        return f"{' | '.join(context_parts)} | Error: {self.message}"

# --- Core Logic ---

def _transform_raw_card_to_model(
    raw_card_model: _RawYAMLCardEntry,
    deck_name: str,
    deck_tags: Set[str],
    source_file_path: Path,
    assets_root_directory: Path,
    card_index: int,
    skip_media_validation: bool,
    skip_secrets_detection: bool
) -> Union[Card, YAMLProcessingError]:
    """
    Transforms a validated raw card entry into a canonical Card model.
    Performs sanitization, media validation, secrets detection, and final Card instantiation.
    """
    card_q_preview = raw_card_model.q[:50]

    # UUID Handling
    card_uuid_obj: Optional[uuid.UUID] = None
    if raw_card_model.id is not None:
        try:
            card_uuid_obj = uuid.UUID(raw_card_model.id)
        except ValueError:
            return YAMLProcessingError(
                file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
                field_name="id", message=f"Invalid UUID format for 'id': '{raw_card_model.id}'."
            )

    # Text Normalization and Sanitization
    front_normalized = raw_card_model.q.strip()
    back_normalized = raw_card_model.a.strip()

    front_sanitized = bleach.clean(front_normalized, tags=DEFAULT_ALLOWED_HTML_TAGS, attributes=DEFAULT_ALLOWED_HTML_ATTRIBUTES, strip=True)
    back_sanitized = bleach.clean(back_normalized, tags=DEFAULT_ALLOWED_HTML_TAGS, attributes=DEFAULT_ALLOWED_HTML_ATTRIBUTES, strip=True)

    # Secrets Detection
    if not skip_secrets_detection:
        for content, field_name_str in [(front_sanitized, "q (question)"), (back_sanitized, "a (answer)")]:
            for pattern in DEFAULT_SECRET_PATTERNS:
                if pattern.search(content):
                    return YAMLProcessingError(
                        file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
                        field_name=field_name_str, message=f"Potential secret detected in field '{field_name_str}'. Matched pattern: '{pattern.pattern[:50]}...'."
                    )
    # Tags Processing
    final_tags = deck_tags.copy() # Start with deck-level tags (already validated for format by _RawYAMLDeckFile)
    if raw_card_model.tags: # raw_card_model.tags is List[str] validated for kebab-case
        final_tags.update(tag.strip().lower() for tag in raw_card_model.tags)

    # Media Processing
    processed_media_paths: Optional[List[Path]] = None
    if raw_card_model.media: # raw_card_model.media is List[str]
        processed_media_paths = []
        for media_item_str in raw_card_model.media:
            media_path = Path(media_item_str.strip())
            if media_path.is_absolute():
                return YAMLProcessingError(
                    file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
                    field_name="media", message=f"Media path '{media_path}' must be relative."
                )
            
            if not skip_media_validation:
                try:
                    # Construct absolute path and resolve symlinks for robust check
                    full_media_path = (assets_root_directory / media_path).resolve(strict=True)
                    abs_assets_root = assets_root_directory.resolve(strict=True)
                    
                    # Ensure the resolved media path is truly within the assets_root_directory
                    if not str(full_media_path).startswith(str(abs_assets_root)):
                         return YAMLProcessingError(
                            file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
                            field_name="media", message=f"Media path '{media_path}' resolves outside the assets root directory '{assets_root_directory}'."
                        )
                    # is_file() check already done by resolve(strict=True) if it points to a file
                except FileNotFoundError:
                     return YAMLProcessingError(
                        file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
                        field_name="media", message=f"Media file not found at expected path: '{(assets_root_directory / media_path)}'."
                    )
                except Exception as e: # Catch other potential Path errors
                     return YAMLProcessingError(
                        file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
                        field_name="media", message=f"Error validating media path '{media_path}': {e}."
                    )
            processed_media_paths.append(media_path) # Store the original relative path

    # Instantiate canonical Card model
    try:
        card_data = {
            "deck_name": deck_name,
            "front": front_sanitized,
            "back": back_sanitized,
            "tags": final_tags,
            "source_yaml_file": source_file_path.resolve() # Store absolute, resolved path
        }
        if card_uuid_obj: # Only include if explicitly provided in YAML
            card_data["uuid"] = card_uuid_obj
        if raw_card_model.origin_task is not None:
            card_data["origin_task"] = raw_card_model.origin_task
        if processed_media_paths is not None: # Only include if media was specified
            card_data["media"] = processed_media_paths
        
        # `added_at` uses default_factory in Card model.
        # `internal_note` can be added here if specific conditions are met during processing.
        
        card_instance = Card(**card_data)
        return card_instance
    except ValidationError as e:
        # Construct a detailed error message from Pydantic's ValidationError
        error_details = "; ".join([f"{err['loc'][0] if err['loc'] else 'card'}: {err['msg']}" for err in e.errors()])
        return YAMLProcessingError(
            file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
            message=f"Final Card model validation failed. Details: {error_details}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error transforming card data for {source_file_path} at index {card_index}: {raw_card_model}")
        return YAMLProcessingError(
            file_path=source_file_path, card_index=card_index, card_question_snippet=card_q_preview,
            message=f"Unexpected internal error during Card instantiation: {type(e).__name__} - {e}"
        )

def _process_single_yaml_file(
    file_path: Path,
    assets_root_directory: Path,
    skip_media_validation: bool,
    skip_secrets_detection: bool
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Processes a single YAML file, returning successfully created Cards and any card-level errors."""
    cards_in_file: List[Card] = []
    errors_in_file: List[YAMLProcessingError] = []

    try:
        with file_path.open('r', encoding='utf-8') as f:
            raw_yaml_content = yaml.safe_load(f)
    except FileNotFoundError:
        raise YAMLProcessingError(file_path, "File not found.")
    except yaml.YAMLError as e:
        raise YAMLProcessingError(file_path, f"Invalid YAML syntax: {e}")
    except IOError as e:
        raise YAMLProcessingError(file_path, f"Could not read file: {e}")

    if not isinstance(raw_yaml_content, dict):
        raise YAMLProcessingError(file_path, "Top level of YAML must be a dictionary (deck object).")

    # Validate Raw YAML Structure using _RawYAMLDeckFile Pydantic model
    try:
        validated_raw_deck = _RawYAMLDeckFile(**raw_yaml_content)
    except ValidationError as e:
        error_details = "; ".join([f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
        raise YAMLProcessingError(file_path, f"File-level schema validation failed. Details: {error_details}")

    deck_name = validated_raw_deck.deck
    deck_tags: Set[str] = set(t.strip().lower() for t in validated_raw_deck.tags if t) if validated_raw_deck.tags else set()
    
    encountered_fronts_this_file: Set[str] = set()

    for idx, raw_card_entry_model in enumerate(validated_raw_deck.cards):
        transformed_result = _transform_raw_card_to_model(
            raw_card_entry_model,
            deck_name,
            deck_tags,
            file_path,
            assets_root_directory,
            idx,
            skip_media_validation,
            skip_secrets_detection
        )

        if isinstance(transformed_result, Card):
            card = transformed_result
            normalized_front = " ".join(card.front.lower().split())
            if normalized_front in encountered_fronts_this_file:
                errors_in_file.append(YAMLProcessingError(
                    file_path, "Duplicate question front within this YAML file.",
                    card_index=idx, card_question_snippet=card.front[:50]
                ))
            else:
                encountered_fronts_this_file.add(normalized_front)
                cards_in_file.append(card)
        else:
            errors_in_file.append(transformed_result)
            
    return cards_in_file, errors_in_file

def load_and_process_flashcard_yamls(
    source_directory: Path,
    assets_root_directory: Path,
    fail_fast: bool = False,
    skip_media_validation: bool = False,
    skip_secrets_detection: bool = False
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """
    Scans a directory for flashcard YAML files, processes them, and returns
    a list of Card objects and a list of any processing errors.
    """
    all_processed_cards: List[Card] = []
    all_errors: List[YAMLProcessingError] = []

    if not source_directory.is_dir():
        error = YAMLProcessingError(source_directory, "Source directory does not exist or is not a directory.")
        if fail_fast:
            raise error
        return [], [error]
    
    if not assets_root_directory.is_dir():
        error = YAMLProcessingError(assets_root_directory, "Assets root directory does not exist or is not a directory.")
        if fail_fast:
            raise error
        all_errors.append(error)
        if not skip_media_validation:
            return [], all_errors

    yaml_files = list(source_directory.rglob("*.yaml")) + list(source_directory.rglob("*.yml"))

    if not yaml_files:
        logger.info(f"No YAML files found in {source_directory}")
        return [], []

    logger.info(f"Found {len(yaml_files)} YAML files to process in {source_directory}")

    for yaml_file_path in yaml_files:
        logger.debug(f"Processing file: {yaml_file_path}")
        try:
            cards_from_file, errors_from_file = _process_single_yaml_file(
                yaml_file_path,
                assets_root_directory,
                skip_media_validation,
                skip_secrets_detection
            )
            all_processed_cards.extend(cards_from_file)
            all_errors.extend(errors_from_file)
            if fail_fast and errors_from_file:
                raise errors_from_file[0]
        except YAMLProcessingError as e:
            all_errors.append(e)
            if fail_fast:
                raise e
        except Exception as e:
            logger.exception(f"Unexpected error processing file {yaml_file_path}")
            error = YAMLProcessingError(yaml_file_path, f"Unexpected system error: {type(e).__name__} - {e}")
            all_errors.append(error)
            if fail_fast:
                raise error
    
    if not fail_fast or not all_errors:
        final_cards_list: List[Card] = []
        global_encountered_fronts: Dict[str, Path] = {}

        for card in all_processed_cards:
            normalized_front = " ".join(card.front.lower().split())
            if normalized_front in global_encountered_fronts:
                first_occurrence_path = global_encountered_fronts[normalized_front]
                duplicate_error = YAMLProcessingError(
                    card.source_yaml_file if card.source_yaml_file else Path("UnknownFile"),
                    f"Cross-file duplicate question front. First seen in '{first_occurrence_path.name}'.",
                    card_question_snippet=card.front[:50]
                )
                all_errors.append(duplicate_error)
            else:
                global_encountered_fronts[normalized_front] = card.source_yaml_file if card.source_yaml_file else Path("UnknownSource")
                final_cards_list.append(card)
        all_processed_cards = final_cards_list

    if all_errors:
        logger.warning(f"Completed YAML processing with {len(all_errors)} errors.")
    else:
        logger.info(f"Successfully processed {len(all_processed_cards)} cards from {len(yaml_files)} files with no errors.")
        
    return all_processed_cards, all_errors
