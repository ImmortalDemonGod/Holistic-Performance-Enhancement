"""
cultivation/scripts/flashcore/yaml_processor.py

Handles loading, parsing, validation, and transformation of flashcard definitions
from source YAML files into canonical `flashcore.card.Card` Pydantic model instances.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Union
import re
import logging

import yaml       # PyYAML for YAML parsing
import bleach     # For HTML content sanitization
from bleach.css_sanitizer import CSSSanitizer
from pydantic import BaseModel as PydanticBaseModel, Field, field_validator, ConfigDict, constr, ValidationError
from dataclasses import dataclass

# Local project imports
from cultivation.scripts.flashcore.card import Card, KEBAB_CASE_REGEX_PATTERN

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
DEFAULT_CSS_SANITIZER = CSSSanitizer()
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
    internal_note: Optional[str] = Field(default=None)  # Authorable from YAML

    model_config = ConfigDict(extra="forbid")

    @field_validator("tags", mode='before')
    @classmethod
    def normalize_tags(cls, v):
        if isinstance(v, list):
            return [tag.strip().lower() if isinstance(tag, str) else tag for tag in v]
        return v

class _RawYAMLDeckFile(PydanticBaseModel):
    deck: constr(min_length=1) # type: ignore
    tags: Optional[List[constr(pattern=RAW_KEBAB_CASE_PATTERN)]] = Field(default_factory=list) # type: ignore
    cards: List[_RawYAMLCardEntry] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")

# --- Custom Error Reporting Dataclass ---
@dataclass
class YAMLProcessingError(Exception):
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

@dataclass
class _CardProcessingContext:
    """Holds contextual data for processing a single card to reduce argument passing."""
    source_file_path: Path
    assets_root_directory: Path
    card_index: int
    card_q_preview: str
    skip_media_validation: bool
    skip_secrets_detection: bool


def _validate_card_uuid(
    raw_card: _RawYAMLCardEntry, context: _CardProcessingContext
) -> Union[uuid.UUID, YAMLProcessingError]:
    """Validates the raw UUID, returning a UUID object or a YAMLProcessingError."""
    if raw_card.id is None:
        return uuid.uuid4() # Assign a new UUID if none is provided
    try:
        return uuid.UUID(raw_card.id)
    except ValueError:
        return YAMLProcessingError(
            file_path=context.source_file_path, card_index=context.card_index, 
            card_question_snippet=context.card_q_preview,
            field_name="id", message=f"Invalid UUID format for 'id': '{raw_card.id}'."
        )

def _sanitize_card_text(raw_card: _RawYAMLCardEntry) -> Tuple[str, str]:
    """Normalizes and sanitizes card front and back text."""
    front_normalized = raw_card.q.strip()
    back_normalized = raw_card.a.strip()
    front_sanitized = bleach.clean(front_normalized, tags=DEFAULT_ALLOWED_HTML_TAGS, attributes=DEFAULT_ALLOWED_HTML_ATTRIBUTES, css_sanitizer=DEFAULT_CSS_SANITIZER, strip=True)
    back_sanitized = bleach.clean(back_normalized, tags=DEFAULT_ALLOWED_HTML_TAGS, attributes=DEFAULT_ALLOWED_HTML_ATTRIBUTES, css_sanitizer=DEFAULT_CSS_SANITIZER, strip=True)
    return front_sanitized, back_sanitized

def _check_for_secrets(
    front: str, back: str, context: _CardProcessingContext
) -> Optional[YAMLProcessingError]:
    """Scans text for secrets, returning an error if a secret is found."""
    if context.skip_secrets_detection:
        return None
    for pattern in DEFAULT_SECRET_PATTERNS:
        if pattern.search(back):
            return YAMLProcessingError(
                file_path=context.source_file_path, card_index=context.card_index, 
                card_question_snippet=context.card_q_preview, field_name="a", 
                message=f"Potential secret detected in card answer. Matched pattern: '{pattern.pattern[:50]}...'."
            )
        if pattern.search(front):
            return YAMLProcessingError(
                file_path=context.source_file_path, card_index=context.card_index, 
                card_question_snippet=context.card_q_preview, field_name="q", 
                message=f"Potential secret detected in card question. Matched pattern: '{pattern.pattern[:50]}...'."
            )
    return None

def _compile_card_tags(deck_tags: Set[str], card_tags: Optional[List[str]]) -> Set[str]:
    """Combines deck-level and card-level tags into a single set."""
    final_tags = deck_tags.copy()
    if card_tags:
        final_tags.update(tag.strip().lower() for tag in card_tags)
    return final_tags

def _validate_media_paths(
    raw_media: List[str], context: _CardProcessingContext
) -> Union[List[Path], YAMLProcessingError]:
    """Validates all media paths for a card, returning a list of Paths or an error."""
    processed_media_paths = []
    for media_item_str in raw_media:
        media_path = Path(media_item_str.strip())
        if media_path.is_absolute():
            return YAMLProcessingError(
                file_path=context.source_file_path, card_index=context.card_index, 
                card_question_snippet=context.card_q_preview, field_name="media", 
                message=f"Media path must be relative: '{media_path}'."
            )
        
        if not context.skip_media_validation:
            try:
                full_media_path = (context.assets_root_directory / media_path).resolve(strict=False)
                abs_assets_root = context.assets_root_directory.resolve(strict=True)
                if not str(full_media_path).startswith(str(abs_assets_root)):
                    return YAMLProcessingError(
                        file_path=context.source_file_path, card_index=context.card_index, 
                        card_question_snippet=context.card_q_preview, field_name="media", 
                        message=f"Media path '{media_path}' resolves outside the assets root directory."
                    )
                if not full_media_path.exists():
                    return YAMLProcessingError(
                        file_path=context.source_file_path, card_index=context.card_index, 
                        card_question_snippet=context.card_q_preview, field_name="media", 
                        message=f"Media file not found at expected path: '{full_media_path}'."
                    )
            except Exception as e:
                return YAMLProcessingError(
                    file_path=context.source_file_path, card_index=context.card_index, 
                    card_question_snippet=context.card_q_preview, field_name="media", 
                    message=f"Error validating media path '{media_path}': {e}."
                )
        processed_media_paths.append(media_path)
    return processed_media_paths

def _transform_raw_card_to_model(
    raw_card_model: _RawYAMLCardEntry,
    deck_name: str,
    deck_tags: Set[str],
    context: _CardProcessingContext,
) -> Union[Card, YAMLProcessingError]:
    """
    Transforms a validated raw card entry into a canonical Card model by orchestrating
    validation, sanitization, and instantiation steps.
    """
    # --- Field Processing and Validation ---
    uuid_or_error = _validate_card_uuid(raw_card_model, context)
    if isinstance(uuid_or_error, YAMLProcessingError):
        return uuid_or_error

    front, back = _sanitize_card_text(raw_card_model)

    secret_error = _check_for_secrets(front, back, context)
    if secret_error:
        return secret_error

    final_tags = _compile_card_tags(deck_tags, raw_card_model.tags)

    media_paths_or_error: Optional[Union[List[Path], YAMLProcessingError]] = None
    if raw_card_model.media:
        media_paths_or_error = _validate_media_paths(raw_card_model.media, context)
        if isinstance(media_paths_or_error, YAMLProcessingError):
            return media_paths_or_error

    # --- Final Model Instantiation ---
    try:
        card_data = {
            "uuid": uuid_or_error,
            "deck_name": deck_name,
            "front": front,
            "back": back,
            "tags": final_tags,
            "source_yaml_file": context.source_file_path.resolve(),
            "internal_note": raw_card_model.internal_note,
            "origin_task": raw_card_model.origin_task,
            "media": media_paths_or_error or []
        }
        return Card(**card_data)
    except ValidationError as e:
        error_details = "; ".join([f"{err['loc'][0] if err['loc'] else 'card'}: {err['msg']}" for err in e.errors()])
        return YAMLProcessingError(
            file_path=context.source_file_path, card_index=context.card_index,
            card_question_snippet=context.card_q_preview,
            message=f"Final Card model validation failed. Details: {error_details}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error during Card instantiation for {context.source_file_path}")
        return YAMLProcessingError(
            file_path=context.source_file_path, card_index=context.card_index,
            card_question_snippet=context.card_q_preview,
            message=f"Unexpected internal error during Card instantiation: {type(e).__name__} - {e}"
        )

def _load_and_parse_yaml(file_path: Path) -> dict:
    """Loads and parses a YAML file, handling file-level exceptions."""
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
    return raw_yaml_content


def _validate_deck_and_extract_metadata(
    raw_yaml_content: dict, file_path: Path
) -> Tuple[str, Set[str], List[dict]]:
    """Validates the deck structure and extracts metadata."""
    if 'deck' not in raw_yaml_content or not isinstance(raw_yaml_content['deck'], str) or not raw_yaml_content['deck'].strip():
        raise YAMLProcessingError(file_path, "Missing or invalid 'deck' field at top level.")
    deck_name = raw_yaml_content['deck'].strip()

    tags = raw_yaml_content.get('tags', [])
    if tags is not None and not isinstance(tags, list):
        raise YAMLProcessingError(file_path, "'tags' field must be a list if present.")
    deck_tags: Set[str] = set(t.strip().lower() for t in tags if isinstance(t, str)) if tags else set()

    if 'cards' not in raw_yaml_content or not isinstance(raw_yaml_content['cards'], list):
        raise YAMLProcessingError(file_path, "Missing or invalid 'cards' list at top level.")
    cards_list = raw_yaml_content['cards']
    if not cards_list:
        raise YAMLProcessingError(file_path, "No cards found in 'cards' list.")

    return deck_name, deck_tags, cards_list


def _process_raw_cards(
    cards_list: List[dict],
    deck_name: str,
    deck_tags: Set[str],
    file_path: Path,
    assets_root_directory: Path,
    skip_media_validation: bool,
    skip_secrets_detection: bool,
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Processes a list of raw card dictionaries into Card objects and errors."""
    cards_in_file: List[Card] = []
    errors_in_file: List[YAMLProcessingError] = []
    encountered_fronts_this_file: Set[str] = set()
    forbidden_fields = {"added_at"}

    for idx, card_dict in enumerate(cards_list):
        if not isinstance(card_dict, dict):
            errors_in_file.append(YAMLProcessingError(
                file_path=file_path,
                message=f"Card entry at index {idx} is not a dictionary.",
                card_index=idx,
            ))
            continue

        forbidden_present = forbidden_fields.intersection(card_dict.keys())
        if forbidden_present:
            errors_in_file.append(
                YAMLProcessingError(
                    file_path=file_path,
                    message=f"Forbidden field(s) present in card YAML: {', '.join(forbidden_present)}. These will be ignored and replaced by system-assigned values.",
                    card_index=idx,
                    card_question_snippet=str(card_dict.get('q', ''))[:50]
                )
            )
        try:
            raw_card_entry_model = _RawYAMLCardEntry(**card_dict)
        except ValidationError as e:
            error_details = "; ".join([f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
            errors_in_file.append(
                YAMLProcessingError(
                    file_path=file_path,
                    message=f"Card-level schema validation failed. Details: {error_details}",
                    card_index=idx,
                    card_question_snippet=str(card_dict.get('q', ''))[:50]
                )
            )
            continue

        context = _CardProcessingContext(
            source_file_path=file_path,
            assets_root_directory=assets_root_directory,
            card_index=idx,
            card_q_preview=raw_card_entry_model.q[:50],
            skip_media_validation=skip_media_validation,
            skip_secrets_detection=skip_secrets_detection,
        )
        transformed_result = _transform_raw_card_to_model(
            raw_card_entry_model, deck_name, deck_tags, context
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
                cards_in_file.append(card)
                encountered_fronts_this_file.add(normalized_front)
        else:  # It's a YAMLProcessingError
            errors_in_file.append(transformed_result)

    return cards_in_file, errors_in_file


def _process_single_yaml_file(
    file_path: Path,
    assets_root_directory: Path,
    skip_media_validation: bool,
    skip_secrets_detection: bool
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Processes a single YAML file, returning successfully created Cards and any card-level errors."""
    raw_yaml_content = _load_and_parse_yaml(file_path)
    deck_name, deck_tags, cards_list = _validate_deck_and_extract_metadata(
        raw_yaml_content, file_path
    )

    return _process_raw_cards(
        cards_list,
        deck_name,
        deck_tags,
        file_path,
        assets_root_directory,
        skip_media_validation,
        skip_secrets_detection,
    )


def _validate_directories(
    source_directory: Path, 
    assets_root_directory: Path, 
    skip_media_validation: bool
) -> Optional[YAMLProcessingError]:
    """Validates that the source and asset directories exist."""
    if not source_directory.is_dir():
        return YAMLProcessingError(source_directory, "Source directory does not exist or is not a directory.")
    
    if not assets_root_directory.is_dir():
        # This error is only critical if we are validating media
        if not skip_media_validation:
            return YAMLProcessingError(assets_root_directory, "Assets root directory does not exist or is not a directory.")
    
    return None

def _find_yaml_files(source_directory: Path) -> List[Path]:
    """Finds all YAML files in the given directory, sorted for deterministic processing."""
    yaml_files = sorted(
        list(source_directory.rglob("*.yaml")) + list(source_directory.rglob("*.yml"))
    )
    if not yaml_files:
        logger.info(f"No YAML files found in {source_directory}")
    else:
        logger.info(f"Found {len(yaml_files)} YAML files to process in {source_directory}")
    return yaml_files

def _process_all_files(
    yaml_files: List[Path],
    assets_root_directory: Path,
    skip_media_validation: bool,
    skip_secrets_detection: bool,
    fail_fast: bool,
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Processes a list of YAML files, collecting cards and errors."""
    all_processed_cards: List[Card] = []
    all_errors: List[YAMLProcessingError] = []

    for yaml_file_path in yaml_files:
        logger.debug(f"Processing file: {yaml_file_path}")
        try:
            cards_from_file, errors_from_file = _process_single_yaml_file(
                yaml_file_path,
                assets_root_directory,
                skip_media_validation,
                skip_secrets_detection,
            )
            all_processed_cards.extend(cards_from_file)
            all_errors.extend(errors_from_file)
            if fail_fast and errors_from_file:
                raise errors_from_file[0]
        except YAMLProcessingError as e:
            # Re-raise to be caught by the main function's fail_fast logic
            if fail_fast:
                raise
            all_errors.append(e)
        except Exception as e:
            logger.exception(f"Unexpected error processing file {yaml_file_path}")
            error = YAMLProcessingError(
                yaml_file_path, f"Unexpected system error: {type(e).__name__} - {e}"
            )
            if fail_fast:
                raise error
            all_errors.append(error)
            
    return all_processed_cards, all_errors

def _filter_duplicate_cards(
    cards: List[Card],
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Filters a list of cards for duplicates based on the question front."""
    final_cards_list: List[Card] = []
    duplicate_errors: List[YAMLProcessingError] = []
    global_encountered_fronts: Dict[str, Path] = {}

    for card in cards:
        normalized_front = " ".join(card.front.lower().split())
        if normalized_front in global_encountered_fronts:
            first_occurrence_path = global_encountered_fronts[normalized_front]
            error = YAMLProcessingError(
                card.source_yaml_file or Path("UnknownFile"),
                f"Cross-file duplicate question front. First seen in '{first_occurrence_path.name}'.",
                card_question_snippet=card.front[:50],
            )
            duplicate_errors.append(error)
        else:
            global_encountered_fronts[normalized_front] = card.source_yaml_file or Path("UnknownSource")
            final_cards_list.append(card)
            
    return final_cards_list, duplicate_errors

def load_and_process_flashcard_yamls(
    source_directory: Path,
    assets_root_directory: Path,
    fail_fast: bool = False,
    skip_media_validation: bool = False,
    skip_secrets_detection: bool = False,
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """
    Scans a directory for flashcard YAML files, processes them, and returns
    a list of Card objects and a list of any processing errors.
    """
    validation_error = _validate_directories(
        source_directory, assets_root_directory, skip_media_validation
    )
    if validation_error:
        if fail_fast:
            raise validation_error
        return [], [validation_error]

    yaml_files = _find_yaml_files(source_directory)
    if not yaml_files:
        return [], []

    processed_cards, processing_errors = _process_all_files(
        yaml_files,
        assets_root_directory,
        skip_media_validation,
        skip_secrets_detection,
        fail_fast,
    )

    # If there are processing errors and we are not in fail_fast mode,
    # return the cards found so far without checking for duplicates.
    if processing_errors and not fail_fast:
        return processed_cards, processing_errors

    # Otherwise, proceed to check for duplicates.
    final_cards, duplicate_errors = _filter_duplicate_cards(processed_cards)
    all_errors = processing_errors + duplicate_errors

    if all_errors:
        logger.warning(f"Completed YAML processing with {len(all_errors)} errors.")
    else:
        logger.info(f"Successfully processed {len(final_cards)} cards from {len(yaml_files)} files with no errors.")
        
    return final_cards, all_errors
