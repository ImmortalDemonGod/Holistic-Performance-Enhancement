"""
cultivation/scripts/flashcore/yaml_processor.py

Handles loading, parsing, validation, and transformation of flashcard definitions
from source YAML files into canonical `flashcore.card.Card` Pydantic model instances.
"""

import logging
import re
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Set, Tuple, Union

import bleach  # For HTML content sanitization
import yaml  # PyYAML for YAML parsing
from bleach.css_sanitizer import CSSSanitizer
from pydantic import (
    BaseModel as PydanticBaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

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

# --- Type Aliases for Pydantic v2 Validation ---
KebabCaseStr = Annotated[str, StringConstraints(pattern=RAW_KEBAB_CASE_PATTERN)]

# --- Internal Pydantic Models for Raw YAML Validation ---
class _RawYAMLCardEntry(PydanticBaseModel):
    id: Optional[str] = Field(default=None)
    s: Optional[int] = Field(default=None, ge=0, le=4)
    q: str = Field(..., min_length=1)
    a: str = Field(..., min_length=1)
    tags: Optional[List[KebabCaseStr]] = Field(default_factory=list)
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
    deck: str = Field(..., min_length=1)
    tags: Optional[List[KebabCaseStr]] = Field(default_factory=list)
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


@dataclass
class _FileProcessingContext:
    """Holds contextual data for processing a single YAML file."""
    file_path: Path
    assets_root_directory: Path
    deck_name: str
    deck_tags: Set[str]
    skip_media_validation: bool
    skip_secrets_detection: bool


@dataclass
class _ProcessedCardData:
    """Holds validated and sanitized data ready for Card model instantiation."""
    uuid: uuid.UUID
    front: str
    back: str
    tags: Set[str]
    media: List[Path]
    raw_card: _RawYAMLCardEntry


@dataclass
class YAMLProcessorConfig:
    """Configuration for the entire YAML processing workflow."""
    source_directory: Path
    assets_root_directory: Path
    fail_fast: bool = False
    skip_media_validation: bool = False
    skip_secrets_detection: bool = False


@dataclass
class _ProcessingConfig:
    """Internal configuration for file-level processing."""
    assets_root_directory: Path
    skip_media_validation: bool
    skip_secrets_detection: bool
    fail_fast: bool


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
        result = _validate_single_media_path(media_item_str, context)
        if isinstance(result, YAMLProcessingError):
            return result
        processed_media_paths.append(result)
    return processed_media_paths

def _validate_single_media_path(
    media_item_str: str, context: _CardProcessingContext
) -> Union[Path, YAMLProcessingError]:
    """Validates a single media path, returning a Path object or a YAMLProcessingError."""
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
    return media_path

def _build_card_model(
    processed_data: _ProcessedCardData,
    deck_name: str,
    context: _CardProcessingContext,
) -> Card:
    """Constructs the final Card model from processed data."""
    return Card(
        uuid=processed_data.uuid,
        deck_name=deck_name,
        front=processed_data.front,
        back=processed_data.back,
        tags=processed_data.tags,
        media=processed_data.media,
        source_yaml_file=context.source_file_path,
        internal_note=processed_data.raw_card.internal_note,
        origin_task=processed_data.raw_card.origin_task,
    )

def _run_card_validation_pipeline(
    raw_card: _RawYAMLCardEntry, context: _CardProcessingContext, deck_tags: Set[str]
) -> Union[Tuple[uuid.UUID, str, str, Set[str], List[Path]], YAMLProcessingError]:
    """
    Runs the validation pipeline for a raw card, returning processed data or an error.
    Orchestrates UUID validation, text sanitization, secret scanning, tag compilation,
    and media path validation.
    """
    uuid_or_error = _validate_card_uuid(raw_card, context)
    if isinstance(uuid_or_error, YAMLProcessingError):
        return uuid_or_error

    front, back = _sanitize_card_text(raw_card)

    secret_error = _check_for_secrets(front, back, context)
    if secret_error:
        return secret_error

    final_tags = _compile_card_tags(deck_tags, raw_card.tags)

    media_paths: List[Path] = []
    if raw_card.media:
        media_paths_or_error = _validate_media_paths(raw_card.media, context)
        if isinstance(media_paths_or_error, YAMLProcessingError):
            return media_paths_or_error
        media_paths = media_paths_or_error
    
    return uuid_or_error, front, back, final_tags, media_paths


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
    pipeline_result = _run_card_validation_pipeline(raw_card_model, context, deck_tags)
    if isinstance(pipeline_result, YAMLProcessingError):
        return pipeline_result
    
    card_uuid, front, back, final_tags, media_paths = pipeline_result

    # --- Final Model Instantiation ---
    try:
        processed_data = _ProcessedCardData(
            uuid=card_uuid,
            front=front,
            back=back,
            tags=final_tags,
            media=media_paths,
            raw_card=raw_card_model,
        )
        return _build_card_model(
            processed_data=processed_data,
            deck_name=deck_name,
            context=context,
        )
    except ValidationError as e:
        error_details = "; ".join([f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
        return YAMLProcessingError(
            file_path=context.source_file_path,
            message=f"Final Card model validation failed. Details: {error_details}",
            card_index=context.card_index,
            card_question_snippet=context.card_q_preview,
        )
    except Exception as e:
        logger.exception(f"Unexpected error during Card instantiation for {context.source_file_path}")
        return YAMLProcessingError(
            file_path=context.source_file_path,
            card_index=context.card_index,
            card_question_snippet=context.card_q_preview,
            message=f"Unexpected internal error during Card instantiation: {type(e).__name__} - {e}",
        )

def _read_file_content(file_path: Path) -> str:
    """Reads file content, wrapping I/O errors into YAMLProcessingError."""
    try:
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise YAMLProcessingError(file_path, "File not found.")
    except IOError as e:
        raise YAMLProcessingError(file_path, f"Could not read file: {e}")


def _load_and_parse_yaml(file_path: Path) -> dict:
    """Loads and parses a YAML file, handling file-level exceptions."""
    content = _read_file_content(file_path)
    try:
        raw_yaml_content = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise YAMLProcessingError(file_path, f"Invalid YAML syntax: {e}")

    if not isinstance(raw_yaml_content, dict):
        raise YAMLProcessingError(file_path, "Top level of YAML must be a dictionary (deck object).")
    return raw_yaml_content


def _extract_deck_name(raw_yaml_content: dict, file_path: Path) -> str:
    """Extracts and validates the deck name from the raw YAML content."""
    deck_value = raw_yaml_content.get('deck')
    if deck_value is None:
        raise YAMLProcessingError(file_path, "Missing 'deck' field at top level.")
    if not isinstance(deck_value, str):
        raise YAMLProcessingError(file_path, "'deck' field must be a string.")
    if not deck_value.strip():
        raise YAMLProcessingError(file_path, "'deck' field cannot be empty or just whitespace.")
    return deck_value.strip()


def _extract_deck_tags(raw_yaml_content: dict, file_path: Path) -> Set[str]:
    """Extracts and validates deck-level tags from the raw YAML content."""
    tags = raw_yaml_content.get('tags', [])
    if tags is not None and not isinstance(tags, list):
        raise YAMLProcessingError(file_path, "'tags' field must be a list if present.")
    return {t.strip().lower() for t in tags if isinstance(t, str)} if tags else set()


def _extract_cards_list(raw_yaml_content: dict, file_path: Path) -> List[dict]:
    """Extracts and validates the list of cards from the raw YAML content."""
    if 'cards' not in raw_yaml_content or not isinstance(raw_yaml_content['cards'], list):
        raise YAMLProcessingError(file_path, "Missing or invalid 'cards' list at top level.")
    cards_list = raw_yaml_content['cards']
    if not cards_list:
        raise YAMLProcessingError(file_path, "No cards found in 'cards' list.")
    return cards_list


def _validate_deck_and_extract_metadata(
    raw_yaml_content: dict, file_path: Path
) -> Tuple[str, Set[str], List[dict]]:
    """Validates the deck structure and extracts metadata by calling specialized helpers."""
    deck_name = _extract_deck_name(raw_yaml_content, file_path)
    deck_tags = _extract_deck_tags(raw_yaml_content, file_path)
    cards_list = _extract_cards_list(raw_yaml_content, file_path)
    return deck_name, deck_tags, cards_list


def _validate_raw_card_structure(
    card_dict: dict, idx: int, file_path: Path
) -> Union[_RawYAMLCardEntry, YAMLProcessingError]:
    """Validates the structure of a raw card dict using Pydantic."""
    try:
        return _RawYAMLCardEntry.model_validate(card_dict)
    except ValidationError as e:
        error_details = "; ".join([f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()])
        q_preview = card_dict.get('q', 'N/A')
        return YAMLProcessingError(
            file_path=file_path,
            message=f"Card validation failed. Details: {error_details}",
            card_index=idx,
            card_question_snippet=q_preview,
        )


def _process_single_raw_card(
    card_dict: dict, idx: int, file_context: _FileProcessingContext
) -> Union[Card, YAMLProcessingError]:
    """Processes a single raw card dictionary from the YAML file."""
    if not isinstance(card_dict, dict):
        return YAMLProcessingError(
            file_path=file_context.file_path,
            message=f"Card entry at index {idx} is not a dictionary.",
            card_index=idx,
        )

    raw_card_model_or_error = _validate_raw_card_structure(card_dict, idx, file_context.file_path)
    if isinstance(raw_card_model_or_error, YAMLProcessingError):
        return raw_card_model_or_error

    card_context = _CardProcessingContext(
        source_file_path=file_context.file_path,
        assets_root_directory=file_context.assets_root_directory,
        card_index=idx,
        card_q_preview=raw_card_model_or_error.q,
        skip_media_validation=file_context.skip_media_validation,
        skip_secrets_detection=file_context.skip_secrets_detection,
    )

    return _transform_raw_card_to_model(
        raw_card_model=raw_card_model_or_error,
        deck_name=file_context.deck_name,
        deck_tags=file_context.deck_tags,
        context=card_context,
    )


def _process_raw_cards(
    cards_list: List[dict], file_context: _FileProcessingContext
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Processes a list of raw card dictionaries into Card objects and errors."""
    cards_in_file: List[Card] = []
    errors_in_file: List[YAMLProcessingError] = []
    encountered_fronts_this_file: Set[str] = set()

    for idx, card_dict in enumerate(cards_list):
        result = _process_single_raw_card(card_dict, idx, file_context)

        if isinstance(result, Card):
            card = result
            normalized_front = " ".join(card.front.lower().split())
            if normalized_front in encountered_fronts_this_file:
                errors_in_file.append(YAMLProcessingError(
                    file_context.file_path, "Duplicate question front within this YAML file.",
                    card_index=idx, card_question_snippet=card.front[:50]
                ))
            else:
                cards_in_file.append(card)
                encountered_fronts_this_file.add(normalized_front)
        else:
            errors_in_file.append(result)

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

    file_context = _FileProcessingContext(
        file_path=file_path,
        assets_root_directory=assets_root_directory,
        deck_name=deck_name,
        deck_tags=deck_tags,
        skip_media_validation=skip_media_validation,
        skip_secrets_detection=skip_secrets_detection,
    )

    return _process_raw_cards(cards_list, file_context)


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
    yaml_files: List[Path], config: _ProcessingConfig
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Processes a list of YAML files, collecting cards and errors."""
    all_cards: List[Card] = []
    all_errors: List[YAMLProcessingError] = []

    for file_path in yaml_files:
        try:
            cards, errors = _process_single_yaml_file(
                file_path,
                config.assets_root_directory,
                config.skip_media_validation,
                config.skip_secrets_detection,
            )
            all_cards.extend(cards)
            all_errors.extend(errors)
        except YAMLProcessingError as e:
            all_errors.append(e)
        except Exception:
            all_errors.append(YAMLProcessingError(
                file_path=file_path,
                message=f"An unexpected error occurred: {traceback.format_exc()}"
            ))

        if config.fail_fast and all_errors:
            logger.warning(f"Fail-fast enabled. Stopping processing after first error in {file_path}")
            break

    return all_cards, all_errors

def _filter_duplicate_cards(
    cards: List[Card],
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """Filters a list of cards for duplicates based on the question front."""
    final_cards_list: List[Card] = []
    duplicate_errors: List[YAMLProcessingError] = []
    global_encountered_fronts: Dict[str, Path] = {}

    for card in cards:
        # Normalize the front to be case-insensitive and ignore leading/trailing whitespace
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

def _log_processing_summary(
    unique_cards_count: int, errors_count: int, files_count: int
) -> None:
    """Logs a summary of the YAML processing results."""
    if errors_count:
        logger.warning(
            f"YAML processing complete. Found {unique_cards_count} unique cards and {errors_count} errors."
        )
    else:
        logger.info(
            f"Successfully processed {unique_cards_count} cards from {files_count} files with no errors."
        )


def load_and_process_flashcard_yamls(
    config: YAMLProcessorConfig,
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """
    Scans a directory for flashcard YAML files, processes them, and returns
    a list of Card objects and a list of any processing errors.

    The processing pipeline includes:
    1. Finding all YAML files recursively in the source directory.
    2. Validating source and asset directories.
    3. Processing each file to parse YAML, validate schema, and transform cards.
    4. Filtering out cards that are cross-file duplicates based on the question front.
    5. Returns the final list of unique cards and a comprehensive list of errors.

    Args:
        config: A YAMLProcessorConfig object holding all configuration parameters.

    Returns:
        A tuple containing:
        - A list of `Card` model instances.
        - A list of `YAMLProcessingError` instances detailing any issues.
    """
    validation_error = _validate_directories(
        config.source_directory,
        config.assets_root_directory,
        config.skip_media_validation,
    )
    if validation_error:
        return [], [validation_error]

    internal_config = _ProcessingConfig(
        assets_root_directory=config.assets_root_directory,
        skip_media_validation=config.skip_media_validation,
        skip_secrets_detection=config.skip_secrets_detection,
        fail_fast=config.fail_fast,
    )

    yaml_files = _find_yaml_files(config.source_directory)
    all_cards, all_errors = _process_all_files(yaml_files, internal_config)

    if config.fail_fast and all_errors:
        raise all_errors[0]

    # Filter for duplicate cards across all processed files
    unique_cards, duplicate_errors = _filter_duplicate_cards(all_cards)
    all_errors.extend(duplicate_errors)

    _log_processing_summary(len(unique_cards), len(all_errors), len(yaml_files))
        
    return unique_cards, all_errors
