"""
cultivation/scripts/flashcore/yaml_processor.py

Main entry point for the YAML processing pipeline. This module orchestrates the
loading, validation, and transformation of flashcard YAML files into Card objects.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import yaml

from .card import Card
from .yaml_models import (
    _CardProcessingContext,
    _FileProcessingContext,
    _ProcessedCardData,
    _RawYAMLCardEntry,
    YAMLProcessingError,
    YAMLProcessorConfig,
)
from .yaml_validators import (
    run_card_validation_pipeline,
    validate_deck_and_extract_metadata,
    validate_directories,
    validate_raw_card_structure,
)

logger = logging.getLogger(__name__)


class YAMLProcessor:
    """
    Orchestrates the end-to-end processing of flashcard YAML files.
    """

    def __init__(self, config: YAMLProcessorConfig):
        self.config = config
        self.all_cards: List[Card] = []
        self.all_errors: List[YAMLProcessingError] = []
        self.yaml_files: List[Path] = []

    def run(self) -> Tuple[List[Card], List[YAMLProcessingError]]:
        """
        Executes the full YAML processing pipeline.

        Returns:
            A tuple containing the list of unique cards and a list of all errors.
        """
        try:
            self._validate_directories()
            self._find_yaml_files()
            self._process_all_files()
            self._filter_and_finalize()
        except YAMLProcessingError as e:
            # Handle early exit errors from validation
            self.all_errors.append(e)
        except Exception:
            # Catch any other unexpected errors during the process
            self.all_errors.append(
                YAMLProcessingError(
                    Path("N/A"),
                    f"An unexpected critical error occurred: {traceback.format_exc()}",
                )
            )

        self._log_summary()
        return self.all_cards, self.all_errors

    def _validate_directories(self):
        """Validates source and asset directories."""
        error = validate_directories(
            self.config.source_directory,
            self.config.assets_root_directory,
            self.config.skip_media_validation,
        )
        if error:
            raise error

    def _find_yaml_files(self):
        """Finds all YAML files in the source directory."""
        self.yaml_files = sorted(
            list(self.config.source_directory.rglob("*.yaml"))
            + list(self.config.source_directory.rglob("*.yml"))
        )
        if not self.yaml_files:
            logger.info(f"No YAML files found in {self.config.source_directory}")
        else:
            logger.info(
                f"Found {len(self.yaml_files)} YAML files to process in {self.config.source_directory}"
            )

    def _process_all_files(self):
        """Processes all found YAML files."""
        for file_path in self.yaml_files:
            try:
                cards, errors = self._process_single_yaml_file(file_path)
                self.all_cards.extend(cards)
                self.all_errors.extend(errors)
            except YAMLProcessingError as e:
                self.all_errors.append(e)
            except Exception:
                self.all_errors.append(
                    YAMLProcessingError(
                        file_path=file_path,
                        message=f"An unexpected error occurred: {traceback.format_exc()}",
                    )
                )

            if self.config.fail_fast and self.all_errors:
                logger.warning(
                    f"Fail-fast enabled. Stopping processing after first error in {file_path}"
                )
                break

    def _process_single_yaml_file(
        self, file_path: Path
    ) -> Tuple[List[Card], List[YAMLProcessingError]]:
        """Processes a single YAML file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            raw_yaml_content = yaml.safe_load(content)
        except FileNotFoundError:
            raise YAMLProcessingError(file_path, "File not found.")
        except IOError as e:
            raise YAMLProcessingError(file_path, f"Could not read file: {e}")
        except yaml.YAMLError as e:
            raise YAMLProcessingError(file_path, f"Invalid YAML syntax: {e}")

        if not isinstance(raw_yaml_content, dict):
            raise YAMLProcessingError(
                file_path, "Top level of YAML must be a dictionary (deck object)."
            )

        deck_name, deck_tags, cards_list = validate_deck_and_extract_metadata(
            raw_yaml_content, file_path
        )

        file_context = _FileProcessingContext(
            file_path=file_path,
            assets_root_directory=self.config.assets_root_directory,
            deck_name=deck_name,
            deck_tags=deck_tags,
            skip_media_validation=self.config.skip_media_validation,
            skip_secrets_detection=self.config.skip_secrets_detection,
        )

        return self._process_raw_cards(cards_list, file_context)

    def _process_raw_cards(
        self, cards_list: List[Dict], file_context: _FileProcessingContext
    ) -> Tuple[List[Card], List[YAMLProcessingError]]:
        """Processes a list of raw card dictionaries."""
        cards_in_file: List[Card] = []
        errors_in_file: List[YAMLProcessingError] = []
        encountered_fronts: Set[str] = set()

        for idx, card_dict in enumerate(cards_list):
            result = self._process_single_raw_card(card_dict, idx, file_context)

            if isinstance(result, Card):
                card = result
                normalized_front = " ".join(card.front.lower().split())
                if normalized_front in encountered_fronts:
                    errors_in_file.append(
                        YAMLProcessingError(
                            file_context.file_path,
                            "Duplicate question front within this YAML file.",
                            card_index=idx,
                            card_question_snippet=card.front[:50],
                        )
                    )
                else:
                    cards_in_file.append(card)
                    encountered_fronts.add(normalized_front)
            else:
                errors_in_file.append(result)

        return cards_in_file, errors_in_file

    def _process_single_raw_card(
        self, card_dict: Dict, idx: int, file_context: _FileProcessingContext
    ) -> Union[Card, YAMLProcessingError]:
        """Processes a single raw card dictionary from the YAML file."""
        if not isinstance(card_dict, dict):
            return YAMLProcessingError(
                file_path=file_context.file_path,
                message=f"Card entry at index {idx} is not a dictionary.",
                card_index=idx,
            )

        raw_card_model = validate_raw_card_structure(
            card_dict, idx, file_context.file_path
        )
        if isinstance(raw_card_model, YAMLProcessingError):
            return raw_card_model

        card_context = _CardProcessingContext(
            source_file_path=file_context.file_path,
            assets_root_directory=file_context.assets_root_directory,
            card_index=idx,
            card_q_preview=raw_card_model.q,
            skip_media_validation=file_context.skip_media_validation,
            skip_secrets_detection=file_context.skip_secrets_detection,
        )

        return self._transform_raw_card_to_model(
            raw_card_model, file_context, card_context
        )

    def _transform_raw_card_to_model(
        self,
        raw_card_model: _RawYAMLCardEntry,
        file_context: _FileProcessingContext,
        card_context: _CardProcessingContext,
    ) -> Union[Card, YAMLProcessingError]:
        """Transforms a validated raw card into a final Card model."""
        pipeline_result = run_card_validation_pipeline(
            raw_card_model, card_context, file_context.deck_tags
        )
        if isinstance(pipeline_result, YAMLProcessingError):
            return pipeline_result

        card_uuid, front, back, final_tags, media_paths = pipeline_result

        try:
            processed_data = _ProcessedCardData(
                uuid=card_uuid,
                front=front,
                back=back,
                tags=final_tags,
                media=media_paths,
                raw_card=raw_card_model,
            )
            return Card(
                uuid=processed_data.uuid,
                deck_name=file_context.deck_name,
                front=processed_data.front,
                back=processed_data.back,
                tags=processed_data.tags,
                media=processed_data.media,
                source_yaml_file=card_context.source_file_path,
                internal_note=processed_data.raw_card.internal_note,
                origin_task=processed_data.raw_card.origin_task,
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error during Card instantiation for {card_context.source_file_path}"
            )
            return YAMLProcessingError(
                file_path=card_context.source_file_path,
                card_index=card_context.card_index,
                card_question_snippet=card_context.card_q_preview,
                message=f"Unexpected internal error during Card instantiation: {type(e).__name__} - {e}",
            )

    def _filter_and_finalize(self):
        """Filters for cross-file duplicates and finalizes the card list."""
        final_cards: List[Card] = []
        duplicate_errors: List[YAMLProcessingError] = []
        encountered_fronts: Dict[str, Path] = {}

        for card in self.all_cards:
            normalized_front = " ".join(card.front.lower().split())
            if normalized_front in encountered_fronts:
                first_path = encountered_fronts[normalized_front]
                error = YAMLProcessingError(
                    card.source_yaml_file or Path("UnknownFile"),
                    f"Cross-file duplicate question front. First seen in '{first_path.name}'.",
                    card_question_snippet=card.front[:50],
                )
                duplicate_errors.append(error)
            else:
                encountered_fronts[normalized_front] = (
                    card.source_yaml_file or Path("UnknownSource")
                )
                final_cards.append(card)

        self.all_cards = final_cards
        self.all_errors.extend(duplicate_errors)

    def _log_summary(self):
        """Logs a summary of the processing results."""
        if self.all_errors:
            logger.warning(
                f"YAML processing complete. Found {len(self.all_cards)} unique cards and {len(self.all_errors)} errors."
            )
        else:
            logger.info(
                f"Successfully processed {len(self.all_cards)} cards from {len(self.yaml_files)} files with no errors."
            )


def load_and_process_flashcard_yamls(
    config: YAMLProcessorConfig,
) -> Tuple[List[Card], List[YAMLProcessingError]]:
    """
    High-level function to process flashcard YAMLs from a directory.

    This function acts as a simple, clean entry point to the YAML processing
    logic, which is encapsulated within the YAMLProcessor class.

    Args:
        config: A YAMLProcessorConfig object holding all configuration parameters.

    Returns:
        A tuple containing the list of unique cards and a list of all errors.
    """
    processor = YAMLProcessor(config)
    return processor.run()
