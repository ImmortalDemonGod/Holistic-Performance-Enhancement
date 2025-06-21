import pytest
from pathlib import Path
import uuid

# Adjust import path as needed
from cultivation.scripts.flashcore.yaml_processor import (
    YAMLProcessingError,
    YAMLProcessorConfig,
    load_and_process_flashcard_yamls,
)


# --- Test Fixtures ---

@pytest.fixture
def assets_dir(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "image.png").write_text("dummy image content")
    (assets / "subfolder").mkdir()
    (assets / "subfolder" / "audio.mp3").write_text("dummy audio content")
    return assets

def create_yaml_file(base_path: Path, filename: str, content: str) -> Path:
    file_path = base_path / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path

# --- Sample YAML Content Strings ---

VALID_YAML_MINIMAL_CONTENT = '''
deck: Minimal
cards:
  - q: Question 1?
    a: Answer 1.
'''

VALID_YAML_COMPREHENSIVE_CONTENT = '''
deck: Comprehensive::SubDeck
tags: [deck-tag, another-deck-tag]
cards:
  - id: 7b7c2d57-e7b8-4e1f-a793-0ee2a61bcb79
    q: Question One Full?
    a: Answer One with <code>code</code> and **markdown**.
    tags: [card-tag1]
    origin_task: TASK-101
    media:
      - image.png
      - subfolder/audio.mp3
  - q: Question Two Full?
    a: Answer Two.
    tags: [card-tag2, another-card-tag]
'''

YAML_WITH_NO_CARD_ID_CONTENT = '''
deck: NoCardID
cards:
  - q: Q1 no id
    a: A1
  - q: Q2 no id
    a: A2
'''

YAML_WITH_SECRET_CONTENT = '''
deck: SecretsDeck
cards:
  - q: What is the api_key?
    a: The api_key is sk_live_verylongtestkey1234567890 # gitleaks:allow
  - q: Another question
    a: Some normal answer.
'''

YAML_WITH_INTRA_FILE_DUPLICATE_Q_CONTENT = '''
deck: IntraDup
cards:
  - q: Duplicate Question?
    a: Answer A
  - q: Unique Question?
    a: Answer B
  - q: Duplicate Question?
    a: Answer C
'''

INVALID_YAML_SYNTAX_CONTENT = '''
deck: BadSyntax
cards:
  - q: Question
  a: Answer with bad indent
'''

INVALID_YAML_SCHEMA_NO_DECK_CONTENT = '''
# Missing 'deck' key
tags: [oops]
cards:
  - q: Q
    a: A
'''

INVALID_YAML_SCHEMA_CARD_NO_Q_CONTENT = '''
deck: CardNoQ
cards:
  - a: Answer without question
'''





# --- Tests for load_and_process_flashcard_yamls ---

class TestLoadAndProcessFlashcardYamls:
    def test_empty_source_directory(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "empty_src"
        source_dir.mkdir()
        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)
        assert not cards
        assert not errors

    def test_single_valid_file(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        create_yaml_file(source_dir, "deck1.yaml", VALID_YAML_MINIMAL_CONTENT)
        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)
        assert len(cards) == 1
        assert not errors
        assert cards[0].deck_name == "Minimal"

    def test_multiple_valid_files_and_recursion(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_multi"
        source_dir.mkdir()
        sub_dir = source_dir / "subdir"
        sub_dir.mkdir()
        create_yaml_file(source_dir, "deckA.yaml", VALID_YAML_MINIMAL_CONTENT)
        create_yaml_file(sub_dir, "deckB.yml", YAML_WITH_NO_CARD_ID_CONTENT) # .yml extension

        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)
        assert len(cards) == 1 + 2 # 1 from deckA, 2 from deckB
        assert not errors

    def test_error_aggregation_fail_fast_false(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_errors"
        source_dir.mkdir()
        create_yaml_file(source_dir, "valid.yaml", VALID_YAML_MINIMAL_CONTENT)
        create_yaml_file(source_dir, "badsyntax.yaml", INVALID_YAML_SYNTAX_CONTENT)
        create_yaml_file(source_dir, "card_no_q.yaml", INVALID_YAML_SCHEMA_CARD_NO_Q_CONTENT)

        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir, fail_fast=False)
        cards, errors = load_and_process_flashcard_yamls(config)
        assert len(cards) == 1 # Only from valid.yaml
        assert len(errors) == 2
        assert any("Invalid YAML syntax" in str(e) for e in errors if e.file_path.name == "badsyntax.yaml")
        assert any("Card validation failed" in str(e) for e in errors if e.file_path.name == "card_no_q.yaml")

    def test_fail_fast_true_on_file_error(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_fail_fast"
        source_dir.mkdir()
        create_yaml_file(source_dir, "badsyntax.yaml", INVALID_YAML_SYNTAX_CONTENT) # This should cause immediate failure
        create_yaml_file(source_dir, "valid.yaml", VALID_YAML_MINIMAL_CONTENT)
        
        with pytest.raises(YAMLProcessingError, match="Invalid YAML syntax"):
            config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir, fail_fast=True)
            load_and_process_flashcard_yamls(config)

    def test_fail_fast_true_on_card_error(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_fail_fast_card"
        source_dir.mkdir()
        create_yaml_file(source_dir, "valid_first.yaml", VALID_YAML_MINIMAL_CONTENT)
        # File with a card-level error
        content_card_error = '''
deck: CardErrorDeck
cards:
  - q: ValidQ
    a: ValidA
  - q: InvalidQSecret
    a: "api_key: sk_live_verylongtestkey1234567890"
'''
        create_yaml_file(source_dir, "card_error.yaml", content_card_error)

        with pytest.raises(YAMLProcessingError, match="Potential secret detected"):
             config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir, fail_fast=True, skip_secrets_detection=False)
             load_and_process_flashcard_yamls(config)


    def test_cross_file_duplicate_question(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_cross_dup"
        source_dir.mkdir()
        yaml_a_content = '''
deck: DeckA
cards:
  - q: Shared Question?
    a: Answer from A
'''
        yaml_b_content = '''
deck: DeckB
cards:
  - q: Shared Question?
    a: Answer from B
  - q: Unique Question?
    a: Answer from B
'''
        create_yaml_file(source_dir, "deckA.yaml", yaml_a_content)
        create_yaml_file(source_dir, "deckB.yaml", yaml_b_content)

        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)
        
        assert len(cards) == 2 # One "Shared Question?" and one "Unique Question?"
        assert len(errors) == 1
        assert "Cross-file duplicate question front. First seen in" in errors[0].message
        assert errors[0].file_path.name == "deckB.yaml"
        assert errors[0].card_question_snippet == "Shared Question?"

    def test_media_validation_flag(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_media_flag"
        source_dir.mkdir()
        yaml_nonexistent_media = '''
deck: MediaTest
cards:
  - q: Media card
    a: Check media
    media: [nonexistent.png]
'''
        create_yaml_file(source_dir, "media_test.yaml", yaml_nonexistent_media)

        # With skip_media_validation=True, should process without media error
        config_skip = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir, skip_media_validation=True)
        cards_skipped, errors_skipped = load_and_process_flashcard_yamls(config_skip)
        assert len(cards_skipped) == 1
        assert not errors_skipped
        assert cards_skipped[0].media == [Path("nonexistent.png")]

        # With skip_media_validation=False (default), should error
        config_no_skip = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir, skip_media_validation=False)
        cards_not_skipped, errors_not_skipped = load_and_process_flashcard_yamls(config_no_skip)
        assert not cards_not_skipped # Card with bad media should not be processed
        assert len(errors_not_skipped) == 1
        assert "Media file not found" in errors_not_skipped[0].message

    def test_secrets_detection_flag(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src_secrets_flag"
        source_dir.mkdir()
        create_yaml_file(source_dir, "secret_test.yaml", YAML_WITH_SECRET_CONTENT)

        # With skip_secrets_detection=True
        config_skip = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir, skip_secrets_detection=True)
        cards_skipped, errors_skipped = load_and_process_flashcard_yamls(config_skip)
        assert len(cards_skipped) == 2 # Both cards should be processed
        assert not errors_skipped

    def test_comprehensive_file_processing(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        create_yaml_file(source_dir, "comp.yaml", VALID_YAML_COMPREHENSIVE_CONTENT)
        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)
        
        assert not errors
        assert len(cards) == 2
        
        card1 = next(c for c in cards if c.uuid == uuid.UUID("7b7c2d57-e7b8-4e1f-a793-0ee2a61bcb79"))
        card2 = next(c for c in cards if c.uuid != uuid.UUID("7b7c2d57-e7b8-4e1f-a793-0ee2a61bcb79"))

        assert card1.deck_name == "Comprehensive::SubDeck"
        assert "deck-tag" in card1.tags and "card-tag1" in card1.tags
        assert card1.media == [Path("image.png"), Path("subfolder/audio.mp3")]
        
        assert card2.deck_name == "Comprehensive::SubDeck"
        assert "another-deck-tag" in card2.tags and "card-tag2" in card2.tags

    def test_intra_file_duplicate_question(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        create_yaml_file(source_dir, "intradup.yaml", YAML_WITH_INTRA_FILE_DUPLICATE_Q_CONTENT)
        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)

        assert len(cards) == 2 # The two unique cards
        assert len(errors) == 1
        assert "Duplicate question front within this YAML file" in errors[0].message
        assert errors[0].card_index == 2
        assert errors[0].file_path.name == "intradup.yaml"

    def test_invalid_schema_no_deck(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        create_yaml_file(source_dir, "no_deck.yaml", INVALID_YAML_SCHEMA_NO_DECK_CONTENT)
        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)

        assert not cards
        assert len(errors) == 1
        assert "Missing 'deck' field at top level" in errors[0].message

    def test_non_existent_source_dir(self, tmp_path: Path, assets_dir: Path):
        source_dir = tmp_path / "non_existent_src"
        # Do not create source_dir
        config = YAMLProcessorConfig(source_directory=source_dir, assets_root_directory=assets_dir)
        cards, errors = load_and_process_flashcard_yamls(config)
        assert not cards
        assert len(errors) == 1
        assert "Source directory does not exist" in errors[0].message
    
    def test_non_existent_assets_dir_no_skip(self, tmp_path: Path):
        source_dir = tmp_path / "src_for_no_assets"
        source_dir.mkdir()
        create_yaml_file(source_dir, "deck.yaml", VALID_YAML_MINIMAL_CONTENT) # No media needed for this test
        
        non_existent_assets_dir = tmp_path / "non_existent_assets"
        # Do not create non_existent_assets_dir
        
        config = YAMLProcessorConfig(
            source_directory=source_dir, 
            assets_root_directory=non_existent_assets_dir, 
            skip_media_validation=False
        )
        cards, errors = load_and_process_flashcard_yamls(config)
        assert not cards # No cards should be processed if assets_root is invalid and not skipping
        assert len(errors) == 1
        assert "Assets root directory does not exist" in errors[0].message

    def test_non_existent_assets_dir_with_skip(self, tmp_path: Path):
        source_dir = tmp_path / "src_for_no_assets_skip"
        source_dir.mkdir()
        yaml_with_media = """
deck: MediaTest
cards:
  - q: Q
    a: A
    media: [image.png] # This path won't be checked for existence
"""
        create_yaml_file(source_dir, "deck_media.yaml", yaml_with_media)
        
        non_existent_assets_dir = tmp_path / "non_existent_assets_skip"
        
        config = YAMLProcessorConfig(
            source_directory=source_dir,
            assets_root_directory=non_existent_assets_dir,
            skip_media_validation=True
        )
        cards, errors = load_and_process_flashcard_yamls(config)
        assert len(cards) == 1 # Card should be processed as media file existence is skipped
        assert not errors # No error due to non-existent assets dir when skipping validation
        assert cards[0].media == [Path("image.png")]

