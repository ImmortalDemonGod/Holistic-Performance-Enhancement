import json
import pytest
from jsonschema import validate, ValidationError
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent.parent.parent / "cultivation/systems/knowledge/schemas/paper.schema.json"

# Minimal valid metadata example
def valid_metadata():
    return {
        "arxiv_id": "1234.56789",
        "title": "A Valid Paper Title",
        "authors": ["Alice", "Bob"],
        "imported_at": "2025-05-11T19:00:00Z"
    }

# Invalid: missing required field 'title'
def invalid_metadata_missing_title():
    return {
        "arxiv_id": "1234.56789",
        "authors": ["Alice", "Bob"],
        "imported_at": "2025-05-11T19:00:00Z"
    }

# Invalid: wrong type for 'authors'
def invalid_metadata_authors_type():
    return {
        "arxiv_id": "1234.56789",
        "title": "A Valid Paper Title",
        "authors": "Alice",
        "imported_at": "2025-05-11T19:00:00Z"
    }

@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def test_valid_metadata_passes(schema):
    validate(instance=valid_metadata(), schema=schema)

def test_missing_title_fails(schema):
    with pytest.raises(ValidationError):
        validate(instance=invalid_metadata_missing_title(), schema=schema)

def test_authors_wrong_type_fails(schema):
    with pytest.raises(ValidationError):
        validate(instance=invalid_metadata_authors_type(), schema=schema)
