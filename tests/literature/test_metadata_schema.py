import json
import pytest
from jsonschema import validate, ValidationError
from pathlib import Path
import json

SCHEMA_PATH = Path(__file__).parent.parent.parent / "cultivation/schemas/paper.schema.json"

# Minimal valid metadata example
def valid_metadata():
    """
    Returns a minimal valid metadata dictionary for a paper.
    
    The dictionary includes required fields: "arxiv_id", "title", "authors" (as a list), and "imported_at" (ISO 8601 datetime string).
    """
    return {
        "arxiv_id": "1234.56789",
        "title": "A Valid Paper Title",
        "authors": ["Alice", "Bob"],
        "imported_at": "2025-05-11T19:00:00Z"
    }

# Invalid: missing required field 'title'
def invalid_metadata_missing_title():
    """
    Returns a metadata dictionary missing the required "title" field.
    
    This example is intended for testing schema validation failures due to missing required fields.
    """
    return {
        "arxiv_id": "1234.56789",
        "authors": ["Alice", "Bob"],
        "imported_at": "2025-05-11T19:00:00Z"
    }

# Invalid: wrong type for 'authors'
def invalid_metadata_authors_type():
    """
    Returns a metadata dictionary with the 'authors' field incorrectly set as a string.
    
    This example is intended to test schema validation for type errors in the 'authors' field.
    """
    return {
        "arxiv_id": "1234.56789",
        "title": "A Valid Paper Title",
        "authors": "Alice",
        "imported_at": "2025-05-11T19:00:00Z"
    }

@pytest.fixture(scope="module")
def schema():
    """
    Loads and returns the paper metadata JSON schema from the predefined file path.
    
    Returns:
        dict: The loaded JSON schema as a dictionary.
    """
    with open(SCHEMA_PATH) as f:
        return json.load(f)

def test_valid_metadata_passes(schema):
    """
    Tests that valid metadata passes schema validation without raising an exception.
    """
    validate(instance=valid_metadata(), schema=schema)

def test_missing_title_fails(schema):
    """
    Tests that metadata missing the 'title' field fails schema validation.
    
    Asserts that a ValidationError is raised when validating metadata without the required 'title' key.
    """
    with pytest.raises(ValidationError):
        validate(instance=invalid_metadata_missing_title(), schema=schema)

def test_authors_wrong_type_fails(schema):
    """
    Tests that metadata with an incorrectly typed 'authors' field fails schema validation.
    
    Asserts that a ValidationError is raised when the 'authors' field is a string instead of a list.
    """
    with pytest.raises(ValidationError):
        validate(instance=invalid_metadata_authors_type(), schema=schema)
