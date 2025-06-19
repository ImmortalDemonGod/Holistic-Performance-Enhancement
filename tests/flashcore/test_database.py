"""
Minimal test file for flashcore.database to debug pytest silent failures.
(Step 2: Re-adding original imports)
"""

import pytest
import uuid
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Generator
import duckdb

from cultivation.scripts.flashcore.card import Card, Review
from cultivation.scripts.flashcore.database import (
    CardOperationError,
    DatabaseConnectionError,
    FlashcardDatabase,
    DEFAULT_FLASHCORE_DATA_DIR,
    DEFAULT_DATABASE_FILENAME
)


# --- Fixtures ---
@pytest.fixture
def db_path_memory() -> str:
    return ":memory:"

@pytest.fixture
def db_path_file(tmp_path: Path) -> Path:
    return tmp_path / "test_flash.db"

@pytest.fixture(params=["memory", "file"])
def db_manager(request, db_path_memory: str, db_path_file: Path) -> Generator[FlashcardDatabase, None, None]:
    if request.param == "memory":
        db_man = FlashcardDatabase(db_path_memory)
    else:
        db_man = FlashcardDatabase(db_path_file)
    try:
        yield db_man
    finally:
        db_man.close_connection()
        if request.param == "file" and db_path_file.exists():
            try:
                db_path_file.unlink()
            except Exception as e:
                print(f"Error removing temporary DB file in test fixture teardown: {e}")

@pytest.fixture
def initialized_db_manager(db_manager: FlashcardDatabase) -> FlashcardDatabase:
    db_manager.initialize_schema()
    return db_manager

@pytest.fixture
def sample_card1() -> Card:
    return Card(
        uuid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        deck_name="Deck A::Sub1",
        front="Card 1 Front",
        back="Card 1 Back",
        tags={"tag1", "common-tag"},
        added_at=datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        source_yaml_file=Path("source/deck_a.yaml")
    )

@pytest.fixture
def sample_card2() -> Card:
    return Card(
        uuid=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        deck_name="Deck A::Sub2",
        front="Card 2 Front",
        back="Card 2 Back",
        tags={"tag2", "common-tag"},
        added_at=datetime(2023, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
        source_yaml_file=Path("source/deck_a.yaml")
    )

@pytest.fixture
def sample_card3_deck_b() -> Card:
    return Card(
        uuid=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        deck_name="Deck B",
        front="Card 3 DeckB Front",
        back="Card 3 DeckB Back",
        tags={"default-tag"},
        media=None,
        source_yaml_file=Path("source/deck_b.yaml"),
        added_at=datetime(2023, 1, 3, 10, 0, 0, tzinfo=timezone.utc)
    )

@pytest.fixture
def sample_review1(sample_card1: Card) -> Review:
    return Review(
        card_uuid=sample_card1.uuid,
        ts=datetime(2023, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
        rating=3,
        stab_before=1.0, stab_after=2.5, diff=5.0,
        next_due=date(2023, 1, 8),
        elapsed_days_at_review=0, scheduled_days_interval=3
    )

@pytest.fixture
def sample_review2_for_card1(sample_card1: Card) -> Review:
    return Review(
        card_uuid=sample_card1.uuid,
        ts=datetime(2023, 1, 8, 13, 0, 0, tzinfo=timezone.utc),
        rating=2,
        stab_before=2.5, stab_after=6.0, diff=4.8,
        next_due=date(2023, 1, 14),
        elapsed_days_at_review=3, scheduled_days_interval=6
    )

# --- Sample Data Generators ---

def create_sample_card(**overrides) -> Card:
    data = dict(
        uuid=uuid.uuid4(),
        deck_name="Deck A::Sub1",
        front="Sample Front",
        back="Sample Back",
        tags={"tag1", "tag2"},
        added_at=datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        source_yaml_file=Path("source/deck_a.yaml"),
        origin_task="test-task",
        media=[Path("media1.png"), Path("media2.mp3")],
        internal_note="test note"
    )
    data.update(overrides)
    return Card(**data)

def create_sample_review(card_uuid, **overrides) -> Review:
    data = dict(
        card_uuid=card_uuid,
        ts=datetime(2023, 1, 2, 11, 0, 0, tzinfo=timezone.utc),
        rating=2,
        stab_before=1.0,
        stab_after=2.0,
        diff=4.0,
        next_due=date(2023, 1, 5),
        elapsed_days_at_review=0,
        scheduled_days_interval=3,
        resp_ms=1234,
        review_type="learn"
    )
    data.update(overrides)
    return Review(**data)


class TestFlashcardDatabaseConnection:
    def test_connection_context_manager_memory(self, db_path_memory: str):
        with FlashcardDatabase(db_path_memory) as db:
            assert db._connection is not None
            # Execute a simple query to ensure connection is live
            db._connection.execute("SELECT 1")
        # After exiting context, connection should be closed
        # Check if attempting a query raises an error indicative of a closed connection
        with pytest.raises(duckdb.ConnectionException) as excinfo:
            db._connection.execute("SELECT 1")
        assert "connection already closed" in str(excinfo.value).lower()


    def test_connection_context_manager_file(self, db_path_file: Path):
        with FlashcardDatabase(db_path_file) as db:
            assert db._connection is not None
            db._connection.execute("SELECT 1")
        with pytest.raises(duckdb.ConnectionException) as excinfo:
            db._connection.execute("SELECT 1")
        assert "connection already closed" in str(excinfo.value).lower()


    def test_explicit_close_memory(self, db_path_memory: str):
        db = FlashcardDatabase(db_path_memory)
        # Lazily create connection and assert it's open
        conn = db.get_connection()
        assert conn is not None
        conn.execute("SELECT 1")

        # Explicitly close
        db.close_connection()

        # Verify that trying to get a new connection fails with our custom error
        with pytest.raises(DatabaseConnectionError) as excinfo:
            db.get_connection()
        assert "permanently closed" in str(excinfo.value).lower()

        # Verify that using the old connection object fails with duckdb's error
        with pytest.raises(duckdb.ConnectionException):
            conn.execute("SELECT 1")


    def test_explicit_close_file(self, db_path_file: Path):
        db = FlashcardDatabase(db_path_file)
        assert db._connection is not None
        db._connection.execute("SELECT 1")
        db.close_connection()
        with pytest.raises(duckdb.ProgrammingError) as excinfo:
            db._connection.execute("SELECT 1")
        assert "Connection has already been closed" in str(excinfo.value).lower()

    def test_read_only_mode_connection(self, tmp_path: Path):
        # First, create and populate a DB
        db_file = tmp_path / "test_readonly.db"
        with FlashcardDatabase(db_file) as db_write:
            db_write.initialize_schema()
            # Add a dummy card to ensure there's data
            card_data = create_sample_card(uuid=uuid.uuid4(), front="Test RO Card")
            db_write.upsert_cards_batch([card_data])

        # Now, open in read-only mode
        with FlashcardDatabase(db_file, read_only=True) as db_read:
            assert db_read._connection is not None
            # Verify we can read
            rows = db_read._connection.execute("SELECT COUNT(*) FROM cards").fetchall()
            assert rows[0][0] == 1
        # Connection should be closed after context manager
        with pytest.raises(duckdb.ProgrammingError) as excinfo:
            db_read._connection.execute("SELECT 1")
        assert "connection has already been closed" in str(excinfo.value).lower()

    def test_read_only_db_write_attempt(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        # This fixture db_manager might be read-write or in-memory.
        # We need to ensure we test a file-based DB opened in read-only.
        
        # Create a temporary file-based DB for this test
        temp_db_path = initialized_db_manager.db_path
        if initialized_db_manager.db_path == ":memory:":
            # If the fixture gave an in-memory DB, we need to create a file for this test
            # This part is tricky because initialized_db_manager might be memory or file.
            # For simplicity, let's assume we can force a file scenario or this test is more for file DBs.
            # A robust way: create a new file DB, initialize, close, then reopen RO.
            
            # Create a new temp file path for this specific test
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpfile:
                temp_db_path_for_ro = Path(tmpfile.name)
            
            # Initialize it
            with FlashcardDatabase(temp_db_path_for_ro) as db_setup:
                db_setup.initialize_schema()
                db_setup.upsert_cards_batch([sample_card1]) # Add some data
            
            # Now open it in read-only
            db_ro = FlashcardDatabase(temp_db_path_for_ro, read_only=True)
        
        else: # Fixture provided a file-based DB, close and reopen in RO
            initialized_db_manager.close_connection() # Close the potentially R/W connection
            db_ro = FlashcardDatabase(temp_db_path, read_only=True)

        try:
            with pytest.raises(duckdb.InvalidInputException) as excinfo: # DuckDB uses InvalidInputException for RO writes
                card_to_add = create_sample_card(uuid=uuid.uuid4(), front="New Card Front")
                db_ro.upsert_cards_batch([card_to_add])
            assert "Cannot execute write query in read-only mode!" in str(excinfo.value)
        finally:
            db_ro.close_connection()
            if 'temp_db_path_for_ro' in locals() and temp_db_path_for_ro.exists():
                temp_db_path_for_ro.unlink()

def test_sanity_check():
    assert True
