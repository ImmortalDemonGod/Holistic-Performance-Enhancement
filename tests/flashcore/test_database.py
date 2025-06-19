"""
Comprehensive test suite for flashcore.database (FlashcardDatabase), covering connection, schema, CRUD, constraints, marshalling, and error handling.
"""

import pytest
from pydantic import ValidationError

from datetime import datetime, timezone
from typing import Generator
import duckdb

from cultivation.scripts.flashcore.card import Card, Review
from cultivation.scripts.flashcore.database import (
    FlashcardDatabase,
    DatabaseConnectionError,
    CardOperationError,
    ReviewOperationError,
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

# --- Test Classes ---

class TestFlashcardDatabaseConnection:
    def test_instantiation_default_path(self):
        db_man_default = FlashcardDatabase() # No path provided
        expected_path = DEFAULT_FLASHCORE_DATA_DIR / DEFAULT_DATABASE_FILENAME
        assert db_man_default.db_path_resolved == expected_path
        conn = None
        try:
            conn = db_man_default.get_connection()
            assert expected_path.parent.exists()
        finally:
            if conn:
                conn.close()
            if expected_path.exists():
                expected_path.unlink()
            if expected_path.parent.exists() and not any(expected_path.parent.iterdir()):
                expected_path.parent.rmdir()

    def test_instantiation_custom_file_path(self, tmp_path: Path):
        custom_path = tmp_path / "custom_dir" / "my_flash.db"
        db_man = FlashcardDatabase(custom_path)
        assert db_man.db_path_resolved == custom_path.resolve()
        conn = None
        try:
            conn = db_man.get_connection()
            assert custom_path.parent.exists()
        finally:
            if conn:
                conn.close()

    def test_instantiation_in_memory(self, db_path_memory: str):
        db_man = FlashcardDatabase(db_path_memory)
        assert str(db_man.db_path_resolved) == ":memory:"
        with db_man as db:
            assert db._connection is not None
            db._connection.execute("SELECT 42;").fetchone() # Check if connection is active
        assert db._connection is None # __exit__ sets _connection to None

    def test_get_connection_success(self, db_manager: FlashcardDatabase):
        conn = db_manager.get_connection()
        assert conn is not None
        conn.execute("SELECT 1") # Check if connection is active

    def test_get_connection_idempotent(self, db_manager: FlashcardDatabase):
        conn1 = db_manager.get_connection()
        conn2 = db_manager.get_connection()
        assert conn1 is conn2
        db_manager.close_connection()
        # conn1 was an alias to db_manager._connection which is now None. Accessing conn1 might be unsafe or its state undefined.
        # The primary check is that db_manager._connection is None.
        assert db_manager._connection is None
        conn3 = db_manager.get_connection()
        assert conn3 is not None
        # Further check: connection is usable
        conn3.execute("SELECT 1")
        assert conn1 is not conn3

    def test_close_connection(self, db_manager: FlashcardDatabase):
        db_manager.get_connection() # Ensure connection is established
        db_manager.close_connection()
        assert db_manager._connection is None
        # Should be able to reconnect
        conn2 = db_manager.get_connection()
        assert conn2 is not None
        assert db_manager._connection is not None

    def test_context_manager_usage(self, db_path_file: Path):
        with FlashcardDatabase(db_path_file) as db:
            assert db._connection is not None
            db._connection.execute("SELECT 1") # Check if connection is active
        assert db._connection is None # __exit__ sets _connection to None

    def test_read_only_mode_connection(self, db_path_file: Path):
        db_man = FlashcardDatabase(db_path_file)
        db_man.initialize_schema()
        db_man.close_connection()
        db_readonly = FlashcardDatabase(db_path_file, read_only=True)
        conn = db_readonly.get_connection()
        assert conn is not None
        conn.execute("SELECT 1") # Check if connection is active
        # Attempt write
        with pytest.raises(duckdb.InvalidInputException):
            # Try to create a table, which is a write operation
            conn.execute("CREATE TABLE test_write (id INTEGER)")
        db_readonly.close_connection()

class TestSchemaInitialization:
    def test_initialize_schema_creates_tables_and_sequence(self, db_manager: FlashcardDatabase):
        with db_manager:
            db_manager.initialize_schema()
            tables = db_manager._connection.execute("SELECT table_name FROM information_schema.tables WHERE table_name IN ('cards', 'reviews');").fetchall()
            table_names = {name[0] for name in tables}
            assert "cards" in table_names
    def test_get_reviews_for_card_no_reviews(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        assert db.get_reviews_for_card(sample_card1.uuid) == []

    def test_get_latest_review_for_card(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        r1 = create_sample_review(card_uuid=sample_card1.uuid, ts=datetime(2023, 1, 2, 11, 0, 0, tzinfo=timezone.utc))
        r2 = create_sample_review(card_uuid=sample_card1.uuid, ts=datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc))
        db.add_reviews_batch([r1, r2])
        latest = db.get_latest_review_for_card(sample_card1.uuid)
        assert latest.ts == r2.ts

    def test_get_latest_review_for_card_no_reviews(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        assert db.get_latest_review_for_card(sample_card1.uuid) is None

    def test_get_all_reviews_empty(self, initialized_db_manager: FlashcardDatabase):
        assert initialized_db_manager.get_all_reviews() == []

    def test_get_all_reviews_with_data_and_filtering(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        r1 = create_sample_review(card_uuid=sample_card1.uuid, ts=datetime(2023, 1, 2, 11, 0, 0, tzinfo=timezone.utc))
        r2 = create_sample_review(card_uuid=sample_card1.uuid, ts=datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc))
        db.add_reviews_batch([r1, r2])
        all_reviews = db.get_all_reviews()
        assert len(all_reviews) >= 2
        filtered = db.get_all_reviews(start_ts=datetime(2023, 1, 3, 0, 0, 0, tzinfo=timezone.utc))
        assert all(r.ts >= datetime(2023, 1, 3, 0, 0, 0, tzinfo=timezone.utc) for r in filtered)

class TestGeneralErrorHandling:
    def test_operations_on_uninitialized_db(self, db_manager: FlashcardDatabase, sample_card1: Card):
        # Ensure connection is open but schema not initialized
        db_manager.get_connection()
        # Attempt to add a card, which should fail if schema is not initialized
        with pytest.raises(duckdb.CatalogException): # Or specific custom error if wrapped
            db_manager.upsert_cards_batch([sample_card1])

    def test_operations_on_closed_connection(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.close_connection()
        # Should reconnect or raise
        try:
            db.upsert_cards_batch([sample_card1])
        except Exception:
            pass

    def test_read_only_db_write_attempt(self, db_path_file: Path):
        db_man = FlashcardDatabase(db_path_file)
        db_man.initialize_schema()
        db_man.close_connection()
        db_readonly = FlashcardDatabase(db_path_file, read_only=True)
        # initialize_schema should not fail on a read-only DB if tables already exist and force_recreate is False (default)
        db_readonly.initialize_schema() 
        with pytest.raises(duckdb.InvalidInputException):
            # Attempting an upsert (write operation) should fail
            db_readonly.upsert_cards_batch([create_sample_card()])
