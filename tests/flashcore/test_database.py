"""
Comprehensive test suite for flashcore.database (FlashcardDatabase), covering connection, schema, CRUD, constraints, marshalling, and error handling.
"""

import pytest
import uuid
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
from typing import List, Dict, Any
import duckdb

try:
    from cultivation.scripts.flashcore.card import Card, Review
    from cultivation.scripts.flashcore.database import (
        FlashcardDatabase,
        DatabaseConnectionError,
        SchemaInitializationError,
        CardOperationError,
        ReviewOperationError,
        RecordNotFoundError,
        DEFAULT_FLASHCORE_DATA_DIR,
        DEFAULT_DATABASE_FILENAME
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))
    from flashcore.card import Card, Review
    from flashcore.database import (
        FlashcardDatabase,
        DatabaseConnectionError,
        SchemaInitializationError,
        CardOperationError,
        ReviewOperationError,
        RecordNotFoundError,
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
def db_manager(request, db_path_memory: str, db_path_file: Path) -> FlashcardDatabase:
    if request.param == "memory":
        db_man = FlashcardDatabase(db_path_memory)
    else:
        db_man = FlashcardDatabase(db_path_file)
    yield db_man
    if db_man._connection and not db_man._connection.closed:
        db_man.close_connection()

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
        tags={"tag3"},
        added_at=datetime(2023, 1, 3, 10, 0, 0, tzinfo=timezone.utc),
        source_yaml_file=Path("source/deck_b.yaml")
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
        media_paths=[Path("media1.png"), Path("media2.mp3")],
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
            if conn: conn.close()
            if expected_path.exists(): expected_path.unlink()
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
            if conn: conn.close()

    def test_instantiation_in_memory(self, db_path_memory: str):
        db_man = FlashcardDatabase(db_path_memory)
        assert str(db_man.db_path_resolved) == ":memory:"
        with db_man as db:
            assert db._connection is not None and not db._connection.closed
            db._connection.execute("SELECT 42;").fetchone()
        assert db._connection is None or db._connection.closed

    def test_get_connection_success(self, db_manager: FlashcardDatabase):
        conn = db_manager.get_connection()
        assert conn is not None and not conn.closed

    def test_get_connection_idempotent(self, db_manager: FlashcardDatabase):
        conn1 = db_manager.get_connection()
        conn2 = db_manager.get_connection()
        assert conn1 is conn2
        db_manager.close_connection()
        assert conn1.closed
        assert db_manager._connection is None
        conn3 = db_manager.get_connection()
        assert conn3 is not None and not conn3.closed
        assert conn1 is not conn3

    def test_close_connection(self, db_manager: FlashcardDatabase):
        conn = db_manager.get_connection()
        db_manager.close_connection()
        assert conn.closed
        assert db_manager._connection is None
        # Should be able to reconnect
        conn2 = db_manager.get_connection()
        assert conn2 is not None and not conn2.closed

    def test_context_manager_usage(self, db_path_file: Path):
        with FlashcardDatabase(db_path_file) as db:
            assert db._connection is not None and not db._connection.closed
        assert db._connection is None or db._connection.closed

    def test_read_only_mode_connection(self, db_path_file: Path):
        db_man = FlashcardDatabase(db_path_file)
        db_man.initialize_schema()
        db_man.close_connection()
        db_readonly = FlashcardDatabase(db_path_file, read_only=True)
        conn = db_readonly.get_connection()
        assert conn is not None and not conn.closed
        # Attempt write
        with pytest.raises((DatabaseConnectionError, duckdb.IOException, duckdb.ReadOnlyException, Exception)):
            db_readonly.upsert_cards_batch([create_sample_card()])

class TestSchemaInitialization:
    def test_initialize_schema_creates_tables_and_sequence(self, db_manager: FlashcardDatabase):
        with db_manager:
            db_manager.initialize_schema()
            tables = db_manager._connection.execute("SELECT table_name FROM information_schema.tables WHERE table_name IN ('cards', 'reviews');").fetchall()
            table_names = {name[0] for name in tables}
            assert "cards" in table_names
            assert "reviews" in table_names
            seq = db_manager._connection.execute("SELECT sequence_name FROM information_schema.sequences WHERE sequence_name='review_seq';").fetchone()
            assert seq is not None

    def test_initialize_schema_idempotent(self, db_manager: FlashcardDatabase, sample_card1: Card):
        with db_manager:
            db_manager.initialize_schema()
            db_manager.upsert_cards_batch([sample_card1])
            db_manager.initialize_schema()
            retrieved_card = db_manager.get_card_by_uuid(sample_card1.uuid)
            assert retrieved_card is not None
            assert retrieved_card.uuid == sample_card1.uuid

    def test_initialize_schema_force_recreate(self, db_manager: FlashcardDatabase, sample_card1: Card):
        with db_manager:
            db_manager.initialize_schema()
            db_manager.upsert_cards_batch([sample_card1])
            assert db_manager.get_card_by_uuid(sample_card1.uuid) is not None
            db_manager.initialize_schema(force_recreate_tables=True)
            assert db_manager.get_card_by_uuid(sample_card1.uuid) is None

    def test_initialize_schema_on_readonly_db_fails_for_force_recreate(self, db_path_file: Path):
        db_man = FlashcardDatabase(db_path_file)
        db_man.initialize_schema()
        db_man.close_connection()
        db_readonly = FlashcardDatabase(db_path_file, read_only=True)
        with pytest.raises(DatabaseConnectionError):
            db_readonly.initialize_schema(force_recreate_tables=True)

    def test_schema_constraints_rating(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        invalid_review = create_sample_review(card_uuid=sample_card1.uuid, rating=5)
        with pytest.raises(ReviewOperationError):
            db.add_review(invalid_review)

    def test_schema_constraints_fk_cascade_delete(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        review = create_sample_review(card_uuid=sample_card1.uuid)
        db.add_review(review)
        assert len(db.get_reviews_for_card(sample_card1.uuid)) == 1
        db.delete_cards_by_uuids_batch([sample_card1.uuid])
        assert db.get_card_by_uuid(sample_card1.uuid) is None
        assert len(db.get_reviews_for_card(sample_card1.uuid)) == 0

class TestCardOperations:
    def test_upsert_cards_batch_insert_new(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card, sample_card2: Card):
        db = initialized_db_manager
        cards_to_insert = [sample_card1, sample_card2]
        affected_rows = db.upsert_cards_batch(cards_to_insert)
        assert db.get_card_by_uuid(sample_card1.uuid) is not None
        assert db.get_card_by_uuid(sample_card2.uuid) is not None
        retrieved_c1 = db.get_card_by_uuid(sample_card1.uuid)
        assert retrieved_c1.added_at == sample_card1.added_at

    def test_upsert_cards_batch_update_existing(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        original_added_at = db.get_card_by_uuid(sample_card1.uuid).added_at
        updated_card1 = sample_card1.model_copy(update={"front": "Updated Card 1 Front", "tags": {"new-tag"}})
        updated_card1.added_at = datetime.now(timezone.utc) + timedelta(days=1)
        db.upsert_cards_batch([updated_card1])
        retrieved_updated_card = db.get_card_by_uuid(sample_card1.uuid)
        assert retrieved_updated_card is not None
        assert retrieved_updated_card.front == "Updated Card 1 Front"
        assert retrieved_updated_card.tags == {"new-tag"}
        assert retrieved_updated_card.added_at == original_added_at

    def test_upsert_cards_batch_mixed_insert_update(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card, sample_card2: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        updated_card1 = sample_card1.model_copy(update={"front": "Mixed Updated", "tags": {"mixed"}})
        new_card = create_sample_card()
        affected = db.upsert_cards_batch([updated_card1, new_card, sample_card2])
        assert db.get_card_by_uuid(updated_card1.uuid).front == "Mixed Updated"
        assert db.get_card_by_uuid(new_card.uuid) is not None
        assert db.get_card_by_uuid(sample_card2.uuid) is not None

    def test_upsert_cards_batch_empty_list(self, initialized_db_manager: FlashcardDatabase):
        affected = initialized_db_manager.upsert_cards_batch([])
        assert affected == 0

    def test_upsert_cards_batch_data_marshalling(self, initialized_db_manager: FlashcardDatabase):
        db = initialized_db_manager
        card = create_sample_card(tags={"tagX"}, media_paths=[Path("foo.png")], source_yaml_file=Path("foo.yaml"))
        db.upsert_cards_batch([card])
        retrieved = db.get_card_by_uuid(card.uuid)
        assert isinstance(retrieved.tags, set) and "tagX" in retrieved.tags
        assert isinstance(retrieved.media_paths, list) and Path("foo.png") in retrieved.media_paths
        assert isinstance(retrieved.source_yaml_file, Path) and retrieved.source_yaml_file.name == "foo.yaml"

    def test_upsert_cards_batch_transaction_rollback(self, db_manager: FlashcardDatabase):
        db = db_manager
        db.initialize_schema()
        valid_card = create_sample_card()
        # Intentionally break a card (deck_name=None violates NOT NULL)
        broken_card = create_sample_card(deck_name=None)
        with pytest.raises(Exception):
            db.upsert_cards_batch([valid_card, broken_card])
        # Neither should be present
        assert db.get_card_by_uuid(valid_card.uuid) is None

    def test_get_card_by_uuid_exists(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        card = db.get_card_by_uuid(sample_card1.uuid)
        assert card is not None
        assert card.uuid == sample_card1.uuid

    def test_get_card_by_uuid_not_exists(self, initialized_db_manager: FlashcardDatabase):
        assert initialized_db_manager.get_card_by_uuid(uuid.uuid4()) is None

    def test_get_all_cards_empty_db(self, initialized_db_manager: FlashcardDatabase):
        assert initialized_db_manager.get_all_cards() == []

    def test_get_all_cards_multiple_cards(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card, sample_card2: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1, sample_card2])
        cards = db.get_all_cards()
        assert len(cards) >= 2
        uuids = {c.uuid for c in cards}
        assert sample_card1.uuid in uuids
        assert sample_card2.uuid in uuids

    def test_get_all_cards_with_deck_filter(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card, sample_card2: Card, sample_card3_deck_b: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1, sample_card2, sample_card3_deck_b])
        deck_a_cards = db.get_all_cards(deck_name_filter="Deck A::%")
        assert len(deck_a_cards) == 2
        deck_b_cards = db.get_all_cards(deck_name_filter="Deck B")
        assert len(deck_b_cards) == 1
        assert sample_card3_deck_b.uuid == deck_b_cards[0].uuid

    def test_get_due_cards_logic(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card, sample_card2: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1, sample_card2])
        today = date.today()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        due_cards = db.get_due_cards(on_date=today)
        assert len(due_cards) == 2
        # Review card1, make it due tomorrow
        review_c1_not_due = create_sample_review(card_uuid=sample_card1.uuid, next_due=tomorrow)
        db.add_review(review_c1_not_due)
        due_cards_after_review1 = db.get_due_cards(on_date=today)
        assert len(due_cards_after_review1) == 1
        assert due_cards_after_review1[0].uuid == sample_card2.uuid
        # Review card2, make it due yesterday
        review_c2_due_yesterday = create_sample_review(card_uuid=sample_card2.uuid, next_due=yesterday, ts=datetime.combine(yesterday - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
        db.add_review(review_c2_due_yesterday)
        due_cards_final = db.get_due_cards(on_date=today, limit=10)
        assert len(due_cards_final) == 1
        assert due_cards_final[0].uuid == sample_card2.uuid

    def test_delete_cards_by_uuids_batch(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        review = create_sample_review(card_uuid=sample_card1.uuid)
        db.add_review(review)
        count = db.delete_cards_by_uuids_batch([sample_card1.uuid])
        assert count == 1
        assert db.get_card_by_uuid(sample_card1.uuid) is None
        assert db.get_reviews_for_card(sample_card1.uuid) == []

    def test_delete_cards_by_uuids_batch_non_existent(self, initialized_db_manager: FlashcardDatabase):
        count = initialized_db_manager.delete_cards_by_uuids_batch([uuid.uuid4()])
        assert count == 0

    def test_get_all_card_fronts_and_uuids(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        card_variant_front = sample_card1.model_copy(update={"uuid": uuid.uuid4(), "front": sample_card1.front.lower()})
        db.upsert_cards_batch([card_variant_front])
        front_map = db.get_all_card_fronts_and_uuids()
        normalized_front1 = " ".join(sample_card1.front.lower().split())
        assert normalized_front1 in front_map
        assert front_map[normalized_front1] in [sample_card1.uuid, card_variant_front.uuid]
        assert len(front_map) == 1

class TestReviewOperations:
    def test_add_review_success(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        review = create_sample_review(card_uuid=sample_card1.uuid)
        review_id = db.add_review(review)
        assert isinstance(review_id, int) and review_id > 0
        retrieved_reviews = db.get_reviews_for_card(sample_card1.uuid)
        assert len(retrieved_reviews) == 1
        assert retrieved_reviews[0].review_id == review_id
        assert retrieved_reviews[0].rating == review.rating

    def test_add_review_fk_violation(self, initialized_db_manager: FlashcardDatabase):
        db = initialized_db_manager
        bad_review = create_sample_review(card_uuid=uuid.uuid4())
        with pytest.raises(ReviewOperationError):
            db.add_review(bad_review)

    def test_add_review_check_constraint_violation(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        bad_review = create_sample_review(card_uuid=sample_card1.uuid, rating=99)
        with pytest.raises(ReviewOperationError):
            db.add_review(bad_review)

    def test_add_reviews_batch_success(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card, sample_card2: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1, sample_card2])
        r1 = create_sample_review(card_uuid=sample_card1.uuid)
        r2 = create_sample_review(card_uuid=sample_card2.uuid)
        db.add_reviews_batch([r1, r2])
        reviews1 = db.get_reviews_for_card(sample_card1.uuid)
        reviews2 = db.get_reviews_for_card(sample_card2.uuid)
        assert len(reviews1) == 1
        assert len(reviews2) == 1

    def test_add_reviews_batch_partial_failure_transaction(self, db_manager: FlashcardDatabase):
        db = db_manager
        db.initialize_schema()
        card = create_sample_card()
        db.upsert_cards_batch([card])
        valid_review = create_sample_review(card_uuid=card.uuid)
        bad_review = create_sample_review(card_uuid=card.uuid, rating=999)
        with pytest.raises(Exception):
            db.add_reviews_batch([valid_review, bad_review])
        assert db.get_reviews_for_card(card.uuid) == []

    def test_get_reviews_for_card(self, initialized_db_manager: FlashcardDatabase, sample_card1: Card):
        db = initialized_db_manager
        db.upsert_cards_batch([sample_card1])
        r1 = create_sample_review(card_uuid=sample_card1.uuid, ts=datetime(2023, 1, 2, 11, 0, 0, tzinfo=timezone.utc))
        r2 = create_sample_review(card_uuid=sample_card1.uuid, ts=datetime(2023, 1, 3, 12, 0, 0, tzinfo=timezone.utc))
        db.add_reviews_batch([r1, r2])
        reviews_asc = db.get_reviews_for_card(sample_card1.uuid, order_by_ts_desc=False)
        reviews_desc = db.get_reviews_for_card(sample_card1.uuid, order_by_ts_desc=True)
        assert reviews_asc[0].ts < reviews_asc[1].ts
        assert reviews_desc[0].ts > reviews_desc[1].ts

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
    def test_operations_on_uninitialized_db(self, db_manager: FlashcardDatabase):
        # No initialize_schema called
        with pytest.raises(Exception):
            db_manager.upsert_cards_batch([create_sample_card()])

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
        db_readonly.initialize_schema()
        with pytest.raises((DatabaseConnectionError, duckdb.IOException, duckdb.ReadOnlyException, Exception)):
            db_readonly.upsert_cards_batch([create_sample_card()])
