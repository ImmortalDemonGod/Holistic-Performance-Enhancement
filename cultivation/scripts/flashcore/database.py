"""
DuckDB database interactions for flashcore.
Implements the FlashcardDatabase class and supporting exceptions as per the v3.0 technical design.
"""

import duckdb
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Sequence, Union
from datetime import date, datetime, timezone
import logging
import pandas as pd
from pydantic import ValidationError

# Local project imports from previously defined modules
try:
    # Use a relative import for use as a package
    from .card import Card, Review, CardState
except ImportError:
    # Fallback for when script is run directly
    from card import Card, Review, CardState

logger = logging.getLogger(__name__)

DEFAULT_FLASHCORE_DATA_DIR = Path.home() / ".cultivation" / "flashcore_data"
DEFAULT_DATABASE_FILENAME = "flash.db"

# --- Custom Exceptions ---
class DatabaseError(Exception):
    """Base exception for all database operations in this module."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
    def __str__(self):
        if self.original_exception:
            return f"{super().__str__()} (Original exception: {type(self.original_exception).__name__}: {self.original_exception})"
        return super().__str__()

class DatabaseConnectionError(DatabaseError):
    pass
class SchemaInitializationError(DatabaseError):
    pass
class CardOperationError(DatabaseError):
    pass
class ReviewOperationError(DatabaseError):
    pass
class RecordNotFoundError(DatabaseError):
    pass

class FlashcardDatabase:
    """
    Manages database connection and all operations for the flashcard DuckDB database.
    Provides methods for schema initialization, and CRUD-like operations for Card and Review Pydantic models.
    """
    _DB_SCHEMA_SQL = [
        """
        CREATE TABLE IF NOT EXISTS cards (
            uuid                UUID PRIMARY KEY,
            deck_name           VARCHAR NOT NULL,
            front               VARCHAR NOT NULL,
            back                VARCHAR NOT NULL,
            tags                VARCHAR[],
            added_at            TIMESTAMP WITH TIME ZONE NOT NULL,
            modified_at         TIMESTAMP WITH TIME ZONE NOT NULL,
            last_review_id      BIGINT, -- FK to reviews.review_id, can be NULL
            next_due_date       DATE,   -- The next date the card is due for review
            state               VARCHAR, -- Current state: NEW, LEARNING, REVIEW, RELEARNING
            stability           DOUBLE,
            difficulty          DOUBLE,
            origin_task         VARCHAR,
            media_paths         VARCHAR[],
            source_yaml_file    VARCHAR,
            internal_note       VARCHAR
        );
        """,
        "CREATE SEQUENCE IF NOT EXISTS review_seq START 1;",
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id                   BIGINT DEFAULT nextval('review_seq') PRIMARY KEY,
            card_uuid                   UUID NOT NULL,
            ts                          TIMESTAMP WITH TIME ZONE NOT NULL,
            rating                      SMALLINT NOT NULL CHECK (rating >= 0 AND rating <= 3),
            resp_ms                     INTEGER CHECK (resp_ms IS NULL OR resp_ms >= 0),
            stab_before                 DOUBLE,
            stab_after                  DOUBLE NOT NULL CHECK (stab_after >= 0.1),
            diff                        DOUBLE NOT NULL,
            next_due                    DATE NOT NULL,
            elapsed_days_at_review      INTEGER NOT NULL CHECK (elapsed_days_at_review >= 0),
            scheduled_days_interval     INTEGER NOT NULL CHECK (scheduled_days_interval >= 1),
            review_type                 VARCHAR(50),
            FOREIGN KEY (card_uuid) REFERENCES cards(uuid)
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_cards_deck_name ON cards (deck_name);",
        "CREATE INDEX IF NOT EXISTS idx_reviews_card_uuid_ts ON reviews (card_uuid, ts DESC);",
        "CREATE INDEX IF NOT EXISTS idx_reviews_next_due ON reviews (next_due);"
    ]

    def __init__(self, db_path: Optional[Union[str, Path]] = None, read_only: bool = False):
        if db_path is None:
            self.db_path_resolved: Path = DEFAULT_FLASHCORE_DATA_DIR / DEFAULT_DATABASE_FILENAME
            logger.info(f"No DB path provided, using default: {self.db_path_resolved}")
        elif isinstance(db_path, str) and db_path.lower() == ":memory:":
            self.db_path_resolved: Path = Path(":memory:")
            logger.info("Using in-memory DuckDB database.")
        else:
            self.db_path_resolved: Path = Path(db_path).resolve()
            logger.info(f"FlashcardDatabase initialized for DB at: {self.db_path_resolved}")
        self.read_only: bool = read_only
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._is_closed: bool = False

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None or getattr(self._connection, 'closed', False):
            try:
                if str(self.db_path_resolved) != ":memory:":
                    self.db_path_resolved.parent.mkdir(parents=True, exist_ok=True)
                self._connection = duckdb.connect(
                    database=str(self.db_path_resolved),
                    read_only=self.read_only
                )
                logger.debug(f"Successfully connected to DuckDB at {self.db_path_resolved} (read_only={self.read_only}).")
            except duckdb.Error as e:
                logger.error(f"Failed to connect to DuckDB at {self.db_path_resolved}: {e}")
                raise DatabaseConnectionError(f"Failed to connect to database: {e}", original_exception=e)
        return self._connection

    def close_connection(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug(f"DuckDB connection to {self.db_path_resolved} closed.")

    def __enter__(self) -> 'FlashcardDatabase':
        self.get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_connection()

    def initialize_schema(self, force_recreate_tables: bool = False) -> None:
        """
        Initializes the database schema. Skips if in read-only mode unless it's an in-memory DB.
        Can force recreation of tables, which will delete all existing data.
        """
        if self._handle_read_only_initialization(force_recreate_tables):
            return

        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                if force_recreate_tables:
                    self._recreate_tables(cursor)
                
                self._create_schema_from_sql(cursor)
                
                cursor.commit()
            logger.info(f"Database schema at {self.db_path_resolved} initialized successfully (or already exists).")
        except duckdb.Error as e:
            self._handle_schema_initialization_error(conn, e)

    def _handle_read_only_initialization(self, force_recreate_tables: bool) -> bool:
        """Handles the logic for schema initialization in read-only mode. Returns True if initialization should be skipped."""
        if self.read_only:
            if force_recreate_tables:
                raise DatabaseConnectionError("Cannot force_recreate_tables in read-only mode.")
            # For a non-memory DB, it's just a warning. For in-memory, we proceed.
            if str(self.db_path_resolved) != ":memory:":
                logger.warning("Attempting to initialize schema in read-only mode. Skipping.")
                return True
        return False

    def _recreate_tables(self, cursor) -> None:
        """Drops all tables and sequences to force recreation."""
        logger.warning(f"Forcing table recreation for {self.db_path_resolved}. ALL EXISTING DATA WILL BE LOST.")
        cursor.execute("DROP TABLE IF EXISTS reviews CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS cards CASCADE;")
        cursor.execute("DROP SEQUENCE IF EXISTS review_seq;")

    def _create_schema_from_sql(self, cursor) -> None:
        """Executes the SQL statements to create the database schema."""
        for statement in self._DB_SCHEMA_SQL:
            cursor.execute(statement)

    def _handle_schema_initialization_error(self, conn, e: duckdb.Error) -> None:
        """Handles errors during schema initialization, including logging and transaction rollback."""
        logger.error(f"Error initializing database schema at {self.db_path_resolved}: {e}")
        if conn and not getattr(conn, 'closed', True):
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to schema initialization error.")
            except duckdb.Error as rb_err:
                logger.error(f"Failed to rollback transaction: {rb_err}")
        raise SchemaInitializationError(f"Failed to initialize schema: {e}", original_exception=e) from e

    # --- Data Marshalling Helpers (Internal) ---
    def _card_to_db_params_list(self, cards: Sequence['Card']) -> List[Tuple]:
        """Converts a sequence of Card models to a list of tuples for DB insertion."""
        return [
            (
                card.uuid,
                card.deck_name,
                card.front,
                card.back,
                list(card.tags) if card.tags else None,
                card.added_at,
                card.modified_at,
                card.last_review_id,
                card.next_due_date,
                card.state.name if card.state else None,
                card.stability,
                card.difficulty,
                card.origin_task,
                [str(p) for p in card.media] if card.media else None,
                str(card.source_yaml_file) if card.source_yaml_file else None,
                card.internal_note
            )
            for card in cards
        ]

    def _db_row_to_card(self, row_dict: Dict[str, Any]) -> 'Card':
        """
        Converts a database row dictionary to a Card Pydantic model.
        This method handles necessary type transformations from DB types to model types.
        """
        data = self._transform_db_row_for_card(row_dict)
        data = self._clean_pandas_null_values(data)
        
        try:
            return Card(**data)
        except ValidationError as e:
            logger.error(f"Pydantic validation failed while converting DB row to Card: {data}")
            raise CardOperationError(f"Data validation failed for card: {e}", original_exception=e) from e

    def _transform_db_row_for_card(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms raw DB data into a dictionary suitable for the Card model."""
        data = row_dict.copy()
        
        media_paths = data.pop("media_paths", None)
        data["media"] = [Path(p) for p in media_paths] if media_paths is not None else []

        if data.get("source_yaml_file"):
            data["source_yaml_file"] = Path(data["source_yaml_file"])

        tags_val = data.get("tags")
        data["tags"] = set(tags_val) if tags_val is not None else set()

        state_val = data.pop("state", None)
        if state_val:
            data["state"] = CardState[state_val]
            
        return data

    def _clean_pandas_null_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts pandas-specific nulls (e.g., NaT, NaN) to Python None."""
        for key, value in data.items():
            if not isinstance(value, (list, set)) and pd.isna(value):
                data[key] = None
        return data

    def _review_to_db_params_tuple(self, review: 'Review') -> Tuple:
        return (
            review.card_uuid,
            review.ts,
            review.rating,
            review.resp_ms,
            review.stab_before,
            review.stab_after,
            review.diff,
            review.next_due,
            review.elapsed_days_at_review,
            review.scheduled_days_interval,
            review.review_type
        )

    def _db_row_to_review(self, row_dict: Dict[str, Any]) -> 'Review':
        return Review(**row_dict)

    # --- Card Operations ---
    _UPSERT_CARDS_SQL = """
        INSERT INTO cards (uuid, deck_name, front, back, tags, added_at, modified_at,
                           last_review_id, next_due_date, state, stability, difficulty,
                           origin_task, media_paths, source_yaml_file, internal_note)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (uuid) DO UPDATE SET
            deck_name = EXCLUDED.deck_name,
            front = EXCLUDED.front,
            back = EXCLUDED.back,
            tags = EXCLUDED.tags,
            modified_at = EXCLUDED.modified_at,
            last_review_id = EXCLUDED.last_review_id,
            next_due_date = EXCLUDED.next_due_date,
            state = EXCLUDED.state,
            stability = EXCLUDED.stability,
            difficulty = EXCLUDED.difficulty,
            origin_task = EXCLUDED.origin_task,
            media_paths = EXCLUDED.media_paths,
            source_yaml_file = EXCLUDED.source_yaml_file,
            internal_note = EXCLUDED.internal_note;
        """

    def upsert_cards_batch(self, cards: Sequence['Card']) -> int:
        if not cards:
            return 0

        conn = self.get_connection()
        card_params_list = self._card_to_db_params_list(cards)

        try:
            return self._execute_upsert_transaction(conn, card_params_list)
        except duckdb.Error as e:
            self._handle_upsert_error(conn, e)

    def _execute_upsert_transaction(self, conn, card_params_list: List[Tuple]) -> int:
        with conn.cursor() as cursor:
            cursor.begin()
            cursor.executemany(self._UPSERT_CARDS_SQL, card_params_list)
            affected_rows = len(card_params_list)
            cursor.commit()
        logger.info(f"Successfully upserted/processed {affected_rows} out of {len(card_params_list)} cards provided.")
        return affected_rows

    def _handle_upsert_error(self, conn, e: duckdb.Error) -> None:
        logger.error(f"Error during batch card upsert: {e}")
        if conn and not getattr(conn, 'closed', True):
            try:
                conn.rollback()
                logger.info("Transaction rolled back due to error in batch card upsert.")
            except duckdb.Error as rb_err:
                logger.error(f"Error during rollback after upsert failure: {rb_err}")

        if isinstance(e, (duckdb.IOException, duckdb.InvalidInputException)):
            raise e
        raise CardOperationError(f"Batch card upsert failed: {e}", original_exception=e) from e

    def get_card_by_uuid(self, card_uuid: uuid.UUID) -> Optional['Card']:
        conn = self.get_connection()
        sql = "SELECT * FROM cards WHERE uuid = $1;"
        try:
            result = conn.execute(sql, (card_uuid,)).fetch_df()
            if result.empty:
                return None
            row_dict = result.to_dict('records')[0]
            logger.info(f"DEBUG: Fetched row for UUID {card_uuid}: {row_dict}")
            return self._db_row_to_card(row_dict)
        except duckdb.Error as e:
            logger.error(f"Error fetching card by UUID {card_uuid}: {e}")
            raise CardOperationError(f"Failed to fetch card by UUID: {e}", original_exception=e) from e

    def get_all_cards(self, deck_name_filter: Optional[str] = None) -> List['Card']:
        conn = self.get_connection()
        params = []
        sql = "SELECT * FROM cards"
        if deck_name_filter:
            sql += " WHERE deck_name LIKE $1"
            params.append(deck_name_filter)
        sql += " ORDER BY deck_name, front;"
        try:
            result_df = conn.execute(sql, params).fetch_df()
            if result_df.empty:
                return []
            return [self._db_row_to_card(row) for row in result_df.to_dict('records')]
        except duckdb.Error as e:
            logger.error(f"Error fetching all cards (filter: {deck_name_filter}): {e}")
            raise CardOperationError(f"Failed to get all cards: {e}", original_exception=e) from e

    def get_due_cards(self, deck_name: str, on_date: date, limit: Optional[int] = 20) -> List['Card']:
        """
        Fetches cards from a specific deck due for review on or before a given date.
        A card is considered due if its next_due_date is on or before the specified date,
        or if it has never been reviewed (next_due_date is NULL).
        """
        if limit == 0:
            return []
        conn = self.get_connection()
        sql = """
        SELECT * FROM cards
        WHERE deck_name = $1 AND (next_due_date <= $2 OR next_due_date IS NULL)
        ORDER BY next_due_date ASC NULLS FIRST, added_at ASC
        """
        params: List[Any] = [deck_name, on_date]
        if limit is not None and limit > 0:
            sql += f" LIMIT ${len(params) + 1}"
            params.append(limit)

        try:
            result_df = conn.execute(sql, params).fetch_df()
            if result_df.empty:
                return []
            return [self._db_row_to_card(row_dict) for row_dict in result_df.to_dict('records')]
        except duckdb.Error as e:
            logger.error(f"Error fetching due cards for deck '{deck_name}' on date {on_date}: {e}")
            raise CardOperationError(f"Failed to fetch due cards: {e}", original_exception=e) from e

    def get_due_card_count(self, deck_name: str, on_date: date) -> int:
        """
        Efficiently counts the number of cards from a specific deck due for review.
        A card is considered due if its next_due_date is on or before the specified date,
        or if it has never been reviewed (next_due_date is NULL).
        """
        conn = self.get_connection()
        sql = "SELECT COUNT(*) FROM cards WHERE deck_name = $1 AND (next_due_date <= $2 OR next_due_date IS NULL);"
        try:
            result = conn.execute(sql, (deck_name, on_date)).fetchone()
            return result[0] if result else 0
        except duckdb.Error as e:
            logger.error(f"Error counting due cards for deck '{deck_name}' on date {on_date}: {e}")
            raise CardOperationError(f"Failed to count due cards: {e}", original_exception=e) from e

    def delete_cards_by_uuids_batch(self, card_uuids: Sequence[uuid.UUID]) -> int:
        if not card_uuids:
            return 0
        if self.read_only:
            raise DatabaseConnectionError("Cannot delete cards in read-only mode.")
        conn = self.get_connection()
        sql = "DELETE FROM cards WHERE uuid IN (SELECT unnest($1::UUID[])) RETURNING uuid;"
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                cursor.execute(sql, [list(card_uuids)])
                deleted_count = len(cursor.fetchall())
                cursor.commit()
            logger.info(f"Batch deleted {deleted_count} cards for {len(card_uuids)} UUIDs provided.")
            return deleted_count
        except duckdb.Error as e:
            logger.error(f"Error during batch card delete: {e}")
            if conn and not getattr(conn, 'closed', True):
                conn.rollback()
            raise CardOperationError(f"Batch card delete failed: {e}", original_exception=e) from e

    def get_all_card_fronts_and_uuids(self) -> Dict[str, uuid.UUID]:
        """
        Retrieves a dictionary mapping all normalized card fronts to their UUIDs.
        This is used for efficient duplicate checking before inserting new cards.
        """
        conn = self.get_connection()
        sql = "SELECT front, uuid FROM cards;"
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()

            front_to_uuid: Dict[str, uuid.UUID] = {}
            for front, card_uuid in results:
                normalized_front = " ".join(str(front).lower().split())
                if normalized_front not in front_to_uuid:
                    front_to_uuid[normalized_front] = card_uuid
                else:
                    logger.warning(
                        f"Duplicate normalized front found: '{normalized_front}'. "
                        f"Keeping first UUID seen: {front_to_uuid[normalized_front]}. "
                        f"Discarding new UUID: {card_uuid}."
                    )
            return front_to_uuid
        except duckdb.Error as e:
            logger.error(f"Error fetching all card fronts and UUIDs: {e}")
            raise CardOperationError("Could not fetch card fronts and UUIDs.", original_exception=e) from e

    # --- Review Operations ---
    def _insert_review_and_get_id(self, cursor, review: 'Review') -> int:
        """Inserts a review record and returns its new ID."""
        review_params_tuple = self._review_to_db_params_tuple(review)
        sql = """
        INSERT INTO reviews (card_uuid, ts, rating, resp_ms, stab_before, stab_after, diff, next_due,
                             elapsed_days_at_review, scheduled_days_interval, review_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING review_id;
        """
        cursor.execute(sql, review_params_tuple)
        result = cursor.fetchone()
        if not result:
            raise ReviewOperationError("Failed to retrieve review_id after insertion.")
        return result[0]

    def _update_card_after_review(self, cursor, review: 'Review', new_card_state: 'CardState', new_review_id: int) -> None:
        """Updates the card's state and links it to the new review."""
        sql = """
        UPDATE cards
        SET last_review_id = $1, next_due_date = $2, state = $3, stability = $4, difficulty = $5, modified_at = $6
        WHERE uuid = $7;
        """
        params = (
            new_review_id,
            review.next_due,
            new_card_state.name,
            review.stab_after,
            review.diff,
            datetime.now(timezone.utc), # modified_at
            review.card_uuid
        )
        cursor.execute(sql, params)

    def add_review_and_update_card(self, review: 'Review', new_card_state: 'CardState') -> 'Card':
        """
        Atomically adds a review and updates the corresponding card's state and due date.
        Returns the fully updated card object.
        """
        if self.read_only:
            raise DatabaseConnectionError("Cannot add review in read-only mode.")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                # Step 1: Insert the review and get its ID
                new_review_id = self._insert_review_and_get_id(cursor, review)
                # Step 2: Update the card with the new review info
                self._update_card_after_review(cursor, review, new_card_state, new_review_id)
                cursor.commit()

            # Step 3: Fetch and return the updated card
            return self.get_card_by_uuid(review.card_uuid)

        except duckdb.Error as e:
            logger.error(f"Error during review and card update transaction: {e}")
            if conn:
                try:
                    conn.rollback()
                    logger.info("Transaction rolled back due to review/update error.")
                except duckdb.Error as rb_err:
                    logger.error(f"Failed to rollback transaction: {rb_err}")
            raise ReviewOperationError(f"Failed to add review and update card: {e}", original_exception=e) from e

    def get_reviews_for_card(self, card_uuid: uuid.UUID, order_by_ts_desc: bool = True) -> List['Review']:
        conn = self.get_connection()
        order_clause = "ORDER BY ts DESC, review_id DESC" if order_by_ts_desc else "ORDER BY ts ASC, review_id ASC"
        sql = f"SELECT * FROM reviews WHERE card_uuid = $1 {order_clause};"
        try:
            result_df = conn.execute(sql, (card_uuid,)).fetch_df()
            if result_df.empty:
                return []
            return [self._db_row_to_review(row_dict) for row_dict in result_df.to_dict('records')]
        except duckdb.Error as e:
            logger.error(f"Error fetching reviews for card UUID {card_uuid}: {e}")
            raise ReviewOperationError(f"Failed to get reviews for card {card_uuid}: {e}", original_exception=e) from e

    def get_latest_review_for_card(self, card_uuid: uuid.UUID) -> Optional['Review']:
        reviews = self.get_reviews_for_card(card_uuid, order_by_ts_desc=True)
        return reviews[0] if reviews else None

    def get_all_reviews(self, start_ts: Optional[datetime] = None, end_ts: Optional[datetime] = None) -> List['Review']:
        conn = self.get_connection()
        sql = "SELECT * FROM reviews"
        params = []
        conditions = []
        if start_ts:
            conditions.append("ts >= $1")
            params.append(start_ts)
        if end_ts:
            conditions.append(f"ts <= ${len(params) + 1}")
            params.append(end_ts)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY ts ASC, review_id ASC;"
        try:
            result_df = conn.execute(sql, params).fetch_df()
            if result_df.empty:
                return []
            return [self._db_row_to_review(row_dict) for row_dict in result_df.to_dict('records')]
        except duckdb.Error as e:
            logger.error(f"Error fetching all reviews (range: {start_ts} to {end_ts}): {e}")
            raise ReviewOperationError(f"Failed to get all reviews: {e}", original_exception=e) from e
