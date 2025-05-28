"""
DuckDB database interactions for flashcore.
Implements the FlashcardDatabase class and supporting exceptions as per the v3.0 technical design.
"""

import duckdb
import uuid
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any, Sequence, Union
from datetime import datetime, date, timezone
import logging

# Local project imports from previously defined modules
try:
    from cultivation.scripts.flashcore.card import Card, Review
except ImportError:
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Could not import Card/Review from cultivation.scripts.flashcore.card. Attempting relative import.")
    from card import Card, Review

logger = logging.getLogger(__name__)

DEFAULT_FLASHCORE_DATA_DIR = Path.home() / ".cultivation" / "flashcore_data"
DEFAULT_DATABASE_FILENAME = "flash.db"

# --- Custom Exceptions ---
class DatabaseError(Exception):
    """Base exception for all database operations in this module."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """
        Initializes a DatabaseError with an optional original exception.
        
        Args:
            message: Description of the database error.
            original_exception: The underlying exception that caused this error, if any.
        """
        super().__init__(message)
        self.original_exception = original_exception
    def __str__(self):
        """
        Returns a string representation of the exception, including the original exception if present.
        """
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
            card_uuid                   UUID NOT NULL REFERENCES cards(uuid) ON DELETE CASCADE,
            ts                          TIMESTAMP WITH TIME ZONE NOT NULL,
            rating                      SMALLINT NOT NULL CHECK (rating >= 0 AND rating <= 3),
            resp_ms                     INTEGER CHECK (resp_ms IS NULL OR resp_ms >= 0),
            stab_before                 DOUBLE,
            stab_after                  DOUBLE NOT NULL CHECK (stab_after >= 0.1),
            diff                        DOUBLE NOT NULL,
            next_due                    DATE NOT NULL,
            elapsed_days_at_review      INTEGER NOT NULL CHECK (elapsed_days_at_review >= 0),
            scheduled_days_interval     INTEGER NOT NULL CHECK (scheduled_days_interval >= 1),
            review_type                 VARCHAR(50)
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_cards_deck_name ON cards (deck_name);",
        "CREATE INDEX IF NOT EXISTS idx_reviews_card_uuid_ts ON reviews (card_uuid, ts DESC);",
        "CREATE INDEX IF NOT EXISTS idx_reviews_next_due ON reviews (next_due);"
    ]

    def __init__(self, db_path: Optional[Union[str, Path]] = None, read_only: bool = False):
        """
        Initializes a FlashcardDatabase instance with the specified database path and mode.
        
        If no path is provided, uses the default file location. Supports in-memory and read-only modes.
        """
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

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Establishes and returns a DuckDB database connection.
        
        If the connection is not already open, this method creates the necessary parent directories (for file-based databases), opens a new DuckDB connection in the specified mode (read-only or read-write), and returns the connection object. Raises a DatabaseConnectionError if the connection cannot be established.
        """
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
        """
        Closes the active DuckDB database connection if it is open.
        
        Resets the internal connection attribute to None after closing.
        """
        if self._connection and not getattr(self._connection, 'closed', False):
            self._connection.close()
            logger.debug(f"DuckDB connection to {self.db_path_resolved} closed.")
        self._connection = None

    def __enter__(self) -> 'FlashcardDatabase':
        """
        Enters the runtime context for the FlashcardDatabase, ensuring a database connection is open.
        
        Returns:
            The FlashcardDatabase instance with an active connection.
        """
        self.get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Closes the database connection when exiting a context manager block.
        """
        self.close_connection()

    def initialize_schema(self, force_recreate_tables: bool = False) -> None:
        """
        Initializes the database schema for flashcards and reviews.
        
        Creates required tables, sequences, and indexes if they do not exist. If `force_recreate_tables` is True, drops and recreates all schema objects, erasing existing data. Raises an error if forced recreation is attempted in read-only mode. Rolls back changes and raises `SchemaInitializationError` on failure.
        """
        if self.read_only and not force_recreate_tables:
            if str(self.db_path_resolved) != ":memory:":
                logger.warning("Attempting to initialize schema in read-only mode. Skipping.")
                return
        if self.read_only and force_recreate_tables:
            raise DatabaseConnectionError("Cannot force_recreate_tables in read-only mode.")
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                if force_recreate_tables:
                    logger.warning(f"Forcing table recreation for {self.db_path_resolved}. ALL EXISTING DATA WILL BE LOST.")
                    cursor.execute("DROP TABLE IF EXISTS reviews CASCADE;")
                    cursor.execute("DROP TABLE IF EXISTS cards CASCADE;")
                    cursor.execute("DROP SEQUENCE IF EXISTS review_seq;")
                for statement in self._DB_SCHEMA_SQL:
                    cursor.execute(statement)
                cursor.commit()
            logger.info(f"Database schema at {self.db_path_resolved} initialized successfully (or already exists).")
        except duckdb.Error as e:
            logger.error(f"Error initializing database schema at {self.db_path_resolved}: {e}")
            if 'cursor' in locals() and cursor.connection.is_open:
                try:
                    cursor.rollback()
                except duckdb.Error as rb_err:
                    logger.error(f"Error during rollback: {rb_err}")
            raise SchemaInitializationError(f"Schema initialization failed: {e}", original_exception=e)

    # --- Data Marshalling Helpers (Internal) ---
    def _card_to_db_params_list(self, cards: Sequence['Card']) -> List[Tuple]:
        """
        Converts a sequence of Card objects into a list of tuples for database insertion.
        
        Each tuple contains the card's UUID, deck name, front and back text, tags, added timestamp, origin task, media paths as strings, source YAML file path as a string, and internal note.
        """
        params_list = []
        for card in cards:
            params_list.append((
                card.uuid,
                card.deck_name,
                card.front,
                card.back,
                list(card.tags) if card.tags else None,
                card.added_at,
                card.origin_task,
                [str(p) for p in card.media] if card.media else None,
                str(card.source_yaml_file) if card.source_yaml_file else None,
                card.internal_note
            ))
        return params_list

    def _db_row_to_card(self, row_dict: Dict[str, Any]) -> 'Card':
        """
        Converts a database row dictionary into a Card object.
        
        Transforms database fields such as media paths and tags into their appropriate Python types before constructing the Card instance.
        """
        if row_dict.get("media_paths"):
            row_dict["media"] = [Path(p) for p in row_dict.pop("media_paths")]
        else:
            row_dict["media"] = None
        if row_dict.get("source_yaml_file"):
            row_dict["source_yaml_file"] = Path(row_dict["source_yaml_file"])
        row_dict["tags"] = set(row_dict["tags"]) if row_dict.get("tags") else set()
        return Card(**row_dict)

    def _review_to_db_params_tuple(self, review: 'Review') -> Tuple:
        """
        Converts a Review object into a tuple of values for database insertion.
        
        Args:
            review: The Review object to convert.
        
        Returns:
            A tuple containing the Review's fields in the order expected by the database schema.
        """
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
        """
        Converts a database row dictionary into a Review object.
        
        Args:
            row_dict: A dictionary representing a row from the reviews table.
        
        Returns:
            A Review object populated with data from the row.
        """
        return Review(**row_dict)

    # --- Card Operations ---
    def upsert_cards_batch(self, cards: Sequence['Card']) -> int:
        """
        Inserts or updates a batch of flashcards in the database.
        
        If a card with the same UUID already exists, its fields are updated; otherwise, a new card is inserted. Raises an error if the database is in read-only mode.
        
        Args:
            cards: A sequence of Card objects to insert or update.
        
        Returns:
            The number of cards inserted or updated.
        
        Raises:
            DatabaseConnectionError: If called in read-only mode.
            CardOperationError: If the database operation fails.
        """
        if not cards:
            return 0
        if self.read_only:
            raise DatabaseConnectionError("Cannot upsert cards in read-only mode.")
        conn = self.get_connection()
        card_params_list = self._card_to_db_params_list(cards)
        sql = """
        INSERT INTO cards (uuid, deck_name, front, back, tags, added_at, 
                           origin_task, media_paths, source_yaml_file, internal_note)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (uuid) DO UPDATE SET
            deck_name = EXCLUDED.deck_name,
            front = EXCLUDED.front,
            back = EXCLUDED.back,
            tags = EXCLUDED.tags,
            origin_task = EXCLUDED.origin_task,
            media_paths = EXCLUDED.media_paths,
            source_yaml_file = EXCLUDED.source_yaml_file,
            internal_note = EXCLUDED.internal_note;
        """
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                cursor.executemany(sql, card_params_list)
                affected_rows = cursor.rowcount if cursor.rowcount is not None else len(cards)
                cursor.commit()
            logger.info(f"Successfully upserted/processed {affected_rows} out of {len(cards)} cards provided.")
            return affected_rows
        except duckdb.Error as e:
            logger.error(f"Error during batch card upsert: {e}")
            if 'cursor' in locals() and cursor.connection.is_open: cursor.rollback()
            raise CardOperationError(f"Batch card upsert failed: {e}", original_exception=e)

    def get_card_by_uuid(self, card_uuid: uuid.UUID) -> Optional['Card']:
        """
        Retrieves a flashcard by its UUID.
        
        Args:
        	card_uuid: The UUID of the card to retrieve.
        
        Returns:
        	The corresponding Card object if found, or None if no card with the given UUID exists.
        
        Raises:
        	CardOperationError: If a database error occurs during retrieval.
        """
        conn = self.get_connection()
        sql = "SELECT * FROM cards WHERE uuid = $1;"
        try:
            result = conn.execute(sql, (card_uuid,)).fetch_df()
            if result.empty:
                return None
            row_dict = result.to_dict('records')[0]
            return self._db_row_to_card(row_dict)
        except duckdb.Error as e:
            logger.error(f"Error fetching card by UUID {card_uuid}: {e}")
            raise CardOperationError(f"Failed to fetch card by UUID: {e}", original_exception=e)

    def get_all_cards(self, deck_name_filter: Optional[str] = None) -> List['Card']:
        """
        Retrieves all flashcards from the database, optionally filtered by deck name.
        
        Args:
            deck_name_filter: If provided, only cards with deck names matching this filter (using SQL LIKE) are returned.
        
        Returns:
            A list of Card objects representing all matching flashcards.
        """
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
            return [self._db_row_to_card(row_dict) for row_dict in result_df.to_dict('records')]
        except duckdb.Error as e:
            logger.error(f"Error fetching all cards (filter: {deck_name_filter}): {e}")
            raise CardOperationError(f"Failed to fetch all cards: {e}", original_exception=e)

    def get_due_cards(self, on_date: date, limit: Optional[int] = 20) -> List['Card']:
        """
        Retrieves flashcards that are due for review on a specified date.
        
        Cards are considered due if they do not have any future-dated reviews with a next due date after the given date. Results are ordered by the earliest next due date and the card's added date. Optionally limits the number of returned cards.
        
        Args:
            on_date: The date to check for due cards.
            limit: Maximum number of cards to return. If None or non-positive, returns all due cards.
        
        Returns:
            A list of Card objects that are due for review on the specified date.
        
        Raises:
            CardOperationError: If a database error occurs during retrieval.
        """
        conn = self.get_connection()
        sql = f"""
        SELECT c.*
        FROM cards c
        WHERE c.uuid NOT IN (
            SELECT r_future.card_uuid
            FROM reviews r_future
            INNER JOIN (
                SELECT card_uuid, MAX(ts) as max_ts
                FROM reviews
                WHERE next_due > $1
                GROUP BY card_uuid
            ) latest_future_rev ON r_future.card_uuid = latest_future_rev.card_uuid 
                               AND r_future.ts = latest_future_rev.max_ts
            WHERE r_future.next_due > $1
        )
        ORDER BY (
            SELECT MIN(r_order.next_due) 
            FROM reviews r_order 
            WHERE r_order.card_uuid = c.uuid AND r_order.next_due <= $1
        ) ASC NULLS LAST,
        c.added_at ASC
        """
        params: List[Any] = [on_date]
        if limit is not None and limit > 0:
            sql += f" LIMIT ${len(params) + 1}"
            params.append(limit)
        try:
            result_df = conn.execute(sql, params).fetch_df()
            if result_df.empty:
                return []
            return [self._db_row_to_card(row_dict) for row_dict in result_df.to_dict('records')]
        except duckdb.Error as e:
            logger.error(f"Error fetching due cards for date {on_date}: {e}")
            raise CardOperationError(f"Failed to fetch due cards: {e}", original_exception=e)

    def delete_cards_by_uuids_batch(self, card_uuids: Sequence[uuid.UUID]) -> int:
        """
        Deletes multiple cards from the database by their UUIDs in a single batch operation.
        
        Args:
            card_uuids: A sequence of card UUIDs to delete.
        
        Returns:
            The number of cards successfully deleted.
        
        Raises:
            DatabaseConnectionError: If the database is in read-only mode.
            CardOperationError: If the deletion fails due to a database error.
        """
        if not card_uuids:
            return 0
        if self.read_only:
            raise DatabaseConnectionError("Cannot delete cards in read-only mode.")
        conn = self.get_connection()
        sql = "DELETE FROM cards WHERE uuid IN (SELECT unnest($1::UUID[]));"
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                cursor.execute(sql, [list(card_uuids)])
                deleted_count = cursor.rowcount if cursor.rowcount is not None else 0
                cursor.commit()
            logger.info(f"Batch deleted {deleted_count} cards for {len(card_uuids)} UUIDs provided.")
            return deleted_count
        except duckdb.Error as e:
            logger.error(f"Error during batch card delete: {e}")
            if 'cursor' in locals() and cursor.connection.is_open: cursor.rollback()
            raise CardOperationError(f"Batch card delete failed: {e}", original_exception=e)

    def get_all_card_fronts_and_uuids(self) -> Dict[str, uuid.UUID]:
        """
        Retrieves a mapping of normalized card front text to their UUIDs.
        
        Returns:
            A dictionary where each key is the normalized (lowercased and whitespace-normalized) front text of a card, and each value is the corresponding card's UUID. If duplicate normalized fronts are found, a warning is logged and only the first occurrence is retained.
        """
        conn = self.get_connection()
        front_to_uuid: Dict[str, uuid.UUID] = {}
        sql = "SELECT front, uuid FROM cards;"
        try:
            result_df = conn.execute(sql).fetch_df()
            for index, row_series in result_df.iterrows():
                front_text = str(row_series['front'])
                card_uuid_val = row_series['uuid']
                normalized_front = " ".join(front_text.lower().split())
                if normalized_front not in front_to_uuid:
                    front_to_uuid[normalized_front] = card_uuid_val
                else:
                    logger.warning(
                        f"Duplicate normalized front '{normalized_front}' found in database. "
                        f"Existing UUID: {front_to_uuid[normalized_front]}, current card UUID: {card_uuid_val}."
                    )
            return front_to_uuid
        except duckdb.Error as e:
            logger.error(f"Error fetching all card fronts and UUIDs: {e}")
            raise CardOperationError(f"Failed to fetch card fronts and UUIDs: {e}", original_exception=e)

    # --- Review Operations ---
    def add_review(self, review: 'Review') -> int:
        """
        Adds a new review record to the database and returns its unique review ID.
        
        Raises:
            DatabaseConnectionError: If called in read-only mode.
            ReviewOperationError: If the review cannot be inserted or the review ID is not returned.
        
        Returns:
            The unique integer ID of the newly inserted review.
        """
        if self.read_only:
            raise DatabaseConnectionError("Cannot add review in read-only mode.")
        conn = self.get_connection()
        review_params_tuple = self._review_to_db_params_tuple(review)
        sql = """
        INSERT INTO reviews (card_uuid, ts, rating, resp_ms, stab_before, stab_after, diff, next_due,
                             elapsed_days_at_review, scheduled_days_interval, review_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING review_id;
        """
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                result = cursor.execute(sql, review_params_tuple).fetchone()
                if result is None or result[0] is None:
                    cursor.rollback()
                    raise ReviewOperationError("Failed to retrieve review_id after insert: No ID returned.")
                new_review_id = int(result[0])
                cursor.commit()
            logger.info(f"Successfully added review with ID: {new_review_id} for card UUID: {review.card_uuid}")
            return new_review_id
        except duckdb.Error as e:
            logger.error(f"Error adding review for card UUID {review.card_uuid}: {e}")
            if 'cursor' in locals() and cursor.connection.is_open:
                try:
                    cursor.rollback()
                except duckdb.Error as rb_err:
                    logger.error(f"Error during rollback for add_review: {rb_err}")
            raise ReviewOperationError(f"Failed to add review: {e}", original_exception=e)

    def add_reviews_batch(self, reviews: Sequence['Review']) -> List[int]:
        """
        Inserts multiple review records into the database in a single batch operation.
        
        Args:
            reviews: A sequence of Review objects to be added.
        
        Returns:
            A list of newly inserted review IDs, in the order of the input reviews.
        
        Raises:
            DatabaseConnectionError: If called in read-only mode.
            ReviewOperationError: If the batch insertion fails.
        """
        if not reviews:
            return []
        if self.read_only:
            raise DatabaseConnectionError("Cannot add reviews in read-only mode.")
        conn = self.get_connection()
        review_params_list = [self._review_to_db_params_tuple(r) for r in reviews]
        sql = """
        INSERT INTO reviews (card_uuid, ts, rating, resp_ms, stab_before, stab_after, diff, next_due,
                             elapsed_days_at_review, scheduled_days_interval, review_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING review_id;
        """
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                result = cursor.executemany(sql, review_params_list).fetchall()
                cursor.commit()
            review_ids = [int(r[0]) for r in result]
            logger.info(f"Successfully batch-added {len(review_ids)} reviews.")
            return review_ids
        except duckdb.Error as e:
            logger.error(f"Error during batch review add: {e}")
            if 'cursor' in locals() and cursor.connection.is_open:
                try:
                    cursor.rollback()
                except duckdb.Error as rb_err:
                    logger.error(f"Error during rollback for add_reviews_batch: {rb_err}")
            raise ReviewOperationError(f"Batch review add failed: {e}") from e

    def get_reviews_for_card(self, card_uuid: uuid.UUID, order_by_ts_desc: bool = True) -> List['Review']:
        """
        Retrieves all reviews associated with a specific card.
        
        Args:
            card_uuid: The UUID of the card whose reviews are to be fetched.
            order_by_ts_desc: If True, reviews are ordered by timestamp descending; otherwise, ascending.
        
        Returns:
            A list of Review objects for the specified card, ordered by timestamp and review ID.
        """
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
            raise ReviewOperationError(f"Failed to get reviews for card {card_uuid}: {e}", original_exception=e)

    def get_latest_review_for_card(self, card_uuid: uuid.UUID) -> Optional['Review']:
        """
        Retrieves the most recent review for a specified card.
        
        Args:
            card_uuid: The UUID of the card whose latest review is to be fetched.
        
        Returns:
            The latest Review object for the card, or None if no reviews exist.
        """
        reviews = self.get_reviews_for_card(card_uuid, order_by_ts_desc=True)
        return reviews[0] if reviews else None

    def get_all_reviews(self, start_ts: Optional[datetime] = None, end_ts: Optional[datetime] = None) -> List['Review']:
        """
        Retrieves all review records, optionally filtered by a timestamp range.
        
        Args:
            start_ts: If provided, only reviews with a timestamp greater than or equal to this value are returned.
            end_ts: If provided, only reviews with a timestamp less than or equal to this value are returned.
        
        Returns:
            A list of Review objects ordered by timestamp and review ID in ascending order.
        """
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
            raise ReviewOperationError(f"Failed to get all reviews: {e}", original_exception=e)
