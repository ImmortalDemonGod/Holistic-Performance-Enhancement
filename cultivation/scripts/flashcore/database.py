"""
DuckDB database interactions for flashcore.
Implements the FlashcardDatabase class and supporting exceptions as per the v3.0 technical design.
"""

import duckdb
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Sequence, Union
from datetime import datetime, date
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
            card_uuid                   UUID NOT NULL REFERENCES cards(uuid),
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
            if 'cursor' in locals() and not getattr(cursor.connection, 'closed', True):
                try:
                    cursor.rollback()
                except duckdb.Error as rb_err:
                    logger.error(f"Error during rollback: {rb_err}")
            raise SchemaInitializationError(f"Schema initialization failed: {e}", original_exception=e)

    # --- Data Marshalling Helpers (Internal) ---
    def _card_to_db_params_list(self, cards: Sequence['Card']) -> List[Tuple]:
        params_list = []
        for card in cards:
            params_list.append((
                card.uuid,
                card.deck_name,
                card.front,
                card.back,
                list(card.tags) if card.tags else None,
                card.added_at,
                card.modified_at,
                card.origin_task,
                [str(p) for p in card.media] if card.media else None,
                str(card.source_yaml_file) if card.source_yaml_file else None,
                card.internal_note
            ))
        return params_list

    def _db_row_to_card(self, row_dict: Dict[str, Any]) -> 'Card':
        # Handle media: convert string list from 'media_paths' column to Path list for 'media' field
        media_paths_val = row_dict.pop("media_paths", None)
        if media_paths_val is not None:
            row_dict["media"] = [Path(p) for p in media_paths_val]
        else:
            row_dict["media"] = None

        # Handle source_yaml_file: convert string to Path
        if row_dict.get("source_yaml_file"):
            row_dict["source_yaml_file"] = Path(row_dict["source_yaml_file"])

        # Handle tags: convert list to set, ensuring it's never None
        row_dict["tags"] = set(row_dict["tags"]) if row_dict.get("tags") is not None else set()

        return Card(**row_dict)

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
    def upsert_cards_batch(self, cards: Sequence['Card']) -> int:
        if not cards:
            return 0
        # Ensure NO explicit 'if self.read_only: raise ...' check here.
        # DuckDB itself should raise an error if trying to write to a read-only DB.
        
        conn = self.get_connection()
        card_params_list = self._card_to_db_params_list(cards)
        sql = """
        INSERT INTO cards (uuid, deck_name, front, back, tags, added_at, modified_at,
                           origin_task, media_paths, source_yaml_file, internal_note)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (uuid) DO UPDATE SET
            deck_name = EXCLUDED.deck_name,
            front = EXCLUDED.front,
            back = EXCLUDED.back,
            tags = EXCLUDED.tags,
            modified_at = EXCLUDED.modified_at, -- Update modified_at timestamp
            origin_task = EXCLUDED.origin_task,
            media_paths = EXCLUDED.media_paths,
            source_yaml_file = EXCLUDED.source_yaml_file,
            internal_note = EXCLUDED.internal_note;
        """
        try:
            with conn.cursor() as cursor:
                cursor.begin()
                cursor.executemany(sql, card_params_list)
                affected_rows = len(cards)
                cursor.commit()
            logger.info(f"Successfully upserted/processed {affected_rows} out of {len(cards)} cards provided.")
            return affected_rows
        except duckdb.Error as e:  # Catch any DuckDB specific error
            logger.error(f"Error during batch card upsert: {e}")
            if conn and not getattr(conn, 'closed', True):
                try:
                    conn.rollback()
                    logger.info("Transaction rolled back due to error in batch card upsert.")
                except duckdb.Error as rb_err:
                    logger.error(f"Error during rollback after upsert failure: {rb_err}")

            # If the error is due to read-only violation or similar, re-raise it directly.
            # DuckDB might raise IOException or InvalidInputException for read-only writes.
            if isinstance(e, (duckdb.IOException, duckdb.InvalidInputException)):
                raise e
            # Otherwise, wrap it in our custom exception.
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

    def get_due_cards(self, on_date: date, limit: Optional[int] = 20) -> List['Card']:
        if limit == 0:
            return []
        conn = self.get_connection()
        sql = """
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
            raise CardOperationError(f"Failed to fetch due cards: {e}", original_exception=e) from e

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
            raise CardOperationError(f"Failed to fetch card fronts and UUIDs: {e}", original_exception=e) from e

    # --- Review Operations ---
    def add_review(self, review: 'Review') -> int:
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
        except duckdb.ConstraintException as e:
            logger.error(f"Failed to add review due to constraint violation: {e}")
            # On constraint violation, DuckDB automatically rolls back the transaction.
            # Explicitly calling rollback() here would cause a TransactionException.
            raise ReviewOperationError(f"Failed to add review due to constraint violation: {e}") from e
        except duckdb.Error as e:
            logger.error(f"Error adding review for card UUID {review.card_uuid}: {e}")
            conn.rollback()
            raise ReviewOperationError(f"Failed to add review: {e}", original_exception=e) from e

    def add_reviews_batch(self, reviews: Sequence['Review']) -> List[int]:
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
                result_ids = []
                for params in review_params_list:
                    result = cursor.execute(sql, params).fetchone()
                    result_ids.append(int(result[0]))
                cursor.commit()
            review_ids = result_ids
            logger.info("Successfully batch-added %d reviews.", len(review_ids))
            logger.info(f"Successfully batch-added {len(review_ids)} reviews.")
            return review_ids
        except duckdb.ConstraintException as e:
            logger.error(f"Failed to add review in batch due to constraint violation: {e}")
            # On constraint violation, DuckDB automatically rolls back the transaction.
            # No explicit rollback() needed here, as it would cause a TransactionException.
            raise ReviewOperationError(f"Failed to add review in batch due to constraint violation: {e}") from e
        except duckdb.Error as e:
            logger.error(f"Error during batch review add: {e}")
            if 'cursor' in locals() and cursor.connection.is_open:
                try:
                    cursor.rollback()
                except duckdb.Error as rb_err:
                    logger.error(f"Error during rollback for add_reviews_batch: {rb_err}")
            raise ReviewOperationError(f"Batch review add failed: {e}", original_exception=e) from e

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
