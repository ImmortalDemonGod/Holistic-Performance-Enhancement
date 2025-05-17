#!/usr/bin/env python3
"""
FastAPI app for instrumented PDF reader.
Serves static frontend and WebSocket endpoint for telemetry events.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

try:
    from jsonschema import validate, ValidationError
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


app = FastAPI()
# Serve static frontend
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Endpoint to serve index
@app.get("/")
def index():
    """
    Serves the main frontend HTML page.
    """
    return FileResponse(static_dir / "index.html")

# Serve PDFs from literature/pdf
@app.get("/pdfs/{arxiv_id}.pdf")
def get_pdf(arxiv_id: str):
    """
    Serves the PDF file for a given arXiv ID.
    
    Returns a 404 JSON response if the PDF does not exist; otherwise, returns the PDF file.
    """
    pdf_path = (
        Path(__file__).parent.parent / "literature" / "pdf" / f"{arxiv_id}.pdf"
    )
    if not pdf_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"detail": f"PDF for {arxiv_id} not found"},
        )
    return FileResponse(str(pdf_path), media_type="application/pdf")

# Database for sessions and events
DB_PATH = Path(__file__).parent.parent / "literature" / "db.sqlite"

def init_db():
    """
    Initializes the SQLite database with tables for sessions and telemetry events if they do not exist.
    
    Creates the 'sessions' table to track reading sessions and the 'events' table to store telemetry events linked to sessions.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT
        )''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )''')
    conn.commit()
    conn.close()

init_db()

@app.websocket("/ws")
async def telemetry_ws(ws: WebSocket):
    """
    Handles a WebSocket connection to receive and store telemetry events for a PDF reading session.
    
    Accepts a WebSocket connection with an `arxiv_id` query parameter, creates a new session record, and continuously receives JSON-encoded telemetry events, storing each event in the database. Marks the session as finished when the connection is closed.
    """
    await ws.accept()
    arxiv_id = ws.query_params.get("arxiv_id", "")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    start_ts = datetime.utcnow().isoformat() + "Z"
    c.execute("INSERT INTO sessions(paper_id, started_at) VALUES (?, ?)", (arxiv_id, start_ts))
    conn.commit()
    session_id = c.lastrowid
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            etype = data.get("event_type")
            payload = json.dumps(data.get("payload", {}))
            ts = datetime.utcnow().isoformat() + "Z"
            c.execute(
                "INSERT INTO events(session_id, event_type, timestamp, payload) VALUES (?, ?, ?, ?)",
                (session_id, etype, ts, payload)
            )
            conn.commit()
    except WebSocketDisconnect:
        finish_ts = datetime.utcnow().isoformat() + "Z"
        c.execute("UPDATE sessions SET finished_at=? WHERE session_id=?", (finish_ts, session_id))
        conn.commit()
        conn.close()

from fastapi import Query, Response
import csv
from io import StringIO

SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "paper.schema.json"

# Utility: load and validate metadata by arxiv_id
def load_and_validate_metadata(arxiv_id: str):
    """
    Loads metadata for a given paper and validates it against a JSON schema if available.
    
    Args:
        arxiv_id: The arXiv identifier of the paper.
    
    Returns:
        The metadata as a dictionary.
    
    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the metadata fails schema validation.
    """
    metadata_path = Path(__file__).parent.parent / "literature" / "metadata" / f"{arxiv_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    data = json.load(metadata_path.open())
    if _HAS_JSONSCHEMA:
        try:
            schema = json.load(SCHEMA_PATH.open())
            validate(instance=data, schema=schema)
        except ValidationError as ve:
            raise ValueError(f"Metadata schema validation failed: {ve}")
    return data

# API: GET /metadata/{arxiv_id} (returns validated metadata)
from fastapi import HTTPException
@app.get("/metadata/{arxiv_id}")
def get_metadata(arxiv_id: str):
    """
    Retrieves and validates metadata for a given paper by arXiv ID.
    
    Attempts to load the metadata JSON file for the specified arXiv ID and validates it against a schema if available. Returns the metadata as a JSON response.
    
    Raises:
        HTTPException: 404 if the metadata file is not found.
        HTTPException: 422 if metadata validation fails.
        HTTPException: 500 for unexpected errors.
    """
    try:
        data = load_and_validate_metadata(arxiv_id)
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.get("/metrics/{arxiv_id}")
def get_metrics(
    arxiv_id: str,
    event_type: str = Query(None),
    session_id: int = Query(None),
    start_time: str = Query(None),
    end_time: str = Query(None),
):
    """
    Retrieves telemetry events for a specified paper, optionally filtered by event type, session, or time range.
    
    Args:
        arxiv_id: The arXiv identifier of the paper.
        event_type: Optional event type to filter results.
        session_id: Optional session ID to filter results.
        start_time: Optional ISO timestamp to filter events occurring after this time.
        end_time: Optional ISO timestamp to filter events occurring before this time.
    
    Returns:
        A JSON response containing a list of matching telemetry events, each with event type, timestamp, payload, and session ID. Returns a 404 response if no events are found.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = '''
        SELECT e.event_type, e.timestamp, e.payload, e.session_id
        FROM events e
        JOIN sessions s ON e.session_id = s.session_id
        WHERE s.paper_id = ?
    '''
    params = [arxiv_id]
    if event_type:
        query += " AND e.event_type = ?"
        params.append(event_type)
    if session_id:
        query += " AND e.session_id = ?"
        params.append(session_id)
    if start_time:
        query += " AND e.timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND e.timestamp <= ?"
        params.append(end_time)
    query += " ORDER BY e.timestamp ASC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    metrics = [
        {"event_type": r[0], "timestamp": r[1], "payload": json.loads(r[2]), "session_id": r[3]} for r in rows
    ]
    if not metrics:
        return JSONResponse(content={"detail": "No events found for query."}, status_code=404)
    return JSONResponse(content=metrics)

@app.get("/metrics/{arxiv_id}/summary")
def get_metrics_summary(arxiv_id: str, session_id: int = Query(None)):
    """
    Returns a summary of reading activity for a given paper, optionally filtered by session.
    
    Analyzes telemetry events to compute pages read, total time spent reading, most and least viewed pages, and time spent per page. Returns a JSON summary with these analytics. Responds with 404 if no events are found.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = '''
        SELECT e.event_type, e.timestamp, e.payload
        FROM events e
        JOIN sessions s ON e.session_id = s.session_id
        WHERE s.paper_id = ?
    '''
    params = [arxiv_id]
    if session_id:
        query += " AND e.session_id = ?"
        params.append(session_id)
    query += " ORDER BY e.timestamp ASC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    if not rows:
        return JSONResponse(content={"detail": "No events found for summary."}, status_code=404)
    # Analytics
    pages = set()
    page_times = {}
    last_page = None
    last_time = None
    for r in rows:
        etype, ts, payload = r[0], r[1], json.loads(r[2])
        if etype == "page_change":
            if last_page is not None and last_time is not None:
                duration = (datetime.fromisoformat(ts.replace("Z", "")) - datetime.fromisoformat(last_time.replace("Z", ""))).total_seconds()
                page_times[last_page] = page_times.get(last_page, 0) + duration
            last_page = payload.get("new_page_num")
            pages.add(last_page)
            last_time = ts
        elif etype == "view_area_update":
            # Use as fallback for time spent if no page_change
            if "page_num" in payload:
                pages.add(payload["page_num"])
    # Final flush: add time spent on last page until now
    if last_page is not None and last_time is not None:
        duration = (datetime.utcnow() - datetime.fromisoformat(last_time.replace("Z", ""))).total_seconds()
        page_times[last_page] = page_times.get(last_page, 0) + duration
    total_time = sum(page_times.values())
    most_viewed = max(page_times, key=page_times.get) if page_times else None
    least_viewed = min(page_times, key=page_times.get) if page_times else None
    summary = {
        "pages_read": sorted(list(pages)),
        "total_time_seconds": total_time,
        "most_viewed_page": most_viewed,
        "least_viewed_page": least_viewed,
        "per_page_time_seconds": page_times
    }
    return JSONResponse(content=summary)

@app.get("/metrics/{arxiv_id}/csv")
def metrics_csv(
    arxiv_id: str,
    event_type: str = Query(None),
    session_id: int = Query(None),
    start_time: str = Query(None),
    end_time: str = Query(None),
):
    """
    Returns telemetry events for a given paper as a CSV file.
    
    Retrieves events associated with the specified arXiv ID from the database, optionally filtered by event type, session ID, and time range, and returns them in CSV format. If no events are found, returns a CSV header only.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = '''
        SELECT e.event_type, e.timestamp, e.payload, e.session_id
        FROM events e
        JOIN sessions s ON e.session_id = s.session_id
        WHERE s.paper_id = ?
    '''
    params = [arxiv_id]
    if event_type:
        query += " AND e.event_type = ?"
        params.append(event_type)
    if session_id:
        query += " AND e.session_id = ?"
        params.append(session_id)
    if start_time:
        query += " AND e.timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND e.timestamp <= ?"
        params.append(end_time)
    query += " ORDER BY e.timestamp ASC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    if not rows:
        return Response(content="event_type,timestamp,payload,session_id\n", media_type="text/csv")
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["event_type", "timestamp", "payload", "session_id"])
    for r in rows:
        writer.writerow([r[0], r[1], r[2], r[3]])
    return Response(content=output.getvalue(), media_type="text/csv")
