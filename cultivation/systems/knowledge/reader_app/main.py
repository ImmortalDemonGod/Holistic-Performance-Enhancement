#!/usr/bin/env python3
"""
FastAPI app for instrumented PDF reader.
Serves static frontend and WebSocket endpoint for telemetry events.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, UTC
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
    return FileResponse(static_dir / "index.html")

# Serve PDFs from literature/pdf
@app.get("/pdfs/{arxiv_id}.pdf")
def get_pdf(arxiv_id: str):
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

@app.websocket("/ws")  # WebSocket connection/disconnection is logged for systematic debugging
async def telemetry_ws(ws: WebSocket):
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
        c.execute("UPDATE sessions SET finished_at=? WHERE session_id=? AND finished_at IS NULL", (finish_ts, session_id))
        conn.commit()
        conn.close()

from fastapi import Query, Response
import csv
from io import StringIO

SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "paper.schema.json"

# Utility: load and validate metadata by arxiv_id
def load_and_validate_metadata(arxiv_id: str):
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

# List all papers (arxiv_id and title)
@app.get("/papers/list")
def list_papers():
    meta_dir = Path(__file__).parent.parent / "literature" / "metadata"
    papers = []
    for f in meta_dir.glob("*.json"):
        try:
            data = json.load(f.open())
            papers.append({"arxiv_id": data["arxiv_id"], "title": data["title"]})
        except Exception:
            continue
    return JSONResponse(content=papers)

# Paper progress for filtering/resume
@app.get("/papers/progress")
def papers_progress():
    meta_dir = Path(__file__).parent.parent / "literature" / "metadata"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    results = []
    for f in meta_dir.glob("*.json"):
        try:
            data = json.load(f.open())
            arxiv_id = data["arxiv_id"]
            # Get all unique pages visited
            c.execute("""
                SELECT e.payload FROM sessions s JOIN events e ON s.session_id = e.session_id
                WHERE s.paper_id = ? AND e.event_type = 'page_change'
            """, (arxiv_id,))
            page_events = c.fetchall()
            unique_pages = set()
            for r in page_events:
                try:
                    unique_pages.add(json.loads(r[0])["new_page_num"])
                except Exception:
                    continue

            # Get last page read (latest event)
            last_page = None
            if page_events:
                try:
                    last_page = json.loads(page_events[-1][0])["new_page_num"]
                except Exception:
                    pass

            # Get total pages (from metadata if available)
            total_pages = data.get("num_pages")
            # If not in metadata, try to estimate from events
            if not total_pages and unique_pages:
                total_pages = max(unique_pages)

            # Compute completion percentage (unique pages visited / total pages)
            completion = None
            if total_pages and unique_pages:
                completion = round(100 * len(unique_pages) / total_pages, 1)
                # Only show 100% if all pages visited
                if len(unique_pages) == total_pages:
                    completion = 100.0
                elif completion > 100:
                    completion = 100.0
            results.append({
                "arxiv_id": arxiv_id,
                "title": data["title"],
                "last_page": last_page,
                "total_pages": total_pages,
                "completion": completion
            })
        except Exception:
            continue
    conn.close()
    return JSONResponse(content=results)

# Finish session and log metrics
from fastapi import Request
@app.post("/finish_session")
async def finish_session(request: Request):
    data = await request.json()
    arxiv_id = data.get("arxiv_id")
    metrics = data.get("metrics", {})
    if not arxiv_id or not metrics:
        return JSONResponse(content={"error": "Missing arxiv_id or metrics"}, status_code=400)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Find latest open session for this arxiv_id
    c.execute("SELECT session_id FROM sessions WHERE paper_id=? AND finished_at IS NULL ORDER BY started_at DESC LIMIT 1", (arxiv_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return JSONResponse(content={"error": "No open session found for this arxiv_id"}, status_code=404)
    session_id = row[0]
    finish_ts = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    # Only update if finished_at is NULL
    c.execute("UPDATE sessions SET finished_at=? WHERE session_id=? AND finished_at IS NULL", (finish_ts, session_id))
    # Insert metrics as session_summary_user event
    payload = json.dumps(metrics)
    c.execute("INSERT INTO events(session_id, event_type, timestamp, payload) VALUES (?, ?, ?, ?)",
              (session_id, 'session_summary_user', finish_ts, payload))
    conn.commit()
    conn.close()
    return JSONResponse(content={"status": "success", "session_id": session_id})

# API: GET /metadata/{arxiv_id} (returns validated metadata)
from fastapi import HTTPException
@app.get("/metadata/{arxiv_id}")
def get_metadata(arxiv_id: str):
    try:
        data = load_and_validate_metadata(arxiv_id)
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        # Try to fetch the paper and metadata using fetch_arxiv_paper
        try:
            import sys
            from pathlib import Path
            fetch_script_dir = Path(__file__).parent.parent / 'scripts' / 'literature'
            if str(fetch_script_dir) not in sys.path:
                sys.path.insert(0, str(fetch_script_dir))
            from fetch_paper import fetch_arxiv_paper
            success = fetch_arxiv_paper(arxiv_id)
            if success:
                data = load_and_validate_metadata(arxiv_id)
                return JSONResponse(content=data)
            else:
                raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} could not be fetched from arXiv.")
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Error fetching paper {arxiv_id}: {ex}")
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
