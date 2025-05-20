import json
import tempfile
import pathlib
import sys
import csv
from cultivation.scripts.software.dev_daily_reflect import aggregate_daily as agg_mod

# Ensure cultivation is on sys.path for import
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))  # Use parent of tests dir

def test_aggregate_daily_basic_rollup(monkeypatch):
    """Integration: aggregate_daily.py produces correct CSV from enriched JSON."""
    # Prepare a fake enriched commit JSON
    fake_commits = [
        {
            "sha": "abc123",
            "author": "alice",
            "timestamp": "2025-05-16 10:00:00 +0000",
            "message": "Initial commit",
            "added": 10,
            "deleted": 2,
            "py_files_changed_count": 1,
            "total_cc": 5,
            "avg_mi": 75.0,
            "ruff_errors": 1
        },
        {
            "sha": "def456",
            "author": "bob",
            "timestamp": "2025-05-16 12:00:00 +0000",
            "message": "Refactor",
            "added": 20,
            "deleted": 5,
            "py_files_changed_count": 2,
            "total_cc": 7,
            "avg_mi": 65.0,
            "ruff_errors": 0
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = pathlib.Path(tmpdir) / "raw"
        rollup_dir = pathlib.Path(tmpdir) / "rollup"
        raw_dir.mkdir()
        rollup_dir.mkdir()
        # Write fake enriched JSON
        date_tag = "2025-05-16"
        enriched_path = raw_dir / f"git_commits_enriched_{date_tag}.json"
        with open(enriched_path, "w") as f:
            json.dump(fake_commits, f)
        # Monkeypatch module constants to use temp dirs
        monkeypatch.setattr(agg_mod, "RAW_DIR", raw_dir)
        monkeypatch.setattr(agg_mod, "ROLLUP_DIR", rollup_dir)
        # Run main aggregation logic
        agg_mod.main([])
        # Check output CSV
        csv_files = list(rollup_dir.glob("dev_metrics_*.csv"))
        assert csv_files, "No rollup CSV produced"
        with open(csv_files[0]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Should have 2 rows (one per author)
        authors = {row["author"] for row in rows}
        assert "alice" in authors and "bob" in authors
        # Check metrics columns exist
        for row in rows:
            assert "total_cc" in row
            assert "avg_mi" in row
            assert "ruff_errors" in row
            assert "py_files_changed_count" in row
