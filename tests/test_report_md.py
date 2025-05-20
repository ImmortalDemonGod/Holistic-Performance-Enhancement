import tempfile
import pathlib
import sys
from importlib import import_module
import csv

# Ensure cultivation is on sys.path for import
sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
report_mod = import_module("cultivation.scripts.software.dev_daily_reflect.report_md")

def test_report_md_basic(monkeypatch):
    """Integration: report_md.py produces expected Markdown from rollup CSV."""
    # Prepare a fake rollup CSV
    rows = [
        {
            "author": "alice",
            "commits": "1",
            "loc_add": "10",
            "loc_del": "2",
            "loc_net": "8",
            "py_files_changed_count": "1",
            "total_cc": "5",
            "avg_mi": "75.0",
            "ruff_errors": "1",
        },
        {
            "author": "bob",
            "commits": "2",
            "loc_add": "20",
            "loc_del": "5",
            "loc_net": "15",
            "py_files_changed_count": "2",
            "total_cc": "7",
            "avg_mi": "65.0",
            "ruff_errors": "0",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        rollup_dir = pathlib.Path(tmpdir)
        rollup_file = rollup_dir / "dev_metrics_2025-05-16.csv"
        with open(rollup_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        # Monkeypatch module constants to use temp dir
        monkeypatch.setattr(report_mod, "ROLLUP_DIR", rollup_dir)
        # Monkeypatch date_tag to match test file
        monkeypatch.setattr(report_mod, "get_date_tag_and_file", lambda target_date=None: (rollup_file, "2025-05-16"))
        # Debug: show ROLLUP_DIR and its files
        print(f"ROLLUP_DIR used by report_mod: {report_mod.ROLLUP_DIR}")
        print(f"Files in ROLLUP_DIR: {list(report_mod.ROLLUP_DIR.glob('*'))}")
        # Run report generation
        report_mod.main([])
        # Read the generated Markdown file
        report_path = report_mod.REPORTS_DIR / 'dev_report_2025-05-16.md'
        with open(report_path) as f:
            md = f.read()
        print("Generated Markdown contents:\n", md)
        # Check that Markdown contains expected author names and metrics
        assert "alice" in md and "bob" in md
        assert "commits" in md
        assert "Total CC" in md
        assert "Avg MI" in md
        assert "Ruff Errors" in md
