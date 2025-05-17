import os
import sys
import subprocess
import pathlib
import pytest

def test_report_md_fails_on_missing_rollup(monkeypatch):
    """Test that report_md.py exits with code 2 and error if no rollup CSV exists."""
    import tempfile
    import shutil
    # Determine project root
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    # Create a temp subdir inside the project root for isolation
    temp_root = project_root / "tmp_test_report_md_failfast"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir()
    try:
        script = project_root / "cultivation/scripts/software/dev_daily_reflect/report_md.py"
        script = script.resolve()
        utils_path = script.parent / "utils.py"
        import importlib.util
        sys.modules.pop('cultivation.scripts.software.dev_daily_reflect.utils', None)
        spec = importlib.util.spec_from_file_location("cultivation.scripts.software.dev_daily_reflect.utils", utils_path)
        utils_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils_mod)
        monkeypatch.setattr(utils_mod, "get_repo_root", lambda start_path=None: temp_root)
        sys.modules["cultivation.scripts.software.dev_daily_reflect.utils"] = utils_mod
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        # Ensure there are no rollup CSVs
        rollup_dir = temp_root / "cultivation/outputs/software/dev_daily_reflect/rollup"
        rollup_dir.mkdir(parents=True, exist_ok=True)
        for f in rollup_dir.glob("*.csv"):
            f.unlink()
        # Run the script from project root
        result = subprocess.run([sys.executable, str(script)], cwd=project_root, capture_output=True, text=True, env=env)
        assert result.returncode == 2
        assert "No rollup CSV found" in result.stdout or result.stderr
    finally:
        shutil.rmtree(temp_root)
