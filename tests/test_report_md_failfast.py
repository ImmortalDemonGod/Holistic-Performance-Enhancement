import os
import sys
import subprocess
import pathlib

def test_report_md_fails_on_missing_rollup():
    """Test that report_md.py exits with code 2 and error if no rollup CSV exists."""
    import shutil
    project_root = pathlib.Path(__file__).parent.parent.resolve()
    temp_root = project_root / "tmp_test_report_md_failfast"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir()
    try:
        script = project_root / "cultivation/scripts/software/dev_daily_reflect/report_md.py"
        script = script.resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        env["CULTIVATION_REPO_ROOT_OVERRIDE"] = str(temp_root)
        rollup_dir = temp_root / "cultivation/outputs/software/dev_daily_reflect/rollup"
        rollup_dir.mkdir(parents=True, exist_ok=True)
        for f in rollup_dir.glob("*.csv"):
            f.unlink()
        result = subprocess.run([sys.executable, str(script)], cwd=project_root, capture_output=True, text=True, env=env)
        assert result.returncode == 2
        assert "No rollup CSV found" in result.stderr
    finally:
        shutil.rmtree(temp_root)
