# report_md.py
# Writes Markdown report for DevDailyReflect (MVP)
# Reads rollup CSV and writes Markdown report

#!/usr/bin/env python3
import pandas as pd
import sys
import json
from cultivation.scripts.software.dev_daily_reflect.utils import get_repo_root

# --- Configuration ---
REPO_ROOT = get_repo_root()
ROLLUP_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'rollup'
REPORTS_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def get_date_tag():
    rollup_files = sorted(ROLLUP_DIR.glob('dev_metrics_*.csv'))
    if not rollup_files:
        print('[WARN] No rollup CSV found. Exiting.')
        sys.exit(2)
    rollup_file = rollup_files[-1]
    return rollup_file, rollup_file.stem[-10:]

def main():
    # --- Find latest rollup CSV file ---
    rollup_file, date_tag = get_date_tag()

    # --- Load rollup data ---
    try:
        df = pd.read_csv(rollup_file)
    except Exception as e:
        print(f'[ERROR] Failed to read {rollup_file}: {e}')
        sys.exit(1)

    md_lines = []
    md_lines.append(f"# 🗓️ Daily Dev Reflect: {date_tag}")
    md_lines.append("")

    if df.empty:
        md_lines.append("**No commits in the last 24 hours.**")
    else:
        total_commits = df['commits'].sum()
        total_loc = df['loc_net'].sum()
        md_lines.append(f"**Total commits:** {total_commits}  |  **Total net LOC:** {total_loc}")
        # Code quality metrics summary
        if 'py_files_changed_count' in df.columns:
            total_py_files = df['py_files_changed_count'].sum()
            md_lines.append(f"**Total Python files changed:** {total_py_files}")
        if 'total_cc' in df.columns:
            total_cc = df['total_cc'].sum()
            md_lines.append(f"**Total Cyclomatic Complexity:** {total_cc}")
        if 'avg_mi' in df.columns:
            avg_mi = df['avg_mi'].mean()
            md_lines.append(f"**Average Maintainability Index:** {avg_mi:.2f}")
        if 'ruff_errors' in df.columns:
            total_ruff = df['ruff_errors'].sum()
            md_lines.append(f"**Total Ruff errors:** {total_ruff}")
        md_lines.append("")
        # Table header
        table_cols = ["Author", "Commits", "LOC Added", "LOC Deleted", "LOC Net"]
        if 'py_files_changed_count' in df.columns:
            table_cols.append("Py Files Changed")
        if 'total_cc' in df.columns:
            table_cols.append("Total CC")
        if 'avg_mi' in df.columns:
            table_cols.append("Avg MI")
        if 'ruff_errors' in df.columns:
            table_cols.append("Ruff Errors")
        md_lines.append("| " + " | ".join(table_cols) + " |")
        md_lines.append("|" + "|".join(["-"*len(c) for c in table_cols]) + "|")
        for _, row in df.iterrows():
            row_vals = [
                str(row['author']),
                str(row['commits']),
                str(row['loc_add']),
                str(row['loc_del']),
                str(row['loc_net'])
            ]
            if 'py_files_changed_count' in df.columns:
                row_vals.append(str(row['py_files_changed_count']))
            if 'total_cc' in df.columns:
                row_vals.append(str(row['total_cc']))
            if 'avg_mi' in df.columns:
                row_vals.append(f"{row['avg_mi']:.2f}")
            if 'ruff_errors' in df.columns:
                row_vals.append(str(row['ruff_errors']))
            md_lines.append("| " + " | ".join(row_vals) + " |")
        # Per-commit table if enriched data is available
        enriched_path = ROLLUP_DIR.parent / 'raw' / f'git_commits_enriched_{date_tag}.json'
        if enriched_path.exists():
            try:
                with open(enriched_path) as f:
                    commits = json.load(f)
                md_lines.append("\n## Per-Commit Metrics\n")
                commit_cols = ["SHA", "Author", "Added", "Deleted", "Py Files", "CC", "Avg MI", "Ruff"]
                md_lines.append("| " + " | ".join(commit_cols) + " |")
                md_lines.append("|" + "|".join(["-"*len(c) for c in commit_cols]) + "|")
                for c in commits:
                    md_lines.append(
                        f"| {c.get('sha','')[:7]} | {c.get('author','')} | {c.get('added',0)} | {c.get('deleted',0)} | "
                        f"{c.get('py_files_changed_count','')} | {c.get('total_cc','')} | "
                        f"{c.get('avg_mi',''):.2f} | {c.get('ruff_errors','')} |"
                    )
            except Exception as e:
                md_lines.append(f"\n[WARN] Could not load per-commit metrics: {e}")

    # --- Write report ---
    report_file = REPORTS_DIR / f'dev_report_{date_tag}.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'[✓] wrote {report_file}')

if __name__ == "__main__":
    main()
