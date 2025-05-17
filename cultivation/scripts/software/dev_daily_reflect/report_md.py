# report_md.py
# Writes Markdown report for DevDailyReflect (MVP)
# Reads rollup CSV and writes Markdown report

import pandas as pd
import pathlib
import datetime
import sys

# --- Configuration ---
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ROLLUP_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'rollup'
REPORTS_DIR = REPO_ROOT / 'cultivation' / 'outputs' / 'software' / 'dev_daily_reflect' / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Find latest rollup CSV file ---
rollup_files = sorted(ROLLUP_DIR.glob('dev_metrics_*.csv'))
if not rollup_files:
    print('[WARN] No rollup CSV found. Exiting.')
    sys.exit(0)
rollup_file = rollup_files[-1]
date_tag = rollup_file.stem[-10:]

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
    md_lines.append("")
    md_lines.append("| Author | Commits | LOC Added | LOC Deleted | LOC Net |")
    md_lines.append("|--------|---------|-----------|-------------|---------|")
    for _, row in df.iterrows():
        md_lines.append(f"| {row['author']} | {row['commits']} | {row['loc_add']} | {row['loc_del']} | {row['loc_net']} |")

# --- Write report ---
report_file = REPORTS_DIR / f'dev_report_{date_tag}.md'
with open(report_file, 'w') as f:
    f.write('\n'.join(md_lines) + '\n')
print(f'[✓] wrote {report_file}')
