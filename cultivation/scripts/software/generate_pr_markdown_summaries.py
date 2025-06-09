#!/usr/bin/env python3
"""
Generate Markdown summaries for each pull request from the full PR JSON file.
- Reads: github_automation_output/pull_requests/pull_requests_full_pretty.json
- Writes: github_automation_output/pr_markdown_summaries/pr_<number>_<title_slug>.md
- Usage: .venv/bin/python cultivation/scripts/software/generate_pr_markdown_summaries.py
"""
import os
import json
import re
from datetime import datetime

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PR_JSON_PATH = os.path.join(SCRIPT_DIR, "github_automation_output", "pull_requests", "pull_requests_full_pretty.json")
OUTPUT_DIR = "/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/outputs/software/pr_markdown_summaries"

# Utility functions
def slugify(text, maxlen=40):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text).strip('-')
    return text[:maxlen]

def format_dt(dtstr):
    if not dtstr:
        return "-"
    try:
        return datetime.fromisoformat(dtstr.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
    except Exception:
        return dtstr

def truncate(text, maxlen=500):
    if text is None:
        return "-"
    if len(text) <= maxlen:
        return text
    return text[:maxlen] + "... (truncated)"

def extract_top_comments(comments, max_comments=3):
    if not comments:
        return "-"
    lines = []
    for c in comments[:max_comments]:
        author = c.get('author', {}).get('login', '-')
        body = truncate(c.get('body', ''), 300)
        lines.append(f"- **{author}**: {body}")
    return '\n'.join(lines)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PR_JSON_PATH, "r") as f:
        pr_list = json.load(f)

    for pr in pr_list:
        number = pr.get("number")
        title = pr.get("title", "Untitled")
        title_slug = slugify(title)
        author = pr.get("author", {}).get("login", "-")
        state = pr.get("state", "-")
        created = format_dt(pr.get("createdAt"))
        closed = format_dt(pr.get("closedAt"))
        merged = format_dt(pr.get("mergedAt"))
        base = pr.get("baseRefName", "-")
        head = pr.get("headRefName", "-")
        files = pr.get("changedFiles", "-")
        additions = pr.get("additions", "-")
        deletions = pr.get("deletions", "-")
        body = pr.get("body", "-")
        comments = extract_top_comments(pr.get("comments", []))

        # Find log file by PR number
        pr_logs_dir = os.path.join(SCRIPT_DIR, "github_automation_output", "pr_logs")
        log_file = None
        for fname in os.listdir(pr_logs_dir):
            if fname.startswith(f"pr{number}_") and fname.endswith(".log"):
                log_file = os.path.join(pr_logs_dir, fname)
                break
        if log_file and os.path.exists(log_file):
            with open(log_file, "r") as lf:
                lines = lf.readlines()
            commit_lines = [line.rstrip() for line in lines if line.lstrip().startswith('* ')]
            if commit_lines:
                log_content = '\n'.join(commit_lines)
            else:
                log_content = '> No commit summary lines found in log.'
            log_section = f"## Git Commit Log\n\n```text\n{log_content}\n```\n"
        else:
            log_section = "## Git Commit Log\n\n> No commit log found for this PR.\n"

        md = f"""# PR #{number}: {title}

- **Author:** {author}
- **State:** {state}
- **Created:** {created}
- **Closed:** {closed}
- **Merged:** {merged}
- **Base branch:** `{base}`
- **Head branch:** `{head}`
- **Files changed:** {files}
- **Additions:** {additions}
- **Deletions:** {deletions}

## Summary
{body}

## Top-level Comments
{comments}

{log_section}
"""
        outname = f"pr_{number}_{title_slug}.md"
        outfile = os.path.join(OUTPUT_DIR, outname)
        with open(outfile, "w") as outf:
            outf.write(md)
    print(f"Wrote {len(pr_list)} PR markdown summaries to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
