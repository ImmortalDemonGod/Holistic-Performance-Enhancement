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

def extract_coderabbit_walkthrough(comments):
    if not comments:
        return "> No CodeRabbit Walkthrough comment found."

    start_html_marker = '<!-- walkthrough_start -->'
    end_html_marker = '<!-- walkthrough_end -->'
    walkthrough_md_header = '## Walkthrough'

    for comment in comments:
        author_login = comment.get('author', {}).get('login', '').lower()
        body = comment.get('body', '')

        is_coderabbit_comment = 'coderabbitai' in author_login or 'coderabbit-ai' in author_login

        if not is_coderabbit_comment:
            continue

        # Find the start of the HTML comment block for the walkthrough
        html_start_index = body.find(start_html_marker)
        if html_start_index == -1:
            continue

        # Adjust index to be after the start marker
        content_start_index = html_start_index + len(start_html_marker)

        # Find the end of the HTML comment block for the walkthrough
        html_end_index = body.find(end_html_marker, content_start_index)
        if html_end_index == -1:
            # If no end marker, this might be a malformed comment or not the one we want
            continue
        
        # Extract the segment between the HTML markers
        walkthrough_segment = body[content_start_index:html_end_index]

        # Within this segment, find the '## Walkthrough' markdown header
        md_header_start_index = walkthrough_segment.find(walkthrough_md_header)

        if md_header_start_index != -1:
            # Return the content from '## Walkthrough' onwards, stripped
            return walkthrough_segment[md_header_start_index:].strip()

    return "> No CodeRabbit Walkthrough comment found."

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
        comments_text = extract_top_comments(pr.get("comments", []))
        coderabbit_walkthrough_text = extract_coderabbit_walkthrough(pr.get("comments", []))

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
{comments_text}

## CodeRabbit Walkthrough
{coderabbit_walkthrough_text}

{log_section}
"""
        outname = f"pr_{number}_{title_slug}.md"
        outfile = os.path.join(OUTPUT_DIR, outname)
        with open(outfile, "w") as outf:
            outf.write(md)
    print(f"Wrote {len(pr_list)} PR markdown summaries to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
