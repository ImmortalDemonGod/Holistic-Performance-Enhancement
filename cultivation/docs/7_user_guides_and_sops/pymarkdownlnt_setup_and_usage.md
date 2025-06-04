# PyMarkdown Lint Setup and Usage Guide

This guide explains how to set up and use [PyMarkdown Lint (`pymarkdownlnt`)](https://github.com/jackdewinter/pymarkdown) in the Cultivation project to ensure Markdown documentation quality and consistency.

---

## 1. Installation

PyMarkdown Lint is installed in the project’s local virtual environment (`.venv`). If you need to reinstall:

```bash
.venv/bin/pip install pymarkdownlnt
```

---

## 2. Configuration

A `.pymarkdown.json` configuration file is located at the project root. It enables recommended rules and sets the line length to 120 characters. You can customize rules further by editing this file. See `docs/rules/` and `newdocs/src/plugins/` for rule documentation.

---

## 3. Running the Linter

To lint all documentation in `cultivation/docs/`:

```bash
.venv/bin/pymarkdown scan cultivation/docs/
```

You can also scan other folders (e.g., `docs/`, `newdocs/`) by changing the path.

---

## 4. Fixing Lint Issues

- Review the output for rule violations (e.g., missing headings, line too long, improper emphasis).
- Fix issues directly in your Markdown files.

---

## 5. Optional: Pre-commit Hook

To enforce Markdown linting before every commit, add this to `.pre-commit-config.yaml`:

```yaml
-   repo: local
    hooks:
      - id: pymarkdown
        name: PyMarkdown Lint
        entry: .venv/bin/pymarkdown scan cultivation/docs/
        language: system
        types: [markdown]
```

Then install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

---

## 6. Optional: GitHub Actions CI

Add this step to your workflow YAML to lint Markdown on every PR:

```yaml
- name: Lint Markdown
  run: .venv/bin/pymarkdown scan cultivation/docs/
```

---

## 7. Customizing Rules

Edit `.pymarkdown.json` to enable/disable rules or change rule settings. Reference:
- [PyMarkdown Official Rule Docs](https://github.com/jackdewinter/pymarkdown/blob/main/docs/rules.md)
- Project’s `docs/rules/` and `newdocs/src/plugins/`

---

## 8. Contributor Instructions

> **Before submitting a PR:**
> - Run `.venv/bin/pymarkdown scan cultivation/docs/` and fix any issues.
> - See `.pymarkdown.json` for rule settings.

---

## 9. Contact

For help or to propose rule changes, contact the maintainers or open an issue.
