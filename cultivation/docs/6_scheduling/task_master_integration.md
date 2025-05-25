*📄 Save this file as **`docs/2_requirements/task_master_integration.md`** (or wherever your repo’s ToC points).  
It consolidates **every relevant Task Master detail** into one self‑contained reference, trimmed of repetition and cross‑linked to the rest of the Cultivation stack.*

---

# 🗂️ Task Master × Cultivation — Integration & Usage Guide v 1.0

> **Mission** – turn high‑level PRDs into sprint‑ready tasks, keep them version‑controlled, and let Cursor/Claude automate the drudge work—without polluting the codebase or slowing CI.

---

## 0 · Why Task Master?

| Problem | Task Master answer |
|---------|-------------------|
| Backlogs rot in Notion/Jira silos. | Tasks live in Git, evolve by PR, reviewed like code. |
| AI agents forget context. | MCP endpoints expose a single canonical `tasks.json`. |
| Dependency spaghetti stalls sprints. | CLI enforces DAGs, auto‑expands complex tasks via Claude/Perplexity. |

---

## 1 · High‑level design

```text
docs/                               # ← human docs
├─ 2_requirements/
│  └─ task_master_integration.md    # ← THIS FILE
├─ 3_planning/                      # ← generated road‑maps, graphs
├─ 4_analysis/                      # ← post‑mortems & metrics
.taskmaster/                        # ← machine state (single source of truth)
│  ├─ tasks.json
│  ├─ task‑complexity‑report.json
│  └─ .env                          # secrets (git‑ignored)
tools/                              # optional git‑submodule here
scripts/                            # tiny shims & hooks
package.json                        # adds task‑master‑ai as dev‑dep
```

*Separation:*  
`3_planning/` looks *forward* (road‑maps) — `4_analysis/` looks *back* (lessons learned).

---

## 2 · Installation & bootstrap

```bash
# 1 Add dependency
npm i -D task-master-ai

# 2 Scaffold state
npx task-master init --project "Cultivation" --path .taskmaster

# 3 Turn your PRD into tasks
npx task-master parse-prd docs/3_planning/prd.md

# 4 Generate individual task files (optional, nice for code search)
npx task-master generate
```

> Commit `.taskmaster/tasks.json` and the first task files so reviewers see the roadmap diff.

---

## 3 · Configuration (.env)

```dotenv
ANTHROPIC_API_KEY=
# optional extras
PERPLEXITY_API_KEY=
MODEL=claude-3-sonnet
MAX_TOKENS=4000
TEMPERATURE=0.7
```

---

## 4 · Daily CLI cheat‑sheet

```bash
# list tasks (with dependency status glyphs)
npx task-master list --with-subtasks

# next actionable task
npx task-master next

# mark done / defer
npx task-master set-status --id 8 --status done

# break down a hairy task into 5 AI‑generated subtasks
npx task-master expand --id 12 --num 5 --prompt "security‑first"

# analyse complexity, then expand everything with smart counts
npx task-master analyze-complexity --research
npx task-master expand --all
```

---

## 5 · Cursor / MCP usage

1. **Add MCP server** in Cursor →  
   *Name:* “Task Master”  *Type:* Command  *Cmd:* `npx -y task-master-mcp`
2. Open chat (Agent mode) and say:  

   ```
   Parse the PRD at docs/3_planning/prd.md and show me the next task.
   ```

The agent proxies `parse-prd`, `next`, `expand`, etc. automatically.

---

## 6 · CI integration

### 6.1 Fast lint (every PR)

```yaml
# .github/workflows/ci.yml (snippet)
- uses: actions/setup-node@v4
  with: { node-version: 20, cache: npm }

- run: npm ci            # installs task-master-ai
- run: npx task-master list --status pending
```

### 6.2 Artifact diff (optional)

```yaml
- uses: actions/upload-artifact@v4
  with:
    name: taskmaster-state
    path: .taskmaster/tasks.json
```

### 6.3 Nightly complexity refresh (heavy)

```yaml
# .github/workflows/taskmaster-nightly.yml
schedule: [{cron: "0 22 * * *"}]
jobs:
  complexity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci
      - run: npx task-master analyze-complexity --research
      - run: npx task-master expand --all
```

Fails don’t block the main pipeline; they ping `#taskmaster-alerts`.

---

## 7 · Hooks into Flash‑Memory layer

| Flow | Script | Effect |
|------|--------|--------|
| **Task → Card** | `scripts/post_task_complete.py` | On `status=done` & description contains `[[fc]]`, run `tm‑fc add …`. |
| **Card → Task** | Nightly analytics | If recall lapses > 3 on code touched in last 30 d, open a GH issue referencing the task. |

---

## 8 · Developer conventions

1. **Truth = `.taskmaster/tasks.json`** (edit via CLI, *not* by hand).  
2. Always run `npx task-master generate` after modifying tasks to keep per‑file views fresh.  
3. PR template line:  
   > `[ ] Updated Task Master backlog / statuses if this PR adds or finishes work`.

---

## 9 · Best‑practice workflow

| Stage | Command / Action |
|-------|------------------|
| New feature idea | Add to PRD, run `parse-prd`, review generated tasks. |
| Sprint planning  | `analyze-complexity` → expand heavy tasks → triage priorities. |
| During sprint    | `next` → implement → `set-status done`. |
| Scope drift      | `update --from 15 --prompt "switch to Express"` |
| Retrospective    | Move post‑mortem to `docs/4_analysis/` and link task IDs. |

---

## 10 · Open roadmap (import into GH Projects)

| # | Title | Tags | ETA |
|---|-------|------|-----|
| 45 | Add Task Master dev‑dependency + wrapper scripts | tooling | Day 1 |
| 46 | Commit baseline tasks + first roadmap | planning | Day 1 |
| 52 | CI artifact diff & alert bot | ci | Day 3 |
| 57 | Cursor MCP boilerplate docs | docs | Day 4 |

---

## 11 · FAQ

| Q | A |
|---|---|
| *Can I keep Task Master outside `devDependencies`?* | Yes, use a git submodule or Docker mode; update CI paths accordingly. |
| *How do I change models?* | Set `MODEL` in `.taskmaster/.env` or export it in CI secrets. |
| *Circular dependency?* | `npx task-master validate-dependencies` shows and fixes loops. |
| *I hate the generated subtasks.* | `clear-subtasks --id 12`, then `expand` with a new prompt. |

---

### ⚡TL;DR

1. `npm i -D task-master-ai`  
2. `npx task-master init --path .taskmaster`  
3. PRDs in `docs/3_planning/`, tasks in `.taskmaster/`, CI lint is cheap.  
4. Cursor + MCP lets Claude juggle the backlog for you.

Now drop this file in the repo, wire the three tiny scripts, and your planning loop is automation‑ready. 🚀
