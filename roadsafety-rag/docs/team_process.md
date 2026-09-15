# Team Process

## Branch / commit conventions
- `main` is always demo-able. Never commit directly to `main`.
- Branch naming: `<area>/<short-description>` e.g. `ingestion/pdf-parsing`,
  `eval/stage-precision-metric`, `ui/streamlit-sidebar`.
- Commit messages: `<area>: <what changed>` e.g. `chunking: reduce chunk size to 150 words`.
- Open a PR into `main` for every change, even solo work - keeps a review
  trail for the report's "who did what" and catches breakage early.

## Ownership split (rough - everyone still touches eval)
- **Ingestion / retrieval**: data/sources.csv upkeep, src/ingestion.py,
  src/chunking.py, src/embed_store.py, src/retrieval.py
- **Generation / UI**: src/generation.py, app.py, prompt iteration
- **Evaluation framework**: eval/test_set_template.csv, eval/evaluate_full.py -
  everyone contributes test questions for their persona/area; this is the
  project's core differentiator so treat it as first-class, not an
  afterthought bolted on at the end.

## Weekly check-in (paste into your sprint board each week)
- What shipped this week (link PRs)
- What's blocked / needs another team member
- Any eval numbers that changed (paste `evaluate_full.py` summary if you ran it)
- Individual contribution log entry (name - what you did - roughly how long)

## Sprint board
Set this up in GitHub Projects from Week 1 with columns: Backlog / In
Progress / In Review / Done, and one card per checklist item from the
project brief so progress against the brief is directly visible.
