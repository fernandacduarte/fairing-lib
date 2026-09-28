# Progress

## Status
- [x] #1 Project setup — PR #17

## Next
- #2 Synthetic meshes and OBJ I/O (Phase 1)

## Decisions
- **Issue numbers = plan step numbers.** All issues were created before any PR, so issue #N is step N of PLAN.md §3 (M1 is #16).
- **Topology helpers live in `fairing/mesh.py`**, as in the PLAN.md §1 layout; there is no separate module.
- **Figures are named `docs/img/<NN>-<name>.png`** with a zero-padded issue number, so they sort in order.
- **Local environment:** `.venv` with Python 3.11 (`pip install -e ".[dev]"`); CI uses Python 3.11 too.

## Gotchas
- The book PDFs must never enter the repo (it is public). `.gitignore` excludes `*.pdf` as a safety net.
