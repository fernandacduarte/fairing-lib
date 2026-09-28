# fairing-lib

A small Python library for studying **mesh fairing** as presented in *Polygon Mesh Processing* (Botsch et al.), Chapters 3–4 and Appendix A. Each step is small, cites the equation it implements, and comes with a test and a figure. See [PLAN.md](PLAN.md) for the roadmap and [docs/PROGRESS.md](docs/PROGRESS.md) for the current status.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
