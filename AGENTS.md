# AGENTS.md

Instructions for agents working in this repository. Full details in
`desarollo.md`.

> Language note: this file and `memory.md` are in English. The rest of the repo
> (README, `desarollo.md`, docstrings and code comments) is in Spanish. Keep
> new code comments and docs in Spanish to match.

## Hard rules

1. **Never change the logic of `colab/py/red_neuronal_ganado.py`.** It is the
   training source of truth. Comments and docstrings may be improved; code,
   configuration values and behavior must stay untouched. If the backend needs
   logic from it, copy the minimal code into `backend/src/` and
   leave the original intact.
2. **Do not reorganize training or move the checkpoint.** It stays at
   `data/NN/cattle_training/best_cattle_breed_model.pth`.
3. **Simplicity first:** the shortest diff that works. No abstractions,
   layers, options or "for later" code. Prefer the standard library or what is
   already installed over adding dependencies.
4. **Do not add tests for now.** If a change breaks `tests/test_inference.py`,
   fix it or flag it; do not grow the suite.
5. If you make a relevant decision (architecture, dependency, contract, path),
   record it in `memory.md`: date, decision, reason, outcome.

## Minimal map

| Path | What it is |
| --- | --- |
| `colab/py/red_neuronal_ganado.py` | Training and evaluation. Read-only |
| `backend/src/` | FastAPI: `main.py` (endpoints), `inference.py` (model), `config.py` (paths/env), `schemas.py` |
| `frontend/app`, `frontend/components`, `frontend/lib/api.ts` | Next.js: upload and results |
| `data/NN/cattle_training/` | Checkpoint and metrics |
| `desarollo.md` | Detailed explanation and design decisions (Spanish) |
| `memory.md` | Agent decision log (English) |

## Commands

```bash
# API (reuses the root .venv; do not create another one)
uv pip install --python .venv/bin/python -e backend
.venv/bin/uvicorn src.main:app --reload --port 8000

# Frontend
cd frontend && bun install && bun run dev

# Quality
.venv/bin/ruff check backend tests
.venv/bin/python -m pytest tests -q   # only to check you broke nothing
```

## API contract

`POST /api/v1/predictions` with multipart `file` returns:

```json
{ "model": "efficientnet_b0",
  "predictions": [{ "breed": "jersey-cattle", "confidence": 0.93 }] }
```

Do not change the contract without updating `frontend/lib/api.ts` and
`memory.md`.

## Conventions

- Backend: Python 3.12, single quotes (ruff), paths resolved from `config.py`
  (never from the cwd). The model loads once in `lifespan` and is stored in
  `app.state.engine`.
- Frontend: App Router, strict TypeScript, plain CSS in `app/globals.css`, no
  styling framework. Every API call goes through `lib/api.ts`.
- Never send the checkpoint to the browser; only labels and probabilities.
