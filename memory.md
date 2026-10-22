# 🧠 Agent Memory — Redes Neuronales

> Language note: this file and `AGENTS.md` are in English. Everything else in
> the repo (README, `desarollo.md`, docstrings and code comments) is in Spanish.

## Current project state

- **Problem type:** Image classification (cattle breeds)
- **Dataset:** `data/NN/cattle_dataset` (5 breeds)
- **Architecture:** EfficientNet-B0 (torchvision)
- **Training status:** Done; best epoch 15, test macro F1 0.94
- **Last iteration:** Inference API + Next.js frontend working

---

## Decision history

| Date | Decision | Reason | Outcome |
|------|----------|--------|---------|
| 2026-10-22 | Training lives in `colab/py/red_neuronal_ganado.py`, backend kept separate | Keep the Colab script untouched and stop the API from depending on data preparation | ✅ Independent API, script unchanged |
| 2026-10-22 | FastAPI backend in `backend/src/cattle_api` with `POST /api/v1/predictions` | Separate model loading from the prediction service | ✅ Verified on CPU and CUDA |
| 2026-10-22 | API dependencies in `backend/pyproject.toml`, installed into the root `.venv` (no second environment) | Isolate dependencies without duplicating several GB of PyTorch | ✅ Single environment |
| 2026-10-22 | Checkpoint and device configurable via `CATTLE_*` env vars; paths resolved in `config.py` | The service must start from any working directory | ✅ Not dependent on cwd |
| 2026-10-22 | Next.js frontend (App Router, TypeScript, plain CSS) that only calls the API | The browser must never receive the model | ✅ Build and types pass |
| 2026-10-22 | No new tests for now; three exist in `tests/test_inference.py` | Favor a simple delivery | ✅ `pytest` green |
| 2026-10-22 | Allowed explanatory comments in `colab/py/red_neuronal_ganado.py`; logic unchanged | User needs to understand the network; the script stays the training source of truth | ✅ Compiles, ruff unchanged |

> **Note:** Record every relevant decision here with date, description and
> outcome. This lets the next agent learn from past iterations and avoid
> repeating mistakes.

---

## Historical metrics per iteration

| Iteration | Architecture | Epochs | Best val macro F1 | Test macro F1 | Comment |
|-----------|-------------|--------|-------------------|---------------|---------|
| 1 | EfficientNet-B0 | 15 | 0.966 | 0.941 | Test accuracy 0.942; 7 misclassified of 121 |

---

## Known issues and fixes

| Problem | Root cause | Fix applied | Effective? |
|---------|-----------|-------------|------------|
| — | — | — | — |

---

## Agent notes

Free-form notes during development:

- [x] Define the problem type (image classification)
- [x] Choose a suitable dataset
- [x] Set up the training environment (Google Colab)
- [x] Train and evaluate EfficientNet-B0
- [x] Implement the inference API and frontend
- [ ] Define deployment (server, authentication, limits)
