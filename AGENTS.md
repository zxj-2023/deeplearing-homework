# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core LinearRAG implementation (`LinearRAG.py`, `embedding_store.py`, `evaluate.py`, `ner.py`, `utils.py`).
- `run.py`, `run_build_index_all.py`, `run_acc.py`: Entry points for single-run QA, batch indexing, and batch QA+evaluation.
- `import/`: Cached indexes and embeddings (`*.parquet`, `*.graphml`, `ner_results.json`).
- `results/`: Logs, timing summaries, and prediction outputs.
- `dataset/<name>/`: Expected input data (`questions.json`, `chunks.json`).
- `model/`: Local embedding model (created by `download_embedding_model.py`).

## Build, Test, and Development Commands
- `python -m venv .venv` / `pip install -r requirements.txt`: Set up Python environment.
- `python download_embedding_model.py`: Download `all-mpnet-base-v2` into `model/`.
- `python check_torch.py`: Verify PyTorch/CUDA availability.
- `python run_build_index_all.py`: Build indexes for all datasets in `dataset/`.
- `python run.py`: Run QA + evaluation for one dataset (config in `run.py`).
- `python run_acc.py`: Batch QA + evaluation across datasets, output in `results/`.

## Coding Style & Naming Conventions
- Language: Python 3; use 4-space indentation and UTF-8 files.
- Naming: `snake_case` for functions/files, `CamelCase` for classes (e.g., `LinearRAGConfig`).
- Keep configs in code for now (`get_config()` in scripts); avoid hidden globals.

## Testing Guidelines
- No dedicated automated tests are present.
- Validate changes by running the relevant script(s) and checking outputs in `results/`
  (e.g., `results/<dataset>/<timestamp>/predictions.json` and log files).
- If you add tests, keep them lightweight and runnable with a single command.

## Commit & Pull Request Guidelines
- This workspace has no Git history available; use a clear convention such as
  `type(scope): summary` (e.g., `feat(eval): add timing stats`).
- PRs should include: purpose, datasets used, commands run, and any output samples
  (e.g., metrics from `results/qa_eval_summary.json`).
- If you modify data artifacts under `import/` or `results/`, call this out explicitly.

## Security & Configuration Tips
- LLM access uses `DASHSCOPE_API_KEY` via `.env` (loaded in `src/utils.py`).
- Avoid committing secrets or large generated artifacts; keep datasets and embeddings local.
