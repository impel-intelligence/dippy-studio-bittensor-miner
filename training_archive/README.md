# Training Archive

This directory keeps the legacy fine-tuning stack for reference. None of it is wired into the current miner build or Makefile.

Contents:
- `training_server.py`, `config_generator.py`, and `run.py` for the legacy API entrypoint
- `toolkit/`, `jobs/`, and `extensions_built_in/` with the full training implementation
- Archived test `tests/e2e/async_inference/test_race_conditions.py`

Notes:
- Use the root `requirements.fluxdev.txt` if you need to recreate the old environment.
- Run tooling from inside this folder so imports resolve locally (`python training_server.py`, etc.).
- Expect to maintain these pieces yourself; they are no longer part of active development.
