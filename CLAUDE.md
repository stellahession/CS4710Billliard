# CLAUDE.md

Guidance for AI coding assistants working in this repo. See README.md for the project overview.

## Project

CS 4710 course project with four people and about three weeks (target finish: Dec 3, 2026).
The **AI agent (expectimax) is graded most heavily**. Prefer simple, readable code that a
teammate can explain in a presentation over clever or general code.

## Architecture

The modules talk to each other **only** through the types in `core/state.py`. Each module
provides one function:

- vision: `detect(image_path) -> TableState`
- physics: `simulate(state, shot, record_paths=False) -> ShotResult`
- agent: `recommend(state, skill) -> Recommendation`

The app uses `app/stubs.py` for any module that isn't available yet.

## Ownership

Each module has one owner: vision = Victoria, physics = Todd, agent = Nik, app/integration and
experiments = Stella. Keep changes inside the module you were asked to work on. If a task needs
changes to another person's module, or to anything in `core/`, say so explicitly rather than
quietly editing it. `core/` changes have to be announced to the whole team.

## Conventions

- **Units:** metres. The origin is the bottom-left of the playing surface and y points up.
  Never pass pixels between modules.
- **Ball ids:** 0 cue, 1–7 solids, 8 eight, 9–15 stripes. `pocket` is an index into `POCKETS`.
- **Skill names** are exactly `"Beginner"`, `"Intermediate"`, `"Advanced"`, matching the app.
- **`simulate` must not mutate its input** and must be fast (about 20 ms or less per call),
  because the agent calls it about 1,100 times per recommendation.
- Shared example tables live in `tests/fixtures/*.json` and load with `TableState.from_json`.

## Commands

Run everything from the repo root with Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app/app.py
pytest
```

Run modules as packages (`python -m physics.demo ...`), not as scripts, so imports like
`from core.state import ...` resolve.

## Out of scope (intentionally)

Spin and English, throw, real photos of pool tables as input, opponent modelling, and multi-shot search. These are
discussed in the final project presentation as extensions. Don't add them unless asked.
