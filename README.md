# Pool Shot Recommender

CS 4710 (Artificial Intelligence) project. Upload a screenshot of a mobile pool game, pick your
skill level, and get a recommended next shot. **Expectimax** over noisy shot execution means
beginners get safer, more achievable shots and advanced players get harder, higher-value ones.

**Team:** Stella Hession, Nikhil Kapadia, Todd Burged, Victoria Lovelace

## How it works

```
screenshot ──► vision ──► TableState ──► agent (expectimax) ──► Recommendation ──► app (draws it)
                                               │
                                               └─► physics (simulates each sampled shot)
```

1. **Vision** finds the table and balls in a top-down screenshot and classifies each ball as
   cue, eight, solid, or stripe.
2. **Physics** is a simplified 2D simulator. It handles ball-ball collisions, cushion rebounds,
   rolling friction, pocketing, and scratches. It ignores spin and throw (see *Limitations*).
3. **Agent** generates candidate shots by aiming at the "ghost ball" for each target and pocket.
   It then samples each shot with skill-dependent aim and power noise (the chance node) and
   picks the shot with the highest expected value (the max node).
4. **App** is a Streamlit UI with an editable ball table, a skill selector, and a drawing of the
   recommended shot and its alternatives.

## Setup

Requires **Python 3.10+** (the code uses `X | None` type syntax).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run from the repo root:

```bash
streamlit run app/app.py
pytest
```

Until a module is implemented, the app uses the stand-ins in `app/stubs.py`.

## Repo layout

| Path | What | Owner |
| --- | --- | --- |
| `core/` | Shared types (`Ball`, `TableState`, `Shot`, `ShotResult`, `Recommendation`) and table geometry | everyone (announce changes) |
| `vision/` | Screenshot → `TableState` | Victoria |
| `physics/` | Simulator, table constants, debug viewer | Todd |
| `agent/` | Candidate shots, expectimax, evaluation function, skill noise models, baselines | Nik |
| `app/` | Streamlit app, stubs, random table generator | Stella |
| `experiments/` | Experiment runner | Stella + Nik |
| `tests/` | Tests, plus shared example tables in `tests/fixtures/*.json` | everyone |

## Conventions

- **Units:** metres on a 9-ft table (2.54 × 1.27 m). The origin is the bottom-left corner of the
  playing surface, x runs along the long rail, and y points up.
- **Ball ids:** 0 = cue, 1–7 solids, 8 = eight ball, 9–15 stripes.
- **Shot:** `angle` is in radians (the cue ball's starting direction). `power` runs 0–1 and the
  physics module scales it to a maximum cue-ball speed.
- **Pockets:** `pocket` is an index into `core.state.POCKETS`.

## Workflow

- `main` always runs. Work on a branch and merge through a pull request.
- Changes to `core/` affect everyone, so post them in the group chat first.
- Every module must run on its own against the shared example tables, e.g.
  `python -m physics.demo tests/fixtures/simple_cut.json`.

## Limitations and possible extensions

- **Input:** only top-down screenshots from one mobile game (Billiard Shot Studio, iPhone).
  Real table photos would need perspective correction, lighting-robust detection, and a
  real ball classifier.
- **Physics:** no spin, English, cut-induced throw, or jump shots. Friction is a single constant
  deceleration. [pooltool](https://github.com/ekiefl/pooltool)
  ([JOSS paper](https://theoj.org/joss-papers/joss.07301/10.21105.joss.07301.pdf)) shows what a
  full model looks like.
- **Agent:** looks one shot ahead plus a position heuristic. It does not search the opponent's
  replies and does not plan safety play.
