"""Timing script: how long one simulate() call takes.

    python -m physics.bench [--n N]

Prints mean, p95 and max milliseconds per call for mid-game tables and for the
full-power break. The set-up matches the speed check in tests/test_physics_perf.py
(issue #8): seed 4710, 10 random tables of 8 balls, shots mostly aimed near an object
ball, default params and record_paths=False, which is how the agent calls it. --n sets
the number of mid-game shots and of break runs (default 250).

This is a dev tool; it imports only core and physics.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState
from physics.simulate import simulate

D = 2 * BALL_R
SEED = 4710
N_TABLES = 10
BALLS_PER_TABLE = 8             # cue + 7 object balls


def mid_game_table(rng: random.Random) -> TableState:
    """Cue plus 7 random object balls, kept off the cushions and clear of each other."""
    ids = [0] + rng.sample(range(1, 16), BALLS_PER_TABLE - 1)
    margin = BALL_R + 0.05
    placed: list[tuple[float, float]] = []
    while len(placed) < BALLS_PER_TABLE:
        p = (rng.uniform(margin, TABLE_W - margin), rng.uniform(margin, TABLE_H - margin))
        if all(math.dist(p, q) > D + 0.01 for q in placed):
            placed.append(p)
    return TableState([Ball(bid, x, y) for bid, (x, y) in zip(ids, placed)], "open")


def varied_shot(rng: random.Random, state: TableState) -> Shot:
    """Mostly aimed near a random object ball, like the agent's shots; some aimed anywhere."""
    cue = state.balls[0]
    if rng.random() < 0.8:
        target = rng.choice(state.balls[1:])
        angle = math.atan2(target.y - cue.y, target.x - cue.x) + rng.uniform(-0.05, 0.05)
    else:
        angle = rng.uniform(-math.pi, math.pi)
    return Shot(angle, rng.uniform(0.1, 1.0))


def mid_game_cases(n: int) -> list[tuple[TableState, Shot]]:
    """n shots spread over N_TABLES tables."""
    rng = random.Random(SEED)
    per_table = math.ceil(n / N_TABLES)
    cases = []
    for _ in range(N_TABLES):
        state = mid_game_table(rng)
        cases += [(state, varied_shot(rng, state)) for _ in range(per_table)]
    return cases[:n]


def break_table() -> TableState:
    """Cue on the head string, tight 15-ball rack with the apex on the foot spot."""
    balls = [Ball(0, TABLE_W / 4, TABLE_H / 2)]
    ids = iter([1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])
    for row in range(5):
        for k in range(row + 1):
            x = 3 * TABLE_W / 4 + row * (D * math.sqrt(3) / 2 + 1e-4)
            y = TABLE_H / 2 + (k - row / 2) * (D + 1e-4)
            balls.append(Ball(next(ids), x, y))
    return TableState(balls)


def time_cases(cases) -> dict:
    """Time simulate() once per (state, shot). Returns mean, p95 and max in ms."""
    simulate(*cases[0])                 # warm up
    times = []
    for state, shot in cases:
        start = time.perf_counter()
        simulate(state, shot)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"n": len(times), "mean": sum(times) / len(times),
            "p95": times[math.ceil(0.95 * len(times)) - 1], "max": times[-1]}


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m physics.bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=positive_int, default=250,
                        help="mid-game shots and break runs to time (default 250)")
    args = parser.parse_args(argv)

    runs = [("mid-game (8 balls, varied)", mid_game_cases(args.n)),
            ("break (16 balls, full power)", [(break_table(), Shot(0.0, 1.0))] * args.n)]
    for name, cases in runs:
        s = time_cases(cases)
        print(f"{name:30s} n={s['n']:<5d} mean={s['mean']:.2f} ms  "
              f"p95={s['p95']:.2f} ms  max={s['max']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
