"""Tests for issue #8: simulate() is fast enough for the agent.

The agent calls simulate() about 1,100 times per recommendation, so:

  - Mid-game: on tables of about 8 balls, over 200+ varied shots, the mean time per call
    is at most 10 ms and the p99 at most 20 ms. Most agent calls look like this.
  - Full-power break: the 16-ball break is timed and printed, but never fails the test.

The timing tests are marked @pytest.mark.slow. The `slow` marker is registered in
pyproject.toml and deselected by default, so a plain `pytest` stays fast and
`pytest -m slow` runs the speed check. The timings are printed even without -s, so they
can be posted for Nik and Stella.

Timings use the default params and record_paths=False, which is how the agent calls it.
"""
import math
import random
import statistics
import time

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState
from tests.test_physics_simulate import break_table

D = 2 * BALL_R
SEED = 4710
N_TABLES = 10
SHOTS_PER_TABLE = 25            # 250 shots in total
BALLS_PER_TABLE = 8             # cue + 7 object balls
MEAN_MAX_MS = 10.0
P99_MAX_MS = 20.0
CALLS_PER_RECOMMENDATION = 1100
BREAK_RUNS = 20


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


# --- helpers -----------------------------------------------------------------------

def mid_game_table(rng: random.Random) -> TableState:
    """Cue plus 7 random object balls, kept off the cushions and clear of each other."""
    ids = [0] + rng.sample([i for i in range(1, 16)], BALLS_PER_TABLE - 1)
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


def mid_game_cases() -> list[tuple[TableState, Shot]]:
    rng = random.Random(SEED)
    cases = []
    for _ in range(N_TABLES):
        state = mid_game_table(rng)
        cases += [(state, varied_shot(rng, state)) for _ in range(SHOTS_PER_TABLE)]
    return cases


def time_ms(simulate, state, shot) -> float:
    start = time.perf_counter()
    simulate(state, shot)
    return (time.perf_counter() - start) * 1000


def p99(times: list[float]) -> float:
    return statistics.quantiles(times, n=100, method="inclusive")[98]


def report(capsys, text: str):
    with capsys.disabled():
        print(f"\n{text}")


# --- marker set-up (not slow, so it runs on every plain pytest) ---------------------

def test_slow_marker_is_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(m.split(":")[0].strip() == "slow" for m in markers), (
        "register the slow marker under [tool.pytest.ini_options] markers in pyproject.toml")


def test_plain_pytest_deselects_slow_tests(pytestconfig):
    addopts = " ".join(pytestconfig.getini("addopts"))
    assert "not slow" in addopts, (
        "add -m \"not slow\" to addopts in pyproject.toml so a plain pytest stays fast")


def test_mid_game_cases_are_what_the_speed_check_expects():
    cases = mid_game_cases()
    assert len(cases) >= 200
    assert all(len(state.balls) == BALLS_PER_TABLE for state, _ in cases)
    assert len({(s.angle, s.power) for _, s in cases}) == len(cases)


# --- speed check -------------------------------------------------------------------

@pytest.mark.slow
def test_mid_game_mean_and_p99_are_fast_enough(simulate, capsys):
    cases = mid_game_cases()
    simulate(*cases[0])                                  # warm up
    times = [time_ms(simulate, state, shot) for state, shot in cases]
    mean, tail = statistics.fmean(times), p99(times)
    total_s = mean * CALLS_PER_RECOMMENDATION / 1000
    report(capsys, f"mid-game ({len(times)} shots, {BALLS_PER_TABLE} balls): "
                   f"mean {mean:.2f} ms, median {statistics.median(times):.2f} ms, "
                   f"p99 {tail:.2f} ms, max {max(times):.2f} ms; "
                   f"about {total_s:.1f} s per recommendation "
                   f"({CALLS_PER_RECOMMENDATION} calls)")
    assert mean <= MEAN_MAX_MS, f"mean {mean:.2f} ms is over {MEAN_MAX_MS} ms"
    assert tail <= P99_MAX_MS, f"p99 {tail:.2f} ms is over {P99_MAX_MS} ms"


@pytest.mark.slow
def test_full_power_break_is_timed(simulate, capsys):
    state, shot = break_table(), Shot(0.0, 1.0)
    simulate(state, shot)                                # warm up
    times = [time_ms(simulate, state, shot) for _ in range(BREAK_RUNS)]
    report(capsys, f"full-power break (16 balls, {BREAK_RUNS} runs): "
                   f"mean {statistics.fmean(times):.2f} ms, max {max(times):.2f} ms")
    # Measured and printed only; a slow break doesn't fail the test.
