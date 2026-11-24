"""Tests for issue #8: simulate() is fast enough for the agent.

The agent calls simulate() about 1,100 times per recommendation, so:

  - Mid-game: on tables of about 8 balls, over 200+ varied shots, the mean time per call
    is at most 10 ms and the p99 at most 20 ms. Most agent calls look like this.
  - Full-power break: the 16-ball break, the worst case, has a mean of at most
    BREAK_MEAN_MAX_MS. That is CLAUDE.md's "about 20 ms" plus some slack for slower
    laptops (it measures about 18 ms on Todd's).

The timing tests are marked @pytest.mark.slow. The `slow` marker is registered in
pyproject.toml and deselected by default, so a plain `pytest` stays fast and
`pytest -m slow` runs the speed check. The timings are printed even without -s, so they
can be posted for Nik and Stella.

Timings use the default params and record_paths=False, which is how the agent calls it.
"""
import statistics
import time

import pytest

from core.state import Shot
from physics import bench
from physics.bench import BALLS_PER_TABLE, N_TABLES, break_table

SHOTS_PER_TABLE = 25            # 250 shots in total
MEAN_MAX_MS = 10.0
P99_MAX_MS = 20.0
CALLS_PER_RECOMMENDATION = 1100
BREAK_RUNS = 20
BREAK_MEAN_MAX_MS = 25.0


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


# --- helpers -----------------------------------------------------------------------

def mid_game_cases():
    """The same 250 shots that python -m physics.bench times by default."""
    return bench.mid_game_cases(N_TABLES * SHOTS_PER_TABLE)


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
def test_full_power_break_is_fast_enough(simulate, capsys):
    state, shot = break_table(), Shot(0.0, 1.0)
    simulate(state, shot)                                # warm up
    times = [time_ms(simulate, state, shot) for _ in range(BREAK_RUNS)]
    mean = statistics.fmean(times)
    report(capsys, f"full-power break (16 balls, {BREAK_RUNS} runs): "
                   f"mean {mean:.2f} ms, max {max(times):.2f} ms")
    assert mean <= BREAK_MEAN_MAX_MS, f"break mean {mean:.2f} ms is over {BREAK_MEAN_MAX_MS} ms"
