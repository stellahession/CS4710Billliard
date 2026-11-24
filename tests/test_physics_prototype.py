"""Failing tests for issue #1: prototype the stepping method and measure a first timing.

The prototype is throwaway, so these tests are too: delete this file along with
physics/prototype.py once the real simulator (issues #2-#8) replaces it.

Interface the prototype is expected to provide (physics/prototype.py):

    DECEL: float        rolling deceleration, m/s^2 (constant, no spin)
    MAX_SPEED: float    cue-ball speed at power 1.0, m/s

    safe_step(balls) -> float
        balls is a list of (x, y, vx, vy) tuples. Returns the largest time step (s)
        that can't skip a ball-ball collision or step past a ball stopping
        (conservative advancement).

    run(state, shot) -> (TableState, first_contact)
        Moves balls with friction until everything stops, resolving ball-ball
        contacts as equal-mass elastic collisions. first_contact is the id of the
        first ball the cue ball touches, or None. Must not mutate `state`.
        Cushions and pockets are not required for the prototype; the tests keep
        balls away from the rails.

    time_shots(state, shots) -> dict
        Calls run() once per shot, timed with time.perf_counter, and returns
        {"n": int, "mean_ms": float, "p95_ms": float, "max_ms": float}.
"""
import copy
import math
from pathlib import Path

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def proto():
    # Imported here so each test fails on its own until the prototype exists.
    from physics import prototype
    return prototype


def load(name: str) -> TableState:
    return TableState.from_json((FIXTURES / name).read_text())


def power_for_distance(proto, d: float) -> float:
    """Power that makes a lone ball roll distance d before stopping (d = v^2 / 2a)."""
    return math.sqrt(2 * proto.DECEL * d) / proto.MAX_SPEED


def ball(state: TableState, bid: int) -> Ball:
    return next(b for b in state.balls if b.id == bid)


def assert_no_overlap(state: TableState, tol: float = 1e-3):
    live = [b for b in state.balls if not b.pocketed]
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            gap = math.hypot(a.x - b.x, a.y - b.y) - 2 * BALL_R
            assert gap > -tol, f"balls {a.id} and {b.id} overlap by {-gap:.4f} m"


# --- safe_step: the conservative-advancement step size -----------------------

def test_safe_step_cannot_skip_an_approaching_collision(proto):
    # Two balls 0.10 m apart (surface to surface), closing head-on at 2 m/s total.
    gap, v = 0.10, 1.0
    x1 = TABLE_W / 2
    x2 = x1 + 2 * BALL_R + gap
    y = TABLE_H / 2
    dt = proto.safe_step([(x1, y, v, 0.0), (x2, y, -v, 0.0)])

    time_to_contact = gap / (2 * v)
    time_to_stop = v / proto.DECEL
    assert 0 < dt <= time_to_contact + 1e-9
    # It should be a real step, not a tiny fixed one.
    assert dt >= 0.5 * min(time_to_contact, time_to_stop)


def test_safe_step_is_large_when_balls_are_far_apart(proto):
    # A slow ball 1 m from the only other ball: no reason to take millisecond steps.
    y = TABLE_H / 2
    dt = proto.safe_step([(0.5, y, 0.3, 0.0), (1.5, y, 0.0, 0.0)])
    assert dt > 0.01


def test_safe_step_does_not_step_past_a_ball_stopping(proto):
    # Lone slow ball: the step must not go beyond the moment it stops.
    v = 0.2
    dt = proto.safe_step([(TABLE_W / 2, TABLE_H / 2, v, 0.0)])
    assert 0 < dt <= v / proto.DECEL + 1e-9


# --- run: friction and ball-ball contact -------------------------------------

def test_lone_ball_rolls_the_friction_distance(proto):
    # Exact between events: d = v^2 / (2 * DECEL), along the shot angle.
    d, angle = 0.30, math.radians(30)
    cue = Ball(0, TABLE_W / 2, TABLE_H / 2)
    state = TableState([cue])
    final, first = proto.run(state, Shot(angle, power_for_distance(proto, d)))

    end = ball(final, 0)
    assert first is None
    assert end.x == pytest.approx(cue.x + d * math.cos(angle), abs=1e-3)
    assert end.y == pytest.approx(cue.y + d * math.sin(angle), abs=1e-3)


def test_run_does_not_mutate_input(proto):
    state = load("simple_cut.json")
    before = copy.deepcopy(state)
    proto.run(state, Shot(0.0, 0.5))
    assert state == before


def test_straight_shot_transfers_all_speed_to_object_ball(proto):
    # Cue (0.5, 0.635) straight at ball 3 (1.5, 0.635). Equal masses, head-on, no
    # spin: the cue ball stops at contact and ball 3 rolls on with its speed.
    state = load("straight_shot.json")
    gap = 1.0 - 2 * BALL_R
    extra = 0.5  # how far ball 3 should roll after contact
    final, first = proto.run(state, Shot(0.0, power_for_distance(proto, gap + extra)))

    assert first == 3
    cue, obj = ball(final, 0), ball(final, 3)
    assert cue.x == pytest.approx(1.5 - 2 * BALL_R, abs=5e-3)
    assert cue.y == pytest.approx(0.635, abs=1e-3)
    assert obj.x == pytest.approx(1.5 + extra, abs=5e-3)
    assert obj.y == pytest.approx(0.635, abs=1e-3)


def test_cut_shot_sends_object_ball_along_line_of_centres(proto):
    # Aim the cue at a ghost ball beside ball 1. At contact the cue's centre is at
    # the ghost position, so ball 1 must leave along ghost -> ball 1.
    state = load("simple_cut.json")
    cue, obj = ball(state, 0), ball(state, 1)
    cut = math.radians(25)
    line = math.atan2(obj.y - cue.y, obj.x - cue.x)
    ghost = (obj.x - 2 * BALL_R * math.cos(line + cut),
             obj.y - 2 * BALL_R * math.sin(line + cut))
    aim = math.atan2(ghost[1] - cue.y, ghost[0] - cue.x)
    final, first = proto.run(state, Shot(aim, power_for_distance(proto, 1.3)))

    assert first == 1
    moved = ball(final, 1)
    got = math.atan2(moved.y - obj.y, moved.x - obj.x)
    want = math.atan2(obj.y - ghost[1], obj.x - ghost[0])
    assert math.hypot(moved.x - obj.x, moved.y - obj.y) > 0.05
    assert abs(math.remainder(got - want, 2 * math.pi)) < math.radians(1)


def test_miss_has_no_first_contact_and_leaves_other_balls_still(proto):
    state = load("straight_shot.json")
    final, first = proto.run(state, Shot(math.pi, power_for_distance(proto, 0.2)))
    assert first is None
    assert ball(final, 3).x == pytest.approx(1.5)
    assert ball(final, 3).y == pytest.approx(0.635)


@pytest.mark.parametrize("deg", range(0, 360, 30))
def test_balls_never_end_up_overlapping(proto, deg):
    state = load("simple_cut.json")
    final, _ = proto.run(state, Shot(math.radians(deg), 0.4))
    assert_no_overlap(final)


def test_full_power_break_finishes_without_overlaps(proto):
    final, first = proto.run(proto.break_table(), Shot(0.0, 1.0))
    assert first == 1
    assert_no_overlap(final)


# --- time_shots: the measurement harness -------------------------------------

def test_time_shots_reports_mean_p95_max(proto):
    shots = [Shot(math.radians(a), 0.5) for a in range(-10, 10)]
    stats = proto.time_shots(load("simple_cut.json"), shots)

    assert stats["n"] == len(shots)
    assert 0 < stats["mean_ms"] <= stats["max_ms"]
    assert 0 < stats["p95_ms"] <= stats["max_ms"]


def test_run_raises_if_it_hits_the_step_cap(proto, monkeypatch):
    # Hitting MAX_STEPS must be loud, not a silent mid-motion result.
    monkeypatch.setattr(proto, "MAX_STEPS", 3)
    with pytest.raises(RuntimeError):
        proto.run(proto.break_table(), Shot(0.0, 1.0))
