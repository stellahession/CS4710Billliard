"""Failing tests for issue #2: module structure, simulate() contract, and input handling.

Interface expected:

    physics/simulate.py
        simulate(state, shot, record_paths=False) -> ShotResult

    physics/params.py
        V_MAX, DECEL, E_BALL, E_CUSHION, POCKET_R, V_STOP, MAX_TIME
        The simulator reads these as params.NAME at call time, so a test can
        override one with monkeypatch.setattr(params, "NAME", value).

These tests cover the contract only. Rolling, cushions, collisions, pockets and paths
have their own tests in issues #3-#7, so nothing here checks exact distances.
"""
import copy
import math
from pathlib import Path

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, ShotResult, TableState
from physics.bench import break_table  # noqa: F401  (other test files import it from here)

FIXTURES = Path(__file__).parent / "fixtures"
PARAM_NAMES = ["V_MAX", "DECEL", "E_BALL", "E_CUSHION", "POCKET_R", "V_STOP", "MAX_TIME"]


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


@pytest.fixture
def params():
    from physics import params
    return params


def load(name: str) -> TableState:
    return TableState.from_json((FIXTURES / name).read_text())


def ball(state: TableState, bid: int) -> Ball:
    return next(b for b in state.balls if b.id == bid)


def near_corner_pocket() -> TableState:
    """Ball 1 sits just off the top-right corner pocket, cue lined up behind it at 45°."""
    return TableState([Ball(0, 2.0, 0.73), Ball(1, 2.40, 1.13), Ball(9, 0.5, 0.3)], "solids")


def with_pocketed_ball_in_the_way() -> TableState:
    """Ball 3 is already pocketed but its stale coordinates sit on the cue's line to ball 5."""
    return TableState([Ball(0, 0.5, 0.635), Ball(3, 1.0, 0.635, pocketed=True),
                       Ball(5, 1.5, 0.635)], "solids")


def assert_all_finite(result: ShotResult):
    for b in result.final.balls:
        assert math.isfinite(b.x) and math.isfinite(b.y), f"ball {b.id} at ({b.x}, {b.y})"


def assert_inside_cushions(state: TableState, tol: float = 1e-9):
    for b in state.balls:
        if b.pocketed:
            continue
        assert BALL_R - tol <= b.x <= TABLE_W - BALL_R + tol, f"ball {b.id} x={b.x}"
        assert BALL_R - tol <= b.y <= TABLE_H - BALL_R + tol, f"ball {b.id} y={b.y}"


def assert_no_overlap(state: TableState, tol: float = 1e-6):
    live = [b for b in state.balls if not b.pocketed]
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            gap = math.hypot(a.x - b.x, a.y - b.y) - 2 * BALL_R
            assert gap > -tol, f"balls {a.id} and {b.id} overlap by {-gap:.6f} m"


# --- module structure ----------------------------------------------------------

def test_simulate_is_importable_from_the_path_the_app_uses():
    from physics.simulate import simulate
    assert callable(simulate)


def test_params_defines_every_tunable_constant(params):
    for name in PARAM_NAMES:
        value = getattr(params, name)
        assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0, name


def test_simulate_returns_a_shot_result(simulate):
    result = simulate(load("simple_cut.json"), Shot(0.0, 0.5))
    assert isinstance(result, ShotResult)
    assert isinstance(result.final, TableState)
    assert isinstance(result.pocketed, list)
    assert isinstance(result.scratch, bool)


# --- doesn't change its input ----------------------------------------------------

@pytest.mark.parametrize("make_state, shot", [
    (lambda: load("simple_cut.json"), Shot(-0.13, 0.5)),
    (lambda: load("straight_shot.json"), Shot(0.0, 1.0)),
    (near_corner_pocket, Shot(math.pi / 4, 0.5)),        # aimed to pocket ball 1
    (with_pocketed_ball_in_the_way, Shot(0.0, 0.6)),
])
@pytest.mark.parametrize("record_paths", [False, True])
def test_simulate_does_not_mutate_input(simulate, make_state, shot, record_paths):
    state = make_state()
    before = copy.deepcopy(state)
    shot_before = copy.deepcopy(shot)
    simulate(state, shot, record_paths=record_paths)
    assert state == before
    assert shot == shot_before


def test_result_does_not_share_ball_objects_with_input(simulate):
    # Even with power 0 (nothing moves) the result must be a copy, or a later
    # edit to the result would quietly change the agent's search state.
    state = load("simple_cut.json")
    result = simulate(state, Shot(0.0, 0.0))
    input_ids = {id(b) for b in state.balls}
    assert result.final is not state
    assert result.final.balls is not state.balls
    assert not any(id(b) in input_ids for b in result.final.balls)


# --- repeatable ------------------------------------------------------------------

@pytest.mark.parametrize("record_paths", [False, True])
@pytest.mark.parametrize("deg", [0, 37, 140, 265])
def test_same_inputs_give_identical_results(simulate, deg, record_paths):
    state, shot = load("simple_cut.json"), Shot(math.radians(deg), 0.7)
    first = simulate(state, shot, record_paths=record_paths)
    for _ in range(3):
        assert simulate(state, shot, record_paths=record_paths) == first


def test_full_power_break_is_repeatable(simulate):
    shot = Shot(0.0, 1.0)
    assert simulate(break_table(), shot) == simulate(break_table(), shot)


# --- final: same ids, same order -------------------------------------------------

@pytest.mark.parametrize("make_state, shot", [
    (lambda: load("simple_cut.json"), Shot(-0.13, 0.5)),
    (near_corner_pocket, Shot(math.pi / 4, 0.5)),
    (with_pocketed_ball_in_the_way, Shot(0.0, 0.6)),
])
def test_final_lists_every_input_ball_in_the_same_order(simulate, make_state, shot):
    state = make_state()
    result = simulate(state, shot)
    assert [b.id for b in result.final.balls] == [b.id for b in state.balls]
    assert result.final.group == state.group


def test_final_keeps_input_order_even_when_cue_is_not_first(simulate):
    state = TableState([Ball(8, 2.1, 0.4), Ball(1, 1.6, 0.5), Ball(0, 0.6, 0.63)])
    result = simulate(state, Shot(0.0, 0.4))
    assert [b.id for b in result.final.balls] == [8, 1, 0]


# --- balls already pocketed in the input -----------------------------------------

def test_already_pocketed_ball_is_ignored(simulate):
    state = with_pocketed_ball_in_the_way()
    result = simulate(state, Shot(0.0, 0.6))

    # The cue passes through ball 3's stale position and hits ball 5.
    assert result.first_contact == 5
    gone = ball(result.final, 3)
    assert gone.pocketed is True
    assert (gone.x, gone.y) == (1.0, 0.635)
    assert 3 not in result.pocketed


def test_already_pocketed_ball_has_no_path(simulate):
    result = simulate(with_pocketed_ball_in_the_way(), Shot(0.0, 0.6), record_paths=True)
    path = (result.paths or {}).get(3)
    assert path is None or all(p == (1.0, 0.635) for p in path)


def test_pocketed_list_never_repeats_an_input_pocketed_ball(simulate):
    state = near_corner_pocket()
    state.balls.append(Ball(12, 1.2, 0.6, pocketed=True))
    result = simulate(state, Shot(math.pi / 4, 0.5))
    assert 12 not in result.pocketed
    assert len(result.pocketed) == len(set(result.pocketed))


# --- no cue ball -----------------------------------------------------------------

def test_missing_cue_ball_raises_value_error(simulate):
    state = TableState([Ball(1, 1.6, 0.5), Ball(8, 2.1, 0.4)])
    with pytest.raises(ValueError, match="(?i)cue"):
        simulate(state, Shot(0.0, 0.5))


def test_pocketed_cue_ball_raises_value_error(simulate):
    state = TableState([Ball(0, 0.6, 0.63, pocketed=True), Ball(1, 1.6, 0.5)])
    with pytest.raises(ValueError, match="(?i)cue"):
        simulate(state, Shot(0.0, 0.5))


# --- power and angle -------------------------------------------------------------

@pytest.mark.parametrize("given, clamped", [(1.5, 1.0), (100.0, 1.0), (-0.3, 0.0), (-5.0, 0.0)])
def test_power_is_clamped_to_zero_one(simulate, given, clamped):
    state = load("straight_shot.json")
    assert simulate(state, Shot(0.0, given)) == simulate(state, Shot(0.0, clamped))


def test_zero_power_moves_nothing(simulate):
    state = load("simple_cut.json")
    result = simulate(state, Shot(0.0, 0.0))
    assert result.first_contact is None
    assert result.pocketed == []
    assert result.scratch is False
    assert [(b.x, b.y) for b in result.final.balls] == [(b.x, b.y) for b in state.balls]


@pytest.mark.parametrize("turns", [-3, -1, 1, 5])
def test_any_real_angle_is_accepted(simulate, turns):
    # Adding whole turns must give the same shot (up to float rounding in cos/sin).
    state, base = load("straight_shot.json"), 0.2
    want = simulate(state, Shot(base, 0.5))
    got = simulate(state, Shot(base + turns * 2 * math.pi, 0.5))
    assert got.first_contact == want.first_contact
    for g, w in zip(got.final.balls, want.final.balls):
        assert g.x == pytest.approx(w.x, abs=1e-6)
        assert g.y == pytest.approx(w.y, abs=1e-6)


# --- messy input from vision -----------------------------------------------------

@pytest.mark.parametrize("make_state", [
    # Two balls overlapping by half a radius.
    lambda: TableState([Ball(0, 0.6, 0.63), Ball(1, 1.6, 0.5), Ball(2, 1.6 + BALL_R * 1.5, 0.5)]),
    # Two balls at exactly the same spot: no direction to separate them along.
    lambda: TableState([Ball(0, 0.6, 0.63), Ball(1, 1.6, 0.5), Ball(2, 1.6, 0.5)]),
    # Cue overlapping an object ball.
    lambda: TableState([Ball(0, 1.0, 0.6), Ball(4, 1.0 + BALL_R, 0.6)]),
    # A cluster of three all touching the same point.
    lambda: TableState([Ball(0, 0.6, 0.63), Ball(1, 1.6, 0.5), Ball(2, 1.61, 0.5),
                        Ball(3, 1.605, 0.51)]),
])
@pytest.mark.parametrize("power", [0.0, 0.6])
def test_overlapping_balls_are_nudged_apart(simulate, make_state, power):
    state = make_state()
    before = copy.deepcopy(state)
    result = simulate(state, Shot(0.3, power))
    assert state == before                      # nudging happens on a copy
    assert_all_finite(result)
    assert_no_overlap(result.final)


@pytest.mark.parametrize("bad", [
    (-0.1, 0.6), (TABLE_W + 0.2, 0.6), (1.0, -0.05), (1.0, TABLE_H + 0.3),
    (0.0, 0.6), (1.0, TABLE_H),                 # centre on the cushion line
])
@pytest.mark.parametrize("power", [0.0, 0.6])
def test_balls_outside_cushions_are_nudged_back_inside(simulate, bad, power):
    state = TableState([Ball(0, 1.27, 0.635), Ball(6, *bad)])
    before = copy.deepcopy(state)
    result = simulate(state, Shot(math.pi / 2, power))
    assert state == before
    assert_all_finite(result)
    assert_inside_cushions(result.final)


def test_out_of_bounds_cue_ball_is_nudged_back_inside(simulate):
    state = TableState([Ball(0, -0.2, 0.6), Ball(1, 1.6, 0.5)])
    result = simulate(state, Shot(0.0, 0.0))
    assert_all_finite(result)
    assert_inside_cushions(result.final)


def test_messy_table_is_repeatable(simulate):
    def messy():
        return TableState([Ball(0, -0.1, 0.63), Ball(1, 1.6, 0.5), Ball(2, 1.6, 0.5),
                           Ball(3, 1.62, 0.51), Ball(8, TABLE_W + 0.1, 0.9)])
    shot = Shot(0.0, 0.8)
    assert simulate(messy(), shot) == simulate(messy(), shot)


# --- params are read at call time ------------------------------------------------

def test_monkeypatching_v_max_changes_the_shot(simulate, params, monkeypatch):
    state, shot = load("straight_shot.json"), Shot(0.0, 0.6)
    normal = simulate(state, shot)
    assert normal.first_contact == 3

    monkeypatch.setattr(params, "V_MAX", 0.0)
    still = simulate(state, shot)
    assert still.first_contact is None
    assert ball(still.final, 0).x == pytest.approx(0.5)


def test_monkeypatching_decel_changes_how_far_the_cue_rolls(simulate, params, monkeypatch):
    state, shot = TableState([Ball(0, 0.5, 0.635)]), Shot(0.0, 0.05)
    normal = ball(simulate(state, shot).final, 0).x

    monkeypatch.setattr(params, "DECEL", params.DECEL * 2)
    shorter = ball(simulate(state, shot).final, 0).x
    assert 0.5 < shorter < normal


# --- regressions from the physics-mvp review ---------------------------------------

@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("gap", [5e-5, 2e-5, 1e-3])
def test_balls_rolling_side_by_side_settle_without_tiny_steps(params, monkeypatch, gap):
    # Two balls rolling the same way just apart never get closer. The step size must
    # not shrink to gap / speed, or the shot runs out of steps. A shot can't start two
    # balls moving together, so this drives the stepping loop directly.
    import physics.simulate as sim
    monkeypatch.setattr(sim, "MAX_STEPS", 2_000)
    v, y0 = 0.5, 0.4                      # rolls about 1.3 m, clear of the cushions
    balls = [(0.5, y0, v, 0.0), (0.5, y0 + 2 * BALL_R + gap, v, 0.0)]
    sim._run(balls, [0, 1], [])
    roll = v * v / (2 * params.DECEL)
    assert balls[0][0] == pytest.approx(0.5 + roll, abs=1e-6)
    assert balls[1][0] == pytest.approx(0.5 + roll, abs=1e-6)


def test_max_steps_guard_warns_and_still_returns_a_result(simulate, monkeypatch):
    import physics.simulate as sim
    monkeypatch.setattr(sim, "MAX_STEPS", 3)
    with pytest.warns(RuntimeWarning, match="steps"):
        result = simulate(break_table(), Shot(0.0, 1.0))
    assert isinstance(result, ShotResult)
    assert_all_finite(result)


def test_pocketed_ball_coordinates_are_floats(simulate):
    # POCKETS mixes ints and floats; the final table must not.
    state = TableState([Ball(0, 0.5, BALL_R + 0.001)])
    result = simulate(state, Shot(math.pi, 0.5))
    cue = ball(result.final, 0)
    assert cue.pocketed
    assert isinstance(cue.x, float) and isinstance(cue.y, float)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("corner", [(0.0, 0.0), (TABLE_W, TABLE_H), (0.0, TABLE_H)])
def test_balls_piled_into_a_corner_are_pushed_fully_apart(simulate, corner):
    # Clamping to the cushions used to undo the push, leaving clustered balls overlapping.
    state = TableState([Ball(bid, *corner) for bid in range(6)])
    final = simulate(state, Shot(0.0, 0.0)).final.balls
    for i, a in enumerate(final):
        for b in final[i + 1:]:
            assert math.hypot(a.x - b.x, a.y - b.y) >= 2 * BALL_R - 1e-12, (a.id, b.id)
