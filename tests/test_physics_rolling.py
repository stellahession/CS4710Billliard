"""Failing tests for issue #3: rolling motion and friction.

Model expected (physics/simulate.py, constants from physics/params.py):

    v0 = power * params.V_MAX          cue-ball starting speed
    params.DECEL                       constant deceleration (m/s^2) opposite to velocity
    params.V_STOP                      a ball slower than this is at rest
    params.MAX_TIME                    simulated-time guard; reaching it stops every ball
                                       where it is and issues a RuntimeWarning

So a lone ball that hits no cushion rolls v0^2 / (2 * DECEL) along the aim line and stops.
Distances are worked out from params at test time, so tuning them in issue #10 won't
break these tests.
"""
import math

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState
from tests.test_physics_simulate import break_table, load

# The guard now warns instead of raising, so turn its warning into an error here: the
# "never hits the guard" tests below must still fail if a normal shot trips it.
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

STOP_TOL = 1e-3     # m, the issue's "within 1 mm"


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


@pytest.fixture
def params():
    from physics import params
    return params


@pytest.fixture
def no_pockets(params, monkeypatch):
    # Keeps long rolls on the table once issue #6 makes balls drop near pockets.
    monkeypatch.setattr(params, "POCKET_R", 0.0)


def cue_at(x: float, y: float) -> TableState:
    return TableState([Ball(0, x, y)])


def roll_distance(params, power: float) -> float:
    v0 = power * params.V_MAX
    return v0 * v0 / (2 * params.DECEL)


def power_for(params, distance: float) -> float:
    """Power that should roll a lone ball exactly `distance` metres."""
    power = math.sqrt(2 * params.DECEL * distance) / params.V_MAX
    assert 0 < power <= 1, "test distance is out of reach with the current params"
    return power


def final_cue(result) -> tuple[float, float]:
    cue = next(b for b in result.final.balls if b.id == 0)
    return cue.x, cue.y


# --- stopping distance -----------------------------------------------------------

# (start x, start y, angle in degrees, distance) -- none of these reach a cushion.
NO_CUSHION_ROLLS = [
    (0.3, 0.635, 0, 0.05),
    (0.3, 0.635, 0, 0.4),
    (0.3, 0.635, 0, 1.5),
    (2.2, 0.635, 180, 1.2),
    (1.27, 0.2, 90, 0.8),
    (1.27, 1.0, 270, 0.6),
    (0.5, 0.3, 30, 0.8),
    (2.0, 1.0, 200, 0.6),
    (1.0, 1.0, -45, 0.5),
]


@pytest.mark.parametrize("x0, y0, deg, distance", NO_CUSHION_ROLLS)
def test_lone_cue_stops_at_v0_squared_over_two_decel(simulate, params, x0, y0, deg, distance):
    angle = math.radians(deg)
    power = power_for(params, distance)
    want = (x0 + distance * math.cos(angle), y0 + distance * math.sin(angle))
    assert BALL_R < want[0] < TABLE_W - BALL_R and BALL_R < want[1] < TABLE_H - BALL_R

    got = final_cue(simulate(cue_at(x0, y0), Shot(angle, power)))
    assert math.dist(got, want) < STOP_TOL, f"stopped at {got}, expected {want}"


@pytest.mark.parametrize("v_max, decel, power", [(2.0, 0.5, 0.5), (4.0, 1.0, 0.25), (6.0, 0.3, 0.1)])
def test_stopping_distance_follows_the_params(simulate, params, monkeypatch, v_max, decel, power):
    monkeypatch.setattr(params, "V_MAX", v_max)
    monkeypatch.setattr(params, "DECEL", decel)
    x, y = final_cue(simulate(cue_at(0.3, 0.635), Shot(0.0, power)))
    assert x - 0.3 == pytest.approx(roll_distance(params, power), abs=STOP_TOL)
    assert y == pytest.approx(0.635, abs=STOP_TOL)


def test_starting_speed_is_linear_in_power(simulate, params):
    # Doubling power doubles v0, so the roll is four times as long.
    start = 0.3
    p = power_for(params, 0.3)
    short = final_cue(simulate(cue_at(start, 0.635), Shot(0.0, p)))[0] - start
    long = final_cue(simulate(cue_at(start, 0.635), Shot(0.0, 2 * p)))[0] - start
    assert long == pytest.approx(4 * short, abs=STOP_TOL)


# --- direction -------------------------------------------------------------------

def test_angle_zero_moves_cue_in_plus_x(simulate, params):
    x, y = final_cue(simulate(cue_at(1.0, 0.635), Shot(0.0, power_for(params, 0.5))))
    assert x > 1.0 + 0.4
    assert y == pytest.approx(0.635, abs=1e-9)


def test_angle_half_pi_moves_cue_in_plus_y(simulate, params):
    x, y = final_cue(simulate(cue_at(1.27, 0.3), Shot(math.pi / 2, power_for(params, 0.5))))
    assert y > 0.3 + 0.4                     # y points up
    assert x == pytest.approx(1.27, abs=1e-9)


@pytest.mark.parametrize("deg, sx, sy", [(180, -1, 0), (270, 0, -1), (45, 1, 1), (135, -1, 1),
                                         (225, -1, -1), (315, 1, -1)])
def test_other_angles_move_the_way_cos_and_sin_say(simulate, params, deg, sx, sy):
    x0, y0 = TABLE_W / 2, TABLE_H / 2
    x, y = final_cue(simulate(cue_at(x0, y0), Shot(math.radians(deg), power_for(params, 0.3))))
    for moved, sign in ((x - x0, sx), (y - y0, sy)):
        if sign == 0:
            assert moved == pytest.approx(0.0, abs=1e-9)
        else:
            assert moved * sign > 0.1


def test_cue_stays_on_the_aim_line(simulate, params):
    x0, y0, angle = 0.4, 0.3, math.radians(25)
    x, y = final_cue(simulate(cue_at(x0, y0), Shot(angle, power_for(params, 1.0))))
    # Perpendicular distance from the aim line.
    off_line = abs((x - x0) * math.sin(angle) - (y - y0) * math.cos(angle))
    assert off_line < 1e-9


# --- more power, longer roll -------------------------------------------------------

def test_higher_power_rolls_strictly_further(simulate, params):
    start = 0.3
    max_power = power_for(params, TABLE_W - BALL_R - start - 0.05)   # stays off the far cushion
    powers = [max_power * k / 20 for k in range(1, 21)]
    distances = [final_cue(simulate(cue_at(start, 0.635), Shot(0.0, p)))[0] - start
                 for p in powers]
    assert all(b > a for a, b in zip(distances, distances[1:])), distances


def test_tiny_power_still_moves_the_cue(simulate, params):
    x, _ = final_cue(simulate(cue_at(1.0, 0.635), Shot(0.0, 0.001)))
    assert x > 1.0


# --- power 0 ---------------------------------------------------------------------

@pytest.mark.parametrize("make_state", [
    lambda: cue_at(1.0, 0.635),
    lambda: load("simple_cut.json"),
    lambda: load("straight_shot.json"),
    break_table,
])
@pytest.mark.parametrize("deg", [0, 90, 217])
def test_zero_power_leaves_every_ball_where_it_was(simulate, make_state, deg):
    state = make_state()
    result = simulate(state, Shot(math.radians(deg), 0.0))
    assert [(b.id, b.x, b.y, b.pocketed) for b in result.final.balls] == \
           [(b.id, b.x, b.y, b.pocketed) for b in state.balls]
    assert result.pocketed == []
    assert result.first_contact is None
    assert result.scratch is False


# --- everything at rest on return ------------------------------------------------

@pytest.mark.parametrize("make_state, shot", [
    (lambda: cue_at(0.3, 0.635), Shot(0.0, 0.1)),
    (lambda: cue_at(0.3, 0.635), Shot(0.3, 1.0)),          # many cushion bounces
    (lambda: load("simple_cut.json"), Shot(-0.13, 0.5)),
    (lambda: load("straight_shot.json"), Shot(0.0, 0.8)),
    (break_table, Shot(0.0, 1.0)),
])
def test_every_ball_is_at_rest_when_simulate_returns(simulate, params, monkeypatch, make_state, shot):
    # If any ball were still moving, giving the simulator more time would change
    # where it ends up. It must not.
    normal = simulate(make_state(), shot)
    monkeypatch.setattr(params, "MAX_TIME", params.MAX_TIME * 100)
    longer = simulate(make_state(), shot)
    assert longer == normal


def test_stopped_lone_ball_is_exactly_where_friction_puts_it(simulate, params):
    # A ball left crawling below V_STOP would still be short of the stop point.
    distance = 0.7
    x, _ = final_cue(simulate(cue_at(0.3, 0.635), Shot(0.0, power_for(params, distance))))
    assert x == pytest.approx(0.3 + distance, abs=1e-6)


# --- max-time guard --------------------------------------------------------------

def test_max_time_guard_warns_and_still_returns_a_result(simulate, params, monkeypatch):
    # One odd shot must not stop the agent's search, so the guard warns instead of raising.
    monkeypatch.setattr(params, "MAX_TIME", 0.5)
    with pytest.warns(RuntimeWarning, match="MAX_TIME"):
        result = simulate(cue_at(0.3, 0.635), Shot(0.3, 1.0))
    x, y = final_cue(result)
    assert 0.3 < x < TABLE_W and math.isfinite(y)


def test_max_time_leaves_room_for_a_full_power_roll(params):
    # A lone ball at full power rolls for V_MAX / DECEL seconds, cushions or not
    # (a cushion bounce only ever slows it down).
    assert params.MAX_TIME > params.V_MAX / params.DECEL


@pytest.mark.parametrize("deg", [0, 0.3, 17, 45, 90, 133, 200, 271, 333])
def test_full_power_lone_cue_with_many_bounces_never_hits_the_guard(simulate, params,
                                                                   monkeypatch, no_pockets,
                                                                   deg):
    # Tighten the guard to just over the longest possible roll: the simulator must
    # finish in time and not run on after the ball has stopped.
    monkeypatch.setattr(params, "MAX_TIME", params.V_MAX / params.DECEL * 1.001)
    result = simulate(cue_at(0.3, 0.635), Shot(math.radians(deg), 1.0))
    x, y = final_cue(result)
    assert BALL_R - 1e-9 <= x <= TABLE_W - BALL_R + 1e-9
    assert BALL_R - 1e-9 <= y <= TABLE_H - BALL_R + 1e-9


@pytest.mark.parametrize("make_state, deg", [
    (break_table, 0),
    (break_table, 2),
    (lambda: load("simple_cut.json"), -7.4),
    (lambda: load("simple_cut.json"), 160),
    (lambda: load("straight_shot.json"), 0),
    (lambda: load("straight_shot.json"), 75),
])
def test_full_power_shots_with_object_balls_never_hit_the_guard(simulate, make_state, deg):
    # Default MAX_TIME; a RuntimeWarning here means a normal shot tripped the guard.
    simulate(make_state(), Shot(math.radians(deg), 1.0))
