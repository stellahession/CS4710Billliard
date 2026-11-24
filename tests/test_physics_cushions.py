"""Failing tests for issue #4: cushion bounces.

Interface expected (physics/simulate.py):

    reflect(vel, wall) -> vel
        vel is a (vx, vy) tuple in m/s; wall is one of "left", "right", "bottom", "top".
        Flips the velocity component perpendicular to that cushion and scales it by
        params.E_CUSHION (read at call time); keeps the component along the cushion.
        Returns a new tuple and does not mutate its input.

In the full simulation a ball's centre bounces at BALL_R from the cushion line, so it
stays inside [BALL_R, TABLE_W - BALL_R] x [BALL_R, TABLE_H - BALL_R]. Pockets near the
cushions come in issue #6, so every shot here keeps its ball away from the pockets or
only checks that it stays on the table.
"""
import math

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState
from tests.test_physics_simulate import break_table, load

X_MIN, X_MAX = BALL_R, TABLE_W - BALL_R
Y_MIN, Y_MAX = BALL_R, TABLE_H - BALL_R
EDGE_TOL = 1e-9     # m, float slack on the cushion lines
STOP_TOL = 1e-3     # m, same "within 1 mm" as issue #3
CENTRE = (TABLE_W / 2, TABLE_H / 2)

WALLS = ["left", "right", "bottom", "top"]


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


@pytest.fixture
def reflect():
    from physics.simulate import reflect
    return reflect


@pytest.fixture
def params():
    from physics import params
    return params


def cue_at(x: float, y: float) -> TableState:
    return TableState([Ball(0, x, y)])


def final_positions(result) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in result.final.balls if not b.pocketed}


def assert_inside(x: float, y: float):
    assert X_MIN - EDGE_TOL <= x <= X_MAX + EDGE_TOL, f"x={x} is outside the cushions"
    assert Y_MIN - EDGE_TOL <= y <= Y_MAX + EDGE_TOL, f"y={y} is outside the cushions"


def roll_distance(params, speed: float) -> float:
    return speed * speed / (2 * params.DECEL)


def speed_to_roll(params, distance: float) -> float:
    """Speed a lone ball needs to roll exactly `distance` metres and stop."""
    return math.sqrt(2 * params.DECEL * distance)


def power_to_arrive(params, speed: float, distance: float) -> float:
    """Power that brings a lone ball `distance` metres away at `speed` m/s."""
    power = math.sqrt(speed * speed + 2 * params.DECEL * distance) / params.V_MAX
    assert 0 < power <= 1, "test shot is out of reach with the current params"
    return power


def mirror_into_table(x: float, y: float) -> tuple[float, float]:
    """Fold a point from the 'unfolded' straight-line path back into the table.

    With E_CUSHION = 1 a bouncing ball ends exactly where a ball rolling the same distance
    through mirror images of the table would, folded back in.
    """
    def fold(p, lo, hi):
        span = hi - lo
        p = (p - lo) % (2 * span)
        return lo + (p if p <= span else 2 * span - p)
    return fold(x, X_MIN, X_MAX), fold(y, Y_MIN, Y_MAX)


# --- reflect(): straight on ------------------------------------------------------

@pytest.mark.parametrize("wall, vel_in, vel_out", [
    ("left", (-2.0, 0.0), (1.0, 0.0)),
    ("right", (2.0, 0.0), (-1.0, 0.0)),
    ("bottom", (0.0, -2.0), (0.0, 1.0)),
    ("top", (0.0, 2.0), (0.0, -1.0)),
])
def test_head_on_comes_straight_back_scaled_by_e_cushion(reflect, params, monkeypatch,
                                                         wall, vel_in, vel_out):
    monkeypatch.setattr(params, "E_CUSHION", 0.5)
    vx, vy = reflect(vel_in, wall)
    assert vx == pytest.approx(vel_out[0], abs=1e-12)
    assert vy == pytest.approx(vel_out[1], abs=1e-12)


@pytest.mark.parametrize("wall", WALLS)
def test_head_on_speed_is_scaled_by_the_default_e_cushion(reflect, params, wall):
    vel_in = {"left": (-3.0, 0.0), "right": (3.0, 0.0),
              "bottom": (0.0, -3.0), "top": (0.0, 3.0)}[wall]
    vx, vy = reflect(vel_in, wall)
    assert math.hypot(vx, vy) == pytest.approx(3.0 * params.E_CUSHION)
    # Straight back: the direction is exactly reversed.
    assert vx * vel_in[0] <= 0 and vy * vel_in[1] <= 0
    assert vx * vel_in[1] - vy * vel_in[0] == pytest.approx(0.0, abs=1e-12)


# --- reflect(): at an angle ------------------------------------------------------

# Velocities arriving at 45° to each wall, both along-cushion directions.
ANGLED = [
    ("left", (-1.0, 1.0)), ("left", (-1.0, -1.0)),
    ("right", (1.0, 1.0)), ("right", (1.0, -1.0)),
    ("bottom", (1.0, -1.0)), ("bottom", (-1.0, -1.0)),
    ("top", (1.0, 1.0)), ("top", (-1.0, 1.0)),
]
# Inward normal of each cushion (points back onto the table).
NORMAL = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "bottom": (0.0, 1.0), "top": (0.0, -1.0)}


@pytest.mark.parametrize("wall, vel_in", ANGLED)
def test_45_degrees_in_leaves_at_45_degrees_with_speed_kept(reflect, params, monkeypatch,
                                                            wall, vel_in):
    monkeypatch.setattr(params, "E_CUSHION", 1.0)
    vx, vy = reflect(vel_in, wall)
    nx, ny = NORMAL[wall]
    assert math.hypot(vx, vy) == pytest.approx(math.hypot(*vel_in))
    # Leaves away from the cushion, at 45° to its normal.
    normal_part = vx * nx + vy * ny
    assert normal_part > 0
    assert math.degrees(math.acos(normal_part / math.hypot(vx, vy))) == pytest.approx(45.0)
    # Mirror image: the along-cushion part is unchanged, the normal part is flipped.
    assert (vx, vy) == pytest.approx((vel_in[0] if nx == 0 else -vel_in[0],
                                      vel_in[1] if ny == 0 else -vel_in[1]))


@pytest.mark.parametrize("wall, vel_in", ANGLED)
def test_along_cushion_part_is_kept_and_normal_part_scaled(reflect, params, monkeypatch,
                                                           wall, vel_in):
    monkeypatch.setattr(params, "E_CUSHION", 0.7)
    vx, vy = reflect(vel_in, wall)
    if wall in ("left", "right"):
        assert vy == pytest.approx(vel_in[1])
        assert vx == pytest.approx(-0.7 * vel_in[0])
    else:
        assert vx == pytest.approx(vel_in[0])
        assert vy == pytest.approx(-0.7 * vel_in[1])


def test_reflect_reads_e_cushion_at_call_time(reflect, params, monkeypatch):
    monkeypatch.setattr(params, "E_CUSHION", 0.25)
    assert reflect((-4.0, 0.0), "left") == pytest.approx((1.0, 0.0))
    monkeypatch.setattr(params, "E_CUSHION", 1.0)
    assert reflect((-4.0, 0.0), "left") == pytest.approx((4.0, 0.0))


def test_reflect_returns_a_new_value_and_leaves_its_input_alone(reflect):
    vel = [-1.5, 0.5]
    out = reflect(vel, "left")
    assert vel == [-1.5, 0.5]
    assert out is not vel


def test_reflect_rejects_an_unknown_wall(reflect):
    with pytest.raises(ValueError):
        reflect((1.0, 0.0), "middle")


# --- full simulation: where a bounced ball stops ----------------------------------

@pytest.mark.parametrize("wall, start, angle", [
    ("left", (0.8, 0.635), math.pi),
    ("right", (1.74, 0.635), 0.0),
    ("bottom", (0.9, 0.5), -math.pi / 2),
    ("top", (0.9, 0.77), math.pi / 2),
])
def test_head_on_bounce_stops_where_e_cushion_says(simulate, params, wall, start, angle):
    # Roll into the cushion, bounce back at E_CUSHION times the arrival speed, then roll
    # (e * v1)^2 / (2 * DECEL) back out. The bounce happens at BALL_R from the cushion line.
    x0, y0 = start
    to_wall = {"left": x0 - X_MIN, "right": X_MAX - x0,
               "bottom": y0 - Y_MIN, "top": Y_MAX - y0}[wall]
    back = 0.3                                           # m, chosen; power follows
    speed_at_wall = speed_to_roll(params, back) / params.E_CUSHION
    power = power_to_arrive(params, speed_at_wall, to_wall)

    result = simulate(cue_at(x0, y0), Shot(angle, power))
    x, y = final_positions(result)[0]
    expected = {"left": (X_MIN + back, y0), "right": (X_MAX - back, y0),
                "bottom": (x0, Y_MIN + back), "top": (x0, Y_MAX - back)}[wall]
    assert x == pytest.approx(expected[0], abs=STOP_TOL)
    assert y == pytest.approx(expected[1], abs=STOP_TOL)


@pytest.mark.parametrize("deg", [20, 45, 70, 110, 160, 200, 250, 290, 340])
def test_elastic_bounces_end_where_the_mirror_image_says(simulate, params, monkeypatch, deg):
    # With E_CUSHION = 1 the angle in equals the angle out at every cushion, so the stop
    # point is the straight-line stop point folded back into the table.
    monkeypatch.setattr(params, "E_CUSHION", 1.0)
    x0, y0 = 1.0, 0.5
    power = 0.6
    dist = roll_distance(params, power * params.V_MAX)
    a = math.radians(deg)
    expected = mirror_into_table(x0 + dist * math.cos(a), y0 + dist * math.sin(a))
    # Keep clear of pockets (issue #6) so this test only checks the bounces.
    assert all(math.hypot(expected[0] - px, expected[1] - py) > 0.15
               for px, py in [(0, 0), (TABLE_W / 2, 0), (TABLE_W, 0),
                              (0, TABLE_H), (TABLE_W / 2, TABLE_H), (TABLE_W, TABLE_H)])

    result = simulate(cue_at(x0, y0), Shot(a, power))
    x, y = final_positions(result)[0]
    assert x == pytest.approx(expected[0], abs=STOP_TOL)
    assert y == pytest.approx(expected[1], abs=STOP_TOL)


def test_inelastic_bounce_shortens_the_roll(simulate, params, monkeypatch):
    # Same shot into the right cushion: a softer cushion leaves the ball nearer to it.
    # Arrive slowly enough that even the 0.9 cushion sends it only 0.5 m back.
    power = power_to_arrive(params, speed_to_roll(params, 0.5) / 0.9, X_MAX - 1.5)
    shot = Shot(0.0, power)
    monkeypatch.setattr(params, "E_CUSHION", 0.9)
    soft = final_positions(simulate(cue_at(1.5, 0.635), shot))[0][0]
    monkeypatch.setattr(params, "E_CUSHION", 0.3)
    softer = final_positions(simulate(cue_at(1.5, 0.635), shot))[0][0]
    assert softer > soft


# --- full simulation: never leaves the table --------------------------------------

@pytest.mark.parametrize("deg", [0, 7, 33, 45, 61, 90, 123, 180, 211, 270, 299, 315])
def test_ball_stays_inside_the_cushions_at_every_stopping_point(simulate, params, deg):
    # Without recorded paths (issue #7) we sample the trajectory by stopping the same
    # shot at many different points along it: every stop must be inside the cushions.
    for i in range(1, 41):
        power = i / 40
        result = simulate(cue_at(0.7, 0.4), Shot(math.radians(deg), power))
        for x, y in final_positions(result).values():
            assert_inside(x, y)


# Aimed straight at each wall and into each corner.
AIMS = {
    "right": 0.0, "top": math.pi / 2, "left": math.pi, "bottom": -math.pi / 2,
    "top-right": math.atan2(TABLE_H, TABLE_W), "top-left": math.atan2(TABLE_H, -TABLE_W),
    "bottom-left": math.atan2(-TABLE_H, -TABLE_W), "bottom-right": math.atan2(-TABLE_H, TABLE_W),
}
STARTS = [
    CENTRE,
    (X_MIN, Y_MIN + 0.3), (X_MAX, Y_MAX - 0.3),          # touching a side cushion
    (0.5, Y_MIN), (2.0, Y_MAX),                          # touching an end cushion
    (X_MIN + 1e-4, 0.6), (0.6, Y_MAX - 1e-4),            # just off a cushion
]


@pytest.mark.parametrize("aim", list(AIMS))
@pytest.mark.parametrize("start", STARTS)
def test_full_power_never_passes_through_a_cushion(simulate, aim, start):
    result = simulate(cue_at(*start), Shot(AIMS[aim], 1.0))
    for x, y in final_positions(result).values():
        assert_inside(x, y)


@pytest.mark.parametrize("aim", list(AIMS))
def test_full_power_with_elastic_cushions_and_fast_balls_stays_inside(simulate, params,
                                                                      monkeypatch, aim):
    # Harsher than the real table: no energy lost at cushions and a much faster cue.
    monkeypatch.setattr(params, "E_CUSHION", 1.0)
    monkeypatch.setattr(params, "V_MAX", 20.0)
    monkeypatch.setattr(params, "MAX_TIME", 1000.0)
    result = simulate(cue_at(*CENTRE), Shot(AIMS[aim], 1.0))
    assert_inside(*final_positions(result)[0])


def test_ball_aimed_exactly_into_a_corner_comes_back_out(simulate, params):
    # Hits both cushions at once: both components flip. With pockets off the table this
    # can't sit outside either cushion line.
    x0, y0 = X_MAX - 0.4, Y_MAX - 0.4
    result = simulate(cue_at(x0, y0), Shot(math.pi / 4, 0.5))
    assert_inside(*final_positions(result)[0])


# --- several bounces, then at rest ------------------------------------------------

@pytest.mark.parametrize("deg", [3, 17, 45, 88, 135, 222, 300])
def test_many_bounces_still_end_at_rest_inside(simulate, params, deg):
    # Full power rolls far more than a lap of the table, so this bounces several times.
    assert roll_distance(params, params.V_MAX) > 2 * (TABLE_W + TABLE_H)
    result = simulate(cue_at(0.4, 0.3), Shot(math.radians(deg), 1.0))
    assert_inside(*final_positions(result)[0])
    # Settled: replaying from where it stopped with no power changes nothing.
    x, y = final_positions(result)[0]
    again = simulate(result.final, Shot(0.0, 0.0))
    assert final_positions(again)[0] == pytest.approx((x, y))


@pytest.mark.parametrize("make_state, deg", [
    (break_table, 0),
    (break_table, 1.5),
    (lambda: load("simple_cut.json"), 95),
    (lambda: load("straight_shot.json"), 200),
])
def test_object_balls_knocked_into_cushions_stay_inside(simulate, make_state, deg):
    result = simulate(make_state(), Shot(math.radians(deg), 1.0))
    for x, y in final_positions(result).values():
        assert_inside(x, y)
