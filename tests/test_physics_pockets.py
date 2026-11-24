"""Failing tests for issue #6: pockets and scratches.

Model expected (physics/simulate.py, constants from physics.params read at call time):

  - A moving ball is pocketed when its centre comes within params.POCKET_R of a POCKETS
    point. POCKET_R must be larger than sqrt(2)·BALL_R, or corners can't be reached.
  - The pocket check runs before the cushion bounce, so a ball at a pocket mouth drops in
    instead of bouncing off the rail.
  - A pocketed ball is out of play at once: it stops and no longer collides. In `final`
    it has pocketed=True and x, y equal to that pocket's coordinates.
  - Only moving balls are checked: a ball resting inside the capture radius stays put
    unless it is hit.
  - `pocketed` lists the balls sunk this shot in the order they dropped.
  - scratch is True exactly when the cue ball (id 0) is pocketed. Physics never re-spots
    the cue ball; that is left to the agent or app.
"""
import math

import pytest

from core.state import BALL_R, POCKETS, TABLE_H, TABLE_W, Ball, Shot, TableState
from tests.test_physics_cushions import power_to_arrive, speed_to_roll
from tests.test_physics_simulate import ball, break_table, near_corner_pocket

D = 2 * BALL_R
STOP_TOL = 1e-3         # m, same "within 1 mm" as issues #3 to #5
MID_Y = TABLE_H / 2
CORNERS = [0, 2, 3, 5]  # indices into POCKETS
SIDES = [1, 4]
POWERS = [0.1, 0.4, 0.7, 1.0]


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
def pre_tuning_params(params, monkeypatch):
    # These shots were sized for the speed, friction and cushions used before issue #10,
    # so a low-power shot still reaches its target. They check mechanics, not the feel.
    monkeypatch.setattr(params, "V_MAX", 6.0)
    monkeypatch.setattr(params, "DECEL", 0.15)
    monkeypatch.setattr(params, "E_CUSHION", 0.8)


@pytest.fixture
def elastic(params, monkeypatch):
    monkeypatch.setattr(params, "E_BALL", 1.0)


# --- helpers -----------------------------------------------------------------------

def unit(deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return math.cos(a), math.sin(a)


def into_pocket(k: int, off_deg: float = 0.0) -> tuple[float, float]:
    """Unit direction from the table into pocket k, turned by off_deg.

    With off_deg = 0 this is 45° into a corner and straight across the rail into a side.
    """
    px, py = POCKETS[k]
    if k in SIDES:
        base = -90.0 if py == 0 else 90.0
    else:
        base = math.degrees(math.atan2(1 if py else -1, 1 if px else -1))
    return unit(base + off_deg)


def aim(direction: tuple[float, float]) -> float:
    return math.atan2(direction[1], direction[0])


def back_from(point, direction, dist) -> tuple[float, float]:
    return point[0] - dist * direction[0], point[1] - dist * direction[1]


def live_positions(result) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in result.final.balls if not b.pocketed}


def assert_sunk_in(result, bid: int, k: int):
    """Ball bid was sunk this shot and sits at pocket k's coordinates in `final`."""
    assert bid in result.pocketed, f"ball {bid} was not pocketed: {result.pocketed}"
    b = ball(result.final, bid)
    assert b.pocketed is True
    assert (b.x, b.y) == pytest.approx(POCKETS[k], abs=1e-12)


def assert_still_on_table(result, bid: int):
    assert bid not in result.pocketed
    assert ball(result.final, bid).pocketed is False


def object_ball_at_pocket(params, k: int, off_deg: float = 0.0, bid: int = 1) -> tuple:
    """Object ball 0.35 m from pocket k on a line into it, cue 0.4 m behind on the same line.

    Returns (state, shot) with the cue arriving at about 1.5 m/s. That hits the object ball
    into the pocket, and with E_BALL < 1 the cue only creeps a few mm after the hit.
    """
    u = into_pocket(k, off_deg)
    obj = back_from(POCKETS[k], u, 0.35)
    cue = back_from(POCKETS[k], u, 0.75)
    power = power_to_arrive(params, 1.5, 0.4 - D)
    return TableState([Ball(0, *cue), Ball(bid, *obj)]), Shot(aim(u), power)


def along_rail_into_corner(k: int, rail: str) -> tuple[tuple, tuple, tuple[float, float]]:
    """Positions 0.4 m and 0.8 m from corner k on `rail` ("horizontal" or "vertical"),
    with the ball centre touching that rail, and the direction into the corner."""
    px, py = POCKETS[k]
    sx = 1.0 if px else -1.0           # +1 means the corner is on the right
    sy = 1.0 if py else -1.0
    if rail == "horizontal":           # rolling along the top or bottom rail
        y = TABLE_H - BALL_R if py else BALL_R
        return (px - sx * 0.4, y), (px - sx * 0.8, y), (sx, 0.0)
    x = TABLE_W - BALL_R if px else BALL_R
    return (x, py - sy * 0.4), (x, py - sy * 0.8), (0.0, sy)


# --- params ------------------------------------------------------------------------

def test_default_pocket_radius_lets_a_ball_reach_the_corners(params):
    # A centre can't get closer than sqrt(2)·BALL_R (about 0.040 m) to a corner point.
    assert params.POCKET_R > math.sqrt(2) * BALL_R


# --- each pocket can be reached ------------------------------------------------------

@pytest.mark.parametrize("k", range(6))
def test_object_ball_sent_straight_at_each_pocket_is_captured(simulate, params, k):
    state, shot = object_ball_at_pocket(params, k)
    result = simulate(state, shot)
    assert result.pocketed == [1]
    assert_sunk_in(result, 1, k)
    assert_still_on_table(result, 0)
    assert result.scratch is False


@pytest.mark.parametrize("k", range(6))
@pytest.mark.parametrize("power", POWERS)
def test_lone_cue_sent_straight_at_each_pocket_is_captured_at_any_power(simulate, params,
                                                                        k, power,
                                                                        pre_tuning_params):
    # Fast balls take long steps; they still must not skip over the capture circle.
    u = into_pocket(k)
    start = back_from(POCKETS[k], u, 0.5)
    if power * params.V_MAX < speed_to_roll(params, 0.5):
        pytest.skip("too soft to reach the pocket with these params")
    result = simulate(TableState([Ball(0, *start)]), Shot(aim(u), power))
    assert result.pocketed == [0]
    assert_sunk_in(result, 0, k)


# --- corners -------------------------------------------------------------------------

@pytest.mark.parametrize("k", CORNERS)
@pytest.mark.parametrize("rail", ["horizontal", "vertical"])
def test_corner_along_the_rail_object_ball_is_captured(simulate, params, k, rail):
    obj, cue, u = along_rail_into_corner(k, rail)
    power = power_to_arrive(params, 1.5, 0.4 - D)
    result = simulate(TableState([Ball(0, *cue), Ball(2, *obj)]), Shot(aim(u), power))
    assert result.pocketed == [2]
    assert_sunk_in(result, 2, k)
    assert result.scratch is False


@pytest.mark.parametrize("k", CORNERS)
@pytest.mark.parametrize("rail", ["horizontal", "vertical"])
@pytest.mark.parametrize("power", POWERS)
def test_corner_along_the_rail_lone_cue_is_captured_at_any_power(simulate, k, rail, power,
                                                                 pre_tuning_params):
    _, start, u = along_rail_into_corner(k, rail)
    result = simulate(TableState([Ball(0, *start)]), Shot(aim(u), power))
    assert result.pocketed == [0]
    assert_sunk_in(result, 0, k)
    assert result.scratch is True


@pytest.mark.parametrize("k", CORNERS)
@pytest.mark.parametrize("off_deg", [-15, 0, 15])     # 30°, 45° and 60° to the rails
def test_corner_diagonal_object_ball_is_captured(simulate, params, k, off_deg):
    state, shot = object_ball_at_pocket(params, k, off_deg)
    result = simulate(state, shot)
    assert result.pocketed == [1]
    assert_sunk_in(result, 1, k)


# --- sides ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", SIDES)
@pytest.mark.parametrize("off_deg", [-30, 0, 30])
def test_side_straight_in_object_ball_is_captured(simulate, params, k, off_deg):
    state, shot = object_ball_at_pocket(params, k, off_deg)
    result = simulate(state, shot)
    assert result.pocketed == [1]
    assert_sunk_in(result, 1, k)


@pytest.mark.parametrize("k", SIDES)
@pytest.mark.parametrize("power", POWERS)
def test_side_straight_in_from_the_table_centre(simulate, params, k, power, pre_tuning_params):
    if power * params.V_MAX < speed_to_roll(params, MID_Y):
        pytest.skip("too soft to reach the pocket with these params")
    u = into_pocket(k)
    result = simulate(TableState([Ball(0, TABLE_W / 2, MID_Y)]), Shot(aim(u), power))
    assert result.pocketed == [0]
    assert_sunk_in(result, 0, k)


@pytest.mark.parametrize("k", SIDES)
@pytest.mark.parametrize("sx", [1.0, -1.0])           # rolling right or left
def test_side_along_the_rail_object_ball_is_captured(simulate, params, k, sx):
    y = TABLE_H - BALL_R if POCKETS[k][1] else BALL_R
    obj = (TABLE_W / 2 - sx * 0.5, y)
    cue = (TABLE_W / 2 - sx * 0.9, y)
    power = power_to_arrive(params, 1.5, 0.4 - D)
    result = simulate(TableState([Ball(0, *cue), Ball(4, *obj)]),
                      Shot(aim((sx, 0.0)), power))
    assert result.pocketed == [4]
    assert_sunk_in(result, 4, k)


@pytest.mark.parametrize("k", SIDES)
@pytest.mark.parametrize("power", POWERS)
def test_side_along_the_rail_lone_cue_is_captured_at_any_power(simulate, k, power,
                                                               pre_tuning_params):
    # The only thing ahead on the rail is the far cushion, so a step can be long.
    y = TABLE_H - BALL_R if POCKETS[k][1] else BALL_R
    result = simulate(TableState([Ball(0, 0.4, y)]), Shot(0.0, power))
    assert result.pocketed == [0]
    assert_sunk_in(result, 0, k)


@pytest.mark.parametrize("gap", [0.03, 0.05])          # both inside POCKET_R
def test_side_rolling_just_off_the_rail_is_captured(simulate, params, gap):
    assert gap < params.POCKET_R
    result = simulate(TableState([Ball(0, 0.4, gap)]), Shot(0.0, 0.5))
    assert_sunk_in(result, 0, 1)


# --- near misses ---------------------------------------------------------------------

@pytest.mark.parametrize("extra", [0.005, 0.03, 0.1])
def test_ball_passing_a_side_pocket_farther_than_pocket_r_is_not_captured(simulate, params,
                                                                          extra):
    # Rolls along the bottom rail past the side pocket and stops where friction says.
    y = params.POCKET_R + extra
    start, stop_x = 0.6, 2.0
    power = speed_to_roll(params, stop_x - start) / params.V_MAX
    result = simulate(TableState([Ball(0, start, y)]), Shot(0.0, power))
    assert result.pocketed == []
    assert result.scratch is False
    assert live_positions(result)[0] == pytest.approx((stop_x, y), abs=STOP_TOL)


def test_ball_bouncing_beside_a_corner_farther_than_pocket_r_is_not_captured(simulate,
                                                                             params):
    # Hits the left cushion at y = 0.08, about 0.085 m from the corner point.
    y = 0.08
    assert math.hypot(BALL_R, y) > params.POCKET_R
    result = simulate(TableState([Ball(0, 0.4, y)]), Shot(math.pi, 0.1))
    assert result.pocketed == []
    assert result.scratch is False
    assert ball(result.final, 0).pocketed is False


def test_pocket_r_is_read_at_call_time(simulate, params, monkeypatch):
    # Same roll as the near miss above: it drops once the radius is larger.
    y = params.POCKET_R + 0.01
    state = TableState([Ball(0, 0.6, y)])
    shot = Shot(0.0, speed_to_roll(params, 1.4) / params.V_MAX)    # stops at x = 2.0
    monkeypatch.setattr(params, "POCKET_R", y + 0.01)
    assert_sunk_in(simulate(state, shot), 0, 1)
    monkeypatch.setattr(params, "POCKET_R", y - 0.01)
    assert simulate(state, shot).pocketed == []


# --- check order: pocket before cushion ---------------------------------------------

def test_ball_frozen_to_the_rail_at_a_pocket_mouth_drops_when_hit_into_the_rail(simulate,
                                                                              params):
    # Ball 6 rests against the left cushion inside the capture radius of corner (0, 0).
    # Pushed into the cushion, it must drop rather than bounce back out.
    obj = (BALL_R, 0.045)
    assert math.hypot(*obj) < params.POCKET_R
    cue = (BALL_R + 0.4, 0.045)
    power = power_to_arrive(params, 1.0, 0.4 - D)
    result = simulate(TableState([Ball(0, *cue), Ball(6, *obj)]), Shot(math.pi, power))
    assert result.pocketed == [6]
    assert_sunk_in(result, 6, 0)


# --- only moving balls are checked --------------------------------------------------

def resting_in_the_mouth() -> TableState:
    """Ball 7 rests inside the capture radius of corner (0, 0); the cue is far away."""
    return TableState([Ball(0, 1.5, MID_Y), Ball(7, 0.035, 0.035)])


def test_resting_ball_inside_the_capture_radius_is_not_pocketed(simulate, params):
    state = resting_in_the_mouth()
    assert math.hypot(0.035, 0.035) < params.POCKET_R
    result = simulate(state, Shot(0.0, 0.1))           # cue rolls away from it
    assert result.pocketed == []
    b = ball(result.final, 7)
    assert (b.pocketed, b.x, b.y) == (False, 0.035, 0.035)


def test_zero_power_pockets_nothing_even_inside_the_capture_radius(simulate):
    result = simulate(resting_in_the_mouth(), Shot(0.0, 0.0))
    assert result.pocketed == []
    assert result.scratch is False


def test_resting_ball_inside_the_capture_radius_drops_when_hit(simulate, params):
    state = resting_in_the_mouth()
    u = into_pocket(0)                                 # 45° down-left, straight at it
    cue = back_from((0.035, 0.035), u, 0.5)
    state.balls[0] = Ball(0, *cue)
    result = simulate(state, Shot(aim(u), power_to_arrive(params, 0.5, 0.5 - D)))
    assert result.pocketed == [7]
    assert_sunk_in(result, 7, 0)


# --- pocketed balls leave play ------------------------------------------------------

def test_pocketed_ball_no_longer_collides(simulate, params, monkeypatch):
    # With E_BALL = 0 the cue and ball 5 roll down together, touching, into the bottom
    # side pocket. POCKET_R = 0.035 is smaller than 2·BALL_R, so if ball 5 stayed in play
    # (at its pocket point or where it dropped) the cue would hit it before reaching the
    # capture circle and never drop.
    monkeypatch.setattr(params, "E_BALL", 0.0)
    monkeypatch.setattr(params, "POCKET_R", 0.035)
    state = TableState([Ball(0, TABLE_W / 2, 0.7), Ball(5, TABLE_W / 2, 0.3)])
    power = power_to_arrive(params, 3.0, 0.4 - D)
    result = simulate(state, Shot(-math.pi / 2, power))
    assert result.pocketed == [5, 0]
    assert_sunk_in(result, 5, 1)
    assert_sunk_in(result, 0, 1)


def test_pocketed_ball_is_listed_once_and_flagged_in_final(simulate, params):
    state, shot = object_ball_at_pocket(params, 5)
    result = simulate(state, shot)
    assert result.pocketed.count(1) == 1
    assert [b.id for b in result.final.balls] == [0, 1]
    assert [b.pocketed for b in result.final.balls] == [False, True]


def test_existing_near_corner_fixture_sinks_ball_1(simulate):
    result = simulate(near_corner_pocket(), Shot(math.pi / 4, 0.5))
    assert result.pocketed == [1]
    assert_sunk_in(result, 1, 5)
    assert result.scratch is False


# --- pocketed order -----------------------------------------------------------------

def two_ball_cut() -> TableState:
    """Elastic 30° cut: ball 3 goes into the top-right corner, the cue goes off at -60°,
    hits ball 11 full and ball 11 goes into the bottom side pocket.

    C is the cue centre at contact, chosen so that ball 3's line (30° from C + D·u(30))
    passes through (TABLE_W, TABLE_H) and the cue's line (-60° from C) passes through
    (TABLE_W / 2, 0). Ball 11 is 0.2 m down that line. Ball 11 drops after about 0.36 s,
    ball 3 after about 0.98 s, so the order is [11, 3].
    """
    cot30, cot60 = 1 / math.tan(math.radians(30)), 1 / math.tan(math.radians(60))
    cy = (cot30 * (TABLE_H - D / 2) - TABLE_W / 2 + D * math.cos(math.radians(30))) \
        / (cot30 + cot60)
    cx = TABLE_W / 2 - cy * cot60
    a, b = unit(30), unit(-60)
    return TableState([Ball(0, cx - 0.5, cy),
                       Ball(3, cx + D * a[0], cy + D * a[1]),
                       Ball(11, cx + 0.2 * b[0], cy + 0.2 * b[1])])


def test_pocketed_lists_balls_in_the_order_they_dropped(simulate, params, elastic):
    power = power_to_arrive(params, 2.0, 0.5)
    result = simulate(two_ball_cut(), Shot(0.0, power))
    assert result.first_contact == 3
    assert result.pocketed == [11, 3]
    assert_sunk_in(result, 11, 1)
    assert_sunk_in(result, 3, 5)
    assert_still_on_table(result, 0)
    assert result.scratch is False


# --- scratches ----------------------------------------------------------------------

def test_cue_pocketed_with_no_contact_is_a_scratch(simulate):
    u = into_pocket(3)
    start = back_from(POCKETS[3], u, 0.6)
    state = TableState([Ball(0, *start), Ball(8, 2.0, MID_Y)])
    result = simulate(state, Shot(aim(u), 0.3))
    assert result.scratch is True
    assert result.pocketed == [0]
    assert result.first_contact is None
    assert_sunk_in(result, 0, 3)
    assert_still_on_table(result, 8)


@pytest.mark.parametrize("make_shot", [
    lambda p: (TableState([Ball(0, 1.0, MID_Y), Ball(1, 2.0, 0.3)]), Shot(math.pi, 0.05)),
    lambda p: (TableState([Ball(0, 1.0, MID_Y)]), Shot(0.0, 0.2)),  # mid line, no pocket
    lambda p: object_ball_at_pocket(p, 2),
    lambda p: (resting_in_the_mouth(), Shot(0.0, 0.0)),
], ids=["hits-nothing", "lone-cue-off-a-cushion", "sinks-object-ball", "zero-power"])
def test_no_scratch_unless_the_cue_drops(simulate, params, make_shot):
    state, shot = make_shot(params)
    result = simulate(state, shot)
    assert 0 not in result.pocketed
    assert result.scratch is False
    assert ball(result.final, 0).pocketed is False


def test_scratch_matches_the_cue_being_in_pocketed_over_many_shots(simulate):
    # Whatever happens, scratch is exactly "0 in pocketed".
    for deg in range(0, 360, 15):
        for power in (0.3, 1.0):
            result = simulate(break_table(), Shot(math.radians(deg), power))
            assert result.scratch is (0 in result.pocketed)
            assert len(result.pocketed) == len(set(result.pocketed))
            for bid in result.pocketed:
                b = ball(result.final, bid)
                assert b.pocketed is True
                assert (b.x, b.y) in [pytest.approx(p, abs=1e-12) for p in POCKETS]


def cue_follows_object_ball() -> TableState:
    """Elastic 45° cut with the cue aimed -x: ball 2 goes 45° into corner (0, 0) and the
    cue goes off at 135° into corner (0, TABLE_H). Contact is at (H/2, H/2), the only
    point on both diagonals."""
    c = TABLE_H / 2
    a = unit(225)
    return TableState([Ball(0, c + 0.4, c), Ball(2, c + D * a[0], c + D * a[1])])


def test_cue_and_object_ball_sunk_together_lists_both_and_is_a_scratch(simulate, params,
                                                                      elastic):
    power = power_to_arrive(params, 2.0, 0.4)
    result = simulate(cue_follows_object_ball(), Shot(math.pi, power))
    assert sorted(result.pocketed) == [0, 2]
    assert result.scratch is True
    assert result.first_contact == 2
    assert_sunk_in(result, 2, 0)
    assert_sunk_in(result, 0, 3)


def test_scratched_cue_is_not_respotted(simulate):
    # Physics leaves the cue in the pocket; re-spotting is the agent's or app's job.
    u = into_pocket(1)
    result = simulate(TableState([Ball(0, TABLE_W / 2, 0.4), Ball(9, 1.9, 0.9)]),
                      Shot(aim(u), 0.3))
    assert result.scratch is True
    cue = ball(result.final, 0)
    assert cue.pocketed is True
    assert (cue.x, cue.y) == pytest.approx(POCKETS[1], abs=1e-12)
    assert sum(1 for b in result.final.balls if b.id == 0) == 1
