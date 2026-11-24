"""Failing tests for issue #7: recorded ball paths.

Model expected (physics/simulate.py):

  - simulate(..., record_paths=False) gives paths=None. With record_paths=True, paths is
    a dict {ball id: [(x, y), ...]} with an entry for every ball that moved this shot and
    no others. "Moved" means its final position differs from its input position, so a
    shot where nothing moves gives an empty dict, not None.
  - There is no spin, so balls move in straight lines between events. A path is the list
    of event points: the start position, each ball-ball contact and cushion bounce, and
    the end position (the pocket's coordinates if the ball was pocketed). Time steps
    don't add points, and the same point is never listed twice in a row (a corner bounce
    off two cushions at once is one point).
  - Points are (x, y) tuples of floats in metres. Every point is inside the cushion lines
    except the last point of a pocketed ball, which is the pocket itself.
  - record_paths only adds paths: final, pocketed, first_contact and scratch are the same
    either way.
  - The cue ball's path includes the point where it touches the first object ball, so the
    segment after that point gives the direction the cue leaves in. The object ball
    starts at rest, so its first segment gives the direction it leaves in. Issue #8 reads
    both.

Drawing the paths in the app (app/app.py:44-46) is checked by hand, not here.
"""
import math

import pytest

from core.state import BALL_R, POCKETS, TABLE_H, TABLE_W, Ball, Shot, TableState
from tests.test_physics_cushions import power_to_arrive, speed_to_roll
from tests.test_physics_pockets import (
    assert_sunk_in, back_from, aim, into_pocket, object_ball_at_pocket, two_ball_cut,
)
from tests.test_physics_simulate import ball, break_table, load, near_corner_pocket

D = 2 * BALL_R
X_MIN, X_MAX = BALL_R, TABLE_W - BALL_R
Y_MIN, Y_MAX = BALL_R, TABLE_H - BALL_R
MID_Y = TABLE_H / 2
EDGE_TOL = 1e-9         # m, float slack on the cushion lines
EVENT_TOL = 1e-4        # m, how close a recorded event point must be to the exact one
ANGLE_TOL = 0.5         # degrees, same as issue #5's 90° rule


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


@pytest.fixture
def no_pockets(params, monkeypatch):
    monkeypatch.setattr(params, "POCKET_R", 0.0)


# --- helpers -----------------------------------------------------------------------

def unit(deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return math.cos(a), math.sin(a)


def direction(p, q) -> tuple[float, float]:
    """Unit vector from point p to point q."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy)
    assert n > 1e-6, f"segment {p} -> {q} is too short to give a direction"
    return dx / n, dy / n


def angle_between(u, v) -> float:
    """Angle in degrees between two vectors."""
    dot = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def on_cushion(p, tol: float = EVENT_TOL) -> bool:
    x, y = p
    return (abs(x - X_MIN) <= tol or abs(x - X_MAX) <= tol
            or abs(y - Y_MIN) <= tol or abs(y - Y_MAX) <= tol)


def cue_at(x: float, y: float) -> TableState:
    return TableState([Ball(0, x, y)])


def cut_table(deg: float) -> TableState:
    """Cue at (0.5, MID_Y) aimed along +x; ball 1 placed so contact at (1.2, MID_Y) is a
    `deg` cut. Same layout as the issue #5 tests."""
    nx, ny = unit(deg)
    return TableState([Ball(0, 0.5, MID_Y), Ball(1, 1.2 + D * nx, MID_Y + D * ny)])


CUT_CONTACT = (1.2, MID_Y)


def chain_table() -> TableState:
    """Cue, ball 5 and ball 13 spaced out on the mid line; ball 9 far away, never hit."""
    return TableState([Ball(0, 0.5, MID_Y), Ball(5, 1.0, MID_Y), Ball(13, 1.5, MID_Y),
                       Ball(9, 0.3, 0.2)])


# A spread of shots used by the general tests. Each is (name, make_state, shot).
SHOTS = [
    ("simple-cut", lambda: load("simple_cut.json"), Shot(-0.13, 0.5)),
    ("straight", lambda: load("straight_shot.json"), Shot(0.0, 1.0)),
    ("near-corner", near_corner_pocket, Shot(math.pi / 4, 0.5)),
    ("chain", chain_table, Shot(0.0, 0.3)),
    ("two-ball-cut", two_ball_cut, Shot(0.0, 0.6)),
    ("break-soft", break_table, Shot(0.02, 0.4)),
    ("break-full", break_table, Shot(0.0, 1.0)),
    ("break-wide", break_table, Shot(math.radians(170), 1.0)),
    ("lone-many-bounces", lambda: cue_at(0.7, 0.4), Shot(math.radians(33), 1.0)),
]
SHOT_IDS = [name for name, _, _ in SHOTS]
SHOT_ARGS = [(make, shot) for _, make, shot in SHOTS]


def starts(state: TableState) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in state.balls}


def ends(result) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in result.final.balls}


# --- record_paths off --------------------------------------------------------------

@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_paths_are_none_by_default(simulate, make_state, shot):
    assert simulate(make_state(), shot).paths is None


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_paths_are_none_when_record_paths_is_false(simulate, make_state, shot):
    assert simulate(make_state(), shot, record_paths=False).paths is None


# --- which balls get a path --------------------------------------------------------

def test_lone_cue_roll_has_only_a_cue_path(simulate):
    result = simulate(cue_at(0.5, MID_Y), Shot(0.0, 0.1), record_paths=True)
    assert isinstance(result.paths, dict)
    assert set(result.paths) == {0}


def test_miss_has_only_a_cue_path(simulate):
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 2.0, 0.3), Ball(8, 2.0, 1.0)])
    result = simulate(state, Shot(math.pi, 0.05), record_paths=True)
    assert result.first_contact is None
    assert set(result.paths) == {0}


def test_ball_that_is_never_hit_has_no_path(simulate, params, pre_tuning_params):
    power = power_to_arrive(params, 0.5, (1.0 - D) - 0.5)
    result = simulate(chain_table(), Shot(0.0, power), record_paths=True)
    assert set(result.paths) == {0, 5, 13}


def test_zero_power_gives_an_empty_dict_not_none(simulate):
    result = simulate(load("simple_cut.json"), Shot(0.0, 0.0), record_paths=True)
    assert result.paths == {}


def test_already_pocketed_ball_gets_no_entry(simulate):
    state = TableState([Ball(0, 0.5, MID_Y), Ball(3, 1.0, MID_Y, pocketed=True),
                        Ball(5, 1.5, MID_Y)])
    result = simulate(state, Shot(0.0, 0.6), record_paths=True)
    assert 3 not in result.paths
    assert {0, 5} <= set(result.paths)


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_paths_cover_exactly_the_balls_that_moved(simulate, make_state, shot):
    state = make_state()
    result = simulate(state, shot, record_paths=True)
    before, after = starts(state), ends(result)
    moved = {bid for bid in before if before[bid] != after[bid]}
    assert set(result.paths) == moved


@pytest.mark.parametrize("deg", range(0, 360, 30))
def test_paths_cover_exactly_the_balls_that_moved_on_the_break(simulate, deg):
    state = break_table()
    result = simulate(state, Shot(math.radians(deg), 0.8), record_paths=True)
    before, after = starts(state), ends(result)
    assert set(result.paths) == {bid for bid in before if before[bid] != after[bid]}


# --- shape of each path ------------------------------------------------------------

@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_points_are_tuples_of_two_finite_floats(simulate, make_state, shot):
    result = simulate(make_state(), shot, record_paths=True)
    for bid, path in result.paths.items():
        assert isinstance(path, list), f"ball {bid}: path is a {type(path).__name__}"
        assert len(path) >= 2, f"ball {bid}: a ball that moved needs a start and an end"
        for p in path:
            assert isinstance(p, tuple) and len(p) == 2, f"ball {bid}: bad point {p!r}"
            assert all(isinstance(c, float) and math.isfinite(c) for c in p), \
                f"ball {bid}: bad point {p!r}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_each_path_starts_at_the_input_position(simulate, make_state, shot):
    state = make_state()
    result = simulate(state, shot, record_paths=True)
    before = starts(state)
    for bid, path in result.paths.items():
        assert path[0] == pytest.approx(before[bid], abs=1e-12), f"ball {bid}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_each_path_ends_at_the_final_position(simulate, make_state, shot):
    result = simulate(make_state(), shot, record_paths=True)
    after = ends(result)
    for bid, path in result.paths.items():
        assert path[-1] == pytest.approx(after[bid], abs=1e-12), f"ball {bid}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_every_point_is_inside_the_cushions_except_a_pocket_end(simulate, make_state, shot):
    result = simulate(make_state(), shot, record_paths=True)
    for bid, path in result.paths.items():
        points = path[:-1] if bid in result.pocketed else path
        for x, y in points:
            assert X_MIN - EDGE_TOL <= x <= X_MAX + EDGE_TOL, f"ball {bid} at x={x}"
            assert Y_MIN - EDGE_TOL <= y <= Y_MAX + EDGE_TOL, f"ball {bid} at y={y}"
        x, y = path[-1]
        assert 0.0 <= x <= TABLE_W and 0.0 <= y <= TABLE_H, f"ball {bid} ends at {path[-1]}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_no_point_is_listed_twice_in_a_row(simulate, make_state, shot):
    result = simulate(make_state(), shot, record_paths=True)
    for bid, path in result.paths.items():
        for p, q in zip(path, path[1:]):
            assert math.hypot(q[0] - p[0], q[1] - p[1]) > 1e-9, \
                f"ball {bid}: repeated point {p} in {path}"


# --- record_paths doesn't change the result ----------------------------------------

@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_recording_paths_does_not_change_the_result(simulate, make_state, shot):
    off = simulate(make_state(), shot, record_paths=False)
    on = simulate(make_state(), shot, record_paths=True)
    assert on.final == off.final
    assert on.pocketed == off.pocketed
    assert on.first_contact == off.first_contact
    assert on.scratch == off.scratch


@pytest.mark.parametrize("deg", range(0, 360, 20))
def test_recording_paths_does_not_change_the_break(simulate, deg):
    shot = Shot(math.radians(deg), 1.0)
    off = simulate(break_table(), shot)
    on = simulate(break_table(), shot, record_paths=True)
    assert (on.final, on.pocketed, on.first_contact, on.scratch) == \
        (off.final, off.pocketed, off.first_contact, off.scratch)


# --- straight rolls and cushion bounces ----------------------------------------------

@pytest.mark.parametrize("deg", [0, 90, 180, 270, 37, 200])
def test_straight_roll_is_just_start_and_end(simulate, params, deg):
    # Rolls 0.3 m from the table centre and stops well short of every cushion, so there
    # are no events: time steps must not add points.
    start = (TABLE_W / 2, MID_Y)
    power = speed_to_roll(params, 0.3) / params.V_MAX
    result = simulate(cue_at(*start), Shot(math.radians(deg), power), record_paths=True)
    path = result.paths[0]
    assert len(path) == 2, path
    assert path[0] == pytest.approx(start, abs=1e-12)
    assert path[1] == pytest.approx(ends(result)[0], abs=1e-12)


def test_head_on_bounce_adds_one_point_on_the_cushion(simulate, params):
    # Rolls 0.5 m to the right cushion and comes back; stops long before the left one.
    x0 = X_MAX - 0.5
    power = power_to_arrive(params, 0.6, 0.5)
    result = simulate(cue_at(x0, MID_Y), Shot(0.0, power), record_paths=True)
    path = result.paths[0]
    assert len(path) == 3, path
    assert path[1] == pytest.approx((X_MAX, MID_Y), abs=EVENT_TOL)
    assert path[2][0] < X_MAX


def test_angled_bounce_point_is_where_the_aim_line_meets_the_cushion(simulate, params):
    # 60° from (1.0, 0.4) meets the top cushion line at x = 1.0 + (Y_MAX - 0.4) / tan 60°,
    # about 0.22 m from the side pocket. The ball arrives slowly, so there is one bounce.
    x0, y0 = 1.0, 0.4
    to_rail = (Y_MAX - y0) / math.sin(math.radians(60))
    hit = (x0 + (Y_MAX - y0) / math.tan(math.radians(60)), Y_MAX)
    power = power_to_arrive(params, 0.5, to_rail)
    result = simulate(cue_at(x0, y0), Shot(math.radians(60), power), record_paths=True)
    path = result.paths[0]
    assert len(path) == 3, path
    assert path[1] == pytest.approx(hit, abs=EVENT_TOL)
    # It comes back down and to the right.
    out = direction(path[1], path[2])
    assert out[0] > 0 and out[1] < 0


def test_corner_bounce_is_a_single_point(simulate, no_pockets):
    # Hits both cushions at once (pockets off), which is one event, not two.
    x0, y0 = X_MAX - 0.4, Y_MAX - 0.4
    result = simulate(cue_at(x0, y0), Shot(math.pi / 4, 0.3), record_paths=True)
    path = result.paths[0]
    assert path[1] == pytest.approx((X_MAX, Y_MAX), abs=EVENT_TOL)
    # The next point is somewhere else, back out along the diagonal.
    out = direction(path[1], path[2])
    assert angle_between(out, unit(225)) == pytest.approx(0.0, abs=ANGLE_TOL)


@pytest.mark.parametrize("deg", [3, 17, 45, 88, 135, 222, 300])
def test_lone_ball_path_points_are_all_cushion_bounces(simulate, no_pockets, deg):
    # A lone ball has no other events, so every point between start and end is a
    # bounce: it sits on a cushion line, the speed across that cushion flips sign and
    # the speed along it keeps its sign.
    result = simulate(cue_at(0.7, 0.4), Shot(math.radians(deg), 1.0), record_paths=True)
    path = result.paths[0]
    assert len(path) > 3, "a full-power roll should bounce several times"
    for before, p, after in zip(path, path[1:], path[2:]):
        assert on_cushion(p), f"interior point {p} is not on a cushion"
        u_in, u_out = direction(before, p), direction(p, after)
        if abs(p[0] - X_MIN) <= EVENT_TOL or abs(p[0] - X_MAX) <= EVENT_TOL:
            assert u_in[0] * u_out[0] < 0, f"x speed didn't flip at {p}"
        if abs(p[1] - Y_MIN) <= EVENT_TOL or abs(p[1] - Y_MAX) <= EVENT_TOL:
            assert u_in[1] * u_out[1] < 0, f"y speed didn't flip at {p}"
        if not (abs(p[0] - X_MIN) <= EVENT_TOL or abs(p[0] - X_MAX) <= EVENT_TOL):
            assert u_in[0] * u_out[0] > 0, f"x speed flipped at {p}"
        if not (abs(p[1] - Y_MIN) <= EVENT_TOL or abs(p[1] - Y_MAX) <= EVENT_TOL):
            assert u_in[1] * u_out[1] > 0, f"y speed flipped at {p}"


# --- pockets -----------------------------------------------------------------------

@pytest.mark.parametrize("k", range(6))
def test_pocketed_object_ball_path_ends_at_the_pocket(simulate, params, k):
    state, shot = object_ball_at_pocket(params, k)
    result = simulate(state, shot, record_paths=True)
    assert_sunk_in(result, 1, k)
    path = result.paths[1]
    assert path[0] == pytest.approx((state.balls[1].x, state.balls[1].y), abs=1e-12)
    assert path[-1] == pytest.approx(POCKETS[k], abs=1e-12)


def test_scratched_cue_path_ends_at_the_pocket(simulate):
    u = into_pocket(3)
    start = back_from(POCKETS[3], u, 0.6)
    result = simulate(TableState([Ball(0, *start)]), Shot(aim(u), 0.3), record_paths=True)
    assert result.scratch is True
    path = result.paths[0]
    assert path[0] == pytest.approx(start, abs=1e-12)
    assert path[-1] == pytest.approx(POCKETS[3], abs=1e-12)


def test_two_ball_cut_paths_follow_the_planned_lines(simulate, params, elastic):
    # See two_ball_cut(): the cue touches ball 3, goes off at -60°, hits ball 11 full,
    # and both object balls are pocketed.
    state = two_ball_cut()
    power = power_to_arrive(params, 2.0, 0.5)
    result = simulate(state, Shot(0.0, power), record_paths=True)
    assert result.pocketed == [11, 3]
    s = starts(state)
    contact_3 = (s[0][0] + 0.5, s[0][1])
    contact_11 = back_from(s[11], unit(-60), D)
    cue = result.paths[0]
    assert cue[1] == pytest.approx(contact_3, abs=EVENT_TOL)
    assert cue[2] == pytest.approx(contact_11, abs=EVENT_TOL)
    assert result.paths[3][-1] == pytest.approx(POCKETS[5], abs=1e-12)
    assert result.paths[11][-1] == pytest.approx(POCKETS[1], abs=1e-12)


# --- contact points and leaving directions (used by issue #8) ------------------------

@pytest.mark.parametrize("deg", [15, 30, 45, 60, -30])
def test_cue_path_includes_the_contact_point(simulate, params, deg):
    power = power_to_arrive(params, 1.0, CUT_CONTACT[0] - 0.5)
    result = simulate(cut_table(deg), Shot(0.0, power), record_paths=True)
    assert result.first_contact == 1
    cue = result.paths[0]
    assert len(cue) >= 3, cue
    assert cue[1] == pytest.approx(CUT_CONTACT, abs=EVENT_TOL)


@pytest.mark.parametrize("e", [1.0, 0.95, 0.8])
@pytest.mark.parametrize("deg", [15, 30, 45, 60, -30])
def test_object_ball_first_segment_is_along_the_line_of_centres(simulate, params,
                                                               monkeypatch, deg, e):
    monkeypatch.setattr(params, "E_BALL", e)
    power = power_to_arrive(params, 1.0, CUT_CONTACT[0] - 0.5)
    result = simulate(cut_table(deg), Shot(0.0, power), record_paths=True)
    obj = result.paths[1]
    assert angle_between(direction(obj[0], obj[1]), unit(deg)) == \
        pytest.approx(0.0, abs=ANGLE_TOL)


@pytest.mark.parametrize("deg", [15, 30, 45, 60, -30])
def test_elastic_cue_segment_after_contact_is_90_degrees_from_the_object_ball(simulate,
                                                                            params,
                                                                            elastic, deg):
    power = power_to_arrive(params, 1.0, CUT_CONTACT[0] - 0.5)
    result = simulate(cut_table(deg), Shot(0.0, power), record_paths=True)
    cue, obj = result.paths[0], result.paths[1]
    cue_out = direction(cue[1], cue[2])
    obj_out = direction(obj[0], obj[1])
    assert angle_between(cue_out, obj_out) == pytest.approx(90.0, abs=ANGLE_TOL)
    # The cue goes off on the far side of its aim: deg - 90 for a positive cut.
    expected = unit(deg - 90 if deg > 0 else deg + 90)
    assert angle_between(cue_out, expected) == pytest.approx(0.0, abs=ANGLE_TOL)


def test_inelastic_straight_hit_cue_keeps_going_forward_after_contact(simulate, params):
    # Default E_BALL < 1: on a full hit the cue creeps forward along +x after contact.
    power = power_to_arrive(params, 1.0, CUT_CONTACT[0] - 0.5)
    result = simulate(cut_table(0), Shot(0.0, power), record_paths=True)
    cue = result.paths[0]
    assert cue[1] == pytest.approx(CUT_CONTACT, abs=EVENT_TOL)
    assert angle_between(direction(cue[1], cue[2]), (1.0, 0.0)) == \
        pytest.approx(0.0, abs=ANGLE_TOL)


def test_chain_middle_ball_path_includes_its_hit_on_the_last_ball(simulate, params,
                                                                  pre_tuning_params):
    power = power_to_arrive(params, 0.5, (1.0 - D) - 0.5)
    result = simulate(chain_table(), Shot(0.0, power), record_paths=True)
    assert result.paths[0][1] == pytest.approx((1.0 - D, MID_Y), abs=EVENT_TOL)
    middle = result.paths[5]
    assert middle[0] == pytest.approx((1.0, MID_Y), abs=1e-12)
    assert any(p == pytest.approx((1.5 - D, MID_Y), abs=EVENT_TOL) for p in middle[1:])
    assert result.paths[13][0] == pytest.approx((1.5, MID_Y), abs=1e-12)
