"""Tests for issue #8: MVP sanity checks with the default settings.

These are the three shots from the physics MVP objective. Unlike the unit tests for
issues #3 to #7, nothing here overrides physics.params: every test runs on the defaults,
so they show the shots look believable as shipped. That is why the tolerances are wider
than in the unit tests: E_BALL < 1, so the cue creeps a little after a head-on hit and
the split after a cut is a few degrees under 90°.

  - Straight shot goes in: cue, object ball and pocket in a line. The object ball drops,
    the cue doesn't, and the object ball is the first contact.
  - Head-on hit stops the cue: on tests/fixtures/straight_shot.json at moderate power the
    cue travels less than 2 cm after impact and the object ball carries on along the
    original line.
  - 90° split: for a 30° cut, the angle between the two balls' directions after contact,
    read from `paths`, is 90° ± 5°.
"""
import math

import pytest

from core.state import BALL_R, POCKETS, TABLE_H, Ball, Shot, TableState
from tests.test_physics_pockets import aim, back_from, into_pocket
from tests.test_physics_simulate import ball, load

D = 2 * BALL_R
MID_Y = TABLE_H / 2
CREEP_MAX = 0.02        # m, the cue may travel less than this after a head-on hit
LINE_TOL = 1.0          # degrees, how far off the original line the object ball may go
SPLIT_TOL = 5.0         # degrees, around 90° for the cut split
EVENT_TOL = 1e-4        # m, same as issue #7
MODERATE = [0.3, 0.4, 0.5]


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


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


def index_of(path, point) -> int:
    """Index of the path point at `point` (within EVENT_TOL); fails if there is none."""
    for i, p in enumerate(path):
        if math.dist(p, point) <= EVENT_TOL:
            return i
    raise AssertionError(f"{point} is not on the path {path}")


def straight_in(k: int) -> tuple[TableState, tuple[float, float]]:
    """Object ball 0.35 m from pocket k and the cue 0.5 m behind it, all on one line.

    Returns the table and the direction from the cue towards the pocket.
    """
    u = into_pocket(k)
    obj = back_from(POCKETS[k], u, 0.35)
    cue = back_from(POCKETS[k], u, 0.85)
    return TableState([Ball(0, *cue), Ball(1, *obj)], "solids"), u


def cut_table(deg: float) -> TableState:
    """Cue at (0.5, MID_Y) aimed along +x; ball 1 placed so contact at (1.2, MID_Y) is a
    `deg` cut. Same layout as the issue #5 and #7 tests."""
    nx, ny = unit(deg)
    return TableState([Ball(0, 0.5, MID_Y), Ball(1, 1.2 + D * nx, MID_Y + D * ny)])


CUT_CONTACT = (1.2, MID_Y)


# --- straight shot goes in ---------------------------------------------------------

@pytest.mark.parametrize("power", MODERATE)
@pytest.mark.parametrize("k", range(6))
def test_straight_shot_goes_in(simulate, k, power):
    state, u = straight_in(k)
    result = simulate(state, Shot(aim(u), power))
    assert result.pocketed == [1]
    assert result.scratch is False
    assert result.first_contact == 1
    assert ball(result.final, 0).pocketed is False


# --- head-on hit stops the cue ball ------------------------------------------------

STRAIGHT_CONTACT = (1.5 - D, 0.635)     # where the cue touches ball 3 in straight_shot.json


def test_straight_shot_fixture_is_the_expected_layout():
    # The head-on tests below rely on these positions.
    state = load("straight_shot.json")
    assert (ball(state, 0).x, ball(state, 0).y) == (0.5, 0.635)
    assert (ball(state, 3).x, ball(state, 3).y) == (1.5, 0.635)


@pytest.mark.parametrize("power", MODERATE)
def test_head_on_hit_first_contact_is_the_object_ball(simulate, power):
    result = simulate(load("straight_shot.json"), Shot(0.0, power))
    assert result.first_contact == 3
    assert result.scratch is False


@pytest.mark.parametrize("power", MODERATE)
def test_head_on_hit_cue_travels_less_than_2_cm_after_impact(simulate, power):
    result = simulate(load("straight_shot.json"), Shot(0.0, power), record_paths=True)
    path = result.paths[0]
    i = index_of(path, STRAIGHT_CONTACT)
    # The next point is where the cue stopped, or where ball 3 came back off the far
    # cushion and hit it while it was at rest. Either way it's the cue's creep.
    assert i + 1 < len(path), f"cue path ends at the contact point: {path}"
    creep = math.dist(path[i], path[i + 1])
    assert creep < CREEP_MAX, f"cue crept {creep * 100:.2f} cm after impact"


@pytest.mark.parametrize("power", MODERATE)
def test_head_on_hit_cue_creeps_forward_not_backward(simulate, power):
    result = simulate(load("straight_shot.json"), Shot(0.0, power), record_paths=True)
    path = result.paths[0]
    i = index_of(path, STRAIGHT_CONTACT)
    assert path[i + 1][0] >= path[i][0] - EVENT_TOL


@pytest.mark.parametrize("power", MODERATE)
def test_head_on_hit_object_ball_carries_on_along_the_line(simulate, power):
    result = simulate(load("straight_shot.json"), Shot(0.0, power), record_paths=True)
    path = result.paths[3]
    assert len(path) >= 2
    first = direction(path[0], path[1])
    assert angle_between(first, (1.0, 0.0)) < LINE_TOL
    # It travels well past where it started, to the far cushion or beyond.
    assert path[1][0] - path[0][0] > 0.5


@pytest.mark.parametrize("power", MODERATE)
def test_head_on_hit_object_ball_stays_on_the_original_line(simulate, power):
    result = simulate(load("straight_shot.json"), Shot(0.0, power), record_paths=True)
    for x, y in result.paths[3]:
        assert y == pytest.approx(0.635, abs=EVENT_TOL)


# --- 90° split after a cut ---------------------------------------------------------

@pytest.mark.parametrize("power", MODERATE)
@pytest.mark.parametrize("deg", [30, -30])
def test_30_degree_cut_splits_at_about_90_degrees(simulate, deg, power):
    result = simulate(cut_table(deg), Shot(0.0, power), record_paths=True)
    assert result.first_contact == 1
    cue_path, obj_path = result.paths[0], result.paths[1]
    i = index_of(cue_path, CUT_CONTACT)
    assert i + 1 < len(cue_path), f"cue path ends at the contact point: {cue_path}"
    cue_dir = direction(cue_path[i], cue_path[i + 1])
    obj_dir = direction(obj_path[0], obj_path[1])
    split = angle_between(cue_dir, obj_dir)
    assert split == pytest.approx(90.0, abs=SPLIT_TOL), f"split was {split:.2f}°"


@pytest.mark.parametrize("power", MODERATE)
@pytest.mark.parametrize("deg", [30, -30])
def test_30_degree_cut_object_ball_leaves_along_the_line_of_centres(simulate, deg, power):
    result = simulate(cut_table(deg), Shot(0.0, power), record_paths=True)
    obj_path = result.paths[1]
    assert angle_between(direction(obj_path[0], obj_path[1]), unit(deg)) < LINE_TOL


@pytest.mark.parametrize("power", MODERATE)
@pytest.mark.parametrize("deg", [30, -30])
def test_30_degree_cut_cue_goes_to_the_other_side(simulate, deg, power):
    # The object ball goes one side of the aim line and the cue the other.
    result = simulate(cut_table(deg), Shot(0.0, power), record_paths=True)
    cue_path = result.paths[0]
    i = index_of(cue_path, CUT_CONTACT)
    _, dy = direction(cue_path[i], cue_path[i + 1])
    assert math.copysign(1, dy) == -math.copysign(1, deg)
