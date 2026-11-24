"""Tests for issue #10: the physics constants match published pool physics.

The constants were first tuned by eye against Billiard Shot Studio, then re-tuned against
Dr. Dave Alciatore's published measurements (Colorado State, billiards.colostate.edu):

  - TP B.6, "CB table lengths of travel for different speeds, accounting for rail
    rebound and drag losses": how many table lengths a cue ball rolls at a given speed
    on a 9-foot table, using rolling resistance 0.01, rail COR 0.7 and the skid after
    each rail. Its table length is 100 in = 2.54 m, the same as TABLE_W.
  - His table of pool equipment properties (drdavepoolinfo.com/?p=2349): rolling
    resistance 0.005-0.015, ball-ball COR 0.92-0.98, ball-rail COR 0.6-0.9.

Our model has no spin, so E_CUSHION is an effective value: the rail COR (0.7) times the
5/7 of speed a stunned ball keeps once it starts rolling again, about 0.5.

The tests check, on the default params with no monkeypatching:
  - a lone cue ball shot from one short rail along the table travels TP B.6's number of
    table lengths at each of its speeds, within ±10% (TRAVEL_TOL),
  - a head-on cushion bounce keeps about 0.5 of the speed, ±10%,
  - every constant is inside its published range and a full-power shot stops well
    before MAX_TIME,
  - each of the five constants has a one-line comment saying where its value came from,
    and params.py no longer calls them placeholders.

Checked elsewhere or by hand, not here:
  - The #8 sanity tests (tests/test_physics_sanity.py) run on the defaults, so a plain
    `pytest` re-checks them. Run `pytest -m slow` for the speed check too.
  - Watching shots in the dev viewer (#9), e.g.
    python -m physics.demo tests/fixtures/straight_shot.json --power 1.0
"""
import math
import re
from pathlib import Path

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState

# --- published reference values ------------------------------------------------------

MPH = 0.44704                   # m/s per mph
# TP B.6: cue-ball speed (mph) -> table lengths travelled, from a rail, rolling.
TP_B6_LENGTHS = [
    (1.5, 0.903),               # touch
    (3.0, 1.684),               # slow
    (3.465, 2.0),               # rolling lag, exactly two lengths
    (7.0, 3.005),               # medium
    (12.0, 3.69),               # fast
    (20.0, 4.715),              # power
]
CUSHION_SPEED_KEPT = 0.7 * 5 / 7    # rail COR 0.7, then 5/7 kept skidding back to rolling
MU_R = (0.005, 0.015)           # ball-cloth rolling resistance range
E_BALL_RANGE = (0.92, 0.98)     # ball-ball COR range
V_MAX_RANGE = (20 * MPH, 30 * MPH)  # TP B.6 power shot (20 mph) up to a powerful break
G = 9.81                        # m/s^2

# -------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = ROOT / "physics" / "params.py"
TUNED = ["V_MAX", "DECEL", "E_CUSHION", "E_BALL", "POCKET_R"]
PLACEHOLDER_COMMENTS = {        # the comments params.py had before tuning
    "V_MAX": "m/s, cue-ball speed at power 1.0",
    "DECEL": "m/s^2, constant rolling deceleration",
    "E_BALL": "ball-ball restitution (1.0 = perfectly elastic)",
    "E_CUSHION": "fraction of normal speed kept after a cushion bounce",
    "POCKET_R": "m, a ball centre this close to a pocket point drops (used from issue #6)",
}
TRAVEL_TOL = 0.10               # ±10% on table lengths and cushion speed kept
MID_Y = TABLE_H / 2
SPAN = TABLE_W - 2 * BALL_R     # ball-centre travel from one short rail to the other
X_MAX = TABLE_W - BALL_R        # where a ball centre touches the right cushion
CUSHION_ROLL = 0.5              # m the bounce-test ball would roll on past the cushion
STOP_MARGIN = 0.5               # a full-power shot must stop within half of MAX_TIME


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


@pytest.fixture
def params():
    from physics import params
    return params


# --- helpers -----------------------------------------------------------------------

def path_length(path: list[tuple[float, float]]) -> float:
    return sum(math.dist(p, q) for p, q in zip(path, path[1:]))


def lengths_from_rail(simulate, params, speed: float) -> float:
    """Table lengths a lone cue ball travels when shot at `speed` m/s from a short rail.

    It starts against the left rail and runs along the centre line, so it bounces
    head-on off the short cushions and never goes near a pocket. Like TP B.6, one table
    length is the distance from one rail to the other.
    """
    power = speed / params.V_MAX
    assert 0 < power <= 1.001, f"{speed:.2f} m/s is beyond full power ({params.V_MAX} m/s)"
    power = min(power, 1.0)                 # 20 mph rounds to V_MAX = 8.94 m/s
    result = simulate(TableState([Ball(0, BALL_R, MID_Y)]), Shot(0.0, power),
                      record_paths=True)
    assert not result.scratch, "the lone roll should never reach a pocket"
    return path_length(result.paths[0]) / SPAN


def assert_within(actual: float, expected: float, what: str):
    lo, hi = expected * (1 - TRAVEL_TOL), expected * (1 + TRAVEL_TOL)
    assert lo <= actual <= hi, (
        f"{what}: simulated {actual:.3f}, published {expected:.3f} "
        f"(allowed {lo:.3f} to {hi:.3f})")


def params_comments() -> dict[str, str]:
    """The trailing comment on each `NAME = value  # comment` line of params.py."""
    comments = {}
    for line in PARAMS_FILE.read_text().splitlines():
        m = re.match(r"^([A-Z_]+)\s*=\s*[^#]+#\s*(.*\S)\s*$", line)
        if m:
            comments[m.group(1)] = m.group(2)
    return comments


# --- travel matches TP B.6 ---------------------------------------------------------

@pytest.mark.parametrize("mph, expected", TP_B6_LENGTHS)
def test_roll_from_a_rail_travels_the_published_table_lengths(simulate, params, mph,
                                                               expected):
    assert_within(lengths_from_rail(simulate, params, mph * MPH), expected,
                  f"{mph} mph table lengths")


def test_full_power_rolls_at_least_as_far_as_a_power_shot(simulate, params):
    full = lengths_from_rail(simulate, params, params.V_MAX)
    assert full >= TP_B6_LENGTHS[-1][1] * (1 - TRAVEL_TOL)


def test_head_on_cushion_bounce_keeps_the_published_speed(simulate, params):
    # Arrive at the right cushion with just enough speed to roll CUSHION_ROLL metres more.
    x0 = TABLE_W - 0.6
    v_wall = math.sqrt(2 * params.DECEL * CUSHION_ROLL)
    v0 = math.sqrt(v_wall ** 2 + 2 * params.DECEL * (X_MAX - x0))
    power = v0 / params.V_MAX
    assert 0 < power <= 1, "bounce test shot is out of reach with the tuned params"

    result = simulate(TableState([Ball(0, x0, MID_Y)]), Shot(0.0, power), record_paths=True)
    path = result.paths[0]
    assert len(path) == 3, f"expected start, one bounce and an end, got {path}"
    back = path[1][0] - path[2][0]
    kept = math.sqrt(back / CUSHION_ROLL)          # speed out / speed in
    assert_within(kept, CUSHION_SPEED_KEPT, "cushion speed kept")


def test_faster_rolls_further(simulate, params):
    travel = [lengths_from_rail(simulate, params, mph * MPH) for mph, _ in TP_B6_LENGTHS]
    assert travel == sorted(travel)


# --- the constants are sensible ----------------------------------------------------

def test_restitutions_are_fractions(params):
    assert 0 < params.E_BALL <= 1
    assert 0 < params.E_CUSHION < 1


def test_ball_ball_restitution_is_in_the_published_range(params):
    assert E_BALL_RANGE[0] <= params.E_BALL <= E_BALL_RANGE[1]


def test_friction_is_a_published_rolling_resistance(params):
    # DECEL = mu_r * g for a rolling ball.
    assert MU_R[0] * G <= params.DECEL <= MU_R[1] * G


def test_full_power_is_between_a_power_shot_and_a_break(params):
    assert V_MAX_RANGE[0] - 0.01 <= params.V_MAX <= V_MAX_RANGE[1]


def test_pocket_radius_fits_a_ball_but_not_much_more(params):
    # A ball tucked into a corner has its centre BALL_R * sqrt(2) from the pocket point,
    # so a smaller radius could never drop a ball there. Much over 3 ball radii would
    # swallow balls that only pass near a pocket mouth.
    assert BALL_R * math.sqrt(2) < params.POCKET_R <= 3 * BALL_R


def test_full_power_shot_stops_well_before_max_time(params):
    # MAX_TIME means "something went wrong", so a normal shot must never get near it.
    # With low rolling resistance most of the speed is lost at the cushions, so time the
    # real shot: a full-power roll along the table, bouncing off both short rails.
    from physics.simulate import simulate_timeline
    _, frames = simulate_timeline(TableState([Ball(0, BALL_R, MID_Y)]), Shot(0.0, 1.0),
                                  dt=0.5)
    assert frames[-1][0] < STOP_MARGIN * params.MAX_TIME


# --- params.py says where each value came from -------------------------------------

def test_params_no_longer_calls_the_values_placeholders():
    assert "placeholder" not in PARAMS_FILE.read_text().lower()


@pytest.mark.parametrize("name", TUNED)
def test_each_tuned_constant_has_a_one_line_comment(name):
    comments = params_comments()
    assert name in comments, f"{name} needs a value and a comment on one line in params.py"


@pytest.mark.parametrize("name", TUNED)
def test_each_comment_says_where_the_value_came_from(name):
    comment = params_comments().get(name, "")
    assert comment and comment != PLACEHOLDER_COMMENTS[name], (
        f"update the comment on {name} to say where its value came from")
