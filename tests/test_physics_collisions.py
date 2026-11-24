"""Failing tests for issue #5: ball-ball collisions.

Interface expected (physics/simulate.py):

    resolve_collision(p1, v1, p2, v2) -> (v1', v2')
        p1, p2 are (x, y) ball centres in m; v1, v2 are (vx, vy) velocities in m/s.
        Equal masses, no spin, no throw. Along the line through the two centres the
        velocities are exchanged, scaled by params.E_BALL (read at call time):
            k = (1 + E_BALL) / 2 * closing_speed
        and the components across that line are kept. If the balls are not closing
        (moving apart or level), the velocities come back unchanged. Returns new tuples
        and does not mutate its inputs.

simulate() uses resolve_collision() for every ball-ball contact. Recorded paths come in
issue #7, so the full-simulation tests below check final positions; to check "never
overlap" along the way they stop the same shot at many different powers. Pockets come
in issue #6, so tests that need balls to survive a long roll use no_pockets.
"""
import math
import random

import pytest

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState
from tests.test_physics_cushions import power_to_arrive
from tests.test_physics_simulate import ball, break_table

D = 2 * BALL_R
OVERLAP_TOL = 1e-6      # m, the issue's "2·BALL_R − 1e-6"
ANGLE_TOL = 0.5         # degrees, the issue's 90° rule tolerance
STOP_TOL = 1e-3         # m, same "within 1 mm" as issues #3 and #4
MID_Y = TABLE_H / 2


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


@pytest.fixture
def resolve():
    from physics.simulate import resolve_collision
    return resolve_collision


@pytest.fixture
def params():
    from physics import params
    return params


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


def angle_between(u, v) -> float:
    """Angle in degrees between two non-zero vectors."""
    dot = u[0] * v[0] + u[1] * v[1]
    cos = dot / (math.hypot(*u) * math.hypot(*v))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def momentum(v1, v2):
    return v1[0] + v2[0], v1[1] + v2[1]


def energy(v1, v2):
    return v1[0] ** 2 + v1[1] ** 2 + v2[0] ** 2 + v2[1] ** 2


def cut(deg: float, speed: float = 2.0):
    """Cue at the origin moving along +x; object ball touching it at `deg` off the aim."""
    nx, ny = unit(deg)
    return (0.0, 0.0), (speed, 0.0), (D * nx, D * ny), (0.0, 0.0)


def random_collisions(n: int, seed: int = 4710):
    """Touching pairs with random positions and velocities that are closing."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p1 = (rng.uniform(0.2, 2.3), rng.uniform(0.2, 1.0))
        nx, ny = unit(rng.uniform(0, 360))
        p2 = (p1[0] + D * nx, p1[1] + D * ny)
        v1 = (rng.uniform(-5, 5), rng.uniform(-5, 5))
        v2 = (rng.uniform(-5, 5), rng.uniform(-5, 5))
        if (v1[0] - v2[0]) * nx + (v1[1] - v2[1]) * ny > 1e-3:
            out.append((p1, v1, p2, v2))
    return out


def live_positions(result) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in result.final.balls if not b.pocketed}


def assert_no_overlap(result):
    live = [b for b in result.final.balls if not b.pocketed]
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            dist = math.hypot(a.x - b.x, a.y - b.y)
            assert dist >= D - OVERLAP_TOL, (
                f"balls {a.id} and {b.id} are {D - dist:.2e} m inside each other")


def frozen_rack() -> TableState:
    """Like break_table() but every ball in the rack is touching its neighbours."""
    balls = [Ball(0, TABLE_W / 4, MID_Y)]
    ids = iter([1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])
    for row in range(5):
        for k in range(row + 1):
            x = 3 * TABLE_W / 4 + row * D * math.sqrt(3) / 2
            y = MID_Y + (k - row / 2) * D
            balls.append(Ball(next(ids), x, y))
    return TableState(balls)


# --- resolve_collision(): head-on ----------------------------------------------------

@pytest.mark.parametrize("deg", [0, 90, 180, 237])
@pytest.mark.parametrize("speed", [0.1, 1.0, 6.0])
def test_head_on_elastic_cue_stops_and_object_takes_its_velocity(resolve, elastic, deg, speed):
    nx, ny = unit(deg)
    v = (speed * nx, speed * ny)
    v1, v2 = resolve((1.0, 0.6), v, (1.0 + D * nx, 0.6 + D * ny), (0.0, 0.0))
    assert v1 == pytest.approx((0.0, 0.0), abs=1e-12)
    assert v2 == pytest.approx(v, abs=1e-12)


def test_head_on_with_e_ball_below_one_leaves_the_cue_creeping_forward(resolve, params,
                                                                     monkeypatch):
    monkeypatch.setattr(params, "E_BALL", 0.9)
    v1, v2 = resolve((0.0, 0.0), (2.0, 0.0), (D, 0.0), (0.0, 0.0))
    assert v1 == pytest.approx((0.1, 0.0))          # (1 - e) / 2 * 2.0
    assert v2 == pytest.approx((1.9, 0.0))          # (1 + e) / 2 * 2.0


def test_head_on_both_moving_elastic_swaps_velocities(resolve, elastic):
    v1, v2 = resolve((0.0, 0.0), (1.5, 0.0), (D, 0.0), (-0.5, 0.0))
    assert v1 == pytest.approx((-0.5, 0.0))
    assert v2 == pytest.approx((1.5, 0.0))


# --- resolve_collision(): cuts and the 90° rule -------------------------------------

@pytest.mark.parametrize("deg", [15, 30, 60, -15, -30, -60])
def test_elastic_cut_balls_leave_90_degrees_apart(resolve, elastic, deg):
    v1, v2 = resolve(*cut(deg))
    assert angle_between(v1, v2) == pytest.approx(90.0, abs=ANGLE_TOL)


@pytest.mark.parametrize("deg", [15, 30, 60, -30])
def test_elastic_cut_speeds_split_by_cos_and_sin(resolve, elastic, deg):
    speed = 2.0
    v1, v2 = resolve(*cut(deg, speed))
    a = math.radians(deg)
    assert math.hypot(*v2) == pytest.approx(speed * math.cos(a))
    assert math.hypot(*v1) == pytest.approx(speed * abs(math.sin(a)))


@pytest.mark.parametrize("deg", [15, 30, 45, 60, 80, -20, -70])
@pytest.mark.parametrize("e", [1.0, 0.95, 0.6])
def test_object_ball_leaves_along_the_line_of_centres(resolve, params, monkeypatch, deg, e):
    monkeypatch.setattr(params, "E_BALL", e)
    _, v2 = resolve(*cut(deg))
    # 1e-4 degrees: acos() loses precision near 0, so this is as tight as it can usefully go.
    assert angle_between(v2, unit(deg)) == pytest.approx(0.0, abs=1e-4)
    # ...and so not along the cue's original aim (+x).
    assert angle_between(v2, (1.0, 0.0)) == pytest.approx(abs(deg), abs=1e-4)


@pytest.mark.parametrize("deg", [15, 30, 60])
def test_cue_ball_leaves_on_the_far_side_of_its_aim(resolve, elastic, deg):
    # Object ball above the aim line goes up; the cue is pushed below it, and vice versa.
    v1, v2 = resolve(*cut(deg))
    assert v2[1] > 0 and v1[1] < 0
    v1, v2 = resolve(*cut(-deg))
    assert v2[1] < 0 and v1[1] > 0


def test_component_across_the_line_of_centres_is_kept(resolve, params, monkeypatch):
    monkeypatch.setattr(params, "E_BALL", 0.7)
    # Line of centres along +x: only the x parts change.
    v1, v2 = resolve((0.0, 0.0), (2.0, 0.8), (D, 0.0), (-0.4, -1.1))
    assert v1[1] == pytest.approx(0.8)
    assert v2[1] == pytest.approx(-1.1)


def test_positions_need_not_be_exactly_touching(resolve, elastic):
    # Only the direction between the centres matters, not the distance.
    a = resolve((0.0, 0.0), (1.0, 0.0), (D * 0.9999, 0.0), (0.0, 0.0))
    b = resolve((0.0, 0.0), (1.0, 0.0), (D * 1.0001, 0.0), (0.0, 0.0))
    assert a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])
    assert a[1] == pytest.approx((1.0, 0.0))


# --- resolve_collision(): conservation ----------------------------------------------

@pytest.mark.parametrize("e", [1.0, 0.95, 0.8, 0.5, 0.0])
def test_momentum_is_conserved(resolve, params, monkeypatch, e):
    monkeypatch.setattr(params, "E_BALL", e)
    for p1, v1, p2, v2 in random_collisions(200):
        a, b = resolve(p1, v1, p2, v2)
        before, after = momentum(v1, v2), momentum(a, b)
        assert after[0] == pytest.approx(before[0], abs=1e-9)
        assert after[1] == pytest.approx(before[1], abs=1e-9)


def test_elastic_collision_conserves_kinetic_energy(resolve, elastic):
    for p1, v1, p2, v2 in random_collisions(200):
        a, b = resolve(p1, v1, p2, v2)
        assert energy(a, b) == pytest.approx(energy(v1, v2), abs=1e-9)


@pytest.mark.parametrize("e", [0.99, 0.95, 0.8, 0.5, 0.0])
def test_inelastic_collision_never_gains_kinetic_energy(resolve, params, monkeypatch, e):
    monkeypatch.setattr(params, "E_BALL", e)
    for p1, v1, p2, v2 in random_collisions(200, seed=int(e * 100)):
        a, b = resolve(p1, v1, p2, v2)
        assert energy(a, b) <= energy(v1, v2) + 1e-9


def test_after_a_collision_the_balls_are_not_closing(resolve, params, monkeypatch):
    # Otherwise the simulator would resolve the same contact again next step.
    for e in (1.0, 0.95, 0.3):
        monkeypatch.setattr(params, "E_BALL", e)
        for p1, v1, p2, v2 in random_collisions(100):
            a, b = resolve(p1, v1, p2, v2)
            nx, ny = p2[0] - p1[0], p2[1] - p1[1]
            assert (a[0] - b[0]) * nx + (a[1] - b[1]) * ny <= 1e-12


# --- resolve_collision(): contract -------------------------------------------------

@pytest.mark.parametrize("v1, v2", [
    ((-1.0, 0.0), (0.0, 0.0)),       # cue moving away from the object ball
    ((0.0, 0.0), (1.0, 0.0)),        # object ball moving away from the cue
    ((0.0, 1.0), (0.0, 0.0)),        # sliding past, not closing
    ((0.0, 0.0), (0.0, 0.0)),        # both still
])
def test_balls_that_are_not_closing_are_left_alone(resolve, v1, v2):
    a, b = resolve((0.0, 0.0), v1, (D, 0.0), v2)
    assert a == pytest.approx(v1)
    assert b == pytest.approx(v2)


def test_resolve_reads_e_ball_at_call_time(resolve, params, monkeypatch):
    args = ((0.0, 0.0), (2.0, 0.0), (D, 0.0), (0.0, 0.0))
    monkeypatch.setattr(params, "E_BALL", 0.5)
    assert resolve(*args)[1] == pytest.approx((1.5, 0.0))
    monkeypatch.setattr(params, "E_BALL", 1.0)
    assert resolve(*args)[1] == pytest.approx((2.0, 0.0))


def test_resolve_leaves_its_inputs_alone(resolve):
    p1, v1, p2, v2 = [0.0, 0.0], [2.0, 0.3], [D, 0.0], [0.0, 0.0]
    out = resolve(p1, v1, p2, v2)
    assert (p1, v1, p2, v2) == ([0.0, 0.0], [2.0, 0.3], [D, 0.0], [0.0, 0.0])
    assert out[0] is not v1 and out[1] is not v2


def test_resolve_is_symmetric_in_the_two_balls(resolve):
    p1, v1, p2, v2 = (0.3, 0.4), (1.2, -0.5), (0.3 + D * 0.6, 0.4 + D * 0.8), (-0.2, -0.9)
    a, b = resolve(p1, v1, p2, v2)
    b2, a2 = resolve(p2, v2, p1, v1)
    assert a == pytest.approx(a2)
    assert b == pytest.approx(b2)


# --- full simulation: where balls go after contact ----------------------------------

CONTACT_X = 1.2                 # where the cue centre is when it touches the object ball
CUE_START = (0.5, MID_Y)


def cut_table(deg: float) -> TableState:
    """Cue on the mid line aimed along +x; ball 1 placed so contact is a `deg` cut."""
    nx, ny = unit(deg)
    return TableState([Ball(0, *CUE_START), Ball(1, CONTACT_X + D * nx, MID_Y + D * ny)])


def test_full_sim_head_on_elastic_cue_stops_at_contact(simulate, params, elastic):
    arrive = 0.6
    power = power_to_arrive(params, arrive, CONTACT_X - CUE_START[0])
    result = simulate(cut_table(0), Shot(0.0, power))
    pos = live_positions(result)
    assert pos[0] == pytest.approx((CONTACT_X, MID_Y), abs=STOP_TOL)
    roll = arrive ** 2 / (2 * params.DECEL)
    assert pos[1] == pytest.approx((CONTACT_X + D + roll, MID_Y), abs=STOP_TOL)
    assert result.first_contact == 1


@pytest.mark.parametrize("deg", [15, 30, 60, -30])
def test_full_sim_object_ball_ends_on_the_line_of_centres(simulate, params, deg):
    # Default E_BALL: the object ball direction doesn't depend on restitution.
    power = power_to_arrive(params, 0.5, CONTACT_X - CUE_START[0])
    state = cut_table(deg)
    start = (state.balls[1].x, state.balls[1].y)
    end = live_positions(simulate(state, Shot(0.0, power)))[1]
    moved = (end[0] - start[0], end[1] - start[1])
    assert math.hypot(*moved) > 0.05
    assert angle_between(moved, unit(deg)) == pytest.approx(0.0, abs=0.1)
    # Not along the cue's aim (+x).
    assert angle_between(moved, (1.0, 0.0)) == pytest.approx(abs(deg), abs=0.1)


# --- full simulation: no overlap ----------------------------------------------------

POWERS = [i / 20 for i in range(1, 21)]


@pytest.mark.parametrize("deg", [0, 20, 45, 70])
def test_full_sim_cut_never_leaves_balls_overlapping(simulate, deg):
    for power in POWERS:
        assert_no_overlap(simulate(cut_table(deg), Shot(0.0, power)))


@pytest.mark.parametrize("aim_deg", [0, 15, 30, 60, 89, 180, 270])
def test_full_sim_cue_frozen_to_object_ball(simulate, aim_deg):
    # Balls start exactly touching along +x.
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 1.0 + D, MID_Y)])
    for power in POWERS:
        result = simulate(state, Shot(math.radians(aim_deg), power))
        assert_no_overlap(result)


def test_full_sim_cue_frozen_and_shot_away_does_not_touch(simulate):
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 1.0 + D, MID_Y)])
    result = simulate(state, Shot(math.pi, 0.02))      # too soft to come back off a cushion
    assert result.first_contact is None
    assert live_positions(result)[1] == pytest.approx((1.0 + D, MID_Y))


def test_full_sim_cue_frozen_and_shot_into_it_moves_it(simulate):
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 1.0 + D, MID_Y)])
    result = simulate(state, Shot(0.0, 0.05))
    assert result.first_contact == 1
    assert live_positions(result)[1][0] > 1.0 + D + 0.01


def test_full_sim_frozen_line_passes_the_hit_to_the_last_ball(simulate, elastic):
    # Newton's cradle: cue, A, B all touching in a line. Only B should roll away.
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 1.0 + D, MID_Y),
                        Ball(2, 1.0 + 2 * D, MID_Y)])
    result = simulate(state, Shot(0.0, 0.05))         # B stops short of the cushion
    pos = live_positions(result)
    assert_no_overlap(result)
    assert result.first_contact == 1
    assert pos[0][0] == pytest.approx(1.0, abs=STOP_TOL)
    assert pos[1][0] == pytest.approx(1.0 + D, abs=STOP_TOL)
    assert pos[2][0] > 1.0 + 2 * D + 0.1


@pytest.mark.parametrize("make_state", [break_table, frozen_rack], ids=["gapped", "frozen"])
@pytest.mark.parametrize("deg", [0, 0.7, -2.5])
def test_full_sim_16_ball_break_never_leaves_balls_overlapping(simulate, make_state, deg):
    for power in [0.2, 0.5, 0.8, 1.0]:
        assert_no_overlap(simulate(make_state(), Shot(math.radians(deg), power)))


def test_frozen_rack_starts_touching(simulate):
    # Guards the fixture above: neighbours really are at exactly 2·BALL_R.
    state = frozen_rack()
    a, b = ball(state, 1), ball(state, 9)
    assert math.hypot(a.x - b.x, a.y - b.y) == pytest.approx(D, abs=1e-12)
    assert_no_overlap(simulate(state, Shot(0.0, 0.0)))


# --- full simulation: first_contact -------------------------------------------------

def test_first_contact_is_none_when_nothing_is_touched(simulate):
    state = TableState([Ball(0, 0.5, MID_Y), Ball(1, 2.0, 0.3), Ball(9, 2.0, 1.0)])
    result = simulate(state, Shot(math.pi, 0.05))
    assert result.first_contact is None


def test_first_contact_after_a_cushion(simulate, no_pockets):
    # Cue goes down into the bottom cushion, comes straight back up and hits ball 4.
    state = TableState([Ball(0, 1.0, 0.3), Ball(4, 1.0, 0.8), Ball(11, 2.2, 0.3)])
    result = simulate(state, Shot(-math.pi / 2, 0.13))    # one bounce, then a soft hit
    assert result.first_contact == 4
    assert live_positions(result)[4][1] > 0.8 + 0.01


def test_first_contact_ignores_balls_the_cue_touches_later(simulate, params, elastic):
    # Elastic 45° cut on A at CONTACT_X: the cue then leaves at -45° and runs into B.
    a_dir, cue_dir = unit(45), unit(-45)
    contact = (CONTACT_X, MID_Y)
    a = (contact[0] + D * a_dir[0], contact[1] + D * a_dir[1])
    b = (contact[0] + 0.3 * cue_dir[0], contact[1] + 0.3 * cue_dir[1])
    state = TableState([Ball(0, *CUE_START), Ball(12, *b), Ball(3, *a)])   # B listed first
    # Arrive fast enough that the cue still reaches B after losing half its energy.
    power = power_to_arrive(params, 1.2, CONTACT_X - CUE_START[0])
    result = simulate(state, Shot(0.0, power))
    assert result.first_contact == 3
    assert live_positions(result)[12] != pytest.approx(b, abs=0.01)     # B was hit


# --- full simulation: chains --------------------------------------------------------

def chain_table() -> TableState:
    """Cue, A and B spaced out on the mid line."""
    return TableState([Ball(0, 0.5, MID_Y), Ball(5, 1.0, MID_Y), Ball(13, 1.5, MID_Y)])


def test_chain_moves_b_and_first_contact_is_a(simulate):
    result = simulate(chain_table(), Shot(0.0, 0.3))
    assert result.first_contact == 5
    assert live_positions(result)[13][0] > 1.5 + 0.01
    assert_no_overlap(result)


def test_elastic_chain_stops_each_ball_where_it_hits_the_next(simulate, params, elastic):
    arrive = 0.5          # speed the cue reaches A with; A and B keep it (no losses but rolling)
    power = power_to_arrive(params, arrive, (1.0 - D) - 0.5)
    result = simulate(chain_table(), Shot(0.0, power))
    pos = live_positions(result)
    assert result.first_contact == 5
    assert pos[0][0] == pytest.approx(1.0 - D, abs=STOP_TOL)
    assert pos[5][0] == pytest.approx(1.5 - D, abs=STOP_TOL)
    # B starts with A's speed on arrival at B.
    a_at_b = math.sqrt(arrive ** 2 - 2 * params.DECEL * (0.5 - D))
    assert pos[13][0] == pytest.approx(1.5 + a_at_b ** 2 / (2 * params.DECEL), abs=STOP_TOL)
