"""The physics simulator: simulate(state, shot, record_paths=False) -> ShotResult.

There is no spin, so between events every ball moves in a straight line with constant
deceleration. Each loop we:
  1. drop moving balls that are within POCKET_R of a pocket, then resolve any contacts
     (ball-ball or ball-cushion) that are touching and closing,
  2. take the largest step that can't skip a contact, a pocket or a ball stopping
     (_safe_step),
  3. move every ball exactly along its line for that step (_advance).
This is "conservative advancement", chosen in issue #1. Cushion bounces go through
reflect() and ball-ball hits through resolve_collision(), which are small enough to
test and show on their own.

Pockets (issue #6): the pocket check runs before cushion bounces, so a ball at a pocket
mouth drops instead of bouncing off the rail. A dropped ball is removed from play at
once. Only moving balls are checked, so a ball resting near a pocket stays until hit.
Physics never re-spots the cue ball after a scratch; the agent or app does that.

Paths (issue #7): with record_paths=True, each ball's path is its event points. With no
spin a ball moves in a straight line between events, so we only need the start, a point
each time a contact changes its velocity (cushion bounce or ball-ball hit), and the end
(the pocket's coordinates if it dropped). Only balls that moved get a path.

Constants are read from physics.params at call time so tests can monkeypatch them.
"""
from __future__ import annotations

import math

from core.state import BALL_R, POCKETS, TABLE_H, TABLE_W, Ball, Shot, ShotResult, TableState
from physics import params

D = 2 * BALL_R                               # centre distance when two balls touch
X_MIN, X_MAX = BALL_R, TABLE_W - BALL_R      # where a ball centre touches a cushion
Y_MIN, Y_MAX = BALL_R, TABLE_H - BALL_R
CONTACT_TOL = 1e-5      # m, a gap smaller than this counts as touching
MAX_STEPS = 100_000     # backstop in case steps get tiny and MAX_TIME is never reached
NUDGE_ROUNDS = 100      # passes used to push overlapping input balls apart
NUDGE_MARGIN = 1e-9     # m, extra push so float rounding can't leave a pair overlapping
PATH_MIN_GAP = 1e-9     # m, a path point this close to the previous one is not added


def simulate(state: TableState, shot: Shot, record_paths: bool = False) -> ShotResult:
    """Play `shot` on `state` until every ball stops. Never mutates its inputs."""
    cue = next((b for b in state.balls if b.id == 0), None)
    if cue is None:
        raise ValueError("simulate() needs a cue ball (id 0) in state.balls")
    if cue.pocketed:
        raise ValueError("simulate() can't shoot: the cue ball (id 0) is already pocketed")

    # Balls already pocketed in the input take no part in the shot.
    live = [b for b in state.balls if not b.pocketed]
    ids = [b.id for b in live]
    positions = _tidy_positions([(b.x, b.y) for b in live])

    power = min(max(shot.power, 0.0), 1.0)
    v = power * params.V_MAX
    vx, vy = v * math.cos(shot.angle), v * math.sin(shot.angle)
    balls = [(x, y, vx, vy) if bid == 0 else (x, y, 0.0, 0.0)
             for bid, (x, y) in zip(ids, positions)]

    sunk = []                           # (ball id, pocket index), in the order they dropped
    # Each path starts at the input position; if clean-up moved the ball, that's a point too.
    paths = None
    if record_paths:
        paths = {b.id: [(b.x, b.y)] for b in live}
        for bid, (x, y) in zip(ids, positions):
            _add_point(paths[bid], x, y)
    first_contact = _run(balls, ids, sunk, paths)

    on_table = {bid: (x, y) for bid, (x, y, _, _) in zip(ids, balls)}
    in_pocket = dict(sunk)
    out = []
    for b in state.balls:
        if b.id in on_table:
            out.append(Ball(b.id, *on_table[b.id]))
        elif b.id in in_pocket:
            out.append(Ball(b.id, *POCKETS[in_pocket[b.id]], pocketed=True))
        else:                           # already pocketed in the input
            out.append(Ball(b.id, b.x, b.y, pocketed=True))
    pocketed = [bid for bid, _ in sunk]
    if paths is not None:
        paths = _finish_paths(paths, state.balls, out)
    return ShotResult(final=TableState(out, state.group), pocketed=pocketed,
                      first_contact=first_contact, scratch=0 in pocketed, paths=paths)


# --- paths -------------------------------------------------------------------------

def _add_point(path, x, y):
    """Append (x, y) unless it is the point already at the end of the path."""
    lx, ly = path[-1]
    if math.hypot(x - lx, y - ly) > PATH_MIN_GAP:
        path.append((x, y))


def _finish_paths(paths, inputs, finals):
    """Add each ball's end point and keep only the balls that moved."""
    done = {}
    for b_in, b_out in zip(inputs, finals):
        if b_in.id not in paths or (b_in.x, b_in.y) == (b_out.x, b_out.y):
            continue
        path = paths[b_in.id]
        # The pocket's coordinates if it dropped. POCKETS mixes ints and floats, so convert.
        end = (float(b_out.x), float(b_out.y))
        _add_point(path, *end)
        if len(path) > 1:
            path[-1] = end              # an end right on the last event replaces it exactly
        done[b_in.id] = path
    return done


# --- cushions ----------------------------------------------------------------------

def reflect(vel: tuple[float, float], wall: str) -> tuple[float, float]:
    """Bounce a velocity off one cushion: "left", "right", "bottom" or "top".

    The component perpendicular to the cushion flips and is scaled by params.E_CUSHION;
    the component along the cushion is kept. Returns a new (vx, vy).
    """
    vx, vy = vel
    e = params.E_CUSHION
    if wall in ("left", "right"):
        return -e * vx, vy
    if wall in ("bottom", "top"):
        return vx, -e * vy
    raise ValueError(f"unknown wall {wall!r}; expected left, right, bottom or top")


# --- ball-ball collisions ----------------------------------------------------------

Vec = tuple[float, float]


def resolve_collision(p1: Vec, v1: Vec, p2: Vec, v2: Vec) -> tuple[Vec, Vec]:
    """Collide two equal-mass balls with no spin and no throw.

    p1, p2 are the (x, y) centres and v1, v2 the (vx, vy) velocities. Along the line
    through the two centres the balls exchange momentum, scaled by params.E_BALL; the
    components across that line are kept. With E_BALL = 1 the components along the line
    simply swap, which gives the 90° rule for cuts. Balls that are not closing, or whose
    centres coincide (no line to push along), are left alone. Returns new (v1', v2').
    """
    unchanged = (v1[0], v1[1]), (v2[0], v2[1])
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return unchanged
    nx, ny = dx / dist, dy / dist
    closing = (v1[0] - v2[0]) * nx + (v1[1] - v2[1]) * ny
    if closing <= 0:
        return unchanged
    k = (1 + params.E_BALL) / 2 * closing
    return (v1[0] - k * nx, v1[1] - k * ny), (v2[0] + k * nx, v2[1] + k * ny)


def _walls_hit(x, y, vx, vy):
    """The cushions the ball is touching and moving into (a corner can give two)."""
    walls = []
    if x <= X_MIN + CONTACT_TOL and vx < 0:
        walls.append("left")
    elif x >= X_MAX - CONTACT_TOL and vx > 0:
        walls.append("right")
    if y <= Y_MIN + CONTACT_TOL and vy < 0:
        walls.append("bottom")
    elif y >= Y_MAX - CONTACT_TOL and vy > 0:
        walls.append("top")
    return walls


# --- pockets -----------------------------------------------------------------------

def _nearest_pocket(x, y):
    """(distance, k): distance from a ball centre to the nearest pocket point, POCKETS[k]."""
    best, best_k = math.inf, 0
    for k, (px, py) in enumerate(POCKETS):
        dist = math.hypot(x - px, y - py)
        if dist < best:
            best, best_k = dist, k
    return best, best_k


def _drop_pocketed(balls, ids, sunk, v_stop):
    """Remove every moving ball that has reached a pocket. Mutates balls, ids and sunk.

    A ball counts as in when its centre is within POCKET_R (plus CONTACT_TOL, so
    _safe_step never has to creep up on the circle in ever smaller steps).
    """
    capture = params.POCKET_R + CONTACT_TOL
    i = 0
    while i < len(balls):
        x, y, vx, vy = balls[i]
        if math.hypot(vx, vy) > v_stop:
            dist, k = _nearest_pocket(x, y)
            if dist <= capture:
                sunk.append((ids[i], k))
                del balls[i], ids[i]
                continue
        i += 1


# --- input clean-up ----------------------------------------------------------------

def _clamp_inside(x: float, y: float) -> tuple[float, float]:
    return min(max(x, X_MIN), X_MAX), min(max(y, Y_MIN), Y_MAX)


def _tidy_positions(positions):
    """Pull balls inside the cushions and push overlapping balls apart.

    Vision output can be slightly off, so this fixes small errors on a copy. Each pass
    moves both balls of an overlapping pair half the overlap apart along the line
    between their centres (along +x if the centres coincide), then clamps again.
    """
    pos = [_clamp_inside(x, y) for x, y in positions]
    n = len(pos)
    for _ in range(NUDGE_ROUNDS):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                (xi, yi), (xj, yj) = pos[i], pos[j]
                dx, dy = xj - xi, yj - yi
                dist = math.hypot(dx, dy)
                if dist >= D:
                    continue
                nx, ny = (dx / dist, dy / dist) if dist > 1e-12 else (1.0, 0.0)
                push = (D - dist) / 2 + NUDGE_MARGIN
                pos[i] = _clamp_inside(xi - nx * push, yi - ny * push)
                pos[j] = _clamp_inside(xj + nx * push, yj + ny * push)
                moved = True
        if not moved:
            break
    return pos


# --- stepping ----------------------------------------------------------------------

def _run(balls, ids, sunk, paths=None):
    """Move balls (list of (x, y, vx, vy), mutated in place) until all stop.

    Pocketed balls are removed from balls and ids and appended to sunk as (id, pocket).
    If paths is a dict {id: [points]}, each contact point is appended to it.
    Returns first_contact: the id of the first ball the cue ball touches, or None.
    """
    decel, v_stop, pocket_r = params.DECEL, params.V_STOP, params.POCKET_R
    max_time = params.MAX_TIME

    first_contact = None
    t = 0.0
    for _ in range(MAX_STEPS):
        first_contact = _resolve_contacts(balls, ids, sunk, first_contact, v_stop, paths)
        dt = _safe_step(balls, decel, v_stop, pocket_r)
        if dt == math.inf:                  # nothing is moving
            return first_contact
        t += dt
        if t > max_time:
            raise RuntimeError(f"simulate() did not settle within MAX_TIME={max_time} s")
        _advance(balls, dt, decel, v_stop)
    raise RuntimeError(f"simulate() did not settle within {MAX_STEPS} steps")


def _safe_step(balls, decel, v_stop, pocket_r) -> float:
    """Largest dt that can't skip a contact, a pocket or a ball stopping.

    Speeds only decrease, so the current speeds are an upper bound for the whole step.
    """
    speeds = [math.hypot(vx, vy) for _, _, vx, vy in balls]
    dt = math.inf
    for i, (x, y, vx, vy) in enumerate(balls):
        s = speeds[i]
        if s <= v_stop:
            continue
        dt = min(dt, s / decel)                          # don't step past it stopping
        if vx > 0:                                       # cushions
            dt = min(dt, max(X_MAX - x, 0) / vx)
        elif vx < 0:
            dt = min(dt, max(x - X_MIN, 0) / -vx)
        if vy > 0:
            dt = min(dt, max(Y_MAX - y, 0) / vy)
        elif vy < 0:
            dt = min(dt, max(y - Y_MIN, 0) / -vy)
        gap = _nearest_pocket(x, y)[0] - pocket_r        # pockets
        if gap > 0:
            dt = min(dt, gap / s)
    for i in range(len(balls)):
        xi, yi, vxi, vyi = balls[i]
        for j in range(i + 1, len(balls)):
            combined = speeds[i] + speeds[j]
            if combined <= v_stop:
                continue
            xj, yj, vxj, vyj = balls[j]
            dx, dy = xj - xi, yj - yi
            gap = math.hypot(dx, dy) - D
            if gap <= CONTACT_TOL and (vxi - vxj) * dx + (vyi - vyj) * dy <= 0:
                continue                                 # touching but separating
            dt = min(dt, max(gap, 0) / combined)
    return dt


def _resolve_contacts(balls, ids, sunk, first_contact, v_stop, paths=None):
    """Drop balls into pockets, then bounce cushion hits and touching, closing pairs.

    Pockets are checked at the start of every round, so a ball that was just hit at a
    pocket mouth drops before it can bounce off the rail. Mutates balls, ids and sunk,
    and adds each ball whose velocity changes to paths (if given) at its current point.
    """
    for _ in range(4 * len(balls) + 4):     # repeat for chains like a frozen rack
        _drop_pocketed(balls, ids, sunk, v_stop)
        n = len(balls)
        changed = False
        for i in range(n):
            x, y, vx, vy = balls[i]
            walls = _walls_hit(x, y, vx, vy)
            if not walls:
                continue
            for wall in walls:
                vx, vy = reflect((vx, vy), wall)
            balls[i] = (x, y, vx, vy)
            changed = True
            if paths is not None:
                _add_point(paths[ids[i]], x, y)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, vxi, vyi = balls[i]
                xj, yj, vxj, vyj = balls[j]
                if math.hypot(xj - xi, yj - yi) - D > CONTACT_TOL:
                    continue
                vi, vj = resolve_collision((xi, yi), (vxi, vyi), (xj, yj), (vxj, vyj))
                if vi == (vxi, vyi) and vj == (vxj, vyj):
                    continue                    # touching but not closing
                balls[i] = (xi, yi, *vi)
                balls[j] = (xj, yj, *vj)
                changed = True
                if paths is not None:
                    _add_point(paths[ids[i]], xi, yi)
                    _add_point(paths[ids[j]], xj, yj)
                if first_contact is None and 0 in (ids[i], ids[j]):
                    first_contact = ids[j] if ids[i] == 0 else ids[i]
        if not changed:
            return first_contact
    # Ran out of rounds: a ball hit in the last round may be at a pocket mouth.
    _drop_pocketed(balls, ids, sunk, v_stop)
    return first_contact


def _advance(balls, dt, decel, v_stop):
    """Move every ball exactly along its line for dt with constant deceleration."""
    for i, (x, y, vx, vy) in enumerate(balls):
        s = math.hypot(vx, vy)
        if s <= v_stop:                 # at rest: clear any leftover crawl speed
            balls[i] = (x, y, 0.0, 0.0)
            continue
        new_s = s - decel * dt
        dist = s * dt - 0.5 * decel * dt * dt
        ux, uy = vx / s, vy / s
        if new_s <= v_stop:
            balls[i] = (x + ux * dist, y + uy * dist, 0.0, 0.0)
        else:
            balls[i] = (x + ux * dist, y + uy * dist, ux * new_s, uy * new_s)
