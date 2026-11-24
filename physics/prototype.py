"""Throwaway prototype for issue #1: conservative advancement + a first timing.

With no spin, each ball moves in a straight line with constant deceleration between
events. Each loop we:
  1. resolve any contacts (ball-ball or ball-rail) that are touching and closing,
  2. take the largest step that can't skip a contact or a ball stopping (safe_step),
  3. move every ball exactly along its line for that step.

Cushions are a simple velocity reflection so the timing is realistic; pockets are
not modelled. The real simulator is built in issues #2-#8.

Timing: python -m physics.prototype
"""
from __future__ import annotations

import math
import random
import time

from core.state import BALL_R, TABLE_H, TABLE_W, Ball, Shot, TableState

DECEL = 0.15            # m/s^2, rolling deceleration (placeholder until issue #10)
MAX_SPEED = 6.0         # m/s, cue-ball speed at power 1.0
CUSHION_E = 0.8         # fraction of normal speed kept after a rail bounce
CONTACT_TOL = 1e-5      # m, a gap smaller than this counts as touching
STOP_SPEED = 1e-9       # m/s, below this a ball is stopped
MAX_STEPS = 100_000     # safety net against an infinite loop

D = 2 * BALL_R          # centre distance at contact
X_MIN, X_MAX = BALL_R, TABLE_W - BALL_R   # where a ball centre touches a rail
Y_MIN, Y_MAX = BALL_R, TABLE_H - BALL_R


def safe_step(balls) -> float:
    """Largest dt that can't skip a collision or a ball stopping.

    balls: list of (x, y, vx, vy). Speeds only decrease, so today's speeds are an
    upper bound for the whole step.
    """
    speeds = [math.hypot(vx, vy) for _, _, vx, vy in balls]
    dt = math.inf
    for i, (x, y, vx, vy) in enumerate(balls):
        s = speeds[i]
        if s <= STOP_SPEED:
            continue
        dt = min(dt, s / DECEL)                          # don't step past it stopping
        if vx > 0:                                       # rails
            dt = min(dt, max(X_MAX - x, 0) / vx)
        elif vx < 0:
            dt = min(dt, max(x - X_MIN, 0) / -vx)
        if vy > 0:
            dt = min(dt, max(Y_MAX - y, 0) / vy)
        elif vy < 0:
            dt = min(dt, max(y - Y_MIN, 0) / -vy)
    for i in range(len(balls)):
        xi, yi, vxi, vyi = balls[i]
        for j in range(i + 1, len(balls)):
            combined = speeds[i] + speeds[j]
            if combined <= STOP_SPEED:
                continue
            xj, yj, vxj, vyj = balls[j]
            dx, dy = xj - xi, yj - yi
            gap = math.hypot(dx, dy) - D
            if gap <= CONTACT_TOL and (vxi - vxj) * dx + (vyi - vyj) * dy <= 0:
                continue                                 # touching but separating
            dt = min(dt, max(gap, 0) / combined)
    return dt


def _resolve_contacts(balls, ids, first_contact):
    """Bounce touching, closing pairs (equal-mass elastic) and rails. Mutates balls."""
    n = len(balls)
    for _ in range(4 * n + 4):          # repeat for chains like a frozen rack
        changed = False
        for i in range(n):
            x, y, vx, vy = balls[i]
            if (x <= X_MIN + CONTACT_TOL and vx < 0) or (x >= X_MAX - CONTACT_TOL and vx > 0):
                vx, changed = -CUSHION_E * vx, True
            if (y <= Y_MIN + CONTACT_TOL and vy < 0) or (y >= Y_MAX - CONTACT_TOL and vy > 0):
                vy, changed = -CUSHION_E * vy, True
            balls[i] = (x, y, vx, vy)
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, vxi, vyi = balls[i]
                xj, yj, vxj, vyj = balls[j]
                dx, dy = xj - xi, yj - yi
                dist = math.hypot(dx, dy)
                if dist - D > CONTACT_TOL:
                    continue
                nx, ny = dx / dist, dy / dist
                closing = (vxi - vxj) * nx + (vyi - vyj) * ny
                if closing <= 0:
                    continue
                # Equal masses, no spin: swap the velocity components along the normal.
                balls[i] = (xi, yi, vxi - closing * nx, vyi - closing * ny)
                balls[j] = (xj, yj, vxj + closing * nx, vyj + closing * ny)
                changed = True
                if first_contact is None and 0 in (ids[i], ids[j]):
                    first_contact = ids[j] if ids[i] == 0 else ids[i]
        if not changed:
            break
    return first_contact


def _advance(balls, dt):
    """Move every ball exactly along its line for dt with constant deceleration."""
    for i, (x, y, vx, vy) in enumerate(balls):
        s = math.hypot(vx, vy)
        if s <= STOP_SPEED:
            continue
        new_s = s - DECEL * dt
        dist = s * dt - 0.5 * DECEL * dt * dt
        ux, uy = vx / s, vy / s
        if new_s <= STOP_SPEED:
            balls[i] = (x + ux * dist, y + uy * dist, 0.0, 0.0)
        else:
            balls[i] = (x + ux * dist, y + uy * dist, ux * new_s, uy * new_s)


def run(state: TableState, shot: Shot):
    """Play the shot until all balls stop. Returns (final TableState, first_contact)."""
    live = [b for b in state.balls if not b.pocketed]
    ids = [b.id for b in live]
    v = shot.power * MAX_SPEED
    balls = [(b.x, b.y, v * math.cos(shot.angle), v * math.sin(shot.angle)) if b.id == 0
             else (b.x, b.y, 0.0, 0.0) for b in live]

    first_contact = None
    for _ in range(MAX_STEPS):
        first_contact = _resolve_contacts(balls, ids, first_contact)
        dt = safe_step(balls)
        if dt == math.inf:              # nothing is moving
            break
        _advance(balls, dt)

    final = {bid: (x, y) for bid, (x, y, _, _) in zip(ids, balls)}
    out = [Ball(b.id, *final[b.id], False) if b.id in final
           else Ball(b.id, b.x, b.y, b.pocketed) for b in state.balls]
    return TableState(out, state.group), first_contact


def time_shots(state: TableState, shots) -> dict:
    """Time run() once per shot with time.perf_counter. Returns n, mean, p95, max in ms."""
    times = []
    for shot in shots:
        t0 = time.perf_counter()
        run(state, shot)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"n": len(times),
            "mean_ms": sum(times) / len(times),
            "p95_ms": times[math.ceil(0.95 * len(times)) - 1],
            "max_ms": times[-1]}


# --- timing tables -------------------------------------------------------------

def mid_game_table() -> TableState:
    """About 8 balls spread out; the cue is lined up for a cut on ball 2."""
    return TableState([Ball(0, 0.8, 0.45), Ball(2, 1.5, 0.55), Ball(5, 0.4, 1.0),
                       Ball(8, 1.9, 1.0), Ball(10, 2.2, 0.3), Ball(12, 1.2, 1.05),
                       Ball(14, 2.3, 0.9), Ball(7, 0.3, 0.2)])


def break_table() -> TableState:
    """Cue on the head string, tight 15-ball rack with the apex on the foot spot."""
    balls = [Ball(0, TABLE_W / 4, TABLE_H / 2)]
    ids = iter([1, 9, 2, 10, 8, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])
    for row in range(5):
        for k in range(row + 1):
            x = 3 * TABLE_W / 4 + row * (D * math.sqrt(3) / 2 + 1e-4)
            y = TABLE_H / 2 + (k - row / 2) * (D + 1e-4)
            balls.append(Ball(next(ids), x, y))
    return TableState(balls)


def main(n: int = 200, seed: int = 0):
    rng = random.Random(seed)
    mid = mid_game_table()
    cue, obj = mid.balls[0], mid.balls[1]
    to_obj = math.atan2(obj.y - cue.y, obj.x - cue.x)
    # Cut shots: aim within about +/-3 degrees of full-ball on ball 2, medium power.
    mid_shots = [Shot(to_obj + rng.uniform(-0.05, 0.05), rng.uniform(0.3, 0.7)) for _ in range(n)]
    break_shots = [Shot(rng.uniform(-0.01, 0.01), 1.0) for _ in range(n)]

    for name, state, shots in [("mid-game (8 balls, cut)", mid, mid_shots),
                               ("break (16 balls, full power)", break_table(), break_shots)]:
        s = time_shots(state, shots)
        print(f"{name:30s} n={s['n']}  mean={s['mean_ms']:.2f} ms  "
              f"p95={s['p95_ms']:.2f} ms  max={s['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
