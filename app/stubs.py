"""Dumb stand-ins so the app runs before the real modules land.
app.py uses the real module whenever it can be imported, otherwise these."""
import math, random
from core.state import *


def detect(image_path: str) -> TableState:
    # fixed fake table: cue + a few balls
    return TableState([Ball(0, 0.6, 0.63), Ball(1, 1.6, 0.5), Ball(9, 1.9, 0.8), Ball(8, 2.1, 0.4)])


def simulate(s: TableState, shot: Shot, record_paths=False) -> ShotResult:
    balls = {b.id: Ball(b.id, b.x, b.y, b.pocketed) for b in s.balls}
    cue = balls[0]
    dx, dy = math.cos(shot.angle), math.sin(shot.angle)
    best_t, first = shot.power * 2.5, None
    for b in balls.values():
        if b.id == 0 or b.pocketed:
            continue
        rx, ry = b.x - cue.x, b.y - cue.y
        t, perp = rx * dx + ry * dy, abs(rx * dy - ry * dx)
        if 0 < t < best_t and perp < 2 * BALL_R:
            first, best_t = b.id, t
    start = (cue.x, cue.y)
    cue.x = min(max(cue.x + dx * best_t, 0), TABLE_W)
    cue.y = min(max(cue.y + dy * best_t, 0), TABLE_H)
    paths, pocketed = {0: [start, (cue.x, cue.y)]}, []
    if first is not None:
        o = balls[first]
        ox, oy = o.x - cue.x, o.y - cue.y
        n = math.hypot(ox, oy) or 1
        end = (min(max(o.x + ox / n * 0.6 * shot.power, 0), TABLE_W),
               min(max(o.y + oy / n * 0.6 * shot.power, 0), TABLE_H))
        paths[first] = [(o.x, o.y), end]
        o.x, o.y = end
        if any(math.hypot(o.x - px, o.y - py) < 0.08 for px, py in POCKETS):
            o.pocketed = True
            pocketed.append(first)
    return ShotResult(TableState(list(balls.values()), s.group), pocketed, first, False,
                      paths if record_paths else None)


def _legal(s: TableState):
    live = [b for b in s.balls if b.id != 0 and not b.pocketed]
    if s.group == "solids":
        live = [b for b in live if 1 <= b.id <= 7]
    elif s.group == "stripes":
        live = [b for b in live if 9 <= b.id <= 15]
    return live


def recommend(s: TableState, skill: str) -> Recommendation:
    cue = next(b for b in s.balls if b.id == 0)
    targets = _legal(s)
    if not targets:
        raise ValueError("No legal target balls on the table.")
    base = {"Beginner": 0.3, "Intermediate": 0.5, "Advanced": 0.7}.get(skill, 0.5)
    recs = []
    for _ in range(3):
        t, pk = random.choice(targets), random.randrange(6)
        px, py = POCKETS[pk]
        ux, uy = px - t.x, py - t.y
        n = math.hypot(ux, uy) or 1
        gx, gy = t.x - 2 * BALL_R * ux / n, t.y - 2 * BALL_R * uy / n   # ghost ball
        shot = Shot(math.atan2(gy - cue.y, gx - cue.x), 0.6)
        recs.append(Recommendation(shot, t.id, pk, random.random(), min(base + random.random() * 0.3, 1),
                                   simulate(s, shot, record_paths=True)))
    recs.sort(key=lambda r: -r.expected_value)
    recs[0].alternatives = recs[1:]
    return recs[0]
