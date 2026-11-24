"""Dev viewer: watch one shot.

    python -m physics.demo <fixture.json> [--angle A] [--power P] [--save out.gif|out.png]

--angle is in radians, like Shot.angle. By default the cue aims straight at the nearest
object ball. --power runs 0-1 (default 0.5). Without --save an interactive window opens.
With --save a .gif saves the animation and a .png saves the final table with each
ball's path; neither needs a display.

This is a dev tool, not part of the Streamlit app. It imports only core and physics.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from core.state import BALL_R, POCKETS, TABLE_H, TABLE_W, Shot, TableState
from physics import params
from physics.simulate import simulate_timeline

FPS = 30
PAD = 0.08              # m, rail width drawn around the playing surface
CONTACT_TOL = 1e-5      # m, a cue this close to one ball-width away starts touching
CLOTH = "#1f6b3a"
RAIL = "#5b3a1e"
# Standard pool colours for 1-7; stripes 9-15 use the same colour as id - 8.
COLOURS = {1: "#f2c200", 2: "#1a4fd6", 3: "#d62020", 4: "#6a2c91", 5: "#f07800",
           6: "#138a36", 7: "#7a1f1f"}


def default_angle(state: TableState) -> float:
    """Aim straight at the nearest object ball still on the table (0 if there is none)."""
    cue = next((b for b in state.balls if b.id == 0), None)
    if cue is None:
        return 0.0                      # simulate() reports the missing cue ball
    others = [b for b in state.balls if b.id != 0 and not b.pocketed]
    if not others:
        return 0.0
    target = min(others, key=lambda b: math.hypot(b.x - cue.x, b.y - cue.y))
    return math.atan2(target.y - cue.y, target.x - cue.x)


def first_contact_point(result, start: dict):
    """Where the cue was when it first touched a ball, or None if it touched nothing.

    Until that hit only the cue has moved, so the struck ball is still at its start and
    the contact is the first place along the cue's path that is exactly one ball-width
    (2 * BALL_R) from it. The path is straight between its points, so we walk it segment
    by segment and solve for where each segment crosses that circle. If the cue starts
    touching the ball, the contact is the cue's start.
    """
    if result.first_contact not in start or not result.paths or 0 not in result.paths:
        return None
    ox, oy = start[result.first_contact]
    path = result.paths[0]
    d = 2 * BALL_R
    px, py = path[0]
    if math.hypot(px - ox, py - oy) <= d + CONTACT_TOL:
        return px, py
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        # Solve |a + t*(b - a) - o| = d for the smallest t in [0, 1].
        ux, uy = bx - ax, by - ay
        fx, fy = ax - ox, ay - oy
        a = ux * ux + uy * uy
        if a == 0:
            continue
        b = 2 * (fx * ux + fy * uy)
        c = fx * fx + fy * fy - d * d
        disc = b * b - 4 * a * c
        if disc < 0:
            continue
        t = (-b - math.sqrt(disc)) / (2 * a)
        if -1e-9 <= t <= 1 + 1e-9:
            t = min(max(t, 0.0), 1.0)
            return ax + t * ux, ay + t * uy
    return None


def ball_style(bid: int) -> dict:
    """Matplotlib Circle style: cue white, 8 black, solids filled, stripes white with a ring."""
    if bid == 0:
        return {"facecolor": "white", "edgecolor": "#888888", "linewidth": 1}
    if bid == 8:
        return {"facecolor": "black", "edgecolor": "#444444", "linewidth": 1}
    if bid < 8:
        return {"facecolor": COLOURS[bid], "edgecolor": "black", "linewidth": 1}
    return {"facecolor": "white", "edgecolor": COLOURS[bid - 8], "linewidth": 3}


def draw_table(ax):
    """Rails, cloth and pockets to scale, in metres."""
    from matplotlib.patches import Circle, Rectangle

    ax.add_patch(Rectangle((-PAD, -PAD), TABLE_W + 2 * PAD, TABLE_H + 2 * PAD, color=RAIL))
    ax.add_patch(Rectangle((0, 0), TABLE_W, TABLE_H, color=CLOTH))
    for px, py in POCKETS:
        ax.add_patch(Circle((px, py), params.POCKET_R, color="black"))
    ax.set_xlim(-PAD, TABLE_W + PAD)
    ax.set_ylim(-PAD, TABLE_H + PAD + 0.08)     # room for the status line on top
    ax.set_aspect("equal")
    ax.axis("off")


def draw(state: TableState, shot: Shot, plt):
    """Build the figure and the animation pieces. Returns (fig, frames, update)."""

    from matplotlib.patches import Circle

    result, frames = simulate_timeline(state, shot, dt=1 / FPS)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    draw_table(ax)

    # Faint lines for every ball's path, so a still picture shows the whole shot.
    for bid, path in (result.paths or {}).items():
        xs, ys = zip(*path)
        colour = {0: "white", 8: "black"}.get(bid) or COLOURS[bid if bid < 8 else bid - 8]
        ax.plot(xs, ys, color=colour, alpha=0.5, linewidth=1, linestyle="--")

    # Mark where the cue first touches a ball.
    contact = first_contact_point(result, frames[0][1])
    if contact is not None:
        cx, cy = contact
        ax.plot([cx], [cy], marker="x", color="yellow", markersize=10, markeredgewidth=2)
        ax.annotate(f"first contact: {result.first_contact}", (cx, cy),
                    xytext=(8, 8), textcoords="offset points", color="yellow")

    circles, labels = {}, {}
    for bid, (x, y) in frames[0][1].items():
        circles[bid] = ax.add_patch(Circle((x, y), BALL_R, zorder=3, **ball_style(bid)))
        if bid != 0:
            text_colour = "white" if bid == 8 else "black"
            labels[bid] = ax.text(x, y, str(bid), ha="center", va="center", fontsize=6,
                                  color=text_colour, zorder=4)

    status = ax.text(TABLE_W / 2, TABLE_H + PAD + 0.03, "", ha="center", fontsize=11)
    scratch = ax.text(TABLE_W / 2, TABLE_H / 2, "SCRATCH", ha="center", va="center",
                      fontsize=40, color="red", fontweight="bold", alpha=0.8, zorder=5,
                      visible=False)
    fig.tight_layout()

    def update(i):
        t, positions = frames[i]
        for bid, circle in circles.items():
            on_table = bid in positions
            circle.set_visible(on_table)
            if bid in labels:
                labels[bid].set_visible(on_table)
            if on_table:
                circle.center = positions[bid]
                if bid in labels:
                    labels[bid].set_position(positions[bid])
        last = i == len(frames) - 1
        sunk = ", ".join(map(str, result.pocketed)) or "none"
        status.set_text(f"angle {shot.angle:.3f} rad, power {shot.power:.2f}   "
                        f"t = {t:.2f} s" + (f"   pocketed: {sunk}" if last else ""))
        scratch.set_visible(last and result.scratch)
        return [*circles.values(), *labels.values(), status, scratch]

    return fig, frames, update


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m physics.demo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path, help="table JSON, e.g. tests/fixtures/simple_cut.json")
    parser.add_argument("--angle", type=float, help="radians (default: at the nearest ball)")
    parser.add_argument("--power", type=float, default=0.5, help="0-1 (default 0.5)")
    parser.add_argument("--save", type=Path, help="write out.gif (animation) or out.png (final)")
    args = parser.parse_args(argv)

    if not args.fixture.is_file():
        parser.error(f"fixture not found: {args.fixture}")
    if args.save and args.save.suffix.lower() not in (".gif", ".png"):
        parser.error("--save must end in .gif or .png")
    state = TableState.from_json(args.fixture.read_text())
    angle = default_angle(state) if args.angle is None else args.angle
    shot = Shot(angle, args.power)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")           # no display needed to write a file
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, frames, update = draw(state, shot, plt)
    if args.save and args.save.suffix.lower() == ".png":
        update(len(frames) - 1)
        fig.savefig(args.save, dpi=120)
    else:
        anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS,
                             blit=False, repeat=False)
        if args.save:
            anim.save(args.save, writer=PillowWriter(fps=FPS), dpi=60)
        else:
            plt.show()
    if args.save:
        print(f"saved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
