"""Failing tests for issue #9: the dev viewer, its timeline helper, and the timing script.

Model expected:

    physics/simulate.py
        simulate_timeline(state, shot, dt=1/30) -> (ShotResult, frames)
            Internal helper for the viewer, not part of the shared interface. It plays the
            same shot as simulate() and also samples where the balls are over time.
            `frames` is a list of (t, {ball id: (x, y)}) with t = 0, dt, 2*dt, ... and one
            last frame when every ball has stopped. A frame lists the balls still on the
            table at time t, so a ball disappears from every frame after it drops. Balls
            already pocketed in the input never appear. The ShotResult is the same as
            simulate(state, shot, record_paths=True) gives.

    python -m physics.demo <fixture.json> [--angle A] [--power P] [--save out.gif|out.png]
        With --save it writes the file and exits 0 without needing a display.

    python -m physics.bench [--n N]
        Prints mean, p95 and max ms per call for a mid-game table and for the break.

Both scripts import only from core and physics (plus the standard library, matplotlib
and numpy). ShotResult and core/ are not changed.

Checked by hand, not here: the drawing itself (colours, scale, the first-contact mark
and the "SCRATCH" label), the interactive window, and that the straight-shot and cut
fixtures look right when watched.
"""
import ast
import copy
import dataclasses
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.state import BALL_R, POCKETS, TABLE_H, TABLE_W, Ball, Shot, ShotResult, TableState
from tests.test_physics_pockets import aim, back_from, into_pocket, object_ball_at_pocket
from tests.test_physics_simulate import break_table, load, near_corner_pocket

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
MID_Y = TABLE_H / 2
DT = 1 / 30
POS_TOL = 1e-9          # m, float slack when comparing positions
ON_PATH_TOL = 1e-6      # m, how far a sampled position may sit from the recorded path
RUN_TIMEOUT = 120       # s, generous so a slow GIF write doesn't flake


@pytest.fixture
def simulate():
    # Imported here so each test fails on its own until the module exists.
    from physics.simulate import simulate
    return simulate


@pytest.fixture
def simulate_timeline():
    from physics.simulate import simulate_timeline
    return simulate_timeline


@pytest.fixture
def params():
    from physics import params
    return params


# --- helpers -----------------------------------------------------------------------

def run_module(*args: str) -> subprocess.CompletedProcess:
    """Run `python -m <args>` from the repo root with no display available."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY", "MPLBACKEND")}
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=RUN_TIMEOUT)


def live_starts(state: TableState) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in state.balls if not b.pocketed}


def live_ends(result: ShotResult) -> dict[int, tuple[float, float]]:
    return {b.id: (b.x, b.y) for b in result.final.balls if not b.pocketed}


def distance_to_segment(p, a, b) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    n2 = dx * dx + dy * dy
    s = 0.0 if n2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / n2))
    return math.hypot(p[0] - (ax + s * dx), p[1] - (ay + s * dy))


def distance_to_path(p, path) -> float:
    if len(path) == 1:
        return math.hypot(p[0] - path[0][0], p[1] - path[0][1])
    return min(distance_to_segment(p, a, b) for a, b in zip(path, path[1:]))


def scratch_table() -> tuple[TableState, Shot]:
    """Lone cue ball rolling into pocket 3 (top-left corner)."""
    u = into_pocket(3)
    return TableState([Ball(0, *back_from(POCKETS[3], u, 0.6))]), Shot(aim(u), 0.3)


def pocketed_input_table() -> TableState:
    return TableState([Ball(0, 0.5, MID_Y), Ball(3, 1.0, MID_Y, pocketed=True),
                       Ball(5, 1.5, MID_Y)])


# Shots used by the general timeline tests. Each is (name, make_state, shot).
SHOTS = [
    ("simple-cut", lambda: load("simple_cut.json"), Shot(-0.13, 0.5)),
    ("straight", lambda: load("straight_shot.json"), Shot(0.0, 1.0)),
    ("near-corner", near_corner_pocket, Shot(math.pi / 4, 0.5)),
    ("pocketed-input", pocketed_input_table, Shot(0.0, 0.6)),
    ("break-full", break_table, Shot(0.0, 1.0)),
    ("lone-bounces", lambda: TableState([Ball(0, 0.7, 0.4)]), Shot(math.radians(33), 1.0)),
]
SHOT_IDS = [name for name, _, _ in SHOTS]
SHOT_ARGS = [(make, shot) for _, make, shot in SHOTS]


# --- simulate_timeline: same result as simulate() ----------------------------------

@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_timeline_result_matches_simulate(simulate, simulate_timeline, make_state, shot):
    expected = simulate(make_state(), shot, record_paths=True)
    result, _ = simulate_timeline(make_state(), shot)
    assert isinstance(result, ShotResult)
    assert result.final == expected.final
    assert result.pocketed == expected.pocketed
    assert result.first_contact == expected.first_contact
    assert result.scratch == expected.scratch
    assert result.paths == expected.paths


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_timeline_does_not_mutate_input(simulate_timeline, make_state, shot):
    state = make_state()
    before = copy.deepcopy(state)
    simulate_timeline(state, shot)
    assert state == before


def test_timeline_does_not_change_simulate(simulate, simulate_timeline):
    # Calling the viewer helper must leave no state behind that changes simulate().
    shot = Shot(0.0, 1.0)
    before = simulate(break_table(), shot)
    simulate_timeline(break_table(), shot)
    after = simulate(break_table(), shot)
    assert (after.final, after.pocketed, after.first_contact, after.scratch) == \
        (before.final, before.pocketed, before.first_contact, before.scratch)


@pytest.mark.parametrize("dt", [0, 0.0, -0.1])
def test_timeline_rejects_a_dt_that_is_not_positive(simulate_timeline, dt):
    with pytest.raises(ValueError):
        simulate_timeline(load("simple_cut.json"), Shot(0.0, 0.5), dt=dt)


# --- simulate_timeline: frame times ------------------------------------------------

@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_frames_are_time_and_position_pairs(simulate_timeline, make_state, shot):
    _, frames = simulate_timeline(make_state(), shot)
    assert isinstance(frames, list) and len(frames) >= 1
    for t, positions in frames:
        assert isinstance(t, float) and math.isfinite(t)
        assert isinstance(positions, dict)
        for bid, p in positions.items():
            assert isinstance(bid, int)
            assert isinstance(p, tuple) and len(p) == 2, f"ball {bid}: bad point {p!r}"
            assert all(isinstance(c, float) and math.isfinite(c) for c in p), \
                f"ball {bid}: bad point {p!r}"


@pytest.mark.parametrize("dt", [1 / 30, 0.01, 0.25])
@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_frames_are_dt_apart_with_one_last_frame_at_rest(simulate_timeline, make_state,
                                                       shot, dt):
    _, frames = simulate_timeline(make_state(), shot, dt=dt)
    times = [t for t, _ in frames]
    assert times[0] == 0.0
    for k, t in enumerate(times[:-1]):
        assert t == pytest.approx(k * dt, abs=1e-9), f"frame {k} at t={t}"
    if len(times) > 1:
        gap = times[-1] - times[-2]
        assert 0 < gap <= dt + 1e-9, f"last gap {gap} with dt={dt}"


def test_smaller_dt_gives_more_frames(simulate_timeline):
    state, shot = load("simple_cut.json"), Shot(-0.13, 0.5)
    _, coarse = simulate_timeline(state, shot, dt=0.1)
    _, fine = simulate_timeline(state, shot, dt=0.01)
    assert len(fine) > len(coarse) > 2


def test_last_frame_time_is_how_long_the_shot_lasts(simulate_timeline, params):
    # A lone ball at speed v with constant deceleration stops after v / DECEL seconds.
    # The power is chosen so it rolls 1.2 m, so from the table centre it never reaches a
    # cushion whatever the tuned params are.
    shot = Shot(0.0, math.sqrt(2 * params.DECEL * 1.2) / params.V_MAX)
    _, frames = simulate_timeline(TableState([Ball(0, TABLE_W / 2 - 0.6, MID_Y)]), shot)
    stop = shot.power * params.V_MAX / params.DECEL
    assert frames[-1][0] == pytest.approx(stop, rel=1e-3)


# --- simulate_timeline: positions --------------------------------------------------

@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_first_frame_is_the_input_layout(simulate_timeline, make_state, shot):
    state = make_state()
    _, frames = simulate_timeline(state, shot)
    t, positions = frames[0]
    assert set(positions) == set(live_starts(state))
    for bid, p in live_starts(state).items():
        assert positions[bid] == pytest.approx(p, abs=POS_TOL), f"ball {bid}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_last_frame_is_the_final_layout(simulate_timeline, make_state, shot):
    result, frames = simulate_timeline(make_state(), shot)
    _, positions = frames[-1]
    assert set(positions) == set(live_ends(result))
    for bid, p in live_ends(result).items():
        assert positions[bid] == pytest.approx(p, abs=POS_TOL), f"ball {bid}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_every_sampled_ball_is_on_the_table(simulate_timeline, make_state, shot):
    _, frames = simulate_timeline(make_state(), shot)
    for t, positions in frames:
        for bid, (x, y) in positions.items():
            assert BALL_R - POS_TOL <= x <= TABLE_W - BALL_R + POS_TOL, f"ball {bid}, t={t}"
            assert BALL_R - POS_TOL <= y <= TABLE_H - BALL_R + POS_TOL, f"ball {bid}, t={t}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_balls_never_jump_between_frames(simulate_timeline, params, make_state, shot):
    # No ball is ever faster than the cue's starting speed, so in one frame it can
    # move at most V_MAX * dt.
    dt = 0.02
    _, frames = simulate_timeline(make_state(), shot, dt=dt)
    limit = params.V_MAX * dt + 1e-6
    for (t0, a), (t1, b) in zip(frames, frames[1:]):
        for bid in set(a) & set(b):
            step = math.hypot(b[bid][0] - a[bid][0], b[bid][1] - a[bid][1])
            assert step <= limit, f"ball {bid} moved {step:.4f} m between t={t0} and {t1}"


@pytest.mark.parametrize("make_state, shot", SHOT_ARGS, ids=SHOT_IDS)
def test_sampled_positions_lie_on_the_recorded_paths(simulate_timeline, make_state, shot):
    # With no spin a ball moves in straight lines between its path points, so every
    # sample of a ball that stays on the table must sit on its path. (A pocketed ball's
    # path ends at the pocket's coordinates, not where it dropped, so it is skipped.)
    state = make_state()
    result, frames = simulate_timeline(state, shot)
    starts = live_starts(state)
    for t, positions in frames:
        for bid, p in positions.items():
            if bid in result.pocketed:
                continue
            path = result.paths.get(bid, [starts[bid]])
            assert distance_to_path(p, path) <= ON_PATH_TOL, f"ball {bid} off its path at t={t}"


def test_ball_that_is_never_hit_stays_put_in_every_frame(simulate_timeline):
    state = TableState([Ball(0, 0.5, MID_Y), Ball(1, 2.0, 0.3), Ball(8, 2.0, 1.0)])
    _, frames = simulate_timeline(state, Shot(math.pi, 0.05))
    for _, positions in frames:
        assert positions[1] == (2.0, 0.3)
        assert positions[8] == (2.0, 1.0)


def test_already_pocketed_input_ball_is_never_in_a_frame(simulate_timeline):
    _, frames = simulate_timeline(pocketed_input_table(), Shot(0.0, 0.6))
    assert all(3 not in positions for _, positions in frames)


def test_zero_power_gives_a_single_still_frame(simulate_timeline):
    state = load("simple_cut.json")
    _, frames = simulate_timeline(state, Shot(0.0, 0.0))
    assert len(frames) == 1
    t, positions = frames[0]
    assert t == 0.0
    assert positions == live_starts(state)


# --- simulate_timeline: pocketed balls disappear -------------------------------------

def assert_disappears_once(frames, bid: int):
    """bid is in a prefix of the frames (at least the first) and in none after."""
    present = [bid in positions for _, positions in frames]
    assert present[0], f"ball {bid} missing from the first frame"
    gone = present.index(False)
    assert not any(present[gone:]), f"ball {bid} came back after it dropped"
    assert gone >= 2, f"ball {bid} dropped before it could have moved"


@pytest.mark.parametrize("k", range(6))
def test_pocketed_object_ball_disappears_and_stays_gone(simulate_timeline, params, k):
    state, shot = object_ball_at_pocket(params, k)
    result, frames = simulate_timeline(state, shot)
    assert 1 in result.pocketed
    assert_disappears_once(frames, 1)


@pytest.mark.parametrize("k", range(6))
def test_pocketed_ball_drops_near_its_pocket(simulate_timeline, params, k):
    # Its last sampled position is no further from the pocket than one frame of travel
    # plus the pocket radius.
    state, shot = object_ball_at_pocket(params, k)
    _, frames = simulate_timeline(state, shot)
    last = [positions[1] for _, positions in frames if 1 in positions][-1]
    reach = params.POCKET_R + params.V_MAX * DT
    assert math.hypot(last[0] - POCKETS[k][0], last[1] - POCKETS[k][1]) <= reach


def test_scratched_cue_ball_disappears_and_stays_gone(simulate_timeline):
    state, shot = scratch_table()
    result, frames = simulate_timeline(state, shot)
    assert result.scratch is True
    assert_disappears_once(frames, 0)
    assert frames[-1][1] == {}


def test_break_balls_that_drop_disappear_and_the_rest_stay(simulate_timeline):
    result, frames = simulate_timeline(break_table(), Shot(0.0, 1.0))
    for bid in result.pocketed:
        assert_disappears_once(frames, bid)
    for bid in live_ends(result):
        assert all(bid in positions for _, positions in frames), f"ball {bid} vanished"


# --- interface stays the same ------------------------------------------------------

def test_shot_result_fields_are_unchanged():
    names = [f.name for f in dataclasses.fields(ShotResult)]
    assert names == ["final", "pocketed", "first_contact", "scratch", "paths"]


def test_importing_simulate_does_not_load_matplotlib():
    # The agent and app import simulate; plotting must stay in the dev tools.
    code = "import sys, physics.simulate; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=RUN_TIMEOUT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


ALLOWED_THIRD_PARTY = {"matplotlib", "numpy", "PIL"}


def imported_top_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                names.add("physics")        # a relative import stays inside physics
            elif node.module:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("script", ["demo.py", "bench.py"])
def test_script_imports_only_core_physics_and_allowed_libraries(script):
    path = ROOT / "physics" / script
    assert path.exists(), f"physics/{script} does not exist"
    allowed = {"core", "physics", "__future__"} | ALLOWED_THIRD_PARTY | sys.stdlib_module_names
    bad = imported_top_modules(path) - allowed
    assert not bad, f"physics/{script} imports {sorted(bad)}"


# --- the viewer: python -m physics.demo ----------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GIF_MAGICS = (b"GIF87a", b"GIF89a")


def test_demo_saves_a_png_of_simple_cut_headless(tmp_path):
    out = tmp_path / "out.png"
    proc = run_module("physics.demo", str(FIXTURES / "simple_cut.json"), "--save", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes().startswith(PNG_MAGIC)


def test_demo_saves_a_gif_of_simple_cut_headless(tmp_path):
    out = tmp_path / "out.gif"
    proc = run_module("physics.demo", str(FIXTURES / "simple_cut.json"),
                      "--angle", "-0.13", "--power", "0.3", "--save", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:6] in GIF_MAGICS


@pytest.mark.parametrize("fixture", ["simple_cut.json", "straight_shot.json"])
def test_demo_accepts_angle_and_power(tmp_path, fixture):
    out = tmp_path / "out.png"
    proc = run_module("physics.demo", str(FIXTURES / fixture),
                      "--angle", "0.0", "--power", "0.5", "--save", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.stat().st_size > 0


def test_demo_angle_changes_the_picture(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for out, angle in [(a, "0.0"), (b, "1.5")]:
        proc = run_module("physics.demo", str(FIXTURES / "simple_cut.json"),
                          "--angle", angle, "--power", "0.5", "--save", str(out))
        assert proc.returncode == 0, proc.stderr
    assert a.read_bytes() != b.read_bytes()


def test_demo_saves_a_scratch_shot(tmp_path):
    # The cue ball drops, which is when the viewer shows "SCRATCH". The label itself is
    # checked by hand; here the run must just succeed.
    state, shot = scratch_table()
    fixture = tmp_path / "scratch.json"
    fixture.write_text(state.to_json())
    out = tmp_path / "out.png"
    proc = run_module("physics.demo", str(fixture), "--angle", repr(shot.angle),
                      "--power", repr(shot.power), "--save", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.stat().st_size > 0


def test_demo_with_a_missing_fixture_fails(tmp_path):
    out = tmp_path / "out.png"
    proc = run_module("physics.demo", str(tmp_path / "nope.json"), "--save", str(out))
    assert "No module named" not in proc.stderr, proc.stderr
    assert proc.returncode != 0
    assert not out.exists()


@pytest.fixture
def demo():
    from physics import demo
    return demo


def test_first_contact_point_on_a_straight_hit(simulate, demo):
    state = load("straight_shot.json")
    result = simulate(state, Shot(0.0, 0.5), record_paths=True)
    obj = next(b for b in state.balls if b.id == result.first_contact)
    assert demo.first_contact_point(result, live_starts(state)) == \
        pytest.approx((obj.x - 2 * BALL_R, obj.y), abs=1e-4)


def test_first_contact_point_on_a_kick_shot_is_the_hit_not_the_cushion(simulate, demo):
    # The cue rolls right, bounces off the right cushion, comes back and hits ball 1
    # on its right side. Its path's second point is the bounce, not the hit.
    state = TableState([Ball(0, 2.0, MID_Y), Ball(1, 1.0, MID_Y)])
    result = simulate(state, Shot(0.0, 0.5), record_paths=True)
    assert result.first_contact == 1
    assert result.paths[0][1][0] == pytest.approx(TABLE_W - BALL_R, abs=1e-4)
    assert demo.first_contact_point(result, live_starts(state)) == \
        pytest.approx((1.0 + 2 * BALL_R, MID_Y), abs=1e-4)


def test_first_contact_point_is_none_on_a_miss(simulate, demo):
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 2.0, 0.3)])
    result = simulate(state, Shot(math.pi, 0.05), record_paths=True)
    assert demo.first_contact_point(result, live_starts(state)) is None


def test_demo_module_imports_without_loading_matplotlib():
    # matplotlib is loaded only when drawing, so the helpers stay cheap to test.
    code = "import sys, physics.demo; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=RUN_TIMEOUT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# --- the timing script: python -m physics.bench --------------------------------------

STAT = r"([0-9]+(?:\.[0-9]+)?)"


def bench_line(stdout: str, label: str) -> dict[str, float]:
    """The one output line mentioning `label`, with its mean, p95 and max in ms."""
    lines = [ln for ln in stdout.splitlines() if label in ln.lower()]
    assert len(lines) == 1, f"expected one '{label}' line in:\n{stdout}"
    line = lines[0]
    stats = {}
    for name in ("mean", "p95", "max"):
        m = re.search(name + r"\D*?" + STAT + r"\s*ms", line, re.IGNORECASE)
        assert m, f"no '{name} ... ms' in: {line}"
        stats[name] = float(m.group(1))
    return stats


@pytest.fixture(scope="module")
def bench_output():
    proc = run_module("physics.bench", "--n", "10")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize("label", ["mid", "break"])
def test_bench_prints_mean_p95_and_max_for_each_table(bench_output, label):
    stats = bench_line(bench_output, label)
    assert 0 < stats["mean"] <= stats["max"]
    assert 0 < stats["p95"] <= stats["max"]


def test_bench_rejects_a_non_integer_n():
    proc = run_module("physics.bench", "--n", "abc")
    assert "No module named" not in proc.stderr, proc.stderr
    assert proc.returncode != 0



def test_first_contact_point_when_the_cue_starts_touching_the_ball(simulate, demo):
    state = TableState([Ball(0, 1.0, MID_Y), Ball(1, 1.0 + 2 * BALL_R, MID_Y)])
    result = simulate(state, Shot(0.0, 0.3), record_paths=True)
    assert result.first_contact == 1
    assert demo.first_contact_point(result, live_starts(state)) == \
        pytest.approx((1.0, MID_Y), abs=1e-9)


def test_first_contact_point_ignores_a_bounce_near_one_ball_width_away(demo):
    # A made-up path: the cue bounces at a point 2R + 0.5 mm from ball 1 (not touching),
    # then rolls straight into it. The marker must be on the hit, not the bounce.
    o = (1.0, 0.5)
    bounce = (o[0] + 2 * BALL_R + 0.0005, o[1])
    hit = (o[0], o[1] + 2 * BALL_R)
    path = [(1.5, o[1]), bounce, (1.3, 0.8), (o[0], 0.8), hit]
    result = ShotResult(final=TableState([]), pocketed=[], first_contact=1,
                        scratch=False, paths={0: path})
    point = demo.first_contact_point(result, {1: o})
    assert point == pytest.approx(hit, abs=1e-9)
