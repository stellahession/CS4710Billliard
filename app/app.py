"""Run from the repo root:  streamlit run app/app.py"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from core.state import *
from app import stubs
from app.tablegen import random_table

# ---- use the real modules when they exist, else the stubs (adjust import paths to match the repo) ----
try:
    from vision.detect import detect
except Exception:
    detect = stubs.detect
try:
    from physics.simulate import simulate
except Exception:
    simulate = stubs.simulate
try:
    from agent.recommend import recommend
except Exception:
    recommend = stubs.recommend

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
COLORS = {1: "#f5c518", 2: "#1f4fd8", 3: "#d62828", 4: "#6a1b9a", 5: "#f77f00", 6: "#2e7d32", 7: "#7b1e1e"}


def ball_color(i):
    return "white" if i == 0 else "black" if i == 8 else COLORS[i if i < 8 else i - 8]


def draw(state: TableState, rec: Recommendation | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.add_patch(Rectangle((0, 0), TABLE_W, TABLE_H, color="#1b7a43", zorder=0))
    for px, py in POCKETS:
        ax.add_patch(Circle((px, py), 0.06, color="black", zorder=1))
    if rec:
        for alt in rec.alternatives:
            for pts in (alt.preview.paths or {}).values():
                ax.plot(*zip(*pts), color="white", alpha=0.25, lw=1, zorder=2)
        for bid, pts in (rec.preview.paths or {}).items():
            ax.plot(*zip(*pts), color="yellow" if bid == 0 else "cyan", lw=2, ls="--", zorder=3)
        t = next((b for b in state.balls if b.id == rec.target), None)
        if t:
            px, py = POCKETS[rec.pocket]
            n = math.hypot(px - t.x, py - t.y) or 1
            gx, gy = t.x - 2 * BALL_R * (px - t.x) / n, t.y - 2 * BALL_R * (py - t.y) / n
            ax.add_patch(Circle((gx, gy), BALL_R, fill=False, ec="yellow", ls=":", zorder=4))
            ax.plot([t.x, px], [t.y, py], color="cyan", lw=1, alpha=0.6, zorder=3)
    for b in state.balls:
        if b.pocketed:
            continue
        striped = b.id > 8
        ax.add_patch(Circle((b.x, b.y), BALL_R, fc="white" if striped else ball_color(b.id),
                            ec=ball_color(b.id) if striped else "black", lw=3 if striped else 0.8, zorder=5))
        if b.id:
            ax.text(b.x, b.y, str(b.id), ha="center", va="center", fontsize=7,
                    color="black" if striped or b.id == 1 else "white", zorder=6)
    ax.set_xlim(-0.1, TABLE_W + 0.1); ax.set_ylim(-0.1, TABLE_H + 0.1)
    ax.set_aspect("equal"); ax.axis("off")
    return fig


def state_to_df(s):
    return pd.DataFrame([{"id": b.id, "x": b.x, "y": b.y, "pocketed": b.pocketed} for b in s.balls],
                        columns=["id", "x", "y", "pocketed"])


def df_to_state(df, group):
    balls = [Ball(int(r.id), float(r.x), float(r.y), bool(r.pocketed))
             for r in df.dropna(subset=["id", "x", "y"]).itertuples()]
    return TableState(balls, group)


st.set_page_config(page_title="Pool Shot Recommender", layout="wide")
st.title("Pool Shot Recommender")
if "state" not in st.session_state:
    st.session_state.state = TableState([Ball(0, 0.6, 0.63)])
    st.session_state.rec = None

with st.sidebar:
    st.header("Table")
    up = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])
    if up and st.button("Detect balls"):
        tmp = Path("/tmp") / up.name
        tmp.write_bytes(up.getvalue())
        st.session_state.state, st.session_state.rec = detect(str(tmp)), None
    fx = sorted(FIXTURES.glob("*.json"))
    if fx:
        pick = st.selectbox("Example table", fx, format_func=lambda p: p.stem)
        if st.button("Load example"):
            st.session_state.state, st.session_state.rec = TableState.from_json(pick.read_text()), None
    c1, c2 = st.columns(2)
    if c1.button("Random"):
        st.session_state.state, st.session_state.rec = random_table(), None
    if c2.button("Empty"):
        st.session_state.state, st.session_state.rec = TableState([Ball(0, 0.6, 0.63)]), None
    st.header("Player")
    group = st.radio("Your group", ["open", "solids", "stripes"])
    skill = st.select_slider("Skill level", ["Beginner", "Intermediate", "Advanced"], value="Intermediate")

left, right = st.columns([1, 2])
with left:
    st.subheader("Balls")
    st.caption("0=cue, 1-7 solids, 8=eight, 9-15 stripes. x: 0–2.54, y: 0–1.27 (metres). Add rows at the bottom.")
    df = st.data_editor(state_to_df(st.session_state.state), num_rows="dynamic", width="stretch", key="editor")
    state = df_to_state(df, group)
    go = st.button("Recommend", type="primary")

if go:
    try:
        st.session_state.rec = recommend(state, skill)
    except Exception as e:
        st.session_state.rec = None
        st.error(f"Recommend failed: {e}")
st.session_state.state = state

with right:
    rec = st.session_state.rec
    st.pyplot(draw(state, rec))
    if rec:
        m1, m2, m3 = st.columns(3)
        m1.metric("Target", f"Ball {rec.target}")
        m2.metric("Success chance", f"{rec.success_prob:.0%}")
        m3.metric("Expected value", f"{rec.expected_value:.2f}")
        for i, a in enumerate(rec.alternatives, 1):
            st.write(f"Alternative {i}: ball {a.target} → pocket {a.pocket} ({a.success_prob:.0%})")
