"""Tunable physics constants. Values are placeholders until they are tuned in issue #10.

Read these as params.NAME at call time (never `from physics.params import NAME`), so a
test can override one with monkeypatch.setattr(params, "NAME", value).
Table geometry (size, ball radius, pocket positions) lives in core.state.
"""

V_MAX = 6.0        # m/s, cue-ball speed at power 1.0
DECEL = 0.15       # m/s^2, constant rolling deceleration
E_BALL = 0.95      # ball-ball restitution (1.0 = perfectly elastic)
E_CUSHION = 0.8    # fraction of normal speed kept after a cushion bounce
POCKET_R = 0.06    # m, a ball centre this close to a pocket point drops (used from issue #6)
V_STOP = 1e-6      # m/s, a ball slower than this is at rest
MAX_TIME = 120.0   # s of simulated time; reaching it means something went wrong
