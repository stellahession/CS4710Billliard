"""Tunable physics constants, tuned in issue #10 against Dr. Dave Alciatore's published
pool physics (billiards.colostate.edu): technical proof TP B.6, "CB table lengths of
travel for different speeds", and his table of pool equipment properties.

Read these as params.NAME at call time (never `from physics.params import NAME`), so a
test can override one with monkeypatch.setattr(params, "NAME", value).
Table geometry (size, ball radius, pocket positions) lives in core.state.
"""

V_MAX = 8.94       # m/s (20 mph) at power 1.0; TP B.6's "power" shot, about 4.7 table lengths
DECEL = 0.098      # m/s^2 = mu_r * g, typical ball-cloth rolling resistance mu_r = 0.01 (TP B.6)
E_BALL = 0.95      # ball-ball restitution; middle of the measured 0.92-0.98 (Dr. Dave's table)
E_CUSHION = 0.52   # rail COR 0.7 times the 5/7 lost skidding after the rail; fits TP B.6 to 5%
POCKET_R = 0.06    # m; about half a 4.5 in (0.114 m) corner pocket mouth, kept from #6
V_STOP = 1e-6      # m/s, a ball slower than this is at rest
MAX_TIME = 120.0   # s of simulated time; reaching it means something went wrong
