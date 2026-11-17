import random
from core.state import *


def random_table(n_solids=7, n_stripes=7, seed=None) -> TableState:
    rng = random.Random(seed)
    ids = [0, 8] + list(range(1, 1 + n_solids)) + list(range(9, 9 + n_stripes))
    placed = []
    for i in ids:
        for _ in range(1000):
            x = rng.uniform(BALL_R, TABLE_W - BALL_R)
            y = rng.uniform(BALL_R, TABLE_H - BALL_R)
            if all((x - b.x) ** 2 + (y - b.y) ** 2 > (2.2 * BALL_R) ** 2 for b in placed):
                placed.append(Ball(i, x, y))
                break
    return TableState(placed)
