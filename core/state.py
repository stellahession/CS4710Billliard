# core/state.py  -- shared interface. If the repo already has this, USE THE REPO'S VERSION
# and just make sure it has to_json / from_json.
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

TABLE_W, TABLE_H = 2.54, 1.27      # playing surface, metres
BALL_R = 0.028575                  # metres
POCKETS = [(0, 0), (TABLE_W / 2, 0), (TABLE_W, 0),
           (0, TABLE_H), (TABLE_W / 2, TABLE_H), (TABLE_W, TABLE_H)]


@dataclass
class Ball:
    id: int            # 0 = cue, 1-7 solids, 8 = eight ball, 9-15 stripes
    x: float
    y: float
    pocketed: bool = False


@dataclass
class TableState:
    balls: list[Ball]
    group: str = "open"            # "solids" | "stripes" | "open"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "TableState":
        d = json.loads(text)
        return TableState([Ball(**b) for b in d["balls"]], d.get("group", "open"))


@dataclass
class Shot:
    angle: float       # radians
    power: float       # 0..1


@dataclass
class ShotResult:
    final: TableState
    pocketed: list[int]
    first_contact: int | None
    scratch: bool
    paths: dict[int, list[tuple[float, float]]] | None = None


@dataclass
class Recommendation:
    shot: Shot
    target: int
    pocket: int                    # index into POCKETS
    expected_value: float
    success_prob: float
    preview: ShotResult
    alternatives: list["Recommendation"] = field(default_factory=list)
