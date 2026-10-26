"""
Parameter schema for the cube-pulling simulator.

- Board: fixed real-world geometry (NOT tunable).
- Params / BOUNDS / CHOICES: the search space. The agent edits VALUES in
  params.yaml and may add new fields here if it adds a new model structure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Fixed geometry (meters, raw tracker frame)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Board:
    x_min: float = -0.5
    x_max: float = -0.07
    y_min: float = -0.71
    y_max: float = -0.12
    thickness: float = 0.02
    cube_half: float = 0.03       # 6 cm cube
    cube_mass: float = 0.062      # kg
    string_height: float = 0.015  # attachment height above the cube's base (lower half of face)


BOARD = Board()


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
_F = (0.02, 3.0)     # per-motor force scale, N at full activation
_S = (0.001, 0.5)    # per-motor kinematic speed, m/s at full activation

BOUNDS = {
    # --- force model: one independent scale per motor/string ---
    "force_scale_m16": _F,   # left
    "force_scale_m17": _F,   # right
    "force_scale_m18": _F,   # top
    "force_scale_m19": _F,   # bottom
    "action_threshold": (-0.5, 0.5),   # executed action must exceed this to count as pulling
    "action_norm": (0.1, 10.0),        # divisor for proportional / saturating modes
    "current_ref": (0.01, 1.0),        # divisor for force_mode=current
    "friction": (0.05, 2.0),           # sliding friction coeff, cube-on-board
    "damping": (0.0, 5.0),             # viscous drag on the cube, N*s/m
    # --- timing ---
    "row_dt": (0.5, 1.2),              # seconds per step; nominal 0.8 s, real value may differ slightly
    "pull_fraction": (0.1, 1.0),       # fraction of the step during which strings pull; the rest coasts
    "motor_lag_rows": (0, 5),          # int, extra delay between action and effect
    "resync_horizon": (0, 200),        # int, 0 = open loop; N = reset to the real start-of-step state every N rows (1 = one-step prediction)
    # --- kinematic model (alternative structure) ---
    "kin_speed_m16": _S,
    "kin_speed_m17": _S,
    "kin_speed_m18": _S,
    "kin_speed_m19": _S,
    "kin_response": (0.05, 1.0),       # velocity smoothing per row (1 = instant)
}

INT_PARAMS = {"motor_lag_rows", "resync_horizon"}

CHOICES = {
    "model_type": ("force", "kinematic"),
    # how executed action (and optionally motor current) maps to activation in [0, 1]
    # All modes give an activation in [0, 1]. Strings only ever PULL; a released string
    # (executed <= threshold) is slack and exerts zero force.
    "force_mode": ("binary", "proportional", "saturating", "current"),
    # to_anchor: string pulls from the face attachment point toward the motor
    # axis:      string pulls purely along +/-x or +/-y
    "direction_mode": ("to_anchor", "axis"),
}


@dataclass
class Params:
    model_type: str = "force"
    force_mode: str = "proportional"
    direction_mode: str = "to_anchor"

    force_scale_m16: float = 0.06
    force_scale_m17: float = 0.06
    force_scale_m18: float = 0.06
    force_scale_m19: float = 0.06
    action_threshold: float = 0.0
    action_norm: float = 1.0
    current_ref: float = 0.2
    friction: float = 0.15
    damping: float = 0.0

    row_dt: float = 0.8
    pull_fraction: float = 1.0
    motor_lag_rows: int = 0
    resync_horizon: int = 0

    kin_speed_m16: float = 0.05
    kin_speed_m17: float = 0.05
    kin_speed_m18: float = 0.05
    kin_speed_m19: float = 0.05
    kin_response: float = 0.5

    def validate(self) -> "Params":
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r} not in {allowed}")
        for name, (lo, hi) in BOUNDS.items():
            val = getattr(self, name)
            if not (lo <= val <= hi):
                raise ValueError(f"{name}={val} outside bounds [{lo}, {hi}]")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(p: Params) -> Params:
    for f in fields(p):
        val = getattr(p, f.name)
        if f.name in INT_PARAMS:
            setattr(p, f.name, int(val))
        elif f.name in BOUNDS:
            setattr(p, f.name, float(val))
    return p


def params_from_dict(d: dict) -> Params:
    known = {f.name for f in fields(Params)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"Unknown parameters: {sorted(unknown)}")
    return _coerce(Params(**d)).validate()


def load_params(path: str | Path = "params.yaml") -> Params:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return params_from_dict(raw)


def save_params(p: Params, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(p.to_dict(), f, sort_keys=False)