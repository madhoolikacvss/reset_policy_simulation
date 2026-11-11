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

BOUNDS = {
    # Board-centred metres. Envelope contains nominal edge midpoints and
    # corrected near-motor measurements +/- 5 cm; not a board-size change.
    "anchor_x_m16": (0.0, 0.127),
    "anchor_y_m16": (-0.295, -0.114),
    "anchor_x_m17": (0.0, 0.132),
    "anchor_y_m17": (0.228, 0.328),
    "anchor_x_m18": (0.215, 0.338),
    "anchor_y_m18": (-0.068, 0.032),
    "anchor_x_m19": (-0.336, -0.215),
    "anchor_y_m19": (-0.026, 0.074),
    "slack_initial": (0.0, 0.05),     # m: unknown free-length excess at each resync
    "spool_speed": (0.001, 0.1),      # m/s at unit executed action
    "release_ratio": (0.1, 3.0),      # payout speed / winding speed
    "string_stiffness": (1.0, 200.0), # N/m: elastic string tension, torque limited
    # --- force model: one independent scale per motor/string ---
    "force_scale_m16": _F,   # -y
    "force_scale_m17": _F,   # +y
    "force_scale_m18": _F,   # +x
    "force_scale_m19": _F,   # -x
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
}

INT_PARAMS = {"motor_lag_rows", "resync_horizon"}

CHOICES = {
    "string_model": ("direct", "slack"),
    # how executed action (and optionally motor current) maps to activation in [0, 1]
    # All modes give an activation in [0, 1]. Strings only ever PULL; a released string
    # (executed <= threshold) is slack and exerts zero force.
    "force_mode": ("binary", "proportional", "saturating", "current"),
}


@dataclass
class Params:
    # Fixed throughout a simulation; defaults reproduce nominal edge midpoints.
    anchor_x_m16: float = 0.0
    anchor_y_m16: float = -0.295
    anchor_x_m17: float = 0.0
    anchor_y_m17: float = 0.295
    anchor_x_m18: float = 0.215
    anchor_y_m18: float = 0.0
    anchor_x_m19: float = -0.215
    anchor_y_m19: float = 0.0
    string_model: str = "direct"
    slack_initial: float = 0.005
    spool_speed: float = 0.02
    release_ratio: float = 1.0
    string_stiffness: float = 40.0
    force_mode: str = "proportional"

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
