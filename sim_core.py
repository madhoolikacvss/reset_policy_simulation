"""
Headless cube-pulling simulator. No rendering, no matplotlib, no cv2.

Row semantics (each CSV row is one complete transition)
-------------------------------------------------------
  obs_cube_{x,y}_norm  = cube position BEFORE the step   (start state)
  executed_m16..m19    = what the motors did during the step
  cube_{x,y}           = cube position AFTER the step    (end state, raw metres)
So row i holds start state, action, and end state of step i. The start state of
row i+1 equals the end state of row i exactly (verified on episode_0002).

Simulator output pred_norm[i] is the predicted END state of step i, directly
comparable to real_norm[i]. Resync controls where the sim starts each step:
  resync_horizon = 0 : start at row 0's start state, then free-run (open loop)
  resync_horizon = H : additionally reset to the real START state of row i
                       whenever i % H == 0   (H = 1 -> pure one-step prediction)

Physical model
--------------
* 6 cm cube on the board (gravity + friction only; strings do not confine it).
  Three slide joints (x, y, z): no roll or yaw, but contact/friction work normally.
* Four strings, one per vertical face, attached to the lower half of the face.
  String k always belongs to motor k. Motors sit at the middle of their board edge.
  Fixed layout:  m16 -> -y edge, m17 -> +y edge, m18 -> +x edge, m19 -> -x edge
  (m16/m17 are the far-apart pair along raw y; m18/m19 the closer pair along raw x).
* HARD PHYSICAL RULE: a string can only PULL the cube toward its motor. Releasing a string
  makes it slack and gives zero force; it can never push. Every per-motor force must have a
  non-negative component toward its motor (enforced by check_physics.py).
* Each step every motor applies its OWN force on the cube, directed from that
  string's attachment point toward the motor. The four forces are never collapsed
  into two axes; the physics engine sums them (opposing pulls are not "resolved").
* Input is the EXECUTED action (post safety filter). Motor current is loaded too
  (obs_motor_*_current_norm at row i+1 reflects the effect of step i).

Frames (no axis swap)
---------------------
norm_x = (raw_x - x_min)/(x_max - x_min) ; norm_y = (raw_y - y_min)/(y_max - y_min)
board  = normalized position centred on the board middle, in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from params import BOARD, Params, load_params

B = BOARD
HALF_X = (B.x_max - B.x_min) / 2.0   # half extent along raw x, 0.215 m
HALF_Y = (B.y_max - B.y_min) / 2.0   # half extent along raw y, 0.295 m
PHYSICS_DT = 0.02
MOTORS = ["m16", "m17", "m18", "m19"]
EXECUTED_COLS = [f"executed_{m}" for m in MOTORS]
ACTION_COLS = [f"action_{m}" for m in MOTORS]
CURRENT_COLS = [f"obs_motor_{m[1:]}_current_norm" for m in MOTORS]
START_COLS = ["obs_cube_x_norm", "obs_cube_y_norm"]

# Rows ordered m16, m17, m18, m19: unit vector from board centre toward each motor.
_UNIT = np.array([[0.0, -1.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
ANCHORS_XY = _UNIT * np.array([HALF_X, HALF_Y])      # motor positions (edge midpoints)
FACE_OFFSETS_XY = B.cube_half * _UNIT                # attachment point relative to cube centre
AXIS_DIRS = _UNIT


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    name: str
    start_norm: np.ndarray   # (N, 2) normalized [x, y] before each step (obs_cube_*_norm)
    real_norm: np.ndarray    # (N, 2) normalized [x, y] after each step (from cube_x, cube_y)
    executed: np.ndarray     # (N, 4) executed actions (m16..m19)
    current: np.ndarray      # (N, 4) motor current (normalized) at the START of each step


@dataclass
class SimResult:
    pred_norm: np.ndarray    # (N, 2) predicted end-of-step state, aligned with real_norm
    forces: np.ndarray       # (N, 4, 2) per-motor force [N] at the start of step i
    off_board_rows: int      # steps where the cube hit the board edge (model-failure signal)


def raw_to_norm(raw_x, raw_y) -> np.ndarray:
    return np.stack([(raw_x - B.x_min) / (B.x_max - B.x_min),
                     (raw_y - B.y_min) / (B.y_max - B.y_min)], axis=-1)


def board_to_norm(xy) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return np.stack([(xy[..., 0] + HALF_X) / (2 * HALF_X),
                     (xy[..., 1] + HALF_Y) / (2 * HALF_Y)], axis=-1)


def norm_to_board(norm) -> np.ndarray:
    norm = np.asarray(norm, dtype=float)
    return np.stack([norm[..., 0] * 2 * HALF_X - HALF_X,
                     norm[..., 1] * 2 * HALF_Y - HALF_Y], axis=-1)


def load_episode(path: str | Path, source: str = "executed") -> Episode:
    """source='executed' (default) or 'action' for the raw policy output."""
    path = Path(path)
    act_cols = EXECUTED_COLS if source == "executed" else ACTION_COLS
    header = pd.read_csv(path, nrows=0).columns
    has_cur = all(c in header for c in CURRENT_COLS)
    cols = ["cube_x", "cube_y", *START_COLS, *act_cols, *(CURRENT_COLS if has_cur else [])]
    df = pd.read_csv(path, usecols=cols).apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid rows in {path}")
    end = raw_to_norm(df["cube_x"].to_numpy(float), df["cube_y"].to_numpy(float))
    cur = df[CURRENT_COLS].to_numpy(float) if has_cur else np.zeros((len(df), 4))
    return Episode(path.stem, df[START_COLS].to_numpy(float), end,
                   df[act_cols].to_numpy(float), cur)


# ---------------------------------------------------------------------------
# Executed action (+ current) -> activation in [0, 1] per motor
# ---------------------------------------------------------------------------
def compute_activation(ep: Episode, p: Params) -> np.ndarray:
    a = np.nan_to_num(ep.executed, nan=0.0)
    on = a > p.action_threshold
    if p.force_mode == "binary":
        act = on.astype(float)
    elif p.force_mode == "proportional":
        act = np.clip(a / p.action_norm, 0.0, 1.0) * on
    elif p.force_mode == "saturating":
        act = np.tanh(np.clip(a, 0.0, None) / p.action_norm) * on
    else:  # "current": start-of-step motor current at row i (known before the step; no future rows)
        act = np.clip(ep.current / p.current_ref, 0.0, 1.0) * on

    lag = p.motor_lag_rows   # extra delay in whole steps (default 0: action acts within its own row)
    if lag > 0:
        act = np.vstack([np.zeros((lag, 4)), act[:-lag]])
    return np.clip(act, 0.0, 1.0)   # strings never push: activation is always in [0, 1]


def _directions(cube_xy: np.ndarray, p: Params) -> np.ndarray:
    """(4, 2) unit pull direction of each string at the current cube position."""
    if p.direction_mode == "axis":
        return AXIS_DIRS
    d = ANCHORS_XY - (cube_xy + FACE_OFFSETS_XY)   # attachment point -> motor
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-8)


def _motor_forces(act_row, cube_xy, scales, p) -> np.ndarray:
    """(4, 2): one force vector per motor. Never collapsed to 2 axes."""
    return (scales * act_row)[:, None] * _directions(cube_xy, p)


def _resync_now(i: int, p: Params) -> bool:
    return i == 0 or (p.resync_horizon > 0 and i % p.resync_horizon == 0)


# ---------------------------------------------------------------------------
# Force model (MuJoCo)
# ---------------------------------------------------------------------------
def _make_xml(p: Params) -> str:
    z0 = B.thickness + B.cube_half
    c = B.cube_half
    return f"""
<mujoco model="cube_pull">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_DT}" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic"/>
  <worldbody>
    <geom name="board" type="box" size="{HALF_X} {HALF_Y} {B.thickness / 2}"
          pos="0 0 {B.thickness / 2}" friction="{p.friction} 0.01 0.001"/>
    <body name="cube" pos="0 0 {z0}">
      <joint name="cx" type="slide" axis="1 0 0"/>
      <joint name="cy" type="slide" axis="0 1 0"/>
      <joint name="cz" type="slide" axis="0 0 1"/>
      <geom name="cube_geom" type="box" size="{c} {c} {c}" mass="{B.cube_mass}"
            friction="{p.friction} 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _simulate_force(ep: Episode, p: Params) -> SimResult:
    model = mujoco.MjModel.from_xml_string(_make_xml(p))
    data = mujoco.MjData(model)
    model.dof_damping[:2] = p.damping      # viscous drag on x, y only
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    z0 = B.thickness + B.cube_half

    def place(xy):
        data.qpos[:3] = [xy[0], xy[1], z0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    start_board = norm_to_board(ep.start_norm)
    act = compute_activation(ep, p)
    scales = np.array([p.force_scale_m16, p.force_scale_m17, p.force_scale_m18, p.force_scale_m19])
    substeps = max(1, int(round(p.row_dt / PHYSICS_DT)))
    pull_steps = max(1, int(round(p.pull_fraction * substeps)))
    n = len(act)
    pred = np.zeros((n, 2))
    forces = np.zeros((n, 4, 2))
    off_board = 0

    for i in range(n):
        if _resync_now(i, p):
            place(start_board[i])

        hit = False
        for s in range(substeps):
            if s < pull_steps:
                f4 = _motor_forces(act[i], data.qpos[:2], scales, p)
            else:
                f4 = np.zeros((4, 2))        # strings released: cube coasts under friction
            if s == 0:
                forces[i] = f4
            data.xfrc_applied[bid, :] = 0.0
            # No rotation DOFs, so a force at the lower-face attachment point is
            # dynamically identical to the same force at the COM.
            data.xfrc_applied[bid, :2] = f4.sum(axis=0)
            mujoco.mj_step(model, data)

            # Safety net only: physically the cube never leaves the board, so this
            # flags a bad parameter set rather than modelling a real constraint.
            for ax, half in ((0, HALF_X), (1, HALF_Y)):
                if abs(data.qpos[ax]) > half:
                    data.qpos[ax] = np.sign(data.qpos[ax]) * half
                    data.qvel[ax] = 0.0
                    hit = True
        pred[i] = data.qpos[:2]              # END state of step i
        off_board += int(hit)
    return SimResult(board_to_norm(pred), forces, off_board)


# ---------------------------------------------------------------------------
# Kinematic model (pure numpy): each string drives the cube at its own speed
# ---------------------------------------------------------------------------
def _simulate_kinematic(ep: Episode, p: Params) -> SimResult:
    act = compute_activation(ep, p)
    speeds = np.array([p.kin_speed_m16, p.kin_speed_m17, p.kin_speed_m18, p.kin_speed_m19])
    start_board = norm_to_board(ep.start_norm)
    pos, vel = start_board[0].copy(), np.zeros(2)
    n = len(act)
    pred = np.zeros((n, 2))
    forces = np.zeros((n, 4, 2))
    off_board = 0

    for i in range(n):
        if _resync_now(i, p):
            pos, vel = start_board[i].copy(), np.zeros(2)
        v4 = _motor_forces(act[i], pos, speeds, p)   # per-motor velocity contributions
        forces[i] = v4
        vel += p.kin_response * (v4.sum(axis=0) - vel)
        pos = pos + vel * p.row_dt
        clipped = np.clip(pos, [-HALF_X, -HALF_Y], [HALF_X, HALF_Y])
        off_board += int(np.any(clipped != pos))
        pos = clipped
        pred[i] = pos                                 # END state of step i
    return SimResult(board_to_norm(pred), forces, off_board)


# ---------------------------------------------------------------------------
def simulate_episode(ep: Episode, p: Params) -> SimResult:
    return _simulate_force(ep, p) if p.model_type == "force" else _simulate_kinematic(ep, p)


def mean_err(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.linalg.norm(pred - real, axis=1).mean())


if __name__ == "__main__":
    import sys
    episode = load_episode(sys.argv[1])
    params = load_params(sys.argv[2] if len(sys.argv) > 2 else "params.yaml")
    open_loop = simulate_episode(episode, Params(**{**params.to_dict(), "resync_horizon": 0}))
    one_step = simulate_episode(episode, Params(**{**params.to_dict(), "resync_horizon": 1}))
    print(f"{episode.name}: rows={len(episode.real_norm)}")
    print(f"  one-step  : sim={mean_err(one_step.pred_norm, episode.real_norm):.4f}  "
          f"stationary(start state)={mean_err(episode.start_norm, episode.real_norm):.4f}")
    stat_open = np.repeat(episode.start_norm[:1], len(episode.real_norm), axis=0)
    print(f"  open-loop : sim={mean_err(open_loop.pred_norm, episode.real_norm):.4f}  "
          f"stationary(row-0 state)={mean_err(stat_open, episode.real_norm):.4f}  "
          f"off_board_rows={open_loop.off_board_rows}")