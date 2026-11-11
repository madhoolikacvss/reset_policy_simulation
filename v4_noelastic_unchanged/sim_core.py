"""Standalone V4 nonelastic transmission with unchanged V4 parameters.
Initial slack, force-limited take-up and a slipping/stalling spool.
No elastic extension or spring energy; see README.md.
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
    on = a > max(0.0, p.action_threshold)  # a release can never activate a string
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
        delayed = np.zeros_like(act)
        if lag < len(act):
            delayed[lag:] = act[:-lag]
        act = delayed
    return np.clip(act, 0.0, 1.0) * on  # current release overrides delayed pull


def anchors_for(p: Params) -> np.ndarray:
    """Fixed motor locations in centred board metres, ordered m16 through m19."""
    return np.array([[getattr(p, f"anchor_x_{m}"), getattr(p, f"anchor_y_{m}")]
                     for m in MOTORS], dtype=float)


def _directions(cube_xy: np.ndarray, p: Params) -> np.ndarray:
    """(4, 2) unit pull direction of each string at the current cube position."""
    d = anchors_for(p) - (cube_xy + FACE_OFFSETS_XY)   # attachment point -> motor
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-8)


def _motor_forces(act_row, cube_xy, scales, p) -> np.ndarray:
    """(4, 2): one force vector per motor. Never collapsed to 2 axes."""
    return (scales * act_row)[:, None] * _directions(cube_xy, p)


def _resync_now(i: int, p: Params) -> bool:
    return i == 0 or (p.resync_horizon > 0 and i % p.resync_horizon == 0)


def _string_lengths(cube_xy, p=None):
    anchors = ANCHORS_XY if p is None else anchors_for(p)
    return np.linalg.norm(anchors - (cube_xy + FACE_OFFSETS_XY), axis=1)


def _slack_forces(action, activation, cube_xy, lengths, scales, p, pulling=True):
    distance = _string_lengths(cube_xy, p)
    action = np.clip(np.nan_to_num(action), -1., 1.)
    on = (action > max(0., p.action_threshold)) & pulling
    # Reconcile actual payout after motion: any excess distance is spool payout,
    # never elastic cable extension. Shortening geometry leaves real slack.
    lengths[:] = np.maximum(lengths, distance)
    requested = PHYSICS_DT * p.spool_speed * np.where(on, action, 0.)
    gap = np.maximum(lengths - distance, 0.)
    taut = on & (requested >= gap)
    lengths[:] = np.maximum(distance, lengths - requested)
    lengths += PHYSICS_DT * p.spool_speed * p.release_ratio * np.maximum(-action, 0.)
    lengths[~on] = np.maximum(lengths[~on], distance[~on] + 1e-8)
    tension = scales * activation * taut
    return tension[:, None] * _directions(cube_xy, p)


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
        # Slide-joint qpos is relative to the body's XML position, already at z0.
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [xy[0], xy[1], 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    act = compute_activation(ep, p)
    scales = np.array([p.force_scale_m16, p.force_scale_m17, p.force_scale_m18, p.force_scale_m19])
    substeps = max(1, int(round(p.row_dt / PHYSICS_DT)))
    pull_steps = max(1, int(round(p.pull_fraction * substeps)))
    n = len(act)
    pred = np.zeros((n, 2))
    forces = np.zeros((n, 4, 2))
    off_board = 0
    lengths = np.zeros(4)

    for i in range(n):
        if _resync_now(i, p):
            place(norm_to_board(ep.start_norm[i]))
            lengths[:] = _string_lengths(data.qpos[:2], p) + p.slack_initial

        hit = False
        for s in range(substeps):
            if p.string_model == "slack":
                f4 = _slack_forces(ep.executed[i], act[i], data.qpos[:2],
                                   lengths, scales, p, pulling=s < pull_steps)
            elif s < pull_steps:
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
def simulate_episode(ep: Episode, p: Params) -> SimResult:
    return _simulate_force(ep, p)


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
