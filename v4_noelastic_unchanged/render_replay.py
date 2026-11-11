#!/usr/bin/env python3
"""Render motor-log CSV episodes as 3-D MuJoCo MP4 videos.

The simulated row-end positions come from sim_core.simulate_episode, so this uses
the same fitted parameters, executed-action inputs, slack model, and MuJoCo
physics as the benchmark.  Positions are linearly interpolated between logged
row boundaries only to make the video smooth; the interpolation does not affect
the simulated trajectory.

Examples
--------
python render_mujoco_replay.py data/val/episode_0006_motors.csv
python render_mujoco_replay.py data/val/episode_0006_motors.csv --smooth-real 5
python render_mujoco_replay.py data/val --out videos --fps 30
python render_mujoco_replay.py episode.csv --resync 20 --no-real
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import mujoco
import numpy as np

import sim_core as sc
from params import BOARD, load_params


MOTOR_COLORS = (
    (0.90, 0.30, 0.20, 1.0),
    (0.20, 0.65, 0.95, 1.0),
    (0.25, 0.80, 0.35, 1.0),
    (0.80, 0.35, 0.90, 1.0),
)


def _rgba_text(rgba: tuple[float, ...]) -> str:
    return " ".join(str(x) for x in rgba)


def make_visual_xml(params) -> str:
    """Make a display-only scene; trajectory generation remains in sim_core."""
    anchors = sc.anchors_for(params)
    z_cube = BOARD.thickness + BOARD.cube_half
    anchor_geoms = []
    for i, ((x, y), color) in enumerate(zip(anchors, MOTOR_COLORS), start=16):
        anchor_geoms.append(
            f'<geom name="m{i}" type="cylinder" pos="{x} {y} 0.035" '
            f'size="0.020 0.035" rgba="{_rgba_text(color)}"/>'
        )
    return f"""
<mujoco model="cube_replay_visualization">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.75 0.75 0.75"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.86 0.86 0.86"
             rgb2="0.72 0.72 0.72" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="5 7" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="0 -0.4 1.4" dir="0 0.3 -1"/>
    <geom name="board" type="box" size="{sc.HALF_X} {sc.HALF_Y} {BOARD.thickness/2}"
          pos="0 0 {BOARD.thickness/2}" material="floor"/>
    {''.join(anchor_geoms)}
    <body name="sim_cube" pos="0 0 {z_cube}">
      <joint name="sim_x" type="slide" axis="1 0 0"/>
      <joint name="sim_y" type="slide" axis="0 1 0"/>
      <geom type="box" size="{BOARD.cube_half} {BOARD.cube_half} {BOARD.cube_half}"
            rgba="0.90 0.16 0.10 1"/>
    </body>
    <body name="real_cube" mocap="true" pos="0 0 {z_cube}">
      <geom type="box" size="{BOARD.cube_half*1.04} {BOARD.cube_half*1.04} {BOARD.cube_half*1.04}"
            contype="0" conaffinity="0" rgba="0.05 0.35 1.0 0.30"/>
    </body>
    <camera name="comparison" pos="0 -0.90 0.90" xyaxes="1 0 0 0 0.72 0.69"/>
  </worldbody>
</mujoco>
"""


def csv_inputs(value: str) -> list[Path]:
    path = Path(value)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if files:
            return files
        raise SystemExit(f"No CSV files found in {path}")
    raise SystemExit(f"Input does not exist: {path}")


def _put_label(frame: np.ndarray, text: str, xy: tuple[int, int], color) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def smooth_path(path: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding, matching run_sim.py."""
    if window <= 1:
        return path.copy()
    pad = np.pad(path, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.column_stack([
        np.convolve(pad[:, axis], kernel, mode="valid") for axis in (0, 1)
    ])


def render_episode(csv_path: Path, output: Path, params, fps: int, width: int,
                   height: int, show_real: bool, smooth_real: int) -> None:
    episode = sc.load_episode(csv_path)
    result = sc.simulate_episode(episode, params)
    simulated = sc.norm_to_board(result.pred_norm)
    displayed_real_norm = smooth_path(episode.real_norm, smooth_real)
    real = sc.norm_to_board(displayed_real_norm)
    initial = sc.norm_to_board(episode.start_norm[0])

    model = mujoco.MjModel.from_xml_string(make_visual_xml(params))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames_per_row = max(1, int(round(params.row_dt * fps)))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Could not open MP4 writer for {output}")

    previous_sim = initial
    previous_real = initial
    try:
        for row, (sim_end, real_end) in enumerate(zip(simulated, real)):
            for subframe in range(frames_per_row):
                alpha = (subframe + 1) / frames_per_row
                sim_xy = previous_sim + alpha * (sim_end - previous_sim)
                real_xy = previous_real + alpha * (real_end - previous_real)
                data.qpos[:2] = sim_xy
                data.qvel[:] = 0
                data.mocap_pos[0] = [real_xy[0], real_xy[1], BOARD.thickness + BOARD.cube_half]
                if not show_real:
                    data.mocap_pos[0, 2] = -1.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="comparison")
                frame = renderer.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                pulling = [f"m{16+k}" for k, value in enumerate(episode.executed[row])
                           if value > params.action_threshold]
                _put_label(frame, f"{csv_path.stem}  row {row+1}/{len(simulated)}", (24, 32), (255, 255, 255))
                real_label = "   blue: real tracked"
                if smooth_real > 1:
                    real_label += f" ({smooth_real}-step average)"
                _put_label(frame, "red: V4 nonelastic" + (real_label if show_real else ""),
                           (24, 60), (245, 245, 245))
                _put_label(frame, "pulling: " + (", ".join(pulling) if pulling else "none"),
                           (24, 88), (80, 210, 255))
                writer.write(frame)
            previous_sim = sim_end
            previous_real = real_end
    finally:
        writer.release()
        renderer.close()
    raw_error = np.linalg.norm(result.pred_norm - episode.real_norm, axis=1).mean()
    display_error = np.linalg.norm(result.pred_norm - displayed_real_norm, axis=1).mean()
    smoothing_note = (f", error vs {smooth_real}-step averaged real path {display_error:.4f}"
                      if smooth_real > 1 else "")
    print(f"wrote {output} ({len(simulated)} rows, raw mean normalized error {raw_error:.4f}{smoothing_note})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="motor-log CSV file or directory of CSV files")
    parser.add_argument("--out", default="mujoco_videos", help="output directory (default: mujoco_videos)")
    parser.add_argument("--params", default="params.yaml", help="fitted parameter YAML")
    parser.add_argument("--resync", type=int, default=0,
                        help="0=open-loop replay; N=reset from the real position every N rows")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-real", action="store_true", help="hide the translucent real-position cube")
    parser.add_argument("--smooth-real", type=int, default=0, metavar="K",
                        help="DISPLAY ONLY: move the real cube along a K-step centered moving average (e.g. 5)")
    args = parser.parse_args()
    if (args.fps <= 0 or args.width <= 0 or args.height <= 0 or
            args.resync < 0 or args.smooth_real < 0):
        parser.error("fps, width, and height must be positive; resync and smooth-real must be nonnegative")

    params = replace(load_params(args.params), resync_horizon=args.resync)
    inputs = csv_inputs(args.input)
    out_dir = Path(args.out)
    for csv_path in inputs:
        render_episode(csv_path, out_dir / f"{csv_path.stem}_mujoco.mp4", params,
                       args.fps, args.width, args.height, not args.no_real, args.smooth_real)


if __name__ == "__main__":
    main()
