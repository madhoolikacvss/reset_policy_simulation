"""Interactively control the unchanged V4 non-elastic MuJoCo model.

The four GUI inputs are executed actions for m16, m17, m18, and m19.  An
action is sampled at the start of each command interval and held for the
model's fitted ``row_dt``.  Positive values pull, zero is neutral, and
negative values pay out (release) string.  String slack persists between
commands until Reset is pressed.

Run from the repository root in the reset conda environment:

    conda activate reset
    python -B reset_policy_simulation/v4_noelastic_unchanged/interactive_control.py

Use ``--check`` to validate imports/model construction without opening GUI
windows.
"""

from __future__ import annotations

import argparse
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

import sim_core as sc
from params import load_params


HERE = Path(__file__).resolve().parent
MOTOR_NAMES = ("m16", "m17", "m18", "m19")
EXPECTED_DIRECTIONS = ("-y", "+y", "+x", "-x")
MOTOR_COLORS = (
    (0.90, 0.30, 0.20, 1.0),
    (0.20, 0.65, 0.95, 1.0),
    (0.25, 0.80, 0.35, 1.0),
    (0.80, 0.35, 0.90, 1.0),
)


def build_visual_model(params) -> mujoco.MjModel:
    """Build the unchanged physics model plus lights, camera, and motor markers."""
    root = ET.fromstring(sc._make_xml(params))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1100", offheight="800")

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("MuJoCo XML has no worldbody")

    ET.SubElement(world, "light", pos="0 -0.4 1.4", dir="0 0.3 -1")
    ET.SubElement(
        world,
        "camera",
        name="interactive",
        pos="0 -0.90 0.90",
        xyaxes="1 0 0 0 0.72 0.69",
    )

    board_geom = world.find("geom[@name='board']")
    cube_geom = world.find("body/geom[@name='cube_geom']")
    if board_geom is not None:
        board_geom.set("rgba", "0.75 0.77 0.80 1")
    if cube_geom is not None:
        cube_geom.set("rgba", "0.95 0.35 0.10 1")

    for index, (anchor, color) in enumerate(
        zip(sc.anchors_for(params), MOTOR_COLORS, strict=True)
    ):
        ET.SubElement(
            world,
            "geom",
            name=MOTOR_NAMES[index],
            type="cylinder",
            pos=f"{anchor[0]} {anchor[1]} 0.035",
            size="0.015 0.035",
            rgba=" ".join(map(str, color)),
            contype="0",
            conaffinity="0",
        )

    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode")
    )
    model.dof_damping[:2] = params.damping
    return model


def activation_for(actions: np.ndarray, params) -> np.ndarray:
    """Use the saved action-to-activation law for one interactive interval."""
    zeros2 = np.zeros((1, 2), dtype=float)
    episode = sc.Episode(
        "interactive",
        zeros2,
        zeros2,
        np.asarray(actions, dtype=float).reshape(1, 4),
        np.zeros((1, 4), dtype=float),
    )
    return sc.compute_activation(episode, params)[0]


class LiveSimulation:
    """Persistent MuJoCo and non-elastic string state for live commands."""

    def __init__(self, model: mujoco.MjModel, params) -> None:
        if params.resync_horizon != 0:
            raise ValueError("Interactive mode requires resync_horizon: 0")
        if params.motor_lag_rows != 0:
            raise ValueError("This live controller expects motor_lag_rows: 0")
        if params.string_model != "slack":
            raise ValueError("This controller is for the V4 slack-string model")

        self.model = model
        self.params = params
        self.data = mujoco.MjData(model)
        self.cube_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self.scales = np.array(
            [getattr(params, f"force_scale_{name}") for name in MOTOR_NAMES],
            dtype=float,
        )
        self.substeps_per_interval = max(
            1, int(round(params.row_dt / sc.PHYSICS_DT))
        )
        self.pull_substeps = max(
            1,
            int(round(params.pull_fraction * self.substeps_per_interval)),
        )

        self.string_lengths = np.zeros(4, dtype=float)
        self.command: np.ndarray | None = None
        self.activation = np.zeros(4, dtype=float)
        self.command_substep = 0
        self.interval_count = 0
        self.clamp_events = 0
        self.last_forces = np.zeros((4, 2), dtype=float)
        self.reset()

    def reset(self) -> None:
        """Return the cube to center/rest and restore initial fitted slack."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.string_lengths[:] = (
            sc._string_lengths(self.data.qpos[:2], self.params)
            + self.params.slack_initial
        )
        self.command = None
        self.activation[:] = 0.0
        self.command_substep = 0
        self.interval_count = 0
        self.clamp_events = 0
        self.last_forces[:] = 0.0

    def start_command(self, actions: np.ndarray) -> None:
        """Start or replace an interval using a frozen action snapshot."""
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (4,) or not np.all(np.isfinite(actions)):
            raise ValueError("Actions must contain four finite values")
        self.command = np.clip(actions, -1.0, 1.0)
        self.activation = activation_for(self.command, self.params)
        self.command_substep = 0

    def stop_now(self) -> None:
        """Immediately remove all active pull without changing slack state."""
        self.command = None
        self.activation[:] = 0.0
        self.command_substep = 0
        self.data.xfrc_applied[self.cube_body_id, :] = 0.0

    @property
    def command_progress(self) -> float:
        if self.command is None:
            return 0.0
        return self.command_substep / self.substeps_per_interval

    def step(self) -> bool:
        """Advance one 20 ms physics step; return True when an interval ends."""
        if self.command is None:
            actions = np.zeros(4, dtype=float)
            activation = np.zeros(4, dtype=float)
            pulling = False
        else:
            actions = self.command
            activation = self.activation
            pulling = self.command_substep < self.pull_substeps

        forces = sc._slack_forces(
            actions,
            activation,
            self.data.qpos[:2],
            self.string_lengths,
            self.scales,
            self.params,
            pulling=pulling,
        )
        directions = sc._directions(self.data.qpos[:2], self.params)
        if np.any(np.sum(forces * directions, axis=1) < -1e-12):
            raise RuntimeError("A string force pointed away from its motor")
        if np.any(np.linalg.norm(forces[actions <= 0.0], axis=1) > 1e-12):
            raise RuntimeError("A released string produced force")

        self.last_forces[:] = forces
        self.data.xfrc_applied[self.cube_body_id, :] = 0.0
        self.data.xfrc_applied[self.cube_body_id, :2] = forces.sum(axis=0)
        mujoco.mj_step(self.model, self.data)

        clamped = False
        for axis, half_extent in ((0, sc.HALF_X), (1, sc.HALF_Y)):
            if abs(self.data.qpos[axis]) > half_extent:
                self.data.qpos[axis] = (
                    math.copysign(half_extent, self.data.qpos[axis])
                )
                self.data.qvel[axis] = 0.0
                self.clamp_events += 1
                clamped = True
        if clamped:
            mujoco.mj_forward(self.model, self.data)

        ended = False
        if self.command is not None:
            self.command_substep += 1
            if self.command_substep >= self.substeps_per_interval:
                self.command = None
                self.activation[:] = 0.0
                self.command_substep = 0
                self.interval_count += 1
                ended = True
        return ended


class ControlPanel:
    """Tk controls and live readouts for the persistent simulation."""

    def __init__(self, root: tk.Tk, simulation: LiveSimulation) -> None:
        self.root = root
        self.simulation = simulation
        self.closing = False
        self.continuous_var = tk.BooleanVar(value=False)
        self.action_vars = [tk.DoubleVar(value=0.0) for _ in MOTOR_NAMES]
        self.status_var = tk.StringVar(value="Idle — set actions, then apply")
        self.position_var = tk.StringVar(value="x = +0.00 mm    y = +0.00 mm")
        self.motion_var = tk.StringVar(value="speed = 0.00 mm/s")
        self.progress_var = tk.StringVar(value="interval progress: idle")
        self.tension_vars = [tk.StringVar(value="0.0000 N") for _ in MOTOR_NAMES]

        root.title("V4 non-elastic cube — interactive executed actions")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.close)

        outer = ttk.Frame(root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            outer,
            text="Executed actions (−1 release, 0 neutral, +1 strongest pull)",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(outer, text="Motor").grid(row=1, column=0, sticky="w")
        ttk.Label(outer, text="Expected").grid(row=1, column=1, sticky="w")
        ttk.Label(outer, text="Action").grid(row=1, column=2, sticky="w")
        ttk.Label(outer, text="Tension").grid(row=1, column=3, sticky="e")

        for index, (name, direction) in enumerate(
            zip(MOTOR_NAMES, EXPECTED_DIRECTIONS, strict=True), start=2
        ):
            ttk.Label(outer, text=name).grid(row=index, column=0, sticky="w")
            ttk.Label(outer, text=direction).grid(row=index, column=1, sticky="w")
            controls = ttk.Frame(outer)
            controls.grid(row=index, column=2, sticky="ew", padx=(8, 8), pady=3)
            slider = tk.Scale(
                controls,
                variable=self.action_vars[index - 2],
                from_=-1.0,
                to=1.0,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                length=300,
                showvalue=False,
                highlightthickness=0,
            )
            slider.grid(row=0, column=0)
            spinbox = ttk.Spinbox(
                controls,
                textvariable=self.action_vars[index - 2],
                from_=-1.0,
                to=1.0,
                increment=0.01,
                width=7,
                format="%.2f",
            )
            spinbox.grid(row=0, column=1, padx=(6, 0))
            ttk.Label(outer, textvariable=self.tension_vars[index - 2]).grid(
                row=index, column=3, sticky="e"
            )

        button_row = ttk.Frame(outer)
        button_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(10, 4))
        ttk.Button(
            button_row,
            text="Apply one interval",
            command=self.apply_once,
        ).grid(row=0, column=0, padx=(0, 5))
        ttk.Checkbutton(
            button_row,
            text="Continuous repeat",
            variable=self.continuous_var,
            command=self.toggle_continuous,
        ).grid(row=0, column=1, padx=5)
        ttk.Button(button_row, text="Stop now", command=self.stop_now).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(button_row, text="Release all", command=self.release_all).grid(
            row=0, column=3, padx=5
        )
        ttk.Button(button_row, text="Reset cube", command=self.reset).grid(
            row=0, column=4, padx=5
        )
        ttk.Button(button_row, text="Quit", command=self.close).grid(
            row=0, column=5, padx=(5, 0)
        )

        ttk.Separator(outer).grid(
            row=7, column=0, columnspan=4, sticky="ew", pady=8
        )
        ttk.Label(outer, textvariable=self.status_var).grid(
            row=8, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(outer, textvariable=self.position_var).grid(
            row=9, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(outer, textvariable=self.motion_var).grid(
            row=10, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(outer, textvariable=self.progress_var).grid(
            row=11, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(
            outer,
            text=(
                "Sliders are sampled at interval boundaries. Reset restores the "
                "cube, velocity, and initial 4.87 mm slack."
            ),
            wraplength=620,
        ).grid(row=12, column=0, columnspan=4, sticky="w", pady=(8, 0))

        root.bind("<Return>", lambda _event: self.apply_once())
        root.bind("<Escape>", lambda _event: self.stop_now())

    def read_actions(self) -> np.ndarray:
        try:
            values = np.array([var.get() for var in self.action_vars], dtype=float)
        except (tk.TclError, ValueError) as exc:
            raise ValueError("Every motor action must be a number") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("Every motor action must be finite")
        values = np.clip(values, -1.0, 1.0)
        for variable, value in zip(self.action_vars, values, strict=True):
            variable.set(round(float(value), 2))
        return values

    def _start_from_controls(self) -> bool:
        try:
            actions = self.read_actions()
            self.simulation.start_command(actions)
        except ValueError as exc:
            messagebox.showerror("Invalid executed action", str(exc), parent=self.root)
            return False
        formatted = ", ".join(f"{name}={value:+.2f}" for name, value in zip(MOTOR_NAMES, actions, strict=True))
        self.status_var.set(f"Executing: {formatted}")
        return True

    def apply_once(self) -> None:
        self.continuous_var.set(False)
        self._start_from_controls()

    def toggle_continuous(self) -> None:
        if self.continuous_var.get():
            if not self._start_from_controls():
                self.continuous_var.set(False)
        else:
            self.status_var.set("Continuous repeat disabled; current interval will finish")

    def stop_now(self) -> None:
        self.continuous_var.set(False)
        self.simulation.stop_now()
        self.status_var.set("Stopped immediately; slack state preserved")

    def release_all(self) -> None:
        self.continuous_var.set(False)
        for variable in self.action_vars:
            variable.set(-1.0)
        self.simulation.start_command(np.full(4, -1.0))
        self.status_var.set("Releasing all four strings for one interval")

    def reset(self) -> None:
        self.continuous_var.set(False)
        for variable in self.action_vars:
            variable.set(0.0)
        self.simulation.reset()
        self.status_var.set("Reset to board center with initial fitted slack")
        self.update_readouts()

    def on_interval_ended(self) -> None:
        if self.continuous_var.get():
            self._start_from_controls()
        else:
            self.status_var.set(
                f"Idle — completed {self.simulation.interval_count} interval(s)"
            )

    def update_readouts(self) -> None:
        x_mm, y_mm = self.simulation.data.qpos[:2] * 1000.0
        vx_mm, vy_mm = self.simulation.data.qvel[:2] * 1000.0
        speed = math.hypot(vx_mm, vy_mm)
        tensions = np.linalg.norm(self.simulation.last_forces, axis=1)
        self.position_var.set(f"x = {x_mm:+.2f} mm    y = {y_mm:+.2f} mm")
        self.motion_var.set(
            f"speed = {speed:.2f} mm/s    board clamps = {self.simulation.clamp_events}"
        )
        if self.simulation.command is None:
            self.progress_var.set("interval progress: idle")
        else:
            self.progress_var.set(
                f"interval progress: {100.0 * self.simulation.command_progress:5.1f}%"
            )
        for variable, tension in zip(self.tension_vars, tensions, strict=True):
            variable.set(f"{tension:.4f} N")

    def close(self) -> None:
        self.closing = True


def configure_viewer(viewer, model: mujoco.MjModel) -> None:
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "interactive"
    )
    if camera_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id


def run_gui(params_path: Path) -> None:
    params = load_params(params_path)
    model = build_visual_model(params)
    simulation = LiveSimulation(model, params)

    root = tk.Tk()
    panel = ControlPanel(root, simulation)
    root.update_idletasks()
    root.update()

    viewer = mujoco.viewer.launch_passive(model, simulation.data)
    configure_viewer(viewer, model)
    viewer.sync()

    next_step = time.perf_counter()
    readout_counter = 0
    try:
        while viewer.is_running() and not panel.closing:
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                panel.closing = True
                break

            now = time.perf_counter()
            if now < next_step:
                time.sleep(min(next_step - now, 0.005))
                continue

            ended = simulation.step()
            if ended:
                panel.on_interval_ended()
            viewer.sync()

            readout_counter += 1
            if readout_counter >= 3:
                panel.update_readouts()
                readout_counter = 0

            next_step += sc.PHYSICS_DT
            if next_step < now - 0.1:
                next_step = now + sc.PHYSICS_DT
    finally:
        viewer.close()
        try:
            root.destroy()
        except tk.TclError:
            pass


def run_check(params_path: Path) -> None:
    """Headless construction and force-direction smoke test."""
    params = load_params(params_path)
    model = build_visual_model(params)
    simulation = LiveSimulation(model, params)
    simulation.start_command(np.array([1.0, -1.0, -1.0, -1.0]))
    for _ in range(simulation.pull_substeps):
        simulation.step()
    directions = sc._directions(simulation.data.qpos[:2], params)
    directional_components = np.sum(simulation.last_forces * directions, axis=1)
    if np.any(directional_components < -1e-12):
        raise AssertionError("Force-direction check failed")
    if np.any(np.linalg.norm(simulation.last_forces[1:], axis=1) > 1e-12):
        raise AssertionError("Released motors generated force")
    print("interactive_control.py check passed")
    print(f"MuJoCo timestep: {sc.PHYSICS_DT:.3f} s")
    print(
        f"Action interval: {params.row_dt:.6f} s "
        f"({simulation.substeps_per_interval} substeps; "
        f"{simulation.pull_substeps} pulling)"
    )
    print("GUI actions: m16, m17, m18, m19 in [-1, +1]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        default=HERE / "params.yaml",
        help="parameter YAML (default: unchanged V4 non-elastic params.yaml)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="headless import/model/physics smoke test; do not open GUI windows",
    )
    args = parser.parse_args()
    params_path = args.params.resolve()
    if not params_path.exists():
        parser.error(f"parameter file not found: {params_path}")
    if args.check:
        run_check(params_path)
    else:
        run_gui(params_path)


if __name__ == "__main__":
    main()
