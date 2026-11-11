"""Plot saved V4 non-elastic command displacements as 2-D vectors.

This script does not run MuJoCo. It reads ``command_tests/results.json`` and
writes four isolated-motor, four pair, and four triple plots under
``command_tests/vector_plots``.

Run from the repository root:

    python -B reset_policy_simulation/v4_noelastic_unchanged/plot_command_vectors.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "command_tests" / "results.json"
OUTPUT_ROOT = HERE / "command_tests" / "vector_plots"
MOTORS = ("m16", "m17", "m18", "m19")


def action_label(row: dict, active_motors: list[str]) -> str:
    """Return a compact action-strength label in the title's motor order."""
    if len(active_motors) == 1:
        return f"a={row[active_motors[0]]:.2f}"
    values = ", ".join(f"{row[m]:.2f}" for m in active_motors)
    return f"({values})"


def category_limit(rows: list[dict]) -> float:
    """Use one symmetric x/y limit for every plot in a category."""
    largest = max(
        max(abs(float(row["dx_mm"])), abs(float(row["dy_mm"])))
        for row in rows
    )
    # Leave room for arrowheads and labels, while retaining a useful minimum.
    return max(2.5, math.ceil(largest * 1.22 * 2.0) / 2.0)


def configure_axes(ax: plt.Axes, limit: float) -> None:
    """Draw signed millimetre axes and a schematic cube at the origin."""
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d7dee8", linewidth=0.7, alpha=0.8)
    ax.axhline(0.0, color="#263238", linewidth=1.0, zorder=1)
    ax.axvline(0.0, color="#263238", linewidth=1.0, zorder=1)
    ax.set_xlabel("x displacement (mm)")
    ax.set_ylabel("y displacement (mm)")

    # Schematic start cube. It is intentionally a marker, not physical scale.
    marker_size = limit * 0.045
    ax.add_patch(
        Rectangle(
            (-marker_size, -marker_size),
            2 * marker_size,
            2 * marker_size,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )
    )
    ax.plot(0.0, 0.0, "o", color="black", markersize=2.5, zorder=7)

    pad = limit * 0.025
    ax.text(limit - pad, pad, "+x", ha="right", va="bottom", fontsize=9)
    ax.text(-limit + pad, pad, "-x", ha="left", va="bottom", fontsize=9)
    ax.text(pad, limit - pad, "+y", ha="left", va="top", fontsize=9)
    ax.text(pad, -limit + pad, "-y", ha="left", va="bottom", fontsize=9)


def plot_group(
    rows: list[dict],
    category: str,
    group_name: str,
    limit: float,
    duration: float,
) -> Path:
    active_motors = group_name.split("+")
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=max(1, len(rows) - 1))

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    configure_axes(ax, limit)

    for index, row in enumerate(rows):
        dx = float(row["dx_mm"])
        dy = float(row["dy_mm"])
        color = cmap(norm(index))
        magnitude = math.hypot(dx, dy)

        if magnitude < 1e-10:
            # Several low isolated actions produce exactly zero movement.
            angle = 2.0 * math.pi * index / max(1, len(rows))
            offset = (13.0 * math.cos(angle), 13.0 * math.sin(angle))
            ax.plot(0.0, 0.0, "o", color=color, markersize=5, zorder=8)
        else:
            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(0.0, 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.7,
                    "mutation_scale": 12,
                    "alpha": 0.9,
                    "shrinkA": 3,
                    "shrinkB": 0,
                },
                zorder=4,
            )
            # Alternate labels across the vector's perpendicular direction.
            side = 1.0 if index % 2 == 0 else -1.0
            offset = (
                side * -dy / magnitude * 8.0,
                side * dx / magnitude * 8.0,
            )

        ax.annotate(
            action_label(row, active_motors),
            xy=(dx, dy),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            zorder=9,
        )

    action_order = ", ".join(active_motors)
    ax.set_title(
        f"{category.capitalize()}: {group_name}\n"
        f"Final displacement after {duration:.2f} s; action labels ordered {action_order}",
        fontsize=12,
    )
    ax.text(
        0.01,
        0.01,
        "Cube marker denotes the common start at (0, 0); marker is not to physical scale.",
        transform=ax.transAxes,
        fontsize=8,
        color="#455a64",
        ha="left",
        va="bottom",
    )

    output_dir = OUTPUT_ROOT / category
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{group_name.replace('+', '_')}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Saved command results not found: {RESULTS_PATH}")

    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    duration = float(payload["duration"])
    all_results = payload["results"]
    expected_counts = {"isolated": 4, "pairs": 4, "triples": 4}
    written: list[Path] = []

    for category in ("isolated", "pairs", "triples"):
        rows = all_results[category]
        groups = list(dict.fromkeys(row["motors"] for row in rows))
        if len(groups) != expected_counts[category]:
            raise ValueError(
                f"Expected {expected_counts[category]} {category} groups, found {len(groups)}: {groups}"
            )
        limit = category_limit(rows)
        for group_name in groups:
            group_rows = [row for row in rows if row["motors"] == group_name]
            written.append(plot_group(group_rows, category, group_name, limit, duration))

    print(f"Wrote {len(written)} plots to {OUTPUT_ROOT}")
    for path in written:
        print(path.relative_to(HERE))


if __name__ == "__main__":
    main()
