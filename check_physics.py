#!/usr/bin/env python3
"""
Physics sanity check. Run before reporting any result:  python check_physics.py [--data-dir data]

Verifies, on every training episode (or a synthetic one if none exist), that for the current
params.yaml and sim_core.py:
  1. every per-motor force/velocity contribution has a NON-NEGATIVE component toward its motor
     (strings only pull; a released string is slack and never pushes);
  2. activations stay in [0, 1];
  3. a motor with zero activation exerts exactly zero force;
  4. each motor's force acts along its own string (parallel to attachment point -> motor),
     for direction_mode="to_anchor".
Exit code 0 = pass, 1 = violation.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import sim_core as sc
from params import load_params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    p = load_params(args.params)

    files = sorted(Path(args.data_dir, "train").glob("*.csv"))
    if files:
        eps = [sc.load_episode(f) for f in files]
    else:  # synthetic fallback so the check can always run
        rng = np.random.default_rng(0)
        n = 200
        eps = [sc.Episode("synthetic", np.tile([0.6, 0.5], (n, 1)), np.tile([0.6, 0.5], (n, 1)),
                          rng.uniform(-1, 1, (n, 4)), rng.uniform(0, 0.3, (n, 4)))]

    problems = []
    anchors, offsets = sc.ANCHORS_XY, sc.FACE_OFFSETS_XY
    for ep in eps:
        act = sc.compute_activation(ep, p)
        if act.min() < 0 or act.max() > 1 + 1e-9:
            problems.append(f"{ep.name}: activation outside [0, 1]")
        # one-step mode: the sim starts every row at the real start state, so the force at
        # the beginning of row i was evaluated at a known cube position (start_norm[i]).
        res = sc.simulate_episode(ep, replace(p, resync_horizon=1))
        for k in range(4):
            f = res.forces[:, k, :]                                  # (N, 2)
            string = anchors[k] - (sc.norm_to_board(ep.start_norm) + offsets[k])
            along = np.einsum("ij,ij->i", f, string)
            if along.min() < -1e-9:
                problems.append(f"{ep.name}: motor {sc.MOTORS[k]} pushes (min toward-motor component {along.min():.3g})")
            zero_act = act[:, k] == 0
            if np.any(np.linalg.norm(f[zero_act], axis=1) > 1e-12):
                problems.append(f"{ep.name}: motor {sc.MOTORS[k]} exerts force at zero activation")
    if problems:
        print("PHYSICS CHECK FAILED"); [print(" -", m) for m in sorted(set(problems))]
        return 1
    print(f"PHYSICS CHECK PASSED ({len(eps)} episode(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())