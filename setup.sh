#!/usr/bin/env bash
# Codex cloud: paste this as the environment's SETUP SCRIPT (internet is available during setup
# only; it is off by default while the agent runs, so everything must be installed here).
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import mujoco, numpy, pandas, yaml, optuna, cmaes; print('mujoco', mujoco.__version__)"
python check_physics.py          # uses a synthetic episode if data/ is empty