# reset_policy_simulation# Cube-pulling simulator tuning package

Read `AGENTS.md` first: it holds the rules, the metric and the workflow.

## Commands
    python make_splits.py --src /path/to/motor_logs --n-train 40 --n-val 40   # once, you run this
    python freeze.py write                      # once, after data/ is built, then commit
    python check_physics.py                     # must pass before reporting any result
    python benchmark.py --per-episode           # evaluate params.yaml on data/train
    python benchmark.py --split val             # held-out; ONE run, at the very end
    python freeze.py verify                     # after the agent finishes: frozen files unchanged?

## Layout
    AGENTS.md          rules (frozen)            benchmark.py      metric (frozen)
    check_physics.py   physics rules (frozen)    data/             train + val (frozen)
    params.yaml        values (agent edits)      params.py         schema and bounds
    sim_core.py        the simulator             requirements.txt, setup.sh   environment

## Codex (ChatGPT) cloud setup
1. Push this repo to GitHub (with `data/` populated) and connect it in Codex.
2. In the environment settings, use `setup.sh` as the setup script. Internet is available during
   setup only, so all packages (mujoco, optuna, cmaes) install there. No wheels are needed.
3. Leave agent internet access off. Nothing needs the network after setup.
4. No GPU or rendering is used, so there is no MuJoCo GL setup.