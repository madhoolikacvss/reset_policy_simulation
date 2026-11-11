# V4 nonelastic, unchanged parameters

Clean, standalone simulator package using the original V4 parameter values.
No fitted nonelastic force scales are included. Motor 18 is unchanged too.

The cable starts with 4.87 mm slack. Winding takes up slack; once taut it
transmits the motor's available pull force immediately. There is no material
extension or spring energy. The motor/spool can stall or pay out under load.
This is not a locked-length constraint, and friction can still prevent motion.
The retained `string_stiffness` YAML field is unused, for schema compatibility.

`sim_core.py`, `params.py`, `params.yaml`, and `render_replay.py` are local copies;
this package does not import the root simulator or the fitted experiment.
Install/use the existing `reset` environment (MuJoCo, NumPy, pandas, PyYAML,
OpenCV). The validation CSVs remain in `../data/val`.

## Videos

`videos/` contains ten open-loop 3D MuJoCo MP4s, 960 x 720 at 30 fps:
red = nonelastic simulation; translucent blue = raw tracked cube.
Real coordinates are not smoothed. Display frames interpolate row endpoints;
these are trajectory replays, not recordings of every internal physics substep.
Each row occupies 23 frames (about 0.767 s), close to V4's 0.7603 s action interval.

The first ten usable validation episodes in filename order were selected,
independent of performance. `manifest.json` records input files, configuration
hashes, frame counts and decoding verification. Sidecar CSVs hold exact endpoint
predictions, JSONs hold replay metadata, and PNGs are first-frame previews.

From the workspace root, to render a new single episode into a new directory:

```powershell
conda run -n reset python -B reset_policy_simulation/v4_noelastic_unchanged/render_replay.py reset_policy_simulation/data/val/episode_0005_motors.csv --params reset_policy_simulation/v4_noelastic_unchanged/params.yaml --out reset_policy_simulation/v4_noelastic_unchanged/new_replay
```

`render_batch.py` reproduces the ten-video batch and refuses to overwrite existing
MP4s. The comparison experiment and report remain in `../v4_noelastic/`.
