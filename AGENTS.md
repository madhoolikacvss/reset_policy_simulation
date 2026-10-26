# AGENTS.md — tuning the cube-pulling simulator

## 1. Goal

Make `sim_core.py` reproduce the real cube's motion as closely as possible from the logged
**executed** motor actions. Success is measured only by `benchmark.py` (section 6). Beating the
"cube never moves" baseline on held-out episodes is the minimum bar; getting one-step error
close to the `linear_ref` printed by the benchmark is the target.

You may change both **parameter values** and **model structure**, subject to the hard physical
rules in section 3. Understanding *why* a model fits is worth more than a lower number you
cannot explain.

## 2. The physical setup

- A rectangular board. A motor sits at the **middle of each edge**, fixed in place.
- A 6 x 6 x 6 cm cube (62 g) rests on the board under gravity. It does not roll or rotate.
- Each of the cube's four vertical faces has one string attached in the lower half of the face.
  Each string runs to one motor, and the face-to-motor assignment never changes.
- A motor can **pull** its string, or **release** it. Releasing makes the string slack. Slack
  strings exert no force and give the other motors room to pull. The strings only slide the cube
  across the board; they do not hold it on the board (gravity does).
- Motor layout (**fixed, not tunable**; frame in section 4): m16 and m17 are the far-apart
  pair, on the raw-y edges (m16 at -y, m17 at +y). m18 and m19 are the closer pair, on the
  raw-x edges (m18 at +x, m19 at -x). The 16/17 pair therefore acts along y, and the 18/19
  pair along x. Do not swap or relabel motors, even if a fit seems to prefer it; report it.

## 3. Hard physical rules (never violate; `check_physics.py` enforces most of them)

1. **A string can only pull the cube toward its own motor.** The force on the cube from motor k
   always has a non-negative component toward motor k. Releasing a string gives **zero** force.
   A motor can **never push** the cube. Do not introduce negative forces, signed forces, or
   per-motor sign flips, and do not model release as a push.
2. **Four separate forces.** Keep one force per string, applied for its own face and directed
   along its own string. Do not collapse them into "horizontal" and "vertical" forces. Do not
   try to resolve opposing pulls analytically; let the physics sum them.
3. **The cube never rotates and never leaves the board.** The board-edge safety clamp and
   its `off_board_rows` counter must stay. It is how a bad parameter set is detected.
4. **Fixed constants stay fixed:** cube mass, cube size, board bounds, motor positions
   (`Board` in `params.py`). Only the friction and force parameters are free.
5. **Forces are always evaluated at the cube's current simulated position** (the string
   direction changes as the cube moves).
6. **The model stays physics-based and interpretable.** The backbone is the MuJoCo rigid-body
   simulation with pull-only string forces and friction. Do not replace it, or add to it,
   any neural network, learned black-box correction, lookup table of logged trajectories, or
   other component fitted to reproduce positions without a physical meaning. New structure
   must be a physical mechanism you can name and describe in `REPORT.md` (for example string
   slack, string stretch, stiction, motor response lag).

## 4. Data and frames

Each CSV row is **one complete transition**:

| What | Column(s) | Meaning |
|---|---|---|
| start state | `obs_cube_x_norm`, `obs_cube_y_norm` | cube position **before** the step |
| action | `executed_m16`..`executed_m19` | what the motors actually did (use these, not `action_m*`) |
| end state | `cube_x`, `cube_y` (raw metres) | cube position **after** the step |

The start state of row i+1 equals the end state of row i exactly. There is no row shifting:
`pred_norm[i]` is the predicted end state of step i and is compared with `real_norm[i]`.

Normalisation (true values, no axis swap):
`x_norm = (x - x_min) / (x_max - x_min)`, `y_norm = (y - y_min) / (y_max - y_min)` with
x in [-0.50, -0.07], y in [-0.71, -0.12] metres. The simulator works in board coordinates
(metres, centred), converted by the helpers in `sim_core.py`.

**Real-world rule: no future data.** The simulator must behave like the real system, where
nothing from the future is available. When simulating step i it may use only:
- the executed action of step i (`executed_m*` in row i; it is known before the step runs);
- observations recorded at the **start** of step i (row i): `obs_motor_*_current_norm`,
  `obs_motor_*_pos_delta_norm`, `obs_*_tension_norm`, `obs_motor_*_target_error_norm`;
- anything from rows **before** i (past actions and observations), as history.

**Forbidden inputs:** anything from rows **after** i (for example the current or tension
reading in row i+1); `cube_x`/`cube_y` of row i or later; anything derived from the end
state (`dist_to_goal*`, reward and penalty columns, which are removed from `data/` anyway);
and `obs_cube_*` at any row other than the resync start rows the harness uses. The simulator
may be started from a real start state only at resync points, as `sim_core.py` already does.
A quick self-test: if the model could not run live, one step at a time, it is using forbidden data.

## 5. What you may and may not edit

| File | Rule |
|---|---|
| `params.yaml` | **Edit freely** (values must stay inside the bounds in `params.py`). |
| `sim_core.py` | **Editable**: the physics and the action-to-force mapping. Do **not** change `load_episode`, the coordinate helpers, the `Episode`/`SimResult` fields, or the meaning of `pred_norm`. |
| `params.py` | **Add** new parameters, bounds and choices for new structure. **Do not widen or remove existing bounds.** If a value sits on a bound, report it as evidence the structure is wrong; do not move the bound. |
| `tune.py`, `REPORT.md`, other new files | Create freely. |
| `benchmark.py`, `check_physics.py`, `data/`, `AGENTS.md` | **Frozen. Never modify.** |

## 6. The metric

All errors are mean Euclidean distance in normalised [x, y] per row.

- `one_step`: the sim restarts from the real start state at **every** row.
- `rollout`: the sim restarts every **20** rows and free-runs in between.
- `long`: the sim restarts every **100** rows (episodes are about 400 steps), so errors that
  accumulate over a long run are scored. This term matters most: the sim will be used to
  reason about whole episodes, and a cube that barely moves must not look good.
- `open`: diagnostic only (never resynced after row 0).
- `off`: fraction of long-rollout steps where the cube hit the board edge.

`score = (one_step + rollout + long) / 3 + 0.1 * off` — lower is better.

Every run also prints the **stationary** baseline for each horizon (cube stays at its last
resynced state), the `skill` (`1 - score / stationary_score`; positive means better than doing
nothing) and `linear_ref`, a linear map from executed actions to one-step displacement fitted on
`data/train`. That last number is a diagnostic ceiling for one-step error, not something to copy.

The tracker is noisy: per-step displacements are small (around 0.01 to 0.03) and the stationary
one-step error is about 0.012 on the full training set. Over a whole episode the real cube
drifts a long way (open-loop stationary error is about 0.15), so the long-horizon terms are where
a good model can win clearly. Differences of less than about 1% of the stationary score are
noise. Judge improvements on all training episodes together, and look at the per-horizon
numbers, not just the total.

## 7. Workflow

1. **Baseline.** Run `python benchmark.py --per-episode` and `python check_physics.py`. Record
   both. Do this before changing anything.
2. **Diagnose with one-step error first.** It has no accumulated drift, so it isolates the
   action-to-force mapping. Fit the structural choices (`force_mode`, `direction_mode`,
   `model_type`) with coarse sweeps before any fine tuning. Do this on a **fixed subset**
   of 10 train episodes (see the budget below).
3. **Optimise continuous values** with a real optimiser (random search, then CMA-ES or Optuna)
   written in `tune.py`, calling `benchmark.evaluate` on train data. Do not hand-tweak. Fix the
   random seed and log every evaluation to `results.csv` (the benchmark does this).
4. **Revisit structure** if the optimised parameters sit on bounds or are physically odd
   (see section 8).
5. **Validate once.** After you have chosen your final configuration on train data, run
   `python benchmark.py --split val` **one time** (the val split holds about 40 episodes, so the
   number is reliable). Report it whatever it is. Do not adjust anything in response to it.
6. **Report** in `REPORT.md` (section 10).

**Runtime:** a full-train evaluation (about 40 episodes) takes roughly 30 s to a minute on one core.
In `tune.py` you may call `benchmark.evaluate(..., include_open=False)` (same score, about a
third faster) and parallelise across episodes with `multiprocessing`.

**Budget:**
- at most **200 full-train evaluations** (optimiser trials count);
- at most **300 subset evaluations**, on a fixed subset of 10 train episodes chosen once, up
  front (for example every 4th file in sorted order), for structure sweeps and quick checks;
- exactly **one** validation run.

Stop earlier if the best full-train score has not improved by more than 1% in the last 40
evaluations. Finish with the final configuration re-run once on the full train split.

## 8. Things to watch for

- **Friction is unknown.** Nobody has measured it, so treat it as a free parameter within its
  bounds (0.05 to 2.0) and let the data decide. Because the cube is nearly at breakaway (the net
  force is only slightly above sliding friction), `force_scale_*` and `friction` are strongly
  coupled: mostly their ratio matters. Expect a ridge of equally good fits, and report it
  rather than reading too much into one point on it.
- **Step duration.** `row_dt` is nominally 0.8 s, but the real value may differ slightly, so it
  is tunable within 0.5 to 1.2 s. In the kinematic model displacement is `kin_speed * row_dt`,
  so those two are exactly degenerate. Interpret only their product, and do not trust a `row_dt`
  that is far from 0.8 s as a physical finding.
- **Degenerate parameters in general.** Force scale, friction, `row_dt`, `pull_fraction` and
  damping trade off against each other. Prefer the physically plausible point, and report the
  trade-off instead of picking an extreme.
- **Per-motor scales that differ wildly** (for example one near the lower bound and one near the
  upper) usually mean the geometry or layout is wrong, not that the motors differ.
- **The simulator ignoring actions.** If the best fit is a cube that barely moves, you have
  learned the model has no predictive power. Say so.
- **Open-loop drift** is expected. Do not tune for open-loop error at the cost of one-step error.

## 9. Observations so far (baseline run on 40 training episodes; re-check after your changes)

Untuned defaults on the full train split, next to the stationary baseline:

| Metric | Default sim | Stationary |
|---|---|---|
| one-step | 0.0136 | 0.0121 |
| 20-step rollout | 0.0259 | 0.0251 |
| open loop | 0.130 | 0.154 |
| score / skill | 0.0197 (the old 2-term score; skill -0.06) | 0.0186 |

- The starting point is about as good as doing nothing. Any gain must come from the tuning.
- **Where the sim loses most:** episodes where the real cube barely moves (stationary one-step
  error 0.001 to 0.003, e.g. episodes 0315, 0228, 0230, 0239, 0215). The sim moves the cube
  whenever a motor acts, but the real cube often does not. Strong hint of **slack**: motors wind
  thread without moving the cube, and the cube moves only once a string is taut.
- **Where the sim helps:** over long horizons the default open-loop sim is already better than
  stationary (0.130 vs 0.154). The real cube drifts about 0.15 over an episode, so a model that
  gets the drift direction right can win clearly there.
- Realistic headroom: the linear reference is only about 7% better than stationary for one-step
  error (0.0112 vs 0.0121), and it is fitted on the same data it is scored on. Expect modest one-step
  gains, and look for the larger win on `rollout` and `long`.
- Actions are continuous in [-1, 1] and look like exploration. The safety filter changes about a
  third of rows; the executed value is what the motors did.
- Negative executed values (release) appear to carry information in a linear fit, even though a
  release can never produce a push. Explain it physically, for example through string slack and
  stored tension, string stretch or residual motion. One candidate: a string-length state per
  motor that shortens while pulling, lengthens when released, and pulls only while taut. That
  would produce exactly the "motor acts, cube does not move" behaviour above. Any such model must
  obey section 3.

## 10. Deliverables

- Final `params.yaml` and `sim_core.py`, passing `python check_physics.py`.
- `results.csv` with the full evaluation log (do not delete rows). Mark subset runs in your own
  `tune.py` log, since the benchmark only records the split.
- `REPORT.md` with: the baseline and final train scores; the single val result; a table of
  structural variants tried and their scores; which parameters sit near a bound; what the
  model gets wrong and why you think so; and anything the data says about the physical setup
  (for example whether a motor's effect has the wrong direction or axis, which would point to a
  layout or wiring problem to investigate by hand). Include the
  **stationary baseline** next to every number.
- State clearly if the simulator does **not** beat the stationary baseline on validation. That
  is a legitimate result.