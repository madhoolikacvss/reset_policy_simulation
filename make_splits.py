#!/usr/bin/env python3
"""
Build data/train and data/val from your full set of episode logs.

    python make_splits.py --src /path/to/motor_logs --n-train 40 --n-val 40 --seed 0

* Splits by WHOLE EPISODE (never by row), train and val disjoint.
* Stratified by length: episodes shorter than --short-below rows are sampled in the same
  proportion as in the full pool, so train and val both contain early-terminating episodes.
* Writes TRIMMED copies that keep only columns usable as model inputs or targets. Reward,
  distance-to-goal and penalty columns (which leak the end state) are dropped on purpose.
* Writes data/manifest.json (seed, counts, file lists) so the split is reproducible.
Episodes you do not select are left untouched in --src: keep them as your private test set.
"""
import argparse, json, random, shutil
from pathlib import Path

import pandas as pd

KEEP = (
    ["step", "cube_x", "cube_y", "obs_cube_x_norm", "obs_cube_y_norm"]
    + [f"action_m{m}" for m in (16, 17, 18, 19)]
    + [f"executed_m{m}" for m in (16, 17, 18, 19)]
    + [f"obs_motor_{m}_{k}" for k in ("pos_delta_norm", "current_norm", "target_error_norm") for m in (16, 17, 18, 19)]
    + ["obs_horizontal_tension_norm", "obs_vertical_tension_norm", "obs_total_tension_norm"]
)
REQUIRED = ["cube_x", "cube_y", "obs_cube_x_norm", "obs_cube_y_norm"] + [f"executed_m{m}" for m in (16, 17, 18, 19)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--short-below", type=int, default=380, help="rows below this = early-terminating")
    ap.add_argument("--min-rows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = {"full": [], "short": []}
    skipped = []
    for f in sorted(Path(args.src).glob("*.csv")):
        try:
            df = pd.read_csv(f)
            if any(c not in df.columns for c in REQUIRED):
                raise ValueError("missing required columns")
            df = df.dropna(subset=REQUIRED)
        except Exception as e:
            skipped.append((f.name, str(e))); continue
        if len(df) < args.min_rows:
            skipped.append((f.name, f"only {len(df)} valid rows")); continue
        pool["short" if len(df) < args.short_below else "full"].append(f)

    total = len(pool["full"]) + len(pool["short"])
    need = args.n_train + args.n_val
    if total < need:
        raise SystemExit(f"Only {total} usable episodes, need {need}")
    frac_short = len(pool["short"]) / total
    rng = random.Random(args.seed)
    for k in pool:
        rng.shuffle(pool[k])

    def take(n):
        n_short = min(round(n * frac_short), len(pool["short"]))
        n_full = n - n_short
        if n_full > len(pool["full"]):
            n_full, n_short = len(pool["full"]), n - len(pool["full"])
        got = [pool["short"].pop() for _ in range(n_short)] + [pool["full"].pop() for _ in range(n_full)]
        return sorted(got)

    chosen = {"train": take(args.n_train), "val": take(args.n_val)}
    out = Path(args.out)
    for split, files in chosen.items():
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in files:
            df = pd.read_csv(f)
            df[[c for c in KEEP if c in df.columns]].to_csv(d / f.name, index=False)
    manifest = {"seed": args.seed, "short_below": args.short_below, "unused_in_pool": {k: len(v) for k, v in pool.items()},
                "frac_short_in_pool": round(frac_short, 3), "skipped": skipped,
                **{s: [f.name for f in fs] for s, fs in chosen.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"usable={total} short_fraction={frac_short:.2f} skipped={len(skipped)}")
    for s, fs in chosen.items():
        print(f"  {s}: {len(fs)} episodes")


if __name__ == "__main__":
    main()