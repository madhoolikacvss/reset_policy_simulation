#!/usr/bin/env python3
"""Record / verify checksums of the frozen files so silent edits are detectable.
    python freeze.py write     # once, before handing over
    python freeze.py verify    # after the agent finishes (exit 1 if anything changed)
"""
import hashlib, sys
from pathlib import Path

FROZEN = ["benchmark.py", "check_physics.py", "AGENTS.md"]
OUT = Path("FROZEN.sha256")


def files():
    fs = [Path(f) for f in FROZEN if Path(f).exists()]
    fs += sorted(p for p in Path("data").rglob("*") if p.is_file() and p.name != ".gitkeep")
    return fs


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    now = {str(p): digest(p) for p in files()}
    if mode == "write":
        OUT.write_text("".join(f"{h}  {p}\n" for p, h in now.items()))
        print(f"wrote {OUT} ({len(now)} files)")
        sys.exit(0)
    old = dict(reversed(l.strip().split("  ", 1)) for l in OUT.read_text().splitlines() if l.strip())
    bad = [p for p in set(old) | set(now) if old.get(p) != now.get(p)]
    if bad:
        print("FROZEN FILES CHANGED:"); [print(" -", p) for p in sorted(bad)]
        sys.exit(1)
    print(f"OK: {len(now)} frozen files unchanged")