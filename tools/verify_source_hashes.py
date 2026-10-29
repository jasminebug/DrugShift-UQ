"""Validate original code files against audited production run manifest (stdlib only)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "evidence/run_manifest.json").read_text(encoding="utf8"))
for path, expected in manifest["code_sha256"].items():
    actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
    status = "OK" if actual == expected else "MISMATCH"
    print(f"{status:8} {path}")
    if actual != expected:
        raise SystemExit(f"Source differs from original research run: {path}")
print("All original research code hashes match recorded run", manifest["run_id"])
