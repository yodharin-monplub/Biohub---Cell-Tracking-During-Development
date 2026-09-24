"""Paired full-repair replay of the frozen model112 notebook."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_replay", ROOT / "model106/replay.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = ROOT / "model112"
module.EXPECTED_MOTION_RELINK = True
module.EXPECTED_SHORT_TRACK_FILTER = False

if __name__ == "__main__":
    module.main()
