"""Use the parity-gated cost1 full-repair runner with model116's corrected notebook."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cost_replay", ROOT / "model115/replay.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = ROOT / "model116"

if __name__ == "__main__":
    module.main()
