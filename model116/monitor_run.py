"""Quiet ten-minute supervisor for model116."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quiet_monitor", ROOT / "model106/monitor_run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.FOLDER = ROOT / "model116"

if __name__ == "__main__":
    module.main()
