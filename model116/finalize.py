"""Apply paired model107-control gates to model116."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_finalizer", ROOT / "model112/finalize.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = ROOT / "model116"

if __name__ == "__main__":
    module.main()
