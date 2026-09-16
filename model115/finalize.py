"""Apply model112's paired gates to model115 versus model1 and model107."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_finalizer", ROOT / "model112/finalize.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = ROOT / "model115"

if __name__ == "__main__":
    module.main()
