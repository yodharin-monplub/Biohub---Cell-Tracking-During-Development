"""Quiet ten-minute supervision of local inference, repair and scoring."""
import importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('quiet_monitor',root/'model96/monitor_run.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.FOLDER=root/'model100'
if __name__=='__main__':
    module.main()
