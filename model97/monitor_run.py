#!/usr/bin/env python3
"""Reuse the quiet local supervisor with model97 paths only."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model96 import monitor_run

if __name__=='__main__':
    monitor_run.FOLDER=Path(__file__).resolve().parent
    monitor_run.main()
