"""Run case #10 with shared air-data implementation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'01_Air_Pollution'))
from pipeline import run_case2
if __name__=='__main__':
    print(run_case2())
