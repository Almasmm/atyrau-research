"""Build both air decks from their shared source-controlled generator."""
from pathlib import Path
import runpy
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'01_Air_Pollution/build_slides_v2.py'),run_name='__main__')
