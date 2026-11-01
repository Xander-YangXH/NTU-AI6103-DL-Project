"""CPU-only numerical gates; no model/data download."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gw_eva.tests import main

if __name__=='__main__':main()
