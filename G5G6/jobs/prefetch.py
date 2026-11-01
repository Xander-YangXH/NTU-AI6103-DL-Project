"""Pinned model/data preparation; run from G5G6 after pip install -e ."""
import argparse
from gw_eva.prepare import main
from gw_eva.study import prepare
p=argparse.ArgumentParser();p.add_argument("--tasks",nargs="+",default=["rte","mrpc","sst2"]);a=p.parse_args()
main()
for task in a.tasks:prepare(task)
