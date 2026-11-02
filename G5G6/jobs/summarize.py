"""Read-only summary of exported main-run JSONs."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results');a=p.parse_args()
    runs=[json.loads(f.read_text(encoding='utf-8')) for f in (a.results/'main').glob('*.json')]
    print(f'Completed main runs: {len(runs)}/26')
    print('| Task | Group | n | Accuracy mean +/- SD (%) | F1 mean +/- SD (%) |')
    print('|---|---|---:|---|---|')
    for task in ['rte','mrpc','sst2']:
        for group in ['G5','G6']:
            values=[r for r in runs if r['task']==task and r['group']==group]
            def formatted(key):
                scores=[r['final'][key] for r in values if key in r['final']]
                if not scores:return '-'
                if len(scores)==1:return f'{scores[0]*100:.2f} (single seed)'
                return f'{statistics.mean(scores)*100:.2f} +/- {statistics.stdev(scores)*100:.2f}'
            print(f'| {task} | {group} | {len(values)} | {formatted("acc")} | {formatted("f1")} |')
    print('Independent protocol; do not directly pair with shared-protocol G3/G4/G7 results.')


if __name__=='__main__':main()
