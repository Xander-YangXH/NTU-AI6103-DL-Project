"""Aggregate completed H1 runs and render an English layerwise figure."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

TASKS = {'rte': 'RTE', 'mrpc': 'MRPC', 'sst2': 'SST-2', 'sst2_lowres': 'SST-2 (1k)'}
TYPES = ['q', 'k', 'v', 'o', 'FFN input', 'FFN output']


def summarize(root, eva_path):
    runs = [json.loads((root / f'{t}_seed{s}.json').read_text()) for t in TASKS for s in range(3)]
    eva = {}
    if eva_path:
        for row in json.loads(eva_path.read_text()):
            key = (row['task'], row['seed'], row['layer'])
            assert key not in eva and 0 <= row['eva_eva'] <= 1
            eva[key] = row['eva_eva']
    records = []
    for run in runs:
        assert len(run['per_layer']) == 72
        for row in run['per_layer']:
            record = dict(task=run['task'], seed=run['seed'], **row)
            record['eva_eva'] = eva.get((run['task'], run['seed'], row['layer']))
            records.append(record)

    def stats(rows):
        eva_vals = [r['eva_eva'] for r in rows if r['eva_eva'] is not None]
        pearson = [r['pearson'] for r in rows if r['pearson'] is not None]
        return dict(eva_gw=mean(r['eva_gw'] for r in rows), gw_gw=mean(r['gw_gw'] for r in rows),
                    mean_angle_deg=mean(mean(r['principal_angles_degrees']) for r in rows),
                    pearson=mean(pearson) if pearson else None, pearson_n=len(pearson),
                    eva_eva=mean(eva_vals) if eva_vals else None, eva_eva_n=len(eva_vals))

    summary = dict(overall=stats(records), tasks={}, completed=len(runs))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained', sharex=True, sharey=True)
    for ax, (task, title) in zip(axes.flat, TASKS.items()):
        rows = [r for r in records if r['task'] == task]
        summary['tasks'][task] = dict(overall=stats(rows), layer_types={}, blocks={})
        for kind in TYPES:
            summary['tasks'][task]['layer_types'][kind] = stats([r for r in rows if r['layer_type'] == kind])
        for block in range(12):
            summary['tasks'][task]['blocks'][block] = stats([r for r in rows if r['block'] == block])
        for kind, marker in zip(TYPES, ('o', 's', '^', 'v', 'D', 'x')):
            by_depth = [stats([r for r in rows if r['block'] == b and r['layer_type'] == kind]) for b in range(12)]
            for key, color, style in [('eva_gw', 'C0', '-'), ('gw_gw', 'C1', '--'), ('eva_eva', 'C2', ':')]:
                if any(r[key] is not None for r in by_depth):
                    ax.plot(range(12), [r[key] for r in by_depth], color=color, linestyle=style,
                            marker=marker, markersize=3, linewidth=.9)
        ax.set_title(title + (' — GW–GW from full SST-2' if task == 'sst2_lowres' else ''))
        ax.set(xticks=range(12), ylim=(0,1.01))
        ax.grid(alpha=.18)
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=c, linestyle=s, label=l) for c,s,l in
              [('C0','-','EVA–GW'),('C1','--','GW–GW'),('C2',':','EVA–EVA')] if l != 'EVA–EVA' or eva]
    legend += [Line2D([], [], color='.3', marker=m, linestyle='None', label=k)
               for k,m in zip(TYPES, ('o','s','^','v','D','x'))]
    fig.legend(handles=legend, loc='outside upper center', ncol=len(legend), frameon=False)
    fig.supxlabel('Transformer block')
    fig.supylabel('Rank-8 subspace overlap')
    fig.savefig(root / 'h1_layerwise_overlap.png', dpi=250)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf8')
    with (root / 'h1_layerwise.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--eva-baseline', type=Path, help='JSON rows with task, seed, layer, eva_eva')
    args = p.parse_args()
    summarize(args.results, args.eva_baseline)
