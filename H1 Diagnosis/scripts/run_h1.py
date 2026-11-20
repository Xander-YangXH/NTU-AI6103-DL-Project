"""Run one H1 task/seed against the original shared-run data and head checkpoints."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from alloc import subspace_overlap
from gw_eva import study
from gw_eva.calibrate import calibrate
from gw_eva.runtime import MODEL, write
from gw_eva.spectral import compare

MODES = ['eva_centered', 'activation_uncentered', 'gw_uncentered']
TYPES = {'attention.self.query': 'q', 'attention.self.key': 'k',
         'attention.self.value': 'v', 'attention.output.dense': 'o',
         'intermediate.dense': 'FFN input', 'output.dense': 'FFN output'}
LAYERS = [f'roberta.encoder.layer.{block}.{suffix}' for block in range(12) for suffix in TYPES]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(task, seed, checkpoint_root, output):
    source = checkpoint_root / task / 'fit' / f'seed{seed}' / 'initialization'
    head = source / 'head_warmup.pt'
    if not head.is_file():
        raise FileNotFoundError(f'Original classification head required: {head}')
    head_meta = study.read(source / 'head_warmup.json')
    data, collator, split, sha = study.prepare(task)
    assert head_meta['indices'] == split['fit']
    assert (head_meta['task'], head_meta['seed'], head_meta['max_length']) == (task, seed, 128)
    assert head_meta['train_data_sha256'] == sha
    lowres = task == 'sst2_lowres'
    run_root = output / 'cache' / task / f'seed{seed}'

    def sketch(name, pool, modes, count, permutation_start=0):
        directory = run_root / name
        directory.mkdir(parents=True, exist_ok=True)
        # In the existing sampler, the first count entries of this permutation get selected.
        perm = np.random.default_rng(seed).permutation(len(pool))
        positions = perm[permutation_start:permutation_start + count]
        assert len(positions) == count
        if permutation_start:
            order = np.arange(len(pool))
            order[perm[:count]] = positions
            order[positions] = perm[:count]
            pool = [pool[i] for i in order]
        if (directory / 'head_warmup.pt').exists():
            assert digest(directory / 'head_warmup.pt') == digest(head)
        else:
            shutil.copyfile(head, directory / 'head_warmup.pt')
        ctx = {'task': task, 'partition': name, 'directory': directory, 'data': data['train'].select(pool),
               'collator': collator, 'data_sha': sha, 'modes': modes}
        artifact = torch.load(calibrate(seed, 128, count, micro_batch=16, context=ctx),
                              weights_only=False, map_location='cpu')
        ids = [int(pool[i]) for i in artifact['metadata']['indices']]
        return artifact, ids

    primary, ids = sketch('primary', split['fit'], MODES, min(1024, len(split['fit'])))
    if lowres:
        _, _, full_split, full_sha = study.prepare('sst2')
        assert full_sha == sha
        excluded = set(ids)
        noise_pool = [i for i in full_split['fit'] if i not in excluded]
        first, noise_a = sketch('full_noise_a', noise_pool, ['gw_uncentered'], 1024)
    else:
        noise_pool = split['fit']
        first, noise_a = primary, ids
    second, noise_b = sketch('noise_b', noise_pool, ['gw_uncentered'], 1024, 1024)
    assert len(set(noise_a)) == len(set(noise_b)) == 1024
    assert not set(noise_a) & set(noise_b)
    assert set(primary['layers']) == set(first['layers']) == set(second['layers']) == set(LAYERS)
    rows = []
    for j, name in enumerate(LAYERS):
        state = primary['layers'][name]
        eva, gw = (state[m]['components'][:8] for m in (MODES[0], MODES[2]))
        gw_a, gw_b = (a['layers'][name][MODES[2]]['components'][:8] for a in (first, second))
        assert all(v.shape[0] == 8 for v in (eva, gw, gw_a, gw_b))
        same = subspace_overlap(eva, gw)
        noise = subspace_overlap(gw_a, gw_b)
        h1 = primary['h1'][name]['centered_eva_vs_gw']
        assert abs(h1['overlap'] - same) < 1e-5
        pearson = primary['gradient_statistics'][name]['pearson_g_squared_vs_x_norm_squared']
        rows.append(dict(layer=name, block=j//6, layer_type=list(TYPES.values())[j%6],
                         eva_gw=same, gw_gw=noise, difference=same-noise,
                         principal_angles_degrees=h1['principal_angles_degrees'],
                         gw_gw_angles_degrees=compare(gw_a, gw_b)['principal_angles_degrees'],
                         pearson=pearson if pearson is not None and math.isfinite(pearson) else None))
    result = dict(task=task, seed=seed, model='FacebookAI/roberta-base',
                  checkpoint_sha256=digest(head), data_sha256=sha, modes=MODES, rank=8,
                  max_length=128, batch_size=16, sample_ids=ids, noise_sample_ids=[noise_a, noise_b],
                  noise_pool='full_sst2_training_split_excluding_low_resource_subset' if lowres else 'same_training_split',
                  runtime_seconds={'primary':primary['metadata']['seconds'],
                                   'noise_a_additional':first['metadata']['seconds'] if lowres else 0,
                                   'noise_b':second['metadata']['seconds']},
                  mean_eva_gw=mean(r['eva_gw'] for r in rows), mean_gw_gw=mean(r['gw_gw'] for r in rows),
                  mean_principal_angles=[mean(r['principal_angles_degrees'][i] for r in rows) for i in range(8)],
                  per_layer=rows)
    for key, values in [('layer_type', TYPES.values()), ('block', range(12))]:
        stats = []
        for value in values:
            group = [r for r in rows if r[key] == value]
            correlations = [r['pearson'] for r in group if r['pearson'] is not None]
            stats.append({key:value, 'eva_gw':mean(r['eva_gw'] for r in group),
                          'gw_gw':mean(r['gw_gw'] for r in group),
                          'angle_deg':mean(mean(r['principal_angles_degrees']) for r in group),
                          'pearson':mean(correlations) if correlations else None,
                          'valid_pearson_layers':len(correlations)})
        result[key + '_statistics'] = stats
    write(output / f'{task}_seed{seed}.json', result)
    print(task, seed, result['mean_eva_gw'], result['mean_gw_gw'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=study.TASKS, required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('G5G6/results/h1'))
    args = parser.parse_args()
    run(args.task, args.seed, args.checkpoint_root, args.output)
