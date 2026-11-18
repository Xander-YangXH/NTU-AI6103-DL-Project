import json
import unittest
from collections import Counter
from pathlib import Path


class SharedResults(unittest.TestCase):
    def test_runs(self):
        root = Path(__file__).resolve().parents[1] / 'results'
        runs = [json.loads(p.read_text(encoding='utf-8'))
                for stage in ('tune', 'main', 'lowres') for p in (root / stage).glob('*.json')]
        self.assertEqual(Counter(r['stage'] for r in runs), {'tune': 18, 'main': 26, 'lowres': 6})
        self.assertEqual(len({r['run_id'] for r in runs}), 50)
        for r in runs:
            with self.subTest(run=r['run_id']):
                self.assertEqual(r['status'], 'completed')
                self.assertEqual(r['max_length'], 128)
                self.assertEqual(r['effective_batch'], 32)
                self.assertEqual(r['steps'], r['planned_steps'])
                self.assertEqual(r['sample_visits'], r['epochs'] * r['training_samples'])
                self.assertEqual(r['epochs'], {'rte': 10, 'mrpc': 10, 'sst2': 3, 'sst2_lowres': 20}[r['task']])
                self.assertIn(r['lr'], (0.0002, 0.0005, 0.001))
                self.assertFalse(r['shared_config']['gradient_clipping'])
                labels = r['predictions']['labels']
                pred = r['predictions']['predictions']
                self.assertEqual(len(labels), len(pred))
                self.assertEqual(len(labels), r['evaluation_samples'])
                self.assertAlmostEqual(sum(a == b for a, b in zip(labels, pred)) / len(labels), r['final']['acc'])
                if r['task'] == 'mrpc':
                    f1 = 2 * sum(a == b == 1 for a, b in zip(labels, pred)) / (sum(labels) + sum(pred))
                    self.assertAlmostEqual(f1, r['final']['f1'])
                if r['group'] == 'G5':
                    self.assertEqual(sum(r['ranks'].values()), 576)
                else:
                    self.assertEqual(r['lora_params'], 1327104)
