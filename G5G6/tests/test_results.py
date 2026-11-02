"""Independently recompute exported scores and check publication integrity."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


class Results(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main=[read(p) for p in (ROOT/'results/main').glob('*.json')]
        cls.tune=[read(p) for p in (ROOT/'results/tune').glob('*.json')]

    def test_scope_and_protocol(self):
        self.assertEqual(len(self.main),26);self.assertEqual(len(self.tune),18)
        self.assertEqual(len({r['run_id'] for r in self.main+self.tune}),44)
        for task,n in [('rte',5),('mrpc',5),('sst2',3)]:
            for group in ['G5','G6']:
                jobs=[r for r in self.main if r['task']==task and r['group']==group]
                self.assertEqual(sorted(r['seed'] for r in jobs),list(range(n)))
                candidates=[r for r in self.tune if r['task']==task and r['group']==group]
                self.assertEqual(sorted(r['lr'] for r in candidates),[.0001,.0004,.001])
        for r in self.main+self.tune:
            self.assertFalse(r['shared_protocol_compliant'])
            self.assertEqual(r['cfg']['max_len'],512)
            source=r['source_result'];self.assertEqual(source['steps'],source['planned_steps'])
            self.assertEqual(source['sample_visits'],source['training_samples']*source['epochs'])
            self.assertEqual(source['reload_max_logit_difference'],0)
            self.assertEqual(len(r['loss_first_steps']),500)
            self.assertEqual(len(r['b_grad_first_steps']),500)
            self.assertTrue(all(math.isfinite(x) for x in r['loss_first_steps']+r['b_grad_first_steps']))
            self.assertEqual(r['step0_grad_B'],r['b_grad_first_steps'][0])

    def test_predictions_and_metrics(self):
        for r in self.main+self.tune:
            data=r['predictions'];labels=data['labels'];pred=data['predictions']
            self.assertEqual(len(labels),r['source_result']['evaluation_samples'])
            self.assertEqual(pred,[max(range(len(x)),key=lambda i:x[i]) for x in data['logits']])
            acc=sum(a==b for a,b in zip(labels,pred))/len(labels)
            self.assertAlmostEqual(acc,r['final']['acc'],places=12)
            if r['task']=='mrpc':
                tp=sum(p==y==1 for p,y in zip(pred,labels));fp=sum(p==1 and y==0 for p,y in zip(pred,labels));fn=sum(p==0 and y==1 for p,y in zip(pred,labels))
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
                self.assertAlmostEqual(f1,r['final']['f1'],places=12)
                self.assertAlmostEqual(r['source_result']['metrics']['selection_score'],(acc+f1)/2,places=12)

    def test_true_parameter_counts_and_prefixes(self):
        for r in self.main+self.tune:
            rank=r['ranks'];self.assertEqual(len(rank),72)
            self.assertTrue(all(isinstance(x,int) and 0<=x<=16 for x in rank.values()))
            cost=sum((3840 if n.endswith('intermediate.dense') or n.endswith('.output.dense') and '.attention.' not in n else 1536)*v for n,v in rank.items())
            self.assertEqual(cost,r['lora_params']);self.assertEqual(r['head_params'],592130)
            if r['group']=='G5':self.assertEqual(sum(rank.values()),576)
            else:self.assertEqual(cost,1327104)
        for path in (ROOT/'results/initialization').glob('*.json'):
            cache=read(path);self.assertEqual(len(cache['layers']),72)
            for layer in cache['layers'].values():
                scores=layer['scores'];self.assertEqual(len(scores),16)
                self.assertTrue(all(math.isfinite(x) and x>=0 for x in scores))
                self.assertTrue(all(a+1e-12>=b for a,b in zip(scores,scores[1:])))

    def test_summary_statistics(self):
        summary=read(ROOT/'results/summary.json');self.assertEqual(summary['completed'],26)
        for row in summary['rows']:
            jobs=[r for r in self.main if r['task']==row['task'] and r['group']==row['group']]
            values=[r['final']['acc'] for r in jobs]
            self.assertAlmostEqual(statistics.mean(values),row['accuracy_mean'],places=12)
            self.assertAlmostEqual(statistics.stdev(values),row['accuracy_std'],places=12)
            if row['task']=='mrpc':
                f1=[r['final']['f1'] for r in jobs]
                self.assertAlmostEqual(statistics.mean(f1),row['f1_mean'],places=12)
                self.assertAlmostEqual(statistics.stdev(f1),row['f1_std'],places=12)

    def test_split_and_pinned_provenance(self):
        for task,n in [('rte',2490),('mrpc',3668),('sst2',67349)]:
            split=read(ROOT/'configs/splits'/f'{task}.json')
            self.assertFalse(set(split['fit'])&set(split['selection']))
            self.assertEqual(set(split['fit'])|set(split['selection']),set(range(n)))
            self.assertEqual(split['exact_text_overlap'],0)
        for source in read(ROOT/'configs/data_sources.json').values():
            self.assertEqual(len(source['revision']),40)
            self.assertEqual(len(source['files']),2)
            self.assertTrue(all(len(item['sha256'])==64 for item in source['files']))


if __name__=='__main__':unittest.main(verbosity=2)
