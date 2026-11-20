"""Numerical gates against direct second moments and exhaustive allocation."""
import itertools
import json
from pathlib import Path
import unittest
import torch
from .spectral import StreamingSVD,compare
from .allocation import count_allocate,cost_allocate

EVIDENCE={}


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(7)

    def test_exact_uncentered_and_centered(self):
        x=torch.randn(90,12)+torch.arange(12)*.8
        g=torch.randn(90,5);z=x*g.norm(dim=1,keepdim=True)
        for centered in [False,True]:
            svd=StreamingSVD(12,max_rank=12,centered=centered,chunk_rows=19).update(z)
            value=z.double()-z.double().mean(0) if centered else z.double()
            vals,vec=torch.linalg.eigh(value.T@value)
            result=compare(svd.state()['components'][:4],vec[:,-4:].T.flip(0).float())
            EVIDENCE[f'exact_centered_{centered}']=result
            self.assertGreater(result['overlap'],.99999)
            self.assertLess(result['projection_frobenius'],.004)
            self.assertLess(result['orthogonality_error_v'],2e-6)
            self.assertTrue(torch.allclose(svd.singular.double().square(),vals.flip(0),rtol=3e-5,atol=.005))
        uncentered=StreamingSVD(12,4).update(z).state()
        centered=StreamingSVD(12,4,centered=True).update(z).state()
        EVIDENCE['centering_changes_target']=compare(uncentered['components'],centered['components'])

    def test_truncation_error_measured(self):
        x=torch.randn(120,24)*torch.linspace(3,.1,24)
        state=StreamingSVD(24,6,chunk_rows=12).update(x).state()
        vals,vec=torch.linalg.eigh(x.double().T@x.double())
        result=compare(state['components'],vec[:,-6:].T.flip(0).float())
        approx=state['components'].double().T@torch.diag(state['singular_values'].double().square())@state['components'].double()
        result['relative_moment_frobenius_error']=float((x.double().T@x.double()-approx).norm()/(x.double().T@x.double()).norm())
        result['optimal_rank6_tail_energy']=float(vals[:-6].sum()/vals.sum())
        result['streaming_discarded_energy']=state['discarded_merge_energy']
        result['streaming_retained_ratio']=state['retained_energy_ratio']
        EVIDENCE['truncated_approximation']=result
        self.assertLess(result['orthogonality_error_v'],2e-6)
        self.assertGreater(result['relative_moment_frobenius_error'],0)
        self.assertLessEqual(state['retained_energy_ratio'],1.00001)

    def test_edges(self):
        for rows in [torch.zeros(0,8),torch.zeros(7,8),torch.randn(1,8),torch.eye(8)]:
            s=StreamingSVD(8,4).update(rows).state()
            self.assertTrue(torch.isfinite(s['components']).all())
            self.assertLess(float((s['components']@s['components'].T-torch.eye(4)).abs().max()),2e-5)
        with self.assertRaises(ValueError):StreamingSVD(8).update(torch.full((2,8),float('nan')))
        repeated=StreamingSVD(8,8).update(torch.eye(8)).state()
        self.assertGreater(compare(repeated['components'],torch.eye(8))['overlap'],.99999)

    def test_g5_semantics_and_budget(self):
        scores={str(i):torch.linspace(16,1,16).numpy()/136 for i in range(72)}
        ranks=count_allocate(scores)
        self.assertEqual(sum(ranks.values()),576);self.assertTrue(all(r==8 for r in ranks.values()))

    def test_exact_cost_dp(self):
        scores={'a':[.8,.3,.1],'b':[.7,.4,.2],'c':[.6,.5,.05]};costs={'a':2,'b':5,'c':3};budget=13
        ranks,report=cost_allocate(scores,costs,budget)
        reference=max(sum(sum(scores[n][:r]) for n,r in zip(scores,rs)) for rs in itertools.product(range(4),repeat=3) if sum(costs[n]*r for n,r in zip(scores,rs))<=budget)
        self.assertAlmostEqual(report['score'],reference)
        EVIDENCE['allocation_exhaustive_reference']={'ranks':ranks,**report,'brute_force_score':reference}
        scores={str(i):torch.linspace(16,1,16).numpy()/136 for i in range(72)}
        costs={str(i):1536 if i%6<4 else 3840 for i in range(72)}
        ranks,report=cost_allocate(scores,costs)
        self.assertTrue(.99<=report['fraction']<=1)
        self.assertTrue(all(0<=r<=16 for r in ranks.values()))
        EVIDENCE['full_shape_budget']=report

    def test_single_token_gradient_identity(self):
        x=torch.randn(13);g=torch.randn(5);a=torch.randn(3,13)
        actual=torch.outer(g,a@x).square().sum()
        expected=torch.trace(a@(g.square().sum()*torch.outer(x,x))@a.T)
        self.assertTrue(torch.allclose(actual,expected,rtol=1e-5))


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    out=Path(__file__).resolve().parents[2]/'outputs/gw_eva';out.mkdir(exist_ok=True,parents=True)
    (out/'math_tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'evidence':EVIDENCE,'failures':[str(x) for x in result.failures+result.errors]},indent=2),encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
