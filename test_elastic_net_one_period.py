"""Focused correctness tests: python -m unittest test_elastic_net_one_period.py."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from elastic_net_one_period import FEATURES, fit_forecast, bond_exposures, sven_yield


class PipelineTests(unittest.TestCase):
    def test_future_labels_cannot_change_fit_or_forecast(self):
        rng=np.random.default_rng(7)
        months=pd.period_range('2016-01','2025-12',freq='M')
        p=pd.DataFrame([(str(i),m) for m in months for i in range(3)],columns=['cusip','month'])
        for name in FEATURES: p[name]=rng.normal(size=len(p))
        p['target_month']=p.month+1
        p['dy']=.1*p.level+.01*rng.normal(size=len(p))
        p['next_return']=.003-p.dy/100
        p['COUPON']=4.;p['PRICE_EOM']=100.;p['DURATION']=1.
        asof=pd.Period('2024-12','M')
        a=fit_forecast(p,asof,FEATURES,.001,.5)
        changed=p.copy()
        changed.loc[changed.target_month>asof,['dy','next_return']]=1e6
        b=fit_forecast(changed,asof,FEATURES,.001,.5)
        np.testing.assert_allclose(a[0].predicted_dy_pp,b[0].predicted_dy_pp)
        np.testing.assert_allclose(a[3].coefficient_original_units,b[3].coefficient_original_units)
        for row in a[1].itertuples():
            self.assertLess(row.last_training_target,row.first_validation_target)

    def test_exposure_partition_and_positive_convexity(self):
        row=SimpleNamespace(NCOUPS=2,MATURITY=pd.Timestamp('2030-12-31'),
             DATE=pd.Timestamp('2024-12-31'),COUPON=4.,PRICE_EOM=100.,COUPACC=0.)
        D,C,krd=bond_exposures(row,np.array([4.,0,0,0,1.,2.]))
        self.assertGreater(D,0);self.assertGreater(C,0)
        self.assertAlmostEqual(D,krd.sum(),places=5)
        self.assertTrue(np.all(krd>=0))
        np.testing.assert_allclose(sven_yield([1.,5.,10.],[4,0,0,0,1,2]),[4,4,4])


if __name__=='__main__': unittest.main()
