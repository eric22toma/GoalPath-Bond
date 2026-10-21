"""Focused correctness tests: python -m unittest test_elastic_net_one_period.py."""
import unittest
import tempfile
from pathlib import Path
from process_investable_universe import process_all_data
from types import SimpleNamespace
import numpy as np
import pandas as pd
from elastic_net_one_period import FEATURES, fit_forecast, bond_exposures, sven_yield, macro_features, load_panel


class PipelineTests(unittest.TestCase):
    def test_shared_preparation_and_explicit_yield_units(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            bonds=folder/'bonds.csv'; fred=folder/'fred.csv'; curve=folder/'curve.csv'
            bonds.write_text('cusip,DATE,YIELD,MATURITY,RET_EOM,DURATION,COUPON,PRICE_EOM,COUPACC,AMOUNT_OUTSTANDING,T_DVolume,NCOUPS,nextcoup,DEFAULTED,CONV,BOND_TYPE,extra\n'
              '001,2024-01-31,2.78%,2030-01-31,1.69%,5,3,100,0,1000,"$1,234",2,2024-07-31,N,0,CDEB,NA\n'
              '001,2024-02-29,2.88%,2030-01-31,-1%,5,3,100,0,1000,2000,2,2024-07-31,N,0,CDEB,keep\n')
            fred.write_text('Date,Fed_Funds_Rate,VIX,CPI\n2024-02-29,5.1,15,300\n2024-01-31,5,14,299\n')
            curve.write_text('Date,SVENY02,SVENY05,SVENY10,BETA0,BETA1,BETA2,BETA3,TAU1,TAU2,unused\n'
              '2024-01-31,2,2,2,2,0,0,0,1,2,-999.99\n2024-02-29,2,2,2,2,0,0,0,1,2,1\n')
            data=process_all_data(bonds,fred,curve)
            raw,prepared,report=data['bonds']
            self.assertEqual(raw.shape,prepared.shape)
            self.assertEqual(raw.cusip.iloc[0],'001')
            self.assertEqual(raw.extra.iloc[0],'NA')
            self.assertAlmostEqual(prepared.YIELD.iloc[0],.0278)
            self.assertAlmostEqual(prepared.RET_EOM.iloc[0],.0169)
            self.assertTrue(pd.isna(data['curve'][1].unused.iloc[0]))
            macro,params=macro_features(data['curve'][1],data['fred'][1])
            panel=load_panel(prepared,macro,params)
            self.assertAlmostEqual(panel.dy.iloc[0],.1)
            self.assertAlmostEqual(panel.rich_cheap.iloc[0],.78)
            self.assertAlmostEqual(panel.RET_EOM.iloc[0],.0169)
            self.assertAlmostEqual(panel.next_return.iloc[0],-.01)
            self.assertTrue(pd.isna(panel.dy.iloc[1]))

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
