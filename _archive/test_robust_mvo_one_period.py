import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from elastic_net_one_period import save_forecast_outputs
from robust_mvo_one_period import load_forecast_bundle


class ForecastHandoffTests(unittest.TestCase):
    def test_yield_units_bundle_dates_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            asof=pd.Period('2024-12','M')
            now=pd.DataFrame({'cusip':['001'],'month':[asof],
                'DATE':[pd.Timestamp('2024-12-31')], 'MATURITY':[pd.Timestamp('2030-12-31')],
                'nextcoup':[pd.Timestamp('2025-06-30')],
                'YIELD':[.05],'predicted_dy_pp':[-.1],
                'next_yield':[.049], 'next_return':[999.], 'dy':[-.1]})
            panel=pd.DataFrame({'cusip':['001','001'],'month':[asof-1,asof], 'RET_EOM':[.01,.02]})
            errors=pd.DataFrame({'cusip':['001'],'target_month':[asof],'return_error':[.001]})
            errors.to_csv(folder/'historical_oos_errors.csv',index=False)
            params=pd.DataFrame([[4.,0.,0.,0.,1.,2.]],
                   columns=['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2'],index=[asof])
            save_forecast_outputs(folder,now,panel,errors,params,SimpleNamespace(asof='2024-12'))
            f=pd.read_csv(folder/'yield_change_forecasts.csv')
            self.assertAlmostEqual(f.forecast_change_yield_basis_points.iloc[0],-10)
            self.assertAlmostEqual(f.forecast_change_yield_decimal.iloc[0],-.001)
            self.assertAlmostEqual(f.forecast_next_yield_decimal.iloc[0],.049)
            manifest,snapshot,history,err,curve=load_forecast_bundle(folder)
            self.assertEqual(snapshot.cusip.iloc[0],'001')
            self.assertNotIn('next_return',snapshot)
            self.assertTrue(history.next_return.isna().all())
            with (folder/'yield_change_forecasts.csv').open('a') as handle: handle.write('\n')
            with self.assertRaises(ValueError): load_forecast_bundle(folder)


if __name__=='__main__': unittest.main()
