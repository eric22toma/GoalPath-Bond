"""Regression checks for the GoalPath-Bond H=1 data and elastic-net pipeline."""
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from process_investable_universe import load_data
from elastic_net_one_period import (
    FEATURES, estimator, training_rows, tune_elastic_net, fit_forecast,
    rolling_backtest, rolling_metrics, limit_tuning_rows, load_panel,
    monthly_quote_coverage, trustworthy_target_month, first_backtest_target,
)


def synthetic_panel():
    months = pd.period_range("2016-01", "2020-12", freq="M")
    rows = []
    for t, month in enumerate(months):
        for j in range(5):
            rate = 0.021 + 0.00012 * t + 0.00007 * j
            next_rate = rate + 0.00012 if t < len(months) - 1 else np.nan
            rows.append({
                "cusip": f"SAMPLE{j:03d}", "month": month,
                "target_month": month + 1, "YIELD": rate,
                "next_yield": next_rate,
                "dy": 100 * (next_rate - rate),
                "next_return": 0.003, "RET_EOM": 0.003,
                "PRICE_EOM": 100., "COUPON": 3., "DURATION": 4.,
                "T_DVolume": float(j + 1), "level": 1.5 + t / 100,
                "time_to_maturity": 8. - t / 12,
            })
    return pd.DataFrame(rows)


class DataChecks(unittest.TestCase):
    def test_zipped_bonds_with_column_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "sample.csv.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("sample.csv",
                            "DATE,cusip,YIELD,unneeded\n"
                            "2020-01-31,X00001,2.50%,ignore\n")
            raw, prepared, report = load_data(
                Path(temp) / "sample.csv", usecols=["DATE", "cusip", "YIELD"]
            )
            self.assertEqual(len(raw), 1)
            self.assertEqual(list(prepared.columns), ["DATE", "cusip", "YIELD"])
            self.assertAlmostEqual(prepared.loc[0, "YIELD"], 0.025)
            self.assertIn("sample.csv.zip", report["source"])

    def test_h1_joins_only_following_calendar_month(self):
        dates = pd.to_datetime(["2020-01-31", "2020-03-31"])
        from elastic_net_one_period import BOND_COLUMNS
        bonds = pd.DataFrame(index=range(2), columns=BOND_COLUMNS)
        bonds["cusip"] = "BOND0001"
        bonds["DATE"] = dates
        bonds["MATURITY"] = pd.Timestamp("2030-01-31")
        bonds["YIELD"] = [0.03, 0.035]
        bonds["RET_EOM"] = [0.01, 0.01]
        months = pd.period_range("2020-01", "2020-03", freq="M")
        macro = pd.DataFrame(index=months, data={
            "level": [2., 2., 2.], "slope": [1., 1., 1.],
            "curvature": [0., 0., 0.], "lag_level_change": [0., 0., 0.],
            "rate_volatility": [.1, .1, .1],
            "fed_funds_change": [.1, .1, .1], "vix": [10., 10., 10.],
            "cpi_inflation_lag1": [2., 2., 2.]
        })
        params = pd.DataFrame(index=months, data={
            "BETA0": [2., 2., 2.], "BETA1": [.1, .1, .1],
            "BETA2": [.1, .1, .1], "BETA3": [.1, .1, .1],
            "TAU1": [1., 1., 1.], "TAU2": [3., 3., 3.]
        })
        prepared = load_panel(bonds, macro, params)
        self.assertTrue(prepared.dy.isna().all())
        self.assertTrue(prepared.next_yield.isna().all())

    def test_cv_tunes_only_elastic_net_with_earlier_targets(self):
        panel = synthetic_panel()
        known = training_rows(panel, pd.Period("2020-11", freq="M"),
                              ["level", "time_to_maturity"])
        self.assertTrue((known.target_month <= pd.Period("2020-11")).all())
        mix, penalty, score, years, table = tune_elastic_net(
            known, ["level", "time_to_maturity"],
            alphas=(0.2, 0.8), lambdas=(1e-3, 1e-2)
        )
        self.assertEqual(len(table), 4)
        self.assertIn(mix, (0.2, 0.8))
        self.assertIn(penalty, (1e-3, 1e-2))
        self.assertTrue(np.isfinite(score))
        self.assertTrue(years and all(y <= 2020 for y in years))
        now, coeff, count = fit_forecast(
            panel, pd.Period("2020-11", freq="M"),
            ["level", "time_to_maturity"], mix, penalty
        )
        self.assertEqual(len(now), 5)
        self.assertTrue(now.predicted_dy_pp.notna().all())
        self.assertEqual(count, len(known))
        self.assertEqual(len(coeff), 2)

    def test_monthly_backtest_has_no_future_training_targets(self):
        panel = synthetic_panel()
        out = rolling_backtest(
            panel, ["level", "time_to_maturity"],
            pd.Period("2020-11", freq="M"),
            pd.Period("2019-01", freq="M"),
            (0.2, 0.8), (1e-3,),
            fixed=(0.5, 0.001)
        )
        self.assertGreater(len(out), 0)
        self.assertTrue((out.target_month <= pd.Period("2020-11")).all())
        self.assertTrue(out.return_error.notna().all())
        self.assertTrue(np.isfinite(rolling_metrics(out).rmse_yield_bp).all())

    def test_earliest_possible_backtest_uses_prior_12_months(self):
        sample = synthetic_panel()
        initial = first_backtest_target(
            sample, ['level', 'time_to_maturity'],
            pd.Period('2018-06','M')
        )
        self.assertEqual(initial, pd.Period('2017-02','M'))

    def test_early_chronological_cv_before_annual_folds(self):
        panel = synthetic_panel()
        known = training_rows(
            panel, pd.Period('2017-01','M'),
            ['level', 'time_to_maturity']
        )
        self.assertEqual(known.target_month.nunique(), 12)
        mix, penalty, score, folds, table = tune_elastic_net(
            known, ['level','time_to_maturity'],
            alphas=(0.2,0.8), lambdas=(1e-3, 1e-2)
        )
        self.assertEqual(len(table),4)
        self.assertTrue(folds)
        self.assertTrue(all(str(m) <= '2017-01' for m in folds))

    def test_stale_target_prices_trigger_quality_warning(self):
        sample = synthetic_panel()
        sample['PRICE_EOM'] = 100.0
        flat = monthly_quote_coverage(sample)
        self.assertFalse(trustworthy_target_month(flat, pd.Period('2020-06','M')))

        sample['PRICE_EOM'] = (
            100.0 + (sample.month.astype('int64') -
                     sample.month.astype('int64').min())*0.1
        )
        changed = monthly_quote_coverage(sample)
        # Only five synthetic bonds: use direct share to test the computed statistic.
        self.assertEqual(changed.loc[pd.Period('2020-06','M'), 'price_pairs'],5)
        self.assertAlmostEqual(changed.loc[pd.Period('2020-06','M'),'price_move_share'],1.0)


    def test_tuning_limit_does_not_drop_months(self):
        sample = synthetic_panel()
        bounded = limit_tuning_rows(sample, limit=2)
        self.assertEqual(bounded.groupby("month").size().max(), 2)
        self.assertEqual(bounded.month.nunique(), sample.month.nunique())


if __name__ == "__main__":
    unittest.main()
