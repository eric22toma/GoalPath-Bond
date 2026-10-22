"""Pooled elastic net forecasting next-month bond yield changes.

Each bond-month has its own predictors; the model is trained across bonds.
Forecasted changes are in percentage points, with basis-point output.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from process_investable_universe import (process_all_data, history_for_features, PROJECT_ROOT,
                                       DEFAULT_BONDS, DEFAULT_FRED, DEFAULT_CURVE)

FEATURES = ['level', 'slope', 'curvature', 'lag_level_change', 'rate_volatility',
            'fed_funds_change', 'vix', 'time_to_maturity', 'rich_cheap', 'cpi_inflation_lag1']
OPTIONAL_FEATURES = ['forward_5y']
CV_MIXING_GRID = (0.20, 0.40, 0.60, 0.80, 0.95)
CV_PENALTY_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FIRST_OOS_YEAR = 2019
LAST_OOS_YEAR = 2025


def sven_yield(tau, params):
    """Svensson continuously compounded zero yield, percentage-point units."""
    t = np.asarray(tau, dtype=float)
    b0,b1,b2,b3,t1,t2 = params
    if t1 <= 0 or t2 <= 0 or np.any(t <= 0):
        raise ValueError('Positive maturities and Svensson decay parameters required')
    x, z = t/t1, t/t2
    a, c = -np.expm1(-x)/x, -np.expm1(-z)/z
    return b0 + b1*a + b2*(a-np.exp(-x)) + b3*(c-np.exp(-z))


def macro_features(curve, fred, core=None):
    """Build monthly Treasury-curve and macroeconomic predictors."""
    daily = curve.sort_values('Date').set_index('Date')
    if daily.index.isna().any():
        raise ValueError('Curve contains invalid dates; inspect preparation report')
    if daily.index.has_duplicates:
        raise ValueError('Duplicate daily curve dates')
    # Last trading day for each month.
    monthly = daily.groupby(daily.index.to_period('M')).tail(1).copy()
    monthly['month'] = monthly.index.to_period('M')
    monthly = monthly.set_index('month')
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq='M'))
    f = pd.DataFrame(index=monthly.index)
    f['level'] = monthly.SVENY10
    f['slope'] = monthly.SVENY10-monthly.SVENY02
    f['curvature'] = 2*monthly.SVENY05-monthly.SVENY02-monthly.SVENY10
    if 'SVENF05' in monthly:
        f['forward_5y'] = monthly.SVENF05
    f['lag_level_change'] = f.level.shift(1)-f.level.shift(2)
    # Daily changes include the prior month's final trading day.
    f['rate_volatility'] = daily.SVENY10.diff().groupby(daily.index.to_period('M')).std()
    macro = fred.sort_values('Date').set_index('Date')
    if macro.index.isna().any(): raise ValueError('FRED contains invalid dates')
    macro.index = macro.index.to_period('M')
    if macro.index.has_duplicates:
        raise ValueError('Duplicate monthly FRED dates')
    macro = macro.reindex(pd.period_range(macro.index.min(), macro.index.max(), freq='M'))
    f = f.reindex(pd.period_range(f.index.min(), f.index.max(), freq='M'))
    # Lag monthly Fed funds changes for availability.
    f['fed_funds_change'] = macro.Fed_Funds_Rate.diff().shift(1)
    f['vix'] = macro.VIX
    # Check supplied YoY inflation against the CPI series.
    computed_yoy = 100*(macro.CPI/macro.CPI.shift(12)-1)
    inflation_column = next((c for c in macro if c.lower() == 'inflation_yoy'), None)
    if inflation_column is not None:
        supplied_yoy = macro[inflation_column]
        comparable = supplied_yoy.notna() & computed_yoy.notna()
        if not np.allclose(supplied_yoy[comparable], computed_yoy[comparable], atol=1e-6, rtol=1e-6):
            raise ValueError('Inflation_YoY does not match 100*(CPI/CPI.shift(12)-1); verify units')
        inflation = supplied_yoy
    else:
        inflation = computed_yoy
    # At month t, use the latest value recorded by t-1, at most two months old.
    # Filling occurs before the lag, so this never uses a future CPI release.
    f['cpi_inflation_lag1'] = inflation.ffill(limit=2).shift(1)
    if core is not None:
        core = core.sort_values('Date').set_index('Date')
        if core.index.isna().any(): raise ValueError('Core CPI contains invalid dates')
        core.index = core.index.to_period('M')
        if core.index.has_duplicates: raise ValueError('Duplicate core CPI months')
        core = core.reindex(pd.period_range(core.index.min(), core.index.max(), freq='M'))
        f['core_inflation_lag1'] = (core.CPILFESL/core.CPILFESL.shift(12)-1).shift(1)*100
    params = monthly[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']]
    # Verify Svensson parameters against the observed 10-year yield.
    check = params.dropna().loc['2016':'2025']
    error = [abs(sven_yield([10], row.to_numpy())[0]-monthly.loc[m,'SVENY10'])
             for m,row in check.iterrows()]
    if error and max(error) > 0.002:
        raise ValueError('Svensson formula does not reproduce source SVENY10')
    return f, params


def load_panel(prepared_bonds, macro, params):
    cols = ['cusip','DATE','YIELD','MATURITY','RET_EOM','DURATION','COUPON',
            'PRICE_EOM','COUPACC','AMOUNT_OUTSTANDING','T_DVolume',
            'NCOUPS','nextcoup','DEFAULTED','CONV','BOND_TYPE']
    # Input yields are decimal fractions; yield changes use percentage points.
    p = prepared_bonds.loc[:, cols].copy()
    if p.DATE.isna().any() or p.cusip.str.strip().eq('').any():
        raise ValueError('Bond panel requires valid DATE and nonblank CUSIP')
    p['month'] = p.DATE.dt.to_period('M')
    if p.duplicated(['cusip','month']).any():
        raise ValueError('Duplicate bond-month keys')
    p = p.merge(macro, left_on='month', right_index=True, how='left', validate='many_to_one')
    p['time_to_maturity'] = (p.MATURITY-p.DATE).dt.days/365.25
    p['rich_cheap'] = np.nan
    for month, rows in p.groupby('month').groups.items():
        if month not in params.index or params.loc[month].isna().any(): continue
        ix = p.loc[rows].index[p.loc[rows,'time_to_maturity']>0]
        p.loc[ix,'rich_cheap'] = 100*p.loc[ix,'YIELD']-sven_yield(p.loc[ix,'time_to_maturity'],params.loc[month])
    # Require the immediately following calendar month for realized targets.
    future = p[['cusip','month','YIELD','RET_EOM']].copy()
    future['month'] = future.month-1
    future = future.rename(columns={'YIELD':'next_yield','RET_EOM':'next_return'})
    p = p.merge(future,on=['cusip','month'],how='left',validate='one_to_one')
    p['target_month'] = p.month+1
    p['dy'] = 100*(p.next_yield-p.YIELD)  # realized one-month yield change (pp)
    return p


def estimator(alpha, l1_ratio):
    """Fit elastic net with both L1 and L2 regularization."""
    if alpha <= 0 or not 0 < l1_ratio < 1:
        raise ValueError('Elastic net requires alpha > 0 and 0 < l1_ratio < 1')
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, tol=1e-7)
    )


def build_horizon_panel(panel, horizon):
    """Create an exact h-month target for each CUSIP without bridging gaps."""
    if horizon == 1:
        return panel
    future = panel[['cusip', 'month', 'YIELD']].copy()
    future['month'] = future.month - horizon
    future = future.rename(columns={'YIELD': 'next_yield'})
    p = panel.drop(columns=['next_yield', 'next_return', 'target_month', 'dy'])
    p = p.merge(future, on=['cusip', 'month'], how='left',
                validate='one_to_one')
    p['target_month'] = p.month + horizon
    p['dy'] = 100 * (p.next_yield - p.YIELD)
    p['next_return'] = np.nan  # One-month RET_EOM is not an h-month return.
    return p


def training_rows(panel, asof, features):
    """Keep targets already realized by asof, with complete predictor rows."""
    return panel.loc[
        (panel.month >= pd.Period('2016-01', 'M')) &
        (panel.target_month <= asof)
    ].dropna(subset=features + ['dy']).copy()


def tune_elastic_net(known, features, alphas=CV_MIXING_GRID,
                     lambdas=CV_PENALTY_GRID):
    """Tune elastic net through expanding, chronological annual validation.

    Use up to three recent folds; each fold trains on earlier target months.
    """
    if known.empty:
        raise ValueError('No historical yield-change observations for tuning')
    years = sorted(set(known.target_month.dt.year))
    folds = []
    for year in years[1:]:
        first = pd.Period(f'{year}-01', 'M')
        train = known.target_month < first
        valid = known.target_month.between(first, pd.Period(f'{year}-12', 'M'))
        if known.loc[train, 'target_month'].nunique() >= 12 and valid.any():
            folds.append((train.to_numpy(), valid.to_numpy(), year))
    folds = folds[-3:]
    if not folds:
        raise ValueError('At least 12 earlier monthly targets and one later validation year are needed for tuning')

    grid = []
    for mixing in alphas:
        for penalty in lambdas:
            rmses = []
            for train, valid, _ in folds:
                m = estimator(penalty, mixing).fit(
                    known.loc[train, features], known.loc[train, 'dy']
                )
                pred = m.predict(known.loc[valid, features])
                delta = known.loc[valid, 'dy'].to_numpy() - pred
                rmses.append(float(np.sqrt(np.mean(delta ** 2)) * 100))
            grid.append({
                'alpha_mixing': float(mixing),
                'lambda_penalty': float(penalty),
                'mean_validation_rmse_bp': float(np.mean(rmses)),
            })
    scores = pd.DataFrame(grid).sort_values(
        ['mean_validation_rmse_bp', 'alpha_mixing', 'lambda_penalty']
    ).reset_index(drop=True)
    best = scores.iloc[0]
    return (float(best.alpha_mixing), float(best.lambda_penalty),
            float(best.mean_validation_rmse_bp),
            [year for _, _, year in folds], scores)


def fit_forecast(panel, asof, features, mixing, penalty):
    """Refit on an expanding window and forecast bond yields from asof."""
    known = training_rows(panel, asof, features)
    if known.empty:
        raise ValueError(f'No training targets available by {asof}')
    selection = panel.loc[panel.month == asof]
    now = selection.dropna(subset=features).copy()
    if now.empty:
        missing = {k: int(selection[k].isna().sum()) for k in features
                   if k in selection and selection[k].isna().any()}
        ready = sorted(str(x) for x in
                       panel.loc[panel[features].notna().all(axis=1),
                                 'month'].unique())
        raise ValueError(
            f'No bonds with complete predictors in {asof}. '
            f'Bond rows in month: {len(selection):,}; missing fields: {missing}; '
            f'latest usable months: {ready[-6:]}.'
        )

    model = estimator(penalty, mixing).fit(known[features], known.dy)
    now['predicted_dy_pp'] = model.predict(now[features])
    original_intercept = float(
        model[-1].intercept_ -
        np.dot(model[-1].coef_ / model[0].scale_, model[0].mean_)
    )
    coef = pd.DataFrame({
        'feature': features,
        'coefficient_per_standard_deviation': model[-1].coef_,
        'coefficient_original_units': model[-1].coef_ / model[0].scale_,
        'training_mean': model[0].mean_,
        'training_scale': model[0].scale_,
    })
    coef['intercept_standardized_inputs'] = float(model[-1].intercept_)
    coef['intercept_original_inputs'] = original_intercept
    return now, coef, len(known)


def rolling_backtest(panel, features, final_asof, horizon, start_target,
                     alphas, lambdas, fixed=None):
    """Monthly expanding-window elastic-net forecasts, without future targets.

    Retune at the first forecast month of each calendar year. Refit monthly.
    """
    first_asof = start_target - horizon
    last_asof = final_asof - horizon
    results = []
    selected_params = None
    last_tuning_year = None
    for asof in pd.period_range(first_asof, last_asof, freq='M'):
        candidates = panel.loc[panel.month == asof]
        if candidates.empty or not candidates[features].notna().all(axis=1).any():
            continue
        known = training_rows(panel, asof, features)
        if known.empty:
            continue
        if fixed is not None:
            mixing, penalty = fixed
        else:
            if selected_params is None or asof.year != last_tuning_year:
                mixing, penalty, _, _, _ = tune_elastic_net(
                    known, features, alphas, lambdas
                )
                selected_params = (mixing, penalty)
                last_tuning_year = asof.year
            mixing, penalty = selected_params

        now, _, _ = fit_forecast(panel, asof, features, mixing, penalty)
        valid = now.dropna(subset=['dy']).copy()
        if valid.empty:
            continue
        valid['forecast_error_bp'] = (valid.dy - valid.predicted_dy_pp) * 100
        valid['horizon'] = horizon
        valid['alpha_mixing'] = mixing
        valid['lambda_penalty'] = penalty
        if horizon == 1:
            # Historical return errors are retained for the optimizer's Theta.
            valid['predicted_return'] = (
                valid.COUPON / valid.PRICE_EOM / 12 -
                valid.DURATION * valid.predicted_dy_pp / 100
            )
            valid['return_error'] = valid.next_return - valid.predicted_return
        results.append(valid)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def rolling_metrics(backtest):
    if backtest.empty:
        return pd.DataFrame()
    result = backtest.copy()
    result['year'] = result.target_month.dt.year
    return result.groupby(['horizon', 'year'], as_index=False).agg(
        observations=('forecast_error_bp', 'size'),
        rmse_yield_bp=('forecast_error_bp',
                       lambda x: float(np.sqrt(np.mean(x ** 2)))),
        mae_yield_bp=('forecast_error_bp',
                      lambda x: float(np.mean(np.abs(x)))),
    )


def show_selected_bonds(now, coef, asof, cusips=None, horizon=1):
    """Show each selected bond's predictors and next-month yield forecast."""
    if cusips:
        requested = list(dict.fromkeys(cusips))
        missing = [c for c in requested if c not in set(now.cusip)]
        if missing:
            raise ValueError(f'No complete predictor row at {asof} for CUSIP(s): {missing}')
        selected = now.set_index('cusip').loc[requested].reset_index()
    else:
        selected = now.sort_values(['T_DVolume', 'cusip'],
                                   ascending=[False, True]).head(3).copy()

    beta = coef.set_index('feature').loc[FEATURES, 'coefficient_original_units']
    intercept = float(coef.intercept_original_inputs.iloc[0])
    target_month = asof + horizon
    print(f'\n{horizon}-MONTH YIELD CHANGE: {asof} -> {target_month}')
    print(f'Bonds with complete predictors: {len(now):,}; displayed: {len(selected)}.')

    summaries = []
    for row in selected.itertuples(index=False):
        start_yield_pct = row.YIELD * 100
        forecast_change_bp = row.predicted_dy_pp * 100
        forecast_yield_pct = start_yield_pct + row.predicted_dy_pp
        actual_yield_pct = row.next_yield * 100 if pd.notna(row.next_yield) else np.nan
        actual_change_bp = row.dy * 100 if pd.notna(row.dy) else np.nan
        error_bp = actual_change_bp - forecast_change_bp if pd.notna(actual_change_bp) else np.nan

        summary = {
            'cusip': row.cusip,
            'asof_month': str(asof),
            'target_month': str(target_month),
            'current_yield_percent': start_yield_pct,
            'expected_change_yield_bp': forecast_change_bp,
            'expected_next_yield_percent': forecast_yield_pct,
            'realized_next_yield_percent': actual_yield_pct,
            'realized_change_yield_bp': actual_change_bp,
            'forecast_error_bp': error_bp,
            'observed_return_asof_percent': row.RET_EOM * 100,
            'realized_next_month_return_percent': row.next_return * 100,
        }
        summaries.append(summary)

        print(f'\nCUSIP: {row.cusip}')
        print(f'  Observed yield ({asof}):        {start_yield_pct:.4f}%')
        print(f'  Forecast yield change:          {forecast_change_bp:+.2f} bp')
        print(f'  Forecast yield ({target_month}):  {forecast_yield_pct:.4f}%')

        if pd.notna(actual_yield_pct):
            np.testing.assert_allclose(
                actual_change_bp, (actual_yield_pct - start_yield_pct) * 100,
                atol=1e-8
            )
            print(f'  Actual yield ({target_month}):    {actual_yield_pct:.4f}% [evaluation only]')
            print(f'  Actual yield change:            {actual_change_bp:+.2f} bp [evaluation only]')
            print(f'  Forecast error (actual - pred): {error_bp:+.2f} bp')
        else:
            print(f'  Actual yield ({target_month}):    not available')

        values = selected.loc[selected.cusip.eq(row.cusip), FEATURES].iloc[0].astype(float)
        table = pd.DataFrame({
            'predictor': FEATURES,
            'value': values.to_numpy(),
            'beta_original_units': beta.to_numpy(),
            'contribution_bp': values.to_numpy() * beta.to_numpy() * 100,
        })
        print('  Predictors and forecast contributions:')
        print(table.to_string(index=False))
        print(f'  Intercept contribution: {intercept * 100:+.2f} bp')
        np.testing.assert_allclose(
            intercept + values.to_numpy() @ beta.to_numpy(),
            row.predicted_dy_pp, atol=1e-10
        )

    print('Realized future yields are evaluation data, never model inputs.')
    return pd.DataFrame(summaries), selected[['cusip'] + FEATURES].copy()


def main():
    parser = argparse.ArgumentParser(
        description='Monthly pooled elastic net with ordered CV and expanding-window refits'
    )
    parser.add_argument('--bonds', default=str(DEFAULT_BONDS))
    parser.add_argument('--fred', default=str(DEFAULT_FRED))
    parser.add_argument('--curve', default=str(DEFAULT_CURVE))
    parser.add_argument('--asof', default='2025-11',
                        help='Forecast origin YYYY-MM, or latest for last usable month')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Forecast direct yield changes for h=1,...,H months')
    parser.add_argument('--cusips', nargs='+',
                        help='CUSIPs to display; default shows three liquid bonds')
    parser.add_argument('--alpha', type=float, default=.5,
                        help='Elastic-net L1/L2 mixing alpha (0 < alpha < 1)')
    parser.add_argument('--lambda', dest='penalty_lambda', type=float, default=.001,
                        help='Elastic-net regularization strength lambda (> 0)')
    parser.add_argument('--no-tune', action='store_true',
                        help='Use --alpha and --lambda instead of ordered CV tuning')
    parser.add_argument('--backtest', action='store_true',
                        help='Run monthly expanding-window out-of-sample evaluation')
    parser.add_argument('--backtest-start', default='2019-01',
                        help='First out-of-sample target month (default 2019-01)')
    parser.add_argument('--output-dir', type=Path,
                        help='New directory for forecast tables and optional backtest')
    args = parser.parse_args()

    if not 1 <= args.horizon <= 12:
        parser.error('--horizon must be from 1 to 12 months')
    if not 0 < args.alpha < 1 or args.penalty_lambda <= 0:
        parser.error('Require 0 < --alpha < 1 and --lambda > 0')

    data = process_all_data(args.bonds, args.fred, args.curve)
    macro, curve_params = macro_features(
        history_for_features(data, 'curve'),
        history_for_features(data, 'fred')
    )
    months = data['bonds'][1].DATE.dt.to_period('M')
    if months.empty:
        parser.error('No aligned bond observations')
    model_months = pd.period_range(months.min(), months.max(), freq='M')
    macro = macro.reindex(model_months)
    curve_params = curve_params.reindex(model_months)
    features = FEATURES + [name for name in OPTIONAL_FEATURES
                           if name in macro.columns and macro[name].notna().any()]
    panel = load_panel(data['bonds'][1], macro, curve_params)
    del data

    complete = panel[features].notna().all(axis=1)
    eligible_months = panel.loc[complete, 'month']
    if eligible_months.empty:
        counts = {name: int(panel[name].notna().sum()) for name in features}
        raise ValueError(
            f'No complete bond predictor rows anywhere. Nonmissing values by predictor: {counts}'
        )
    asof = (eligible_months.max() if args.asof == 'latest'
            else pd.Period(args.asof, freq='M'))
    backtest_start = pd.Period(args.backtest_start, freq='M')
    print(f'Data range: {months.min()} through {months.max()}')
    print(f'Predictors ({len(features)}): {", ".join(features)}')
    print('Annual time-series CV tunes mixing alpha and penalty lambda; '
          'coefficients are refitted using all data available by each forecast date.')

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)

    alphas = (.2, .4, .6, .8)
    lambdas = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    for horizon in range(1, args.horizon + 1):
        labeled = build_horizon_panel(panel, horizon)
        known = training_rows(labeled, asof, features)
        if args.no_tune:
            mixing, penalty = args.alpha, args.penalty_lambda
            cv_info = 'fixed parameters (tuning disabled)'
        else:
            mixing, penalty, score, folds = tune_elastic_net(
                known, features, alphas, lambdas
            )
            cv_info = f'CV RMSE={score:.3f} bp; validation years={folds}'

        now, coefficients, training_count = fit_forecast(
            labeled, asof, features, mixing, penalty
        )
        print(f'\nHORIZON h={horizon}: {asof} -> {asof + horizon}')
        print(f'Training bond-month rows: {training_count:,}')
        print(f'Selected alpha (mix): {mixing:.2f}; lambda (penalty): {penalty:g}; {cv_info}')
        print('Elastic-net coefficients (original predictor units):')
        print(coefficients[['feature', 'coefficient_original_units']].to_string(index=False))
        show_selected_bonds(now, coefficients, asof, args.cusips, horizon=horizon)

        history_errors = pd.DataFrame()
        if args.backtest:
            history_errors = rolling_backtest(
                labeled, features, asof, horizon, backtest_start,
                alphas, lambdas,
                fixed=(mixing, penalty) if args.no_tune else None
            )
            if history_errors.empty:
                print('No historical backtest observations available for this horizon.')
            else:
                print('\nMONTHLY EXPANDING-WINDOW BACKTEST')
                print(rolling_metrics(history_errors).to_string(index=False))
                earliest = history_errors.sort_values(
                    ['target_month', 'cusip']
                ).head(3).copy()
                print('First three out-of-sample bond predictions:')
                earliest['predicted_change_bp'] = earliest.predicted_dy_pp * 100
                earliest['actual_change_bp'] = earliest.dy * 100
                print(earliest[['cusip', 'month', 'target_month',
                                'predicted_change_bp', 'actual_change_bp',
                                'forecast_error_bp']].to_string(index=False))

        if args.output_dir is not None:
            now.to_csv(args.output_dir / f'forecast_h{horizon}.csv', index=False)
            coefficients.to_csv(args.output_dir / f'coefficients_h{horizon}.csv',
                                index=False)
            if not history_errors.empty:
                history_errors.to_csv(
                    args.output_dir / f'backtest_h{horizon}.csv', index=False
                )
            if horizon == 1 and not history_errors.empty:
                # Compatible with the existing single-period RMVO bundle format.
                import pickle
                errors = history_errors.dropna(
                    subset=['return_error', 'predicted_return', 'next_return']
                )
                history = panel.loc[panel.month.between(asof - 35, asof)].copy()
                bundle = {
                    'asof': str(asof),
                    'target_month': str(asof + 1),
                    'snapshot': now,
                    'history': history,
                    'errors': errors,
                    'curve_params': curve_params,
                }
                with (args.output_dir / 'elastic_net_outputs.pkl').open('wb') as fh:
                    pickle.dump(bundle, fh)
    if args.output_dir is not None:
        print(f'\nForecast outputs saved to {args.output_dir.resolve()}')
    else:
        print('\nForecasts displayed; no output files saved.')


if __name__ == '__main__':
    main()
