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


def fit_forecast(panel, asof, features, alpha, l1_ratio):
    """Train elastic net and evaluate earlier annual out-of-sample periods."""
    model_factory = estimator(alpha, l1_ratio)
    known = panel.loc[
        (panel.target_month <= asof) &
        (panel.month >= pd.Period('2016-01', 'M'))
    ].dropna(subset=features + ['dy']).copy()
    if known.empty:
        raise ValueError('No complete training observations before the forecast month')

    metrics, errors = [], []
    # Start evaluation in 2019; refit annually with expanding training history.
    for year in range(FIRST_OOS_YEAR, min(LAST_OOS_YEAR, asof.year) + 1):
        first = pd.Period(f'{year}-01', 'M')
        last = min(asof, pd.Period(f'{year}-12', 'M'))
        train = known.loc[known.target_month < first]
        valid = known.loc[known.target_month.between(first, last)]
        if train.empty or valid.empty:
            continue
        assert train.target_month.max() < valid.target_month.min()

        predictor = estimator(alpha, l1_ratio).fit(train[features], train.dy)
        pred = predictor.predict(valid[features])
        metrics.append({
            'year': year, 'model': 'elastic_net', 'train_rows': len(train),
            'validation_rows': len(valid),
            'last_training_target': str(train.target_month.max()),
            'first_validation_target': str(valid.target_month.min()),
            'rmse_yield_bp': float(np.sqrt(np.mean((valid.dy - pred)**2)) * 100),
        })

        e = valid[['cusip', 'month', 'target_month', 'dy', 'next_return',
                   'COUPON', 'PRICE_EOM', 'DURATION']].copy()
        e['predicted_dy_pp'] = pred
        # Approximate return forecast for subsequent portfolio risk estimation.
        e['predicted_return'] = e.COUPON / e.PRICE_EOM / 12 - e.DURATION * pred / 100
        e['return_error'] = e.next_return - e.predicted_return
        errors.append(e)

    model = model_factory.fit(known[features], known.dy)
    now = panel.loc[panel.month == asof].dropna(subset=features).copy()
    if now.empty:
        raise ValueError(f'No bonds with complete predictors in {asof}')
    now['predicted_dy_pp'] = model.predict(now[features])
    standardized_intercept = float(model[-1].intercept_)
    original_intercept = float(
        standardized_intercept -
        np.dot(model[-1].coef_ / model[0].scale_, model[0].mean_)
    )
    coef = pd.DataFrame({
        'feature': features,
        'coefficient_per_standard_deviation': model[-1].coef_,
        'coefficient_original_units': model[-1].coef_ / model[0].scale_,
        'training_mean': model[0].mean_,
        'training_scale': model[0].scale_,
    })
    coef['intercept_standardized_inputs'] = standardized_intercept
    coef['intercept_original_inputs'] = original_intercept
    return (now, pd.DataFrame(metrics),
            pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(),
            coef)


def show_selected_bonds(now, coef, asof, cusips=None):
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
    target_month = asof + 1
    print(f'\nMONTHLY YIELD CHANGE: {asof} -> {target_month}')
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

    print('Actual next-month yields are evaluation data, never model inputs.')
    return pd.DataFrame(summaries), selected[['cusip'] + FEATURES].copy()


def main():
    parser=argparse.ArgumentParser(description='One-period pooled elastic net: coefficients and next-month yield changes')
    parser.add_argument('--bonds',default=str(DEFAULT_BONDS))
    parser.add_argument('--fred',default=str(DEFAULT_FRED))
    parser.add_argument('--curve',default=str(DEFAULT_CURVE))
    parser.add_argument('--asof',default='2025-11')
    parser.add_argument('--cusips',nargs='+',help='CUSIPs to display; default: 3 highest dollar-volume bonds. Does not restrict training.')
    parser.add_argument('--alpha',type=float,default=.001)
    parser.add_argument('--l1-ratio',type=float,default=.5)
    args=parser.parse_args()
    data=process_all_data(args.bonds,args.fred,args.curve)
    macro,params=macro_features(history_for_features(data,'curve'),history_for_features(data,'fred'))
    months=data['bonds'][1].DATE.dt.to_period('M')
    model_months=pd.period_range(months.min(),months.max(),freq='M')
    macro=macro.reindex(model_months);params=params.reindex(model_months)
    panel=load_panel(data['bonds'][1],macro,params)
    del data
    asof=pd.Period(args.asof,'M')
    now,metrics,errors,coef=fit_forecast(panel,asof,FEATURES,args.alpha,args.l1_ratio)
    print('ELASTIC-NET COEFFICIENTS (yield change in percentage points)')
    print(coef[['feature','coefficient_original_units']].to_string(index=False))
    print('Intercept:',coef.intercept_original_inputs.iloc[0])
    show_selected_bonds(now,coef,asof,args.cusips)
    print(f'\nOUT-OF-SAMPLE ELASTIC-NET VALIDATION ({FIRST_OOS_YEAR}–{LAST_OOS_YEAR})')
    print('Expanding training history; coefficients refitted at each year start.')
    if metrics.empty:
        print('No historical validation periods before the requested as-of month.')
    else:
        print(metrics[['year', 'train_rows', 'validation_rows', 'rmse_yield_bp']].to_string(index=False))
        early = errors.sort_values(['target_month', 'cusip']).head(3).copy()
        early['forecast_change_bp'] = early.predicted_dy_pp * 100
        early['actual_change_bp'] = early.dy * 100
        print('\nFIRST THREE HISTORICAL OUT-OF-SAMPLE BOND FORECASTS')
        print(early[['cusip', 'month', 'target_month', 'forecast_change_bp',
                     'actual_change_bp']].to_string(index=False))
    print('Forecasts displayed; no output files saved.')


if __name__=='__main__': main()
