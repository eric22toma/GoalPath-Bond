"""One-period elastic net: coefficients and next-month bond yield changes.

Imports the validated processing module unchanged. Saves one local handoff file
for robust_mvo_one_period.py; no portfolio optimization is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from process_investable_universe import (process_all_data, history_for_features, PROJECT_ROOT,
                                       DEFAULT_BONDS, DEFAULT_FRED, DEFAULT_CURVE)

FEATURES = ['level', 'slope', 'curvature', 'lag_level_change', 'rate_volatility',
            'fed_funds_change', 'vix', 'time_to_maturity', 'rich_cheap', 'cpi_inflation_lag1']


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
    """Construct predictors from shared prepared tables; never reload raw CSVs."""
    daily = curve.sort_values('Date').set_index('Date')
    if daily.index.isna().any():
        raise ValueError('Curve contains invalid dates; inspect preparation report')
    if daily.index.has_duplicates:
        raise ValueError('Duplicate daily curve dates')
    # Actual last trading-day row, rather than last nonmissing value per column.
    monthly = daily.groupby(daily.index.to_period('M')).tail(1).copy()
    monthly['month'] = monthly.index.to_period('M')
    monthly = monthly.set_index('month')
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq='M'))
    f = pd.DataFrame(index=monthly.index)
    f['level'] = monthly.SVENY10
    f['slope'] = monthly.SVENY10-monthly.SVENY02
    f['curvature'] = 2*monthly.SVENY05-monthly.SVENY02-monthly.SVENY10
    f['lag_level_change'] = f.level.shift(1)-f.level.shift(2)
    # Daily difference includes the previous trading day's close at month boundary.
    f['rate_volatility'] = daily.SVENY10.diff().groupby(daily.index.to_period('M')).std()
    macro = fred.sort_values('Date').set_index('Date')
    if macro.index.isna().any(): raise ValueError('FRED contains invalid dates')
    macro.index = macro.index.to_period('M')
    if macro.index.has_duplicates:
        raise ValueError('Duplicate monthly FRED dates')
    macro = macro.reindex(pd.period_range(macro.index.min(), macro.index.max(), freq='M'))
    f = f.reindex(pd.period_range(f.index.min(), f.index.max(), freq='M'))
    # Monthly-average Fed funds for month t may be released after month end.
    # Use t-1 minus t-2 as an explicit extra availability-delay assumption.
    f['fed_funds_change'] = macro.Fed_Funds_Rate.diff().shift(1)
    f['vix'] = macro.VIX
    # Bill approved supplied CPI in place of CPILFESL; YoY percent, lagged one month.
    # Reindexed calendar months ensure gaps do not become twelve-observation changes.
    computed_yoy = 100*(macro.CPI/macro.CPI.shift(12)-1)
    inflation_column = next((c for c in macro if c.lower() == 'inflation_yoy'), None)
    if inflation_column is not None:
        supplied_yoy = macro[inflation_column]
        comparable = supplied_yoy.notna() & computed_yoy.notna()
        if not np.allclose(supplied_yoy[comparable], computed_yoy[comparable], atol=1e-6, rtol=1e-6):
            raise ValueError('Inflation_YoY does not match 100*(CPI/CPI.shift(12)-1); verify units')
        f['cpi_inflation_lag1'] = supplied_yoy.shift(1)
    else:
        f['cpi_inflation_lag1'] = computed_yoy.shift(1)
    if core is not None:
        core = core.sort_values('Date').set_index('Date')
        if core.index.isna().any(): raise ValueError('Core CPI contains invalid dates')
        core.index = core.index.to_period('M')
        if core.index.has_duplicates: raise ValueError('Duplicate core CPI months')
        core = core.reindex(pd.period_range(core.index.min(), core.index.max(), freq='M'))
        f['core_inflation_lag1'] = (core.CPILFESL/core.CPILFESL.shift(12)-1).shift(1)*100
    params = monthly[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']]
    # Check source formula/units against the supplied 10-year zero yield.
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
    # Shared preparation contract: YIELD and RET_* are decimal fractions.
    # Preserve these units; convert yield DIFFERENCES explicitly to percentage
    # points for elastic net's target and rich/cheap feature.
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
    # Exact calendar successor; never turn a gap into a one-month target.
    future = p[['cusip','month','YIELD','RET_EOM']].copy()
    future['month'] = future.month-1
    future = future.rename(columns={'YIELD':'next_yield','RET_EOM':'next_return'})
    p = p.merge(future,on=['cusip','month'],how='left',validate='one_to_one')
    p['target_month'] = p.month+1
    p['dy'] = 100*(p.next_yield-p.YIELD)  # percentage points, not decimal yields
    return p


def estimator(kind, alpha, l1_ratio):
    models = {'elastic_net':ElasticNet(alpha=alpha,l1_ratio=l1_ratio,max_iter=50000,tol=1e-7),
              'ridge':Ridge(alpha=1.), 'lasso':Lasso(alpha=alpha,max_iter=50000,tol=1e-7)}
    return make_pipeline(StandardScaler(), models[kind])


def fit_forecast(panel, asof, features, alpha, l1_ratio):
    known = panel.loc[(panel.target_month<=asof)&(panel.month>=pd.Period('2016-01','M'))]
    known = known.dropna(subset=features+['dy']).copy()
    metrics, errors = [], []
    # Expanding training with strictly earlier target months; all bonds in a
    # calendar month remain together. Test year 2025 is never used for fitting.
    for year in [2021,2022,2023,2024]:
        start = pd.Period(f'{year}-01','M')
        train = known.loc[known.target_month<start]
        valid = known.loc[(known.target_month>=start)&(known.target_month<=min(asof,pd.Period(f'{year}-12','M')))]
        if len(train)==0 or len(valid)==0: continue
        assert train.target_month.max()<valid.target_month.min()
        for kind in ['elastic_net','ridge','lasso','no_change']:
            pred = np.zeros(len(valid)) if kind=='no_change' else estimator(kind,alpha,l1_ratio).fit(train[features],train.dy).predict(valid[features])
            metrics.append({'year':year,'model':kind,'train_rows':len(train),'validation_rows':len(valid),
                            'last_training_target':str(train.target_month.max()),
                            'first_validation_target':str(valid.target_month.min()),
                            'rmse_yield_bp':float(np.sqrt(np.mean((valid.dy-pred)**2))*100)})
            if kind=='elastic_net':
                e=valid[['cusip','month','target_month','dy','next_return','COUPON','PRICE_EOM','DURATION']].copy()
                e['predicted_dy_pp']=pred
                # Historical first-order mapping; its realized return error
                # captures coupon/roll/spread/convexity approximation error too.
                e['predicted_return']=e.COUPON/e.PRICE_EOM/12-e.DURATION*pred/100
                e['return_error']=e.next_return-e.predicted_return
                errors.append(e)
    model=estimator('elastic_net',alpha,l1_ratio).fit(known[features],known.dy)
    now=panel.loc[panel.month==asof].dropna(subset=features).copy()
    now['predicted_dy_pp']=model.predict(now[features])
    standardized_intercept=float(model[-1].intercept_)
    original_intercept=float(standardized_intercept-np.dot(model[-1].coef_/model[0].scale_,model[0].mean_))
    coef=pd.DataFrame({'feature':features,'coefficient_per_standard_deviation':model[-1].coef_,
                       'coefficient_original_units':model[-1].coef_/model[0].scale_,
                       'training_mean':model[0].mean_,'training_scale':model[0].scale_})
    coef['intercept_standardized_inputs']=standardized_intercept
    coef['intercept_original_inputs']=original_intercept
    return now,pd.DataFrame(metrics),pd.concat(errors,ignore_index=True),coef


def show_selected_bonds(now, coef, asof, cusips=None):
    """Display actual returns separately from the yield-change forecast.

    CUSIP selection affects display only, never the estimation sample.
    Next-month realized returns are ex-post evaluation, never predictors.
    """
    if cusips:
        requested=list(dict.fromkeys(cusips))
        missing=[c for c in requested if c not in set(now.cusip)]
        if missing:
            raise ValueError(f'No complete predictor row at {asof} for CUSIP(s): {missing}')
        selected=now.set_index('cusip').loc[requested].reset_index()
    else:
        selected=now.sort_values(['T_DVolume','cusip'],ascending=[False,True]).head(3).copy()
    beta=coef.set_index('feature').loc[FEATURES,'coefficient_original_units']
    intercept=float(coef.intercept_original_inputs.iloc[0])
    print('SELECTED BONDS: decision month',asof,'forecast target',asof+1)
    summaries=[]
    for row in selected.itertuples(index=False):
        print(f'\nCUSIP: {row.cusip}')
        summary={'cusip':row.cusip,'current_yield_percent':row.YIELD*100,
            'expected_change_yield_bp':row.predicted_dy_pp*100,
            'expected_next_yield_percent':row.YIELD*100+row.predicted_dy_pp,
            'observed_return_asof_percent':row.RET_EOM*100,
            'realized_next_month_return_percent':row.next_return*100}
        summaries.append(summary)
        for name,value in summary.items():
            if name!='cusip': print(f'  {name}: {value:.6f}' if pd.notna(value) else f'  {name}: unavailable')
        values=selected.loc[selected.cusip.eq(row.cusip),FEATURES].iloc[0].astype(float)
        table=pd.DataFrame({'predictor':FEATURES,'value':values.to_numpy(),
             'beta_original_units':beta.to_numpy(),
             'contribution_to_yield_change_bp':values.to_numpy()*beta.to_numpy()*100})
        print(table.to_string(index=False))
        print(f'  Intercept contribution (bp): {intercept*100:.6f}')
        np.testing.assert_allclose(intercept+values.to_numpy()@beta.to_numpy(),row.predicted_dy_pp,atol=1e-10)
    print('Returns above are observed RET_EOM values, not predicted returns.')
    print('Next-month return is ex-post evaluation only. Elastic net forecasts yield CHANGE, not total return.')
    print('Predictor units: rate levels/changes/spreads and inflation in percentage points; maturity in years; VIX in source units.')
    return pd.DataFrame(summaries),selected[ ['cusip']+FEATURES ].copy()


def main():
    parser=argparse.ArgumentParser(description='One-period pooled elastic net: coefficients and next-month yield changes')
    parser.add_argument('--bonds',default=str(DEFAULT_BONDS))
    parser.add_argument('--fred',default=str(DEFAULT_FRED))
    parser.add_argument('--curve',default=str(DEFAULT_CURVE))
    parser.add_argument('--asof',default='2024-12')
    parser.add_argument('--cusips',nargs='+',help='CUSIPs to display; default: 3 highest dollar-volume bonds. Does not restrict training.')
    parser.add_argument('--alpha',type=float,default=.001)
    parser.add_argument('--l1-ratio',type=float,default=.5)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'03_Results'/'elastic_net_outputs.pkl')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f'Choose a new output filename: {args.output}')
    data=process_all_data(args.bonds,args.fred,args.curve)
    macro,params=macro_features(history_for_features(data,'curve'),history_for_features(data,'fred'))
    months=data['bonds'][1].DATE.dt.to_period('M')
    model_months=pd.period_range(months.min(),months.max(),freq='M')
    macro=macro.reindex(model_months);params=params.reindex(model_months)
    panel=load_panel(data['bonds'][1],macro,params)
    del data
    asof=pd.Period(args.asof,'M')
    now,metrics,errors,coef=fit_forecast(panel,asof,FEATURES,args.alpha,args.l1_ratio)
    print('ELASTIC-NET COEFFICIENTS (original predictor units)')
    print(coef[['feature','coefficient_original_units']].to_string(index=False))
    print('Intercept:',coef.intercept_original_inputs.iloc[0])
    forecasts=pd.DataFrame({'cusip':now.cusip,'current_yield_percent':now.YIELD*100,
        'expected_change_yield_bp':now.predicted_dy_pp*100,
        'expected_next_yield_percent':now.YIELD*100+now.predicted_dy_pp})
    selected_returns,selected_predictors=show_selected_bonds(now,coef,asof,args.cusips)
    print('VALIDATION RMSE (yield basis points):')
    print(metrics[['year','model','rmse_yield_bp']].to_string(index=False))
    history=panel.loc[panel.month.between(asof-35,asof),['cusip','month','RET_EOM']].copy()
    bundle={'asof':str(asof),'target_month':str(asof+1),'coefficients':coef,
        'forecasts':forecasts,'displayed_bond_returns':selected_returns,'displayed_bond_predictors':selected_predictors,
        'snapshot':now.drop(columns=['next_yield','next_return','dy'],errors='ignore'),
        'history':history,'errors':errors[['cusip','target_month','return_error']],
        'curve_params':params.loc[[asof]],'validation':metrics,'monthly_predictors':macro}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.to_pickle(bundle,args.output)
    print('Saved ONE handoff file:',args.output)
    print('Prototype: revised data, unverified eligibility; validation is not evidence of profitable forecasts.')


if __name__=='__main__': main()
