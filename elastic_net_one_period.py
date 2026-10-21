"""Historical prototype: pooled elastic net -> Deliverable 2, section 4.1 solve.

Run from the data folder: python elastic_net_one_period.py
Optional --core-cpi CSV must contain Date,CPILFESL (at least 12 prior months).
Data are revised snapshots: this is NOT a point-in-time verified backtest.
No source files are changed. Outputs go to a new --output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.covariance import LedoitWolf
from process_investable_universe import process_all_data

FEATURES = ['level', 'slope', 'curvature', 'lag_level_change', 'rate_volatility',
            'fed_funds_change', 'vix', 'time_to_maturity', 'rich_cheap']
KEYS = np.array([1., 2., 5., 10., 20., 30.])


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


def bond_exposures(row, params):
    """Approximate regular coupon cash flows, curve+constant-spread valuation.

    Face=100; coupon in percent; calendar schedule from nextcoup backwards
    from maturity. Actual day-count, irregular coupons and options are not
    modeled. These are estimated exposures, NOT source-verified bond analytics.
    """
    freq=int(row.NCOUPS)
    if freq not in [1,2,4,12]: raise ValueError('Unsupported coupon frequency')
    date=row.MATURITY
    dates=[]
    while date>row.DATE:
        dates.append(date)
        date=date-pd.DateOffset(months=12//freq)
    times=np.array([(d-row.DATE).days/365.25 for d in dates])
    cf=np.full(len(times),row.COUPON/freq); cf[0]+=100
    zero=sven_yield(times,params)/100
    dirty=float(row.PRICE_EOM+row.COUPACC)
    if dirty<=0: raise ValueError('Nonpositive dirty price')
    price=lambda spread,bump: np.sum(cf*np.exp(-(zero+spread+bump)*times))
    spread=brentq(lambda s:price(s,0)-dirty,-0.5,5.)
    h=0.0001
    base=price(spread,0)
    d=(price(spread,-h)-price(spread,h))/(2*h*base)
    c=(price(spread,-h)+price(spread,h)-2*base)/(h*h*base)
    hat=np.array([np.interp(times,KEYS,np.eye(len(KEYS))[k]) for k in range(len(KEYS))])
    krd=np.array([(price(spread,-h*hat[k])-price(spread,h*hat[k]))/(2*h*base) for k in range(len(KEYS))])
    return d,c,krd


def solve_period(now,panel,errors,params,args):
    import gurobipy as gp
    from gurobipy import GRB
    asof=pd.Period(args.asof,'M')
    history=panel.loc[(panel.month>=asof-35)&(panel.month<=asof)]
    returns=history.pivot(index='month',columns='cusip',values='RET_EOM').reindex(pd.period_range(asof-35,asof,freq='M'))
    # Historical completeness uses only pre-decision data, never future returns.
    complete=returns.columns[returns.notna().all()]
    candidates=now.loc[now.cusip.isin(complete)&(now.time_to_maturity>1/12)&
                       now.DEFAULTED.eq('N')&now.CONV.eq(0)].copy()
    candidates=candidates.dropna(subset=['PRICE_EOM','COUPACC','COUPON','NCOUPS','AMOUNT_OUTSTANDING','T_DVolume'])
    candidates=candidates.loc[(candidates.PRICE_EOM>0)&(candidates.AMOUNT_OUTSTANDING>0)&
                              candidates.NCOUPS.isin([1,2,4,12])]
    # Deterministic as-of liquidity screen. Callable status is NOT supplied;
    # eligibility for the final conventional-bond model remains unverified.
    candidates=candidates.sort_values(['T_DVolume','cusip'],ascending=[False,True]).head(args.candidates).copy()
    if len(candidates)<args.max_holdings: raise ValueError('Too few candidates')
    exposures=[bond_exposures(r,params.loc[asof].to_numpy()) for r in candidates.itertuples()]
    D=np.array([e[0] for e in exposures]); C=np.array([e[1] for e in exposures]); KRD=np.stack([e[2] for e in exposures],axis=1)
    ids=candidates.cusip.tolist(); n=len(ids)
    b=(candidates.AMOUNT_OUTSTANDING*candidates.PRICE_EOM).to_numpy(); b=b/b.sum()
    current=b.copy()  # Illustrative existing portfolio, not an initial cash purchase.
    carry=(candidates.COUPON/(candidates.PRICE_EOM+candidates.COUPACC)/12).to_numpy()
    dy=candidates.predicted_dy_pp.to_numpy()/100
    mu=carry-D*dy+0.5*C*dy**2
    Q=LedoitWolf().fit(returns[ids].to_numpy()).covariance_
    # Theta uses genuine earlier fixed-configuration validation predictions.
    variances=errors.groupby('cusip').return_error.agg(['var','count'])
    pooled=float(errors.return_error.var())
    theta=np.array([variances.loc[i,'var'] if i in variances.index and variances.loc[i,'count']>=12 else pooled for i in ids])
    theta=np.maximum(theta,1e-12)
    required=(1+args.annual_goal)**(1/12)-1  # Wt=W0, t=0, Y=1, T=12
    m=gp.Model('Deliverable2_one_period'); m.Params.OutputFlag=0; m.Params.TimeLimit=60; m.Params.MIPGap=1e-5
    w=m.addMVar(n,lb=0,ub=1,name='w'); buy=m.addMVar(n,lb=0,name='u_plus'); sell=m.addMVar(n,lb=0,name='u_minus')
    selected=m.addMVar(n,vtype=GRB.BINARY,name='delta'); eta=m.addVar(lb=0,name='eta')
    # SOC epigraph of ||Theta^(1/2) w||. The objective penalizes eta;
    # epsilon_mu > 0 forces tightness at the optimum.
    m.addConstr(w.sum()==1,name='full_investment')
    m.addConstr(w-current==buy-sell,name='trade_definition')
    m.addConstr(w<=selected,name='selection_link'); m.addConstr(selected.sum()<=args.max_holdings,name='cardinality')
    m.addConstr(gp.quicksum(theta[i]*w[i]*w[i] for i in range(n))<=eta*eta,name='robust_norm')
    for name,x,tol in [('duration',D,args.duration_tolerance),('convexity',C,args.convexity_tolerance)]+[(f'krd_{k:g}',KRD[j],args.krd_tolerance) for j,k in enumerate(KEYS)]:
        m.addConstr(x@w-x@b<=tol,name=name+'_upper'); m.addConstr(x@b-x@w<=tol,name=name+'_lower')
    robust=mu@w-args.epsilon_mu*eta
    m.addConstr(robust>=required,name='required_robust_return')
    te=(w-b)@Q@(w-b); tc=args.trade_cost*(buy.sum()+sell.sum())
    m.setObjective(args.gamma*te-robust+tc,GRB.MINIMIZE)
    m.optimize()
    candidates['forecast_return']=mu; candidates['estimated_duration']=D; candidates['estimated_convexity']=C
    candidates['theta_return_variance']=theta; candidates['benchmark_proxy_weight']=b
    for j,k in enumerate(KEYS): candidates[f'estimated_krd_{k:g}']=KRD[j]
    info={'solver_status':m.Status,'solution_count':m.SolCount,'required_monthly_return':required,'annual_goal':args.annual_goal,
          'candidate_count':n,'benchmark':'normalized amount outstanding times clean price of selected candidate subset; NOT AGG',
          'current_portfolio':'same as proxy benchmark, illustrative', 'goal_feasible':False}
    if m.SolCount:
        weights=w.X
        candidates['weight']=weights; candidates['buy']=buy.X; candidates['sell']=sell.X
        true_robust=float(mu@weights-args.epsilon_mu*np.sqrt(theta@(weights**2)))
        checks={'budget_error':abs(weights.sum()-1),'holdings':int((weights>1e-6).sum()),
                'duration_deviation':float(abs(D@(weights-b))), 'convexity_deviation':float(abs(C@(weights-b))),
                'maximum_krd_deviation':float(np.max(abs(KRD@(weights-b)))),
                'robust_goal_slack':true_robust-required,
                'trade_balance_error':float(np.max(abs(weights-current-buy.X+sell.X)))}
        assert checks['budget_error']<1e-5 and checks['holdings']<=args.max_holdings
        assert checks['robust_goal_slack']>=-1e-5 and checks['trade_balance_error']<1e-5
        assert checks['duration_deviation']<=args.duration_tolerance+1e-5
        assert checks['convexity_deviation']<=args.convexity_tolerance+1e-5
        assert checks['maximum_krd_deviation']<=args.krd_tolerance+1e-5
        info.update(goal_feasible=True,optimal=m.Status==GRB.OPTIMAL,mip_gap=m.MIPGap,
                    nominal_monthly_return=float(mu@weights),robust_monthly_return=true_robust,
                    monthly_tracking_error=float(np.sqrt(max(0,(weights-b)@Q@(weights-b)))),
                    transaction_cost_fraction=float(tc.getValue()),turnover=float(0.5*np.sum(buy.X+sell.X)),checks=checks)
        # Ex-post observation is accessed ONLY AFTER solve. Never used for candidate selection.
        actual=panel.loc[panel.month==asof,['cusip','next_return']].set_index('cusip').reindex(ids).next_return
        info['realized_return_coverage']=int(actual.notna().sum())
        if actual.notna().all():
            rp=float(actual.to_numpy()@weights); rb=float(actual.to_numpy()@b)
            info.update(realized_portfolio_return=rp,realized_proxy_return=rb,
                        realized_active_return=rp-rb,
                        realized_wealth_after_cost=args.wealth*(1-info['transaction_cost_fraction'])*(1+rp))
    elif m.Status==GRB.INFEASIBLE:
        m.computeIIS(); m.write(str(Path(args.output)/'infeasible.ilp'))
        info['infeasible_constraints']=[c.ConstrName for c in m.getConstrs() if c.IISConstr]
    return candidates,info,Q,theta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bonds',default='Investable_universe_AGG.csv'); p.add_argument('--fred',default='FRED_Monthly_2015_2025.csv')
    p.add_argument('--curve',default='federal_reserve_nominal_yield_curve (1).csv'); p.add_argument('--core-cpi')
    p.add_argument('--asof',default='2024-12'); p.add_argument('--alpha',type=float,default=.001); p.add_argument('--l1-ratio',type=float,default=.5)
    p.add_argument('--annual-goal',type=float,default=.03); p.add_argument('--wealth',type=float,default=100000)
    p.add_argument('--candidates',type=int,default=40); p.add_argument('--max-holdings',type=int,default=10)
    p.add_argument('--gamma',type=float,default=10); p.add_argument('--epsilon-mu',type=float,default=.5)
    p.add_argument('--trade-cost',type=float,default=.001); p.add_argument('--duration-tolerance',type=float,default=.5)
    p.add_argument('--convexity-tolerance',type=float,default=10); p.add_argument('--krd-tolerance',type=float,default=.5)
    p.add_argument('--forecast-only',action='store_true',help='Stop after predictor construction, beta estimation, and validation')
    p.add_argument('--output',default='outputs/elastic_net_shared_processing')
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    # 1. Prepare every source in one module before constructing predictors.
    data=process_all_data(args.bonds,args.fred,args.curve,args.core_cpi)
    preparation={name: values[2] for name,values in data.items()}
    (out/'preparation_report.json').write_text(json.dumps(preparation,indent=2)+'\n')
    # 2. Build dated predictors from the prepared tables.
    macro,params=macro_features(data['curve'][1],data['fred'][1],
                               data['core_cpi'][1] if 'core_cpi' in data else None)
    features=FEATURES+(['core_inflation_lag1'] if args.core_cpi else [])
    panel=load_panel(data['bonds'][1],macro,params)
    del data  # Release raw-cell copies after constructing the model view.
    print('Loaded panel',panel.shape,flush=True)
    now,metrics,errors,coef=fit_forecast(panel,pd.Period(args.asof,'M'),features,args.alpha,args.l1_ratio)
    print('Forecast fits complete. Beta coefficients (original predictor units):',flush=True)
    print(coef[['feature','coefficient_original_units']].to_string(index=False),flush=True)
    print('Intercept:',coef.intercept_original_inputs.iloc[0],flush=True)
    metrics.to_csv(out/'forecast_validation.csv',index=False); coef.to_csv(out/'coefficients.csv',index=False)
    macro.to_csv(out/'monthly_predictors.csv'); errors.to_csv(out/'historical_oos_errors.csv',index=False)
    now[['cusip','month','target_month']+features+['predicted_dy_pp']].to_csv(out/'asof_predictors_and_forecasts.csv',index=False)
    if args.forecast_only:
        result={'status':'not_run','reason':'forecast-only requested'}
    else:
        holdings,result,Q,theta=solve_period(now,panel,errors,params,args)
        holdings.to_csv(out/'portfolio.csv',index=False)
        np.savez(out/'risk_inputs.npz',Q=Q,theta=theta,bond_ids=holdings.cusip.to_numpy(dtype=str))
    source_paths=[args.bonds,args.fred,args.curve]+([args.core_cpi] if args.core_cpi else [])
    manifest={str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_paths}
    report={'settings':vars(args),'features':features,'missing_requested_predictors':[] if args.core_cpi else ['lagged CPILFESL core inflation'],
            'input_sha256':manifest,'preparation':preparation,
            'beta_equation':'predicted_dy_pp = intercept_original_inputs + sum(coefficient_original_units * original_feature)','forecast_structure':'pooled shared slopes and common intercept; no bond-specific intercepts in configuration 1',
            'hyperparameters':'one prespecified configuration; expanding-year validation, NOT a tuned optimum',
            'limitations':['Historical revised snapshots; releases/vintages not verified. VIX month-end semantics not source-verified.',
             'Fed funds uses one additional month of lag for availability; differs from meeting formula.',
             'Core CPI omitted unless explicitly supplied; headline CPI never silently substituted.',
             'Callable/embedded-option and source conventional-bond eligibility not verified; CONV=0 only screens convertibility.',
             'Proxy benchmark and illustrative existing holdings; no claims of AGG replication.',
             'Estimated regular-coupon cash flows, curve+constant spread, approximate day counts and schedule.',
             'Carry is coupon yield/12; no exact coupon/roll cash accounting.',
             'Historical OOS uncertainty uses first-order return mapping, final uses estimated convexity.',
             'Costs penalized in objective per Deliverable 2; ex-post wealth deduction is explicitly illustrative.',
             'No terminal wealth distribution or goal probability calculated from one rebalance.'],
            'optimization':result}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
