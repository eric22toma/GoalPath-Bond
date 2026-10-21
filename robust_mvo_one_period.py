"""One-period mixed-integer robust MVO, Deliverable 2 section 4.1.

Reads the single saved elastic-net output file. DOES NOT fit elastic net or reload raw data.
Run: python robust_mvo_one_period.py results/elastic_net_outputs.pkl --output new_folder

Return mapping: mu_i = carry_i - D_i*dy_i + .5*C_i*dy_i**2 (dy decimal).
Minimize gamma*(w-b)'Q(w-b) - [mu'w-epsilon_mu*||sqrt(Theta)w||]
         + kappa_buy'u_plus + kappa_sell'u_minus.
Full investment, trade balance, long-only/binary cardinality, duration,
convexity and key-rate exposure bands, are implemented below. Investor wealth/required-return constraints are excluded.
The benchmark/risk-analytics prototype limitations remain in force.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from elastic_net_one_period import sven_yield
from scipy.optimize import brentq
from sklearn.covariance import LedoitWolf

KEYS = np.array([1., 2., 5., 10., 20., 30.])


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
    if not 0 < args.max_weight <= 1 or args.max_holdings*args.max_weight < 1:
        raise ValueError('max_weight must be in (0,1] and max_holdings*max_weight >= 1')
    if args.gamma < 0 or args.epsilon_mu <= 0 or args.trade_cost < 0:
        raise ValueError('Require gamma >= 0, epsilon_mu > 0, trade_cost >= 0')
    m=gp.Model('Deliverable2_one_period_no_goal'); m.Params.OutputFlag=0; m.Params.TimeLimit=60; m.Params.MIPGap=1e-5
    w=m.addMVar(n,lb=0,ub=args.max_weight,name='w'); buy=m.addMVar(n,lb=0,name='u_plus'); sell=m.addMVar(n,lb=0,name='u_minus')
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
    te=(w-b)@Q@(w-b); tc=args.trade_cost*(buy.sum()+sell.sum())
    tracking_variance=m.addVar(lb=0,name='tracking_variance')
    m.addConstr(te<=tracking_variance,name='tracking_variance_epigraph')
    m.setObjective(args.gamma*tracking_variance-robust+tc,GRB.MINIMIZE)
    m.optimize()
    candidates['forecast_return']=mu; candidates['estimated_duration']=D; candidates['estimated_convexity']=C
    candidates['theta_return_variance']=theta; candidates['benchmark_proxy_weight']=b
    for j,k in enumerate(KEYS): candidates[f'estimated_krd_{k:g}']=KRD[j]
    info={'solver_status':m.Status,'solution_count':m.SolCount,'wealth_goal_enabled':False,'max_bond_weight':args.max_weight,
          'candidate_count':n,'benchmark':'normalized amount outstanding times clean price of selected candidate subset; NOT AGG',
          'current_portfolio':'same as proxy benchmark, illustrative', 'portfolio_feasible':False}
    if m.SolCount:
        weights=w.X
        candidates['weight']=weights; candidates['buy']=buy.X; candidates['sell']=sell.X
        true_robust=float(mu@weights-args.epsilon_mu*np.sqrt(theta@(weights**2)))
        checks={'budget_error':abs(weights.sum()-1),'holdings':int((weights>1e-6).sum()),
                'duration_deviation':float(abs(D@(weights-b))), 'convexity_deviation':float(abs(C@(weights-b))),
                'maximum_krd_deviation':float(np.max(abs(KRD@(weights-b)))),
                'maximum_bond_weight':float(weights.max()),
                'trade_balance_error':float(np.max(abs(weights-current-buy.X+sell.X)))}
        assert checks['budget_error']<1e-5 and checks['holdings']<=args.max_holdings
        assert checks['maximum_bond_weight']<=args.max_weight+1e-5 and checks['trade_balance_error']<1e-5
        assert checks['duration_deviation']<=args.duration_tolerance+1e-5
        assert checks['convexity_deviation']<=args.convexity_tolerance+1e-5
        assert checks['maximum_krd_deviation']<=args.krd_tolerance+1e-5
        info.update(portfolio_feasible=True,optimal=m.Status==GRB.OPTIMAL,mip_gap=m.MIPGap,
                    nominal_monthly_return=float(mu@weights),robust_monthly_return=true_robust,
                    monthly_tracking_error=float(np.sqrt(max(0,(weights-b)@Q@(weights-b)))),
                    transaction_cost_fraction=float(tc.getValue()),turnover=float(0.5*np.sum(buy.X+sell.X)),checks=checks)
    elif m.Status==GRB.INFEASIBLE:
        m.computeIIS(); m.write(str(Path(args.output)/'infeasible.ilp'))
        info['infeasible_constraints']=[c.ConstrName for c in m.getConstrs() if c.IISConstr]
    return candidates,info,Q,theta


def load_forecast_bundle(path):
    """Read the locally generated elastic-net output; no raw data reload/refit."""
    b=pd.read_pickle(path)
    asof=pd.Period(b['asof'],'M')
    if pd.Period(b['target_month'],'M')!=asof+1: raise ValueError('Expected one-month forecast horizon')
    now=b['snapshot'].copy();history=b['history'].copy();errors=b['errors'].copy()
    if now.cusip.duplicated().any() or not now.month.eq(asof).all(): raise ValueError('Invalid forecast snapshot')
    if not np.isfinite(now.predicted_dy_pp).all(): raise ValueError('Missing yield forecasts')
    if not history.month.between(asof-35,asof).all(): raise ValueError('History exceeds as-of window')
    if not (pd.PeriodIndex(errors.target_month,freq='M')<=asof).all(): raise ValueError('Future uncertainty observations')
    return b,now,history,errors,b['curve_params']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('forecast_file',type=Path,help='elastic_net_outputs.pkl generated locally by elastic net')
    p.add_argument('--output',type=Path,required=True,help='NEW output directory')
    for name,kind in [('max-weight',float),('candidates',int),('max-holdings',int),
                      ('gamma',float),('epsilon-mu',float),('trade-cost',float),
                      ('duration-tolerance',float),('convexity-tolerance',float),('krd-tolerance',float)]:
        p.add_argument('--'+name,type=kind,default=None)
    cli=p.parse_args()
    manifest,now,history,errors,params=load_forecast_bundle(cli.forecast_file)
    settings=dict(candidates=40,max_holdings=10,gamma=10.,epsilon_mu=.5,trade_cost=.001,
                  duration_tolerance=.5,convexity_tolerance=10.,krd_tolerance=.5,max_weight=.20)
    for name,value in vars(cli).items():
        if value is not None and name not in ['forecast_file','output']: settings[name]=value
    settings['asof']=manifest['asof'];settings['output']=str(cli.output)
    cli.output.mkdir(parents=True,exist_ok=False)
    holdings,result,Q,theta=solve_period(now,history,errors,params,SimpleNamespace(**settings))
    holdings.to_csv(cli.output/'portfolio.csv',index=False)
    report={'forecast_source':str(cli.forecast_file.resolve()),'asof':manifest['asof'],
            'target_month':manifest['target_month'],'settings':settings,'optimization':result,
            'limitations':['Candidate-subset proxy benchmark, not AGG.',
             'Estimated coupon-bond risk analytics; eligibility and release vintages unverified.',
             'No goal-achievement probability from this single-period solve.',
             'No target-period realized returns used or evaluated by this solver.']}
    (cli.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ONE-PERIOD ROBUST MVO — no wealth goal; default maximum bond weight 20%')
    print(json.dumps(result,indent=2))
    print('Selected positive-weight holdings:')
    if 'weight' in holdings:
        print(holdings.loc[holdings.weight>1e-6,['cusip','predicted_dy_pp','forecast_return','weight','buy','sell']].to_string(index=False))


if __name__=='__main__': main()
