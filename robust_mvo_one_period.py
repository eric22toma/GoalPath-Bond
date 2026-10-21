"""One-period mixed-integer robust MVO, Deliverable 2 section 4.1.

Reads an elastic-net output folder. DOES NOT fit elastic net or reload raw data.
Run: python robust_mvo_one_period.py path/to/forecast_folder --output new_folder

Return mapping: mu_i = carry_i - D_i*dy_i + .5*C_i*dy_i**2 (dy decimal).
Minimize gamma*(w-b)'Q(w-b) - [mu'w-epsilon_mu*||sqrt(Theta)w||]
         + kappa_buy'u_plus + kappa_sell'u_minus.
Full investment, trade balance, long-only/binary cardinality, duration,
convexity and key-rate exposure bands, and robust required-return constraint
are implemented by the shared solve_period function, not a second formulation.
The benchmark/risk-analytics prototype limitations remain in force.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from elastic_net_one_period import solve_period


def load_forecast_bundle(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'forecast_manifest.json').read_text())
    required={'yield_change_forecasts.csv','optimizer_snapshot.csv','optimizer_return_history.csv',
              'optimizer_curve_params.json','historical_oos_errors.csv'}
    if not required.issubset(manifest['files']):
        raise ValueError('Incomplete forecast bundle; rerun updated elastic-net script')
    for name in required:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=manifest['files'][name]:
            raise ValueError(f'{name}: differs from saved forecast manifest; regenerate consistent outputs')
    asof=pd.Period(manifest['asof'],freq='M')
    if pd.Period(manifest['target_month'],freq='M')!=asof+1:
        raise ValueError('Forecast horizon must be exactly one month')
    snapshot=pd.read_csv(folder/'optimizer_snapshot.csv',dtype={'cusip':str})
    forecast=pd.read_csv(folder/'yield_change_forecasts.csv',dtype={'cusip':str})
    if snapshot.cusip.duplicated().any() or forecast.cusip.duplicated().any():
        raise ValueError('Duplicate forecast identifiers')
    if set(snapshot.cusip)!=set(forecast.cusip):
        raise ValueError('Snapshot and yield forecasts have different bonds')
    if not forecast.forecast_origin_month.eq(str(asof)).all() or not forecast.forecast_target_month.eq(str(asof+1)).all():
        raise ValueError('Forecast dates differ from manifest')
    # Explicitly use the final yield-change output; never stale betas or a refit.
    snapshot=snapshot.drop(columns='predicted_dy_pp').merge(
        forecast[['cusip','forecast_change_yield_percentage_points']].rename(
            columns={'forecast_change_yield_percentage_points':'predicted_dy_pp'}),
        on='cusip',validate='one_to_one')
    if not np.isfinite(snapshot.predicted_dy_pp).all():
        raise ValueError('Nonfinite yield forecasts')
    for c in ['DATE','MATURITY','nextcoup']:
        snapshot[c]=pd.to_datetime(snapshot[c],errors='raise')
    snapshot['month']=pd.PeriodIndex(snapshot.month,freq='M')
    if not snapshot.month.eq(asof).all(): raise ValueError('Snapshot is not the decision month')
    history=pd.read_csv(folder/'optimizer_return_history.csv',dtype={'cusip':str})
    history['month']=pd.PeriodIndex(history.month,freq='M')
    if not history.month.between(asof-35,asof).all(): raise ValueError('Return history includes out-of-window observations')
    # No realized target-period returns in this solver: evaluation is separate.
    history['next_return']=np.nan
    errors=pd.read_csv(folder/'historical_oos_errors.csv',dtype={'cusip':str},
                       usecols=['cusip','target_month','return_error'])
    if not (pd.PeriodIndex(errors.target_month,freq='M')<=asof).all():
        raise ValueError('Uncertainty sample contains future outcomes')
    curve=json.loads((folder/'optimizer_curve_params.json').read_text())
    params=pd.DataFrame([curve],index=pd.PeriodIndex([asof],freq='M'))
    params=params[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']]
    return manifest,snapshot,history,errors,params


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('forecast_dir',type=Path)
    p.add_argument('--output',type=Path,required=True,help='NEW output directory')
    for name,kind in [('annual-goal',float),('wealth',float),('candidates',int),('max-holdings',int),
                      ('gamma',float),('epsilon-mu',float),('trade-cost',float),
                      ('duration-tolerance',float),('convexity-tolerance',float),('krd-tolerance',float)]:
        p.add_argument('--'+name,type=kind,default=None)
    cli=p.parse_args()
    manifest,now,history,errors,params=load_forecast_bundle(cli.forecast_dir)
    settings=dict(manifest['settings'])
    for name,value in vars(cli).items():
        if value is not None and name not in ['forecast_dir','output']: settings[name]=value
    settings['asof']=manifest['asof'];settings['output']=str(cli.output)
    cli.output.mkdir(parents=True,exist_ok=False)
    holdings,result,Q,theta=solve_period(now,history,errors,params,SimpleNamespace(**settings))
    holdings.to_csv(cli.output/'portfolio.csv',index=False)
    np.savez(cli.output/'risk_inputs.npz',Q=Q,theta=theta,bond_ids=holdings.cusip.to_numpy(dtype=str))
    report={'forecast_source':str(cli.forecast_dir.resolve()),'asof':manifest['asof'],
            'target_month':manifest['target_month'],'settings':settings,'optimization':result,
            'limitations':['Candidate-subset proxy benchmark, not AGG.',
             'Estimated coupon-bond risk analytics; eligibility and release vintages unverified.',
             'No goal-achievement probability from this single-period solve.',
             'No target-period realized returns used or evaluated by this solver.']}
    (cli.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('Selected positive-weight holdings:')
    if 'weight' in holdings:
        print(holdings.loc[holdings.weight>1e-6,['cusip','predicted_dy_pp','forecast_return','weight','buy','sell']].to_string(index=False))


if __name__=='__main__': main()
