"""Prepare multi-period optimizer inputs using the existing Stage 1 data and risk model.
Only as-of-date forecasts and historically realized forecast errors enter the plan.
Run Stage_2_elastic_net_multi_period.py first to refresh the forecast history."""

import gzip
import pickle
import numpy as np
import pandas as pd

from elastic_net_one_period import PROJECT_ROOT, DEFAULT_FORECAST_BUNDLE
from prepare_stage1_inputs import build_inputs, FACTOR_NAMES

# Settings for one decision month; the Stage 2 backtest can call the functions below.
ASOF = '2025-10'
HORIZON = 3
FORECAST_MODEL = 'direct'  # 'constant' (H=1 forecast repeated) or 'direct' as alternative model
CANDIDATE_COUNT = 150
INITIAL_CAPITAL = 100_000.0
MIN_ERROR_OBSERVATIONS = 6

FORECAST_FILE = PROJECT_ROOT / 'Stage_2_outputs' / 'forecasts' / 'Stage_2_bond_forecast_history.csv.gz'
OUTPUT_FILE = PROJECT_ROOT / 'Stage_2_outputs' / 'inputs' / 'Stage_2_inputs.pkl'


def prepare_inputs(bundle, forecasts, pretrade_state=None):
    """Reuse Stage 1's candidates, benchmark and factor covariance for H periods."""
    asof = pd.Period(bundle['asof'], freq='M')
    return_col = 'predicted_return' if FORECAST_MODEL == 'direct' else 'constant_predicted_return'
    required_horizons = list(range(1, HORIZON + 1))
    forecast_matrix = forecasts.pivot(index='cusip', columns='horizon', values=return_col)
    if not set(required_horizons).issubset(forecast_matrix.columns):
        raise ValueError(f'Forecast file does not contain every horizon through H={HORIZON}')
    complete_cusips = forecast_matrix.dropna(subset=required_horizons).index

    # Use Stage 2's H=1 forecast for the existing Stage 1 input.
    h1 = forecasts.loc[forecasts.horizon.eq(1)].set_index('cusip')
    snapshot = bundle['snapshot'].loc[bundle['snapshot'].cusip.isin(complete_cusips)].copy()
    snapshot['predicted_dy_pp'] = snapshot.cusip.map(h1.predicted_dy_pp)
    stage = build_inputs(
        {**bundle, 'snapshot': snapshot},
        candidate_count=CANDIDATE_COUNT,
        initial_capital=INITIAL_CAPITAL,
        pretrade_state=pretrade_state,)
    candidates, benchmark = stage['candidates'], stage['benchmark']

    # Stage 2 tracks benchmark RETURN VARIANCE, not Stage 1's forecast-augmented squared-active-return quadratic.
    B = candidates[[f'B_{factor}' for factor in FACTOR_NAMES]].to_numpy(float)
    psi = candidates.psi.to_numpy(float)
    b = candidates.benchmark_weight.to_numpy(float)
    b_factor = np.asarray(stage['benchmark_factor'], dtype=float)
    sigma_f = np.asarray(stage['sigma_f'], dtype=float)
    stage['risk_linear'] = -2 * (B @ sigma_f @ b_factor + psi * b)
    stage['risk_constant'] = float(b_factor @ sigma_f @ b_factor + np.sum(benchmark.psi.to_numpy(float) * benchmark.benchmark_weight.to_numpy(float)**2))
    
    # The Stage 1 covariance and benchmark composition are held at t when planning future months: no realized future constituents, weights or risk.
    stage['horizon'] = HORIZON
    stage['forecast_model'] = FORECAST_MODEL
    stage['cash_weight'] = float(stage['pretrade_cash_weight'])
    stage['forced_sale_weight'] = float(stage['outside_pretrade_weight'])
    stage['cash_monthly_return'] = 0.0  # Preliminary cash-rate assumption.
    stage['periods'] = []
    maturity_month = pd.to_datetime(candidates.MATURITY).dt.to_period('M')

    for h in required_horizons:
        fh = forecasts.loc[forecasts.horizon.eq(h)].set_index('cusip')
        mu = candidates.cusip.map(fh[return_col]).to_numpy(float)
        benchmark_returns = benchmark.cusip.map(fh[return_col])
        # Preserve all benchmark weights; use current yield carry when a constituent has no horizon forecast.
        benchmark_returns = benchmark_returns.fillna(benchmark.yield_for_risk / 12)
        stage['periods'].append({
            'month': str(asof + h),
            'mu': mu,
            'benchmark_mu': float(benchmark.benchmark_weight @ benchmark_returns),
            'rho': None,  # Filled using past out-of-sample errors below.
            'matures_in_month': maturity_month.eq(asof + h).to_numpy(bool)})

    # Current cash + candidate holdings + bonds requiring liquidation = 100%.
    allocated = candidates.current_weight.sum() + stage['cash_weight'] + stage['forced_sale_weight']
    return stage


def add_forecast_uncertainty(stage, errors):
    """Bond/horizon out-of-sample return-error std.; horizon-pooled fallback."""
    error_col = 'return_error' if FORECAST_MODEL == 'direct' else 'constant_return_error'
    for h, period in enumerate(stage['periods'], start=1):
        past = errors.loc[errors.horizon.eq(h), ['cusip', error_col]].dropna()
        past = past.loc[np.isfinite(past[error_col])]
        stats = past.groupby('cusip')[error_col].agg(['std', 'count'])
        bond_std = stats['std'].where(stats['count'].ge(MIN_ERROR_OBSERVATIONS))
        pooled_std = float(past[error_col].std())
        period['rho'] = stage['candidates'].cusip.map(bond_std).fillna(pooled_std).to_numpy(float)
    return stage


def main():
    with gzip.open(DEFAULT_FORECAST_BUNDLE, 'rb') as file:
        bundle = pickle.load(file)
    if str(bundle['asof']) != ASOF:
        raise ValueError(f'Regenerate the Stage 1 bundle for {ASOF}; found {bundle["asof"]}')

    # Read only the requested origin from the large saved forecast history.
    latest = []
    columns = ['cusip', 'month', 'horizon', 'predicted_dy_pp','predicted_return', 'constant_predicted_return']
    for part in pd.read_csv(FORECAST_FILE, usecols=columns, dtype={'cusip': str}, chunksize=150_000):
        selected = part.loc[part.month.eq(ASOF) & part.horizon.between(1, HORIZON)]
        if not selected.empty:
            latest.append(selected)
    latest = pd.concat(latest, ignore_index=True)
    stage = prepare_inputs(bundle, latest)

    # Read errors only for chosen bonds and only if their target was realized
    # on/before the decision date. No look-ahead from future outcomes.
    error_col = 'return_error' if FORECAST_MODEL == 'direct' else 'constant_return_error'
    cusips = set(stage['candidates'].cusip)
    historical = []
    for part in pd.read_csv(FORECAST_FILE,usecols=['cusip', 'horizon', 'target_month', error_col],dtype={'cusip': str}, chunksize=150_000):
        selected = part.loc[part.cusip.isin(cusips) & part.horizon.between(1, HORIZON) & part.target_month.le(ASOF),['cusip', 'horizon', error_col]]
        if not selected.empty:
            historical.append(selected)
    stage = add_forecast_uncertainty(stage, pd.concat(historical, ignore_index=True))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as file:
        pickle.dump(stage, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSTAGE 2 INPUTS | asof={ASOF}, H={HORIZON}, forecast={FORECAST_MODEL}')
    print('=' * 76)
    print(f'Capital: ${stage["initial_capital"]:,.2f}; eligible bonds: {len(stage["eligible"]):,}; '
          f'tradable candidates: {len(stage["candidates"]):,}; benchmark bonds: {len(stage["benchmark"]):,}')
    print(f'Initial cash: {stage["cash_weight"]:.2%}; forced sales: {stage["forced_sale_weight"]:.2%}')
    print('\nADDITIONAL REQUIRED INPUTS FOR STAGE 2')
    print('1. H-PERIOD EXPECTED RETURNS mu[h]: one monthly return forecast per candidate and planned month.')
    print('   Each forecast combines yield-based carry with the duration effect of predicted yield changes.')
    print(f'   Forecast method: {FORECAST_MODEL}; H={HORIZON} monthly return vectors, '
          f'each with {len(stage["candidates"])} candidates.')
    print('   These vectors let the optimizer compare planned holdings across future months.')
    print('2. FORECAST UNCERTAINTY rho[h]: standard deviation of past out-of-sample '
          'monthly return errors for each candidate and horizon.')
    print('   Measures historical prediction instability, distinct from bond-return covariance Q.')
    print(f'   At least {MIN_ERROR_OBSERVATIONS} errors per bond; otherwise use the '
          f'horizon-pooled estimate. Only errors realized by {ASOF} are used.')
    print('3. TRACKING-RISK TERMS: active variance = x.T Q x + risk_linear.T x + risk_constant.')
    print('   Q measures candidate return covariance; linear and constant terms retain')
    print('   the tracking contribution of the full benchmark, including bonds not traded.')
    print(f'   Q shape: {stage["candidate_covariance"].shape}; '
          f'linear shape: {stage["risk_linear"].shape}; '
          f'constant: {stage["risk_constant"]:.8f}.')
    print('   Q and full benchmark exposures are reused from Stage 1 and held fixed in the plan.')
    print('4. CASH / FUNDING: pre-trade cash plus forced sales; cash earns '
          f'{stage["cash_monthly_return"]:.2%} per month.')
    print('   Cash is available for purchases; sales release funds, while transaction fees consume cash.')
    print('   The optimizer will enforce self-financing and deduct transaction costs.')

    # Preview the first five tradable candidates; each has a matching rho value.
    sample = stage['candidates'][['cusip', 'benchmark_weight', 'current_weight', 'duration', 'MATURITY']].head(5).copy()
    return_col = 'predicted_return' if FORECAST_MODEL == 'direct' else 'constant_predicted_return'
    for h, period in enumerate(stage['periods'], start=1):
        print(f'\nH={h} | planned month: {period["month"]}')
        print(f'  Candidate mu: min={period["mu"].min():+.4%}, '
              f'mean={period["mu"].mean():+.4%}, max={period["mu"].max():+.4%}; '
              f'benchmark mu={period["benchmark_mu"]:+.4%}')
        print(f'  Candidate rho: min={period["rho"].min():.4%}, '
              f'mean={period["rho"].mean():.4%}, max={period["rho"].max():.4%}; '
              f'maturities={period["matures_in_month"].sum()}')

        forecast_h = latest.loc[latest.horizon.eq(h)].set_index('cusip')[return_col]
        candidate_rho = pd.Series(period['rho'], index=stage['candidates'].cusip)
        preview = sample.copy()
        preview['mu'] = preview.cusip.map(forecast_h)
        preview['rho'] = preview.cusip.map(candidate_rho)
        preview['matures'] = pd.to_datetime(preview.MATURITY).dt.to_period('M').eq(pd.Period(period['month'], freq='M'))
        preview = preview[['cusip', 'mu', 'rho', 'benchmark_weight', 'current_weight', 'duration', 'matures']]
        print(f'  First {len(preview)} tradable candidates (monthly decimal returns/weights):')
        print(preview.to_string(index=False, na_rep='n/a',float_format=lambda x: f'{x:,.6f}'))

    print(f'\nSaved: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
