"""Stage 1 inputs for GoalPath-Bond. The following code constructs expected returns, three-factor covariance, 
benchmark weights and starting weights, and duration in a separately before loading into Robust MVO."""

import json
import pickle
import subprocess
import numpy as np
import pandas as pd
from elastic_net_one_period import (sven_yield, PROJECT_ROOT, DEFAULT_FORECAST_BUNDLE,)
from scan_early_redemptions import NOT_REDEEMED_EARLY

DEFAULT_RESULTS = PROJECT_ROOT / 'stage1_outputs'
DEFAULT_STAGE_DIR = DEFAULT_RESULTS / 'inputs'
FACTOR_NAMES = ('level', 'slope', 'curvature')
TENORS = np.arange(1.0, 31.0)
RISK_WINDOW_MONTHS = 36

# Current YTM ceiling for portfolio candidates only (decimal annual yield).
# This differs from the 500 bp historical yield-CHANGE filter in elastic net.
MAX_INVESTABLE_YIELD = 0.15
MIN_FACTOR_OBSERVATIONS = 12
MIN_BOND_RESIDUALS = 12
MIN_BUCKET_RESIDUALS = 30

def filter_ineligible_bonds(eligible):
    """Exclude manually audited non-investable CUSIPs from trading only.
    This ex-post data-quality screen must not affect model training or the
    full benchmark"""
    return eligible.loc[~eligible.cusip.isin(NOT_REDEEMED_EARLY)].copy()


def prepare_pretrade_holdings(eligible, pretrade_state):
    """Map already-settled surviving holdings and cash into Stage 1 inputs.

    The backtest applies realized returns and settles redemptions before
    supplying holdings (cusip, weight) and cash_weight here.
    """
    holdings = pretrade_state['holdings'][['cusip', 'weight']].copy()
    holdings['cusip'] = holdings.cusip.astype(str).str.strip().str.upper()
    holdings['weight'] = pd.to_numeric(holdings.weight, errors='coerce')
    cash_weight = float(pretrade_state.get('cash_weight', 0.0))

    # Ensure the settled bond positions and cash account for the full portfolio.
    if not np.isclose(holdings.weight.sum() + cash_weight, 1.0, atol=1e-5, rtol=0):
        raise ValueError('Pretrade bond and cash weights must sum to one')

    result = eligible.copy()
    result['current_weight'] = result.cusip.map(
        holdings.set_index('cusip').weight
    ).fillna(0.0)
    forced_outside = (holdings.loc[~holdings.cusip.isin(result.cusip)]
                      .rename(columns={'weight': 'current_weight'})
                      .reset_index(drop=True))
    return result, forced_outside, cash_weight


def read_weight_file(path, asof=None):
    """Read explicit CUSIP weights; monthly benchmark files require DATE."""
    table = pd.read_csv(path, dtype={'cusip': str, 'CUSIP': str})
    if 'CUSIP' in table.columns and 'cusip' not in table.columns:
        table = table.rename(columns={'CUSIP': 'cusip'})
    if asof is not None:
        date_name = next((x for x in ('DATE', 'Date', 'date', 'month')
                          if x in table.columns), None)
        periods = pd.to_datetime(table[date_name], errors='coerce').dt.to_period('M')
        table = table.loc[periods == asof].copy()
    table['cusip'] = table.cusip.astype(str).str.strip()
    table['weight'] = pd.to_numeric(table.weight, errors='coerce')
    return table[['cusip', 'weight']]


def curve_factor_model(curve_params, asof, lookback):
    """Return maturity-specific Treasury yield-factor loadings (coef), 
    factor covariance matrix (sigma_f), and monthly factor changes (df)."""
    params = ['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']
    curve = curve_params.loc[:asof, params].dropna().sort_index().tail(lookback + 1)
    y = pd.DataFrame(
        [sven_yield(TENORS, row) for row in curve.to_numpy()],
        index=curve.index, columns=TENORS)
    factors = pd.DataFrame({
        'level': y[10.0],
        'slope': y[10.0] - y[2.0],
        'curvature': 2*y[5.0] - y[2.0] - y[10.0]})
    consecutive = factors.index == factors.index.to_series().shift(1).add(1)
    df = factors.diff().loc[consecutive].dropna()
    dy = y.diff().loc[df.index]
    # Estimate maturity-specific loadings and factor covariance.
    coef = np.linalg.lstsq(df.to_numpy(), dy.to_numpy(), rcond=None)[0].T
    sigma_f = np.cov(df.to_numpy(), rowvar=False, ddof=1)
    return coef, sigma_f, df

def bond_loadings(bonds, tenor_coefs):
    """Return B_i: change in monthly bond return per 1 percentage point factor move."""
    maturity = bonds.time_to_maturity.to_numpy(dtype=float)
    duration = bonds.DURATION.to_numpy(dtype=float)
    factor_sensitivity = np.column_stack([np.interp(maturity, TENORS, tenor_coefs[:, j]) for j in range(3)])
    return -duration[:, None] * factor_sensitivity / 100.0


def residual_variances(history, eligible, tenor_coefs, factor_moves, lookback):
    """Estimate bond-specific idiosyncratic variances (diagonal of Psi)."""
    history = history.loc[
        history.month.isin(factor_moves.index) &
        history.RET_EOM.notna() &
        history.DURATION.gt(0) &
        history.time_to_maturity.gt(0)].copy()
    # Calculate residual returns unexplained by Treasury factors.
    factors = factor_moves.loc[history.month, list(FACTOR_NAMES)].to_numpy()
    B = bond_loadings(history, tenor_coefs)
    history['residual'] = history.RET_EOM.to_numpy() - np.sum(B * factors, axis=1)
    # Estimate idiosyncratic variance for each bond.
    psi = history.groupby('cusip').residual.var()
    pooled_var = history.residual.var()
    return eligible.cusip.map(psi).fillna(pooled_var).to_numpy(dtype=float)



def reduced_tracking_objective(stage):
    """Compute candidate covariance and quadratic tracking terms,
    while preserving full-benchmark exposures."""

    c = stage['candidates']
    sigma_f = np.asarray(stage['sigma_f'], dtype=float)
    B = c[[f'B_{name}' for name in FACTOR_NAMES]].to_numpy(float)
    mu = c.mu.to_numpy(float)
    psi = c.psi.to_numpy(float)
    b = c.benchmark_weight.to_numpy(float)
    factor_b = np.asarray(stage['benchmark_factor'], dtype=float)
    mu_b = float(stage['benchmark_mu'])
    Q = B @ sigma_f @ B.T + np.diag(psi)
    H = Q + np.outer(mu, mu)
    linear = -2.0 * (B @ sigma_f @ factor_b + mu * mu_b + psi * b)
    constant = (factor_b @ sigma_f @ factor_b + mu_b ** 2 + psi @ (b ** 2) + stage['outside_benchmark_psi_constant'])
    return Q, (H+H.T)/2, linear, float(constant)


def build_inputs(bundle, benchmark_weights=None, benchmark_proxy=False, prior_portfolio=None, candidate_count=60, lookback=RISK_WINDOW_MONTHS, initial_capital=100_000.0, use_forecasts=True):
    """Create full-benchmark Stage 1 inputs, with a reduced tradable candidate set."""
    asof = pd.Period(bundle['asof'], freq='M')
    now = bundle['snapshot'].copy()
    required = ['cusip','month','YIELD','predicted_dy_pp','DURATION','time_to_maturity','PRICE_EOM','AMOUNT_OUTSTANDING','T_DVolume']

    # Benchmark is the unscreened point-in-time source universe. 
    # Keep bonds above the proposedyield limit .

    benchmark = bundle['benchmark_snapshot'].copy().reset_index(drop=True)

    if benchmark_weights is not None:
        weights = read_weight_file(benchmark_weights, asof=asof)
        benchmark['benchmark_weight'] = benchmark.cusip.map(
            weights.set_index('cusip').weight).fillna(0.0)
        benchmark_type = 'filtered_AGG_original_weights'
    else:
        amount = pd.to_numeric(benchmark.AMOUNT_OUTSTANDING, errors='coerce')
        price = pd.to_numeric(benchmark.PRICE_EOM, errors='coerce')
        benchmark['benchmark_weight'] = (amount * price).where(
            amount.gt(0) & price.gt(0), 0.0)
        benchmark_type = 'OUTSTANDING_X_PRICE_PROXY_NOT_AGG'
    benchmark = benchmark.loc[benchmark.benchmark_weight.gt(0)].copy().reset_index(drop=True)
    weight_total = benchmark.benchmark_weight.sum()
    benchmark['benchmark_weight'] /= weight_total
    benchmark_matched = len(benchmark)

    # For missing benchmark-only risk inputs, use a disclosed cross-sectional
    # proxy. These proxies do not change index weights or realized returns.
    risk_cols = {
        'duration': benchmark.DURATION.where(benchmark.DURATION.gt(0)),
        'time_to_maturity': benchmark.time_to_maturity.where(benchmark.time_to_maturity.gt(0)),
        'yield_for_risk': benchmark.YIELD,
    }
    proxy_mask = pd.Series(False, index=benchmark.index)
    for key, values in risk_cols.items():
        proxy_mask |= values.isna()
        median = values.median()
        if not np.isfinite(median):
            raise ValueError(f'No valid benchmark {key} values for {asof}')
        benchmark[key] = values.fillna(median)
    benchmark_risk_proxy_count = int(proxy_mask.sum())

    # The current 5% yield cap applies only to portfolio candidates.
    eligible = now.loc[
        now.time_to_maturity.ge(1) &
        now.DURATION.gt(0) &
        now.YIELD.notna() &
        now.YIELD.le(MAX_INVESTABLE_YIELD) &
        now.predicted_dy_pp.notna() &
        now.PRICE_EOM.gt(0)
    ].copy().reset_index(drop=True)
    if benchmark_weights is None:
        eligible = eligible.loc[pd.to_numeric(eligible.AMOUNT_OUTSTANDING, errors='coerce').gt(0)].reset_index(drop=True)
    eligible = eligible.loc[eligible.cusip.isin(benchmark.cusip)].copy().reset_index(drop=True)

    # Ex-post data-quality exclusions apply only to portfolio eligibility.
    eligible = filter_ineligible_bonds(eligible)

    eligible['duration'] = eligible.DURATION.astype(float)
    eligible['mu'] = (eligible.YIELD / 12 - eligible.duration * (eligible.predicted_dy_pp / 100) if use_forecasts else 0.0)

    benchmark['predicted_dy_pp'] = benchmark.cusip.map(now.set_index('cusip').predicted_dy_pp)
    # Where no forecast exists for a benchmark constituent, use zero yield
    # change for expected-return risk terms; keep its full index weight.
    benchmark['mu'] = (
        benchmark.yield_for_risk / 12 -
        benchmark.duration * benchmark.predicted_dy_pp.fillna(0.0) / 100
        if use_forecasts else 0.0)

    tenor_coef, sigma_f, factor_moves = curve_factor_model(
        bundle['curve_params'], asof, lookback)
    # Use the filled duration for benchmark risk loadings; preserve raw DURATION.
    benchmark_loadings = bond_loadings(benchmark.assign(DURATION=benchmark['duration']), tenor_coef)
    for j, name in enumerate(FACTOR_NAMES):
        benchmark[f'B_{name}'] = benchmark_loadings[:, j]
    benchmark['psi'] = residual_variances(
        bundle['history'], benchmark, tenor_coef, factor_moves, lookback
    )
    # Reuse benchmark risk estimates for tradable securities, on identical CUSIPs.
    by_cusip = benchmark.set_index('cusip')
    for name in FACTOR_NAMES:
        eligible[f'B_{name}'] = eligible.cusip.map(by_cusip[f'B_{name}'])
    eligible['psi'] = eligible.cusip.map(by_cusip.psi)
    eligible['benchmark_weight'] = eligible.cusip.map(by_cusip.benchmark_weight)

    if prior_portfolio is None:
        '''Initial investment of $100,000, allocated across at most K bonds. Transaction costs are penalized separately from 
        capital... cash budgeting and self-financing constraints deferred to Stage 2.'''
        eligible['current_weight'] = 0.0
        initial_cash_weight = 1.0
        forced_outside = pd.DataFrame(columns=['cusip', 'current_weight'])
        holdings_type = 'initial_cash_investment_zero_bond_holdings'
    else:
        prior = (prior_portfolio[['cusip', 'weight']].copy()
                 if isinstance(prior_portfolio, pd.DataFrame)
                 else read_weight_file(prior_portfolio))
        prior = prior.loc[prior.weight > 1e-6].copy()
        month_returns = bundle['history'].loc[
            bundle['history'].month.eq(asof), ['cusip', 'RET_EOM']
        ].set_index('cusip').RET_EOM
        r = prior.cusip.map(month_returns)
        if r.isna().any():
            missing = prior.loc[r.isna(), 'cusip'].tolist()
            raise ValueError(f'Missing {asof} realized returns for held bonds: {missing}')
        prior['current_weight'] = prior.weight * (1 + r.to_numpy())
        prior['current_weight'] /= prior.current_weight.sum()
        eligible['current_weight'] = eligible.cusip.map(prior.set_index('cusip').current_weight).fillna(0.)
        forced_outside = prior.loc[~prior.cusip.isin(eligible.cusip),['cusip','current_weight']].copy()
        holdings_type = 'previous_posttrade_weights_drifted_by_current_month_returns'
        initial_cash_weight = 0.0

    liquid = eligible.loc[eligible.T_DVolume.notna()].sort_values(['T_DVolume','cusip'], ascending=[False,True])
    # Retain currently held bonds wherever possible before choosing the
    # remainder by liquidity; K applies to post-trade, not starting holdings.
    retained = liquid.loc[liquid.current_weight.gt(0)].head(candidate_count)
    remainder = liquid.loc[~liquid.cusip.isin(retained.cusip)]
    candidates = pd.concat([retained, remainder.head(max(0, candidate_count-len(retained)))])
    eligible_idx = eligible.set_index('cusip')
    candidates = eligible_idx.loc[candidates.cusip].reset_index()
    benchmark_b = benchmark.benchmark_weight.to_numpy(float)
    bload = benchmark_loadings.T @ benchmark_b
    bmu = float(benchmark.mu.to_numpy(float) @ benchmark_b)
    residual_benchmark_constant = float(
        (benchmark.psi.to_numpy(float) * np.square(benchmark_b)).sum() -
        (candidates.psi.to_numpy(float) *
         np.square(candidates.benchmark_weight.to_numpy(float))).sum())
    # Only actual bond positions omitted from the candidate set must be
    # liquidated. Uninvested initial funding is NOT an outside bond holding.
    outside_current = float(eligible.current_weight.sum() + forced_outside.current_weight.sum()- candidates.current_weight.sum())

    out = {
        'asof': str(asof), 'target_month': str(asof+1),
        'benchmark_type': benchmark_type, 'holdings_type': holdings_type,
        'initial_capital': float(initial_capital),
        'pretrade_cash_weight': float(initial_cash_weight),
        'benchmark': benchmark, 'eligible': eligible, 'candidates': candidates,
        'sigma_f': sigma_f, 'tenor_factor_betas': tenor_coef,
        'benchmark_factor': bload, 'benchmark_mu': bmu,
        'benchmark_duration': float(benchmark.duration @ benchmark.benchmark_weight),
        'benchmark_risk_proxy_count': benchmark_risk_proxy_count,
        'excluded_benchmark_weight_from_trading': float(
            benchmark.loc[~benchmark.cusip.isin(eligible.cusip), 'benchmark_weight'].sum()
        ),
        'outside_benchmark_psi_constant': max(0., residual_benchmark_constant),
        'outside_pretrade_weight': max(0., outside_current),
        'factor_months': [str(x) for x in factor_moves.index],
        'lookback_months': lookback,
        'benchmark_positive_weights': benchmark_matched,
        'forced_outside_holdings': forced_outside
    }
    Q, H, g, offset = reduced_tracking_objective(out)
    # Fail here with a useful message rather than passing NaNs to Gurobi.
    for name, value in (('quadratic_matrix', H), ('quadratic_linear', g),
                        ('quadratic_constant', offset),
                        ('benchmark_factor', bload)):
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite {name} in Stage 1 inputs for {asof}')
    out['candidate_covariance'] = Q
    out['quadratic_matrix'] = H
    out['quadratic_linear'] = g
    out['quadratic_constant'] = offset
    return out


def show_stage1_input_summary(stage, preview=5):
    """Print the required five inputs, without a massive N-by-N covariance."""
    e = stage['eligible']
    c = stage['candidates']
    capital = float(stage['initial_capital'])
    benchmark = stage['benchmark']
    b = benchmark.benchmark_weight.to_numpy(float)
    current = e.current_weight.to_numpy(float)
    factor_cov = np.asarray(stage['sigma_f'], dtype=float)
    candidate_q = np.asarray(stage['candidate_covariance'], dtype=float)

    print('\nThe FIVE REQUIRED INPUTS for stage 1')
    print('=' * 76)
    print(f'Decision month: {stage["asof"]}   Forecast target: {stage["target_month"]}')
    print(f'Capital: ${capital:,.2f}   Eligible bonds: {len(e):,} 'f'Tradable candidates (K): {len(c):,}')

    print('\n1. EXPECTED RETURNS (Yield-based carry variant) mu-hat_t: mu_i = YIELD_i/12 - D_i * forecast_delta_y_i')
    print(f'   Vector: ({len(e):,},)   Range: {e.mu.min():+.6f} to {e.mu.max():+.6f} 'f'per month; weighted benchmark mu = {stage["benchmark_mu"]:+.6f}')

    print('\n2. RETURN COVARIANCE Q_t = B_t Sigma_f,t B_t.T + Psi_t')
    print(f'   Full {len(e):,} x {len(e):,} Q is represented by 3 factor loadings per bond plus diagonal residual variances.')
    print(f'   Materialized candidate covariance Q: {candidate_q.shape}')
    print('   Monthly Treasury factor covariance Sigma_f (3 x 3):')
    print(pd.DataFrame(factor_cov, index=FACTOR_NAMES, columns=FACTOR_NAMES).to_string(float_format=lambda x: f'{x:.7g}'))
    
    print('\n3. BENCHMARK WEIGHTS b_t: full eligible universe calculated as b_{i,t} = (A_{i,t} × P_{i,t}) / Σ_j (A_{j,t} × P_{j,t})')
    print("   where A = amount outstanding, P = end-of-month bond price")
    print(f'   {stage["benchmark_type"]}; sum = {b.sum():.6f}; nonzero constituents = {(b > 0).sum():,}')
    print(f'   Excluded from trading: {stage["excluded_benchmark_weight_from_trading"]:.2%} of benchmark weight; '
          f'benchmark-only risk proxies: {stage["benchmark_risk_proxy_count"]:,} bonds')
    
    print('\n4. PRE-TRADE HOLDINGS w_tilde_t: actual holdings BEFORE rebalancing')
    print(f'   {stage["holdings_type"]}; bond weights sum = {current.sum():.6f}; 'f'cash/funding = {stage["pretrade_cash_weight"]:.6f}')
    print(f'   Pre-trade bond value: ${capital * current.sum():,.2f}; available uninvested funding: ${capital * stage["pretrade_cash_weight"]:,.2f}')

    print('\n5. DURATION D_t: current WRDS bond duration observations')
    print(f'   Individual durations: {e.duration.min():.2f} to {e.duration.max():.2f} years; benchmark duration = {stage["benchmark_duration"]:.4f} years')
   
    show = e[['cusip','mu','benchmark_weight','current_weight','duration']].head(preview).copy()
    show['benchmark_dollars'] = show.benchmark_weight * capital
    show['pretrade_dollars'] = show.current_weight * capital
    print(f'\nSample of first {len(show)} eligible bonds (all rows saved in eligible_inputs.csv):')
    print(show.to_string(index=False, float_format=lambda v: f'{v:,.6f}'))
    print('\nPortfolio convention: optimizer invests 100% of the initial capital in <= K bonds.')
    print('Stage 1 transaction fees are not deducted from the invested $100,000.')

def main():
    """Build Stage 1 inputs from saved forecasts and publish them for robust MVO."""
    forecast_file = DEFAULT_FORECAST_BUNDLE
    output_dir = DEFAULT_STAGE_DIR

    print(f'\nLoading forecasts from {forecast_file}')
    stage = build_inputs(
        pd.read_pickle(forecast_file),
        benchmark_proxy=True,
        candidate_count=20,
        initial_capital=100_000.0,)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [output_dir / 'stage1_inputs.pkl',
        output_dir / 'eligible_inputs.csv',
        output_dir / 'candidate_inputs.csv',
        output_dir / 'candidate_covariance.csv',
        output_dir / 'factor_covariance.csv',
        output_dir / 'stage1_report.json',]
    
    with artifacts[0].open('wb') as handle:
        pickle.dump(stage, handle)
    stage['eligible'].to_csv(artifacts[1], index=False)
    stage['candidates'].to_csv(artifacts[2], index=False)
    labels = stage['candidates'].cusip.tolist()
    pd.DataFrame(stage['candidate_covariance'], index=labels, columns=labels).to_csv(artifacts[3])
    pd.DataFrame(stage['sigma_f'], index=FACTOR_NAMES, columns=FACTOR_NAMES).to_csv(artifacts[4])

    report = {
        'asof': stage['asof'],
        'target_month': stage['target_month'],
        'benchmark': stage['benchmark_type'],
        'initial_capital_dollars': stage['initial_capital'],
        'eligible_bonds': len(stage['eligible']),
        'benchmark_bonds': len(stage['benchmark']),
        'excluded_benchmark_weight_from_trading': stage['excluded_benchmark_weight_from_trading'],
        'benchmark_risk_proxy_count': stage['benchmark_risk_proxy_count'],
        'tradable_candidates': len(stage['candidates']),
        'benchmark_duration': stage['benchmark_duration'],
        'benchmark_expected_return': stage['benchmark_mu'],
        'risk_factor_months': stage['factor_months'],
    }
    artifacts[5].write_text(json.dumps(report, indent=2) + '\n')

    show_stage1_input_summary(stage)
    print(f'\nStage 1 inputs saved to {output_dir}')

if __name__ == '__main__':
    main()
