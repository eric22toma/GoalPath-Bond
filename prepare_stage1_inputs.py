"""Stage 1 inputs for GoalPath-Bond (Deliverable 3, Section 5.2).

Construct expected returns, three-factor covariance, benchmark and
starting weights, and duration in a separate staging bundle. With no
arguments, reuse the existing elastic-net code to forecast H=1 and write
stage1_outputs/inputs/stage1_inputs.pkl, ready for the optimizer.

VS Code: click Run Python File; no arguments required.

Optional saved forecast bundle:
  python prepare_stage1_inputs.py forecasts/elastic_net_outputs.pkl \
      --benchmark-proxy --output-dir stage1_inputs_demo

For an AGG constituent-weight benchmark:
  python prepare_stage1_inputs.py forecasts/elastic_net_outputs.pkl \
      --benchmark-weights eligible_AGG_holdings.csv --output-dir stage1_inputs
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from elastic_net_one_period import (
    sven_yield, FEATURES, BOND_COLUMNS, CV_MIXING_GRID, CV_PENALTY_GRID,
    process_all_data, history_for_features, PROJECT_ROOT,
    macro_features, load_panel, monthly_quote_coverage,
    trustworthy_target_month, training_rows, tune_elastic_net, fit_forecast,
)
DEFAULT_RESULTS = PROJECT_ROOT / 'stage1_outputs'
DEFAULT_STAGE_DIR = DEFAULT_RESULTS / 'inputs'

FACTOR_NAMES = ('level', 'slope', 'curvature')
TENORS = np.arange(1.0, 31.0)
RISK_WINDOW_MONTHS = 36
MIN_FACTOR_OBSERVATIONS = 12
MIN_BOND_RESIDUALS = 12
MIN_BUCKET_RESIDUALS = 30


def read_weight_file(path, asof=None):
    """Read explicit CUSIP weights; monthly benchmark files require DATE."""
    table = pd.read_csv(path, dtype={'cusip': str, 'CUSIP': str})
    if 'CUSIP' in table.columns and 'cusip' not in table.columns:
        table = table.rename(columns={'CUSIP': 'cusip'})
    if not {'cusip', 'weight'}.issubset(table.columns):
        raise ValueError(f'{path}: expected cusip and weight columns')
    if asof is not None:
        date_name = next((x for x in ('DATE', 'Date', 'date', 'month')
                          if x in table.columns), None)
        if date_name is None:
            raise ValueError('Benchmark weights require DATE (or month) to ensure month-specific membership')
        periods = pd.to_datetime(table[date_name], errors='coerce').dt.to_period('M')
        table = table.loc[periods == asof].copy()
        if table.empty:
            raise ValueError(f'No benchmark weights for {asof} in {path}')
    table['cusip'] = table.cusip.astype(str).str.strip()
    table['weight'] = pd.to_numeric(table.weight, errors='coerce')
    if (table.cusip.eq('').any() or table.weight.isna().any()
            or (table.weight < 0).any() or table.cusip.duplicated().any()):
        raise ValueError(f'{path}: duplicate/blank CUSIPs or invalid weights')
    if table.weight.sum() <= 0:
        raise ValueError(f'{path}: nonpositive total weight')
    return table[['cusip', 'weight']]


def curve_factor_model(curve_params, asof, lookback):
    """Three-factor monthly Treasury covariance and maturity-specific loadings.

    Reconstruct fitted 1–30y nominal Treasury yields from historical Svensson
    parameters. Regress monthly fitted yield changes on changes in level
    (10y), slope (10y-2y), and curvature (2*5y-2y-10y). Only use dates <= t.
    """
    curve = curve_params.loc[curve_params.index <= asof].dropna(
        subset=['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']
    ).sort_index().tail(lookback + 1)
    if len(curve) < MIN_FACTOR_OBSERVATIONS + 1:
        raise ValueError('Insufficient Treasury curve observations for factor covariance')
    fitted = np.stack([sven_yield(TENORS, row.to_numpy(dtype=float))
                       for _, row in curve[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']].iterrows()])
    y = pd.DataFrame(fitted, index=curve.index, columns=TENORS)
    factors = pd.DataFrame({
        'level': y[10.0],
        'slope': y[10.0] - y[2.0],
        'curvature': 2 * y[5.0] - y[2.0] - y[10.0],
    })
    # Do not interpret a gap in available curve months as a one-month change.
    is_successor = factors.index == factors.index.to_series().shift(1).add(1)
    df = factors.diff().loc[is_successor].dropna()
    dy = y.diff().reindex(df.index)
    if len(df) < MIN_FACTOR_OBSERVATIONS:
        raise ValueError('Fewer than 12 complete monthly factor changes')
    coef = np.linalg.lstsq(df.to_numpy(), dy.to_numpy(), rcond=None)[0].T
    sigma_f = np.cov(df.to_numpy(), rowvar=False, ddof=1)
    return coef, sigma_f, df


def bond_loadings(bonds, tenor_coefs):
    """Return B_i: change in monthly bond return per 1 pp factor move."""
    maturity = bonds.time_to_maturity.to_numpy(dtype=float)
    duration = bonds.DURATION.to_numpy(dtype=float)
    factor_sensitivity = np.column_stack(
        [np.interp(maturity, TENORS, tenor_coefs[:, j]) for j in range(3)]
    )
    return -duration[:, None] * factor_sensitivity / 100.0


def residual_variances(history, eligible, tenor_coefs, factor_moves, lookback):
    """Estimate diagonal Psi, falling back to rating/maturity and maturity buckets."""
    history = history.loc[history.month >= factor_moves.index.min()].copy()
    history = history.loc[history.month.isin(factor_moves.index)].copy()
    factors = history.month.map(
        {month: tuple(vals) for month, vals in factor_moves.iterrows()}
    )
    good = history.RET_EOM.notna() & history.DURATION.gt(0) & history.time_to_maturity.gt(0)
    history = history.loc[good].copy()
    if history.empty:
        raise ValueError('No historical returns available for factor residual variances')
    factor_rows = np.asarray([factor_moves.loc[month, list(FACTOR_NAMES)].to_numpy(dtype=float)
                              for month in history.month])
    load = bond_loadings(history, tenor_coefs)
    history['residual'] = history.RET_EOM.to_numpy(dtype=float) - np.sum(load * factor_rows, axis=1)
    history['maturity_bucket'] = pd.cut(
        history.time_to_maturity, bins=[0, 3, 7, 15, np.inf],
        labels=['1-3', '3-7', '7-15', '15+'], include_lowest=True
    ).astype(str)
    eligible['maturity_bucket'] = pd.cut(
        eligible.time_to_maturity, bins=[0, 3, 7, 15, np.inf],
        labels=['1-3', '3-7', '7-15', '15+'], include_lowest=True
    ).astype(str)

    has_rating = 'RATING_NUM' in history.columns and 'RATING_NUM' in eligible.columns
    if has_rating:
        history['rating_bucket'] = pd.to_numeric(history.RATING_NUM, errors='coerce').fillna(-1).astype(int).astype(str)
        eligible['rating_bucket'] = pd.to_numeric(eligible.RATING_NUM, errors='coerce').fillna(-1).astype(int).astype(str)
    else:
        history['rating_bucket'] = 'unknown'
        eligible['rating_bucket'] = 'unknown'

    by_bond = history.groupby('cusip').residual.agg(['count', 'var'])
    group_keys = ['rating_bucket', 'maturity_bucket']
    by_group = history.groupby(group_keys, observed=True).residual.agg(['count', 'var'])
    by_maturity = history.groupby('maturity_bucket').residual.agg(['count', 'var'])
    overall = float(history.residual.var())
    if not np.isfinite(overall) or overall <= 0:
        raise ValueError('Unable to estimate positive residual variance from history')

    own = eligible.cusip.map(by_bond['var'])
    own_count = eligible.cusip.map(by_bond['count']).fillna(0)
    grouped_index = pd.MultiIndex.from_frame(eligible[group_keys])
    group_var = pd.Series(by_group['var'].reindex(grouped_index).to_numpy(), index=eligible.index)
    group_count = pd.Series(by_group['count'].reindex(grouped_index).to_numpy(), index=eligible.index)
    maturity_var = eligible.maturity_bucket.map(by_maturity['var'])
    maturity_count = eligible.maturity_bucket.map(by_maturity['count'])

    psi = np.where(own_count >= MIN_BOND_RESIDUALS, own, np.nan)
    psi = np.where(np.isnan(psi) & (group_count >= MIN_BUCKET_RESIDUALS),
                   group_var, psi)
    psi = np.where(np.isnan(psi) & (maturity_count >= MIN_BUCKET_RESIDUALS),
                   maturity_var, psi)
    psi = np.where(np.isnan(psi), overall, psi)
    if not np.isfinite(psi).all():
        raise ValueError('Nonfinite estimated residual variances')
    return np.maximum(psi.astype(float), 1e-10)


def reduced_tracking_objective(stage):
    """Precompute the candidate covariance and FULL-benchmark quadratic terms.

    The reduced matrix is candidate_count squared, never eligible_count squared.
    Noncandidate benchmark exposures are retained in the linear and constant
    terms so tracking is evaluated against the complete eligible benchmark.
    """
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
    constant = (
        factor_b @ sigma_f @ factor_b + mu_b ** 2 +
        psi @ (b ** 2) + stage['outside_benchmark_psi_constant']
    )
    return Q, (H+H.T)/2, linear, float(constant)


def build_inputs(bundle, benchmark_weights=None, benchmark_proxy=False,
                 prior_portfolio=None, candidate_count=60,
                 lookback=RISK_WINDOW_MONTHS, initial_capital=100_000.0):
    """Create full-benchmark Stage 1 inputs, with a reduced tradable candidate set."""
    asof = pd.Period(bundle['asof'], freq='M')
    if pd.Period(bundle['target_month'], freq='M') != asof + 1:
        raise ValueError('Stage 1 requires exactly a one-month forecast')
    now = bundle['snapshot'].copy()
    if now.cusip.duplicated().any():
        raise ValueError('Duplicate CUSIPs in forecast snapshot')
    required = ['cusip','month','YIELD','predicted_dy_pp','DURATION',
                'time_to_maturity','PRICE_EOM','AMOUNT_OUTSTANDING','T_DVolume']
    missing = set(required) - set(now.columns)
    if missing:
        raise ValueError(f'Missing bond columns from elastic-net output: {sorted(missing)}')
    if not now.month.eq(asof).all():
        raise ValueError('Snapshot month does not agree with as-of month')
    if (benchmark_weights is None) == (not benchmark_proxy):
        raise ValueError('Supply exactly one of benchmark_weights or benchmark_proxy=True')
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('initial_capital must be positive and finite')

    # Section 5.2: only bonds with at least one year to maturity are investable.
    eligible = now.loc[
        now.time_to_maturity.ge(1) & now.DURATION.gt(0) &
        now.YIELD.notna() & now.predicted_dy_pp.notna() &
        now.PRICE_EOM.gt(0)
    ].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError('No eligible bonds with positive duration and maturity >= 1 year')
    eligible['duration'] = eligible.DURATION.astype(float)
    # YIELD is a decimal annual yield, predicted_dy_pp is a percentage-point change.
    eligible['mu'] = eligible.YIELD / 12 - eligible.duration * (eligible.predicted_dy_pp / 100)

    if benchmark_weights is not None:
        weights = read_weight_file(benchmark_weights, asof=asof)
        eligible['benchmark_weight'] = eligible.cusip.map(weights.set_index('cusip').weight).fillna(0.)
        benchmark_type = 'filtered_AGG_original_weights'
    else:
        outstanding = pd.to_numeric(eligible.AMOUNT_OUTSTANDING, errors='coerce')
        # Missing/nonpositive outstanding amount cannot establish a proxy
        # benchmark position; those securities receive zero benchmark weight.
        proxy_notional = outstanding.where(outstanding.gt(0), 0.0).fillna(0.0)
        eligible['benchmark_weight'] = proxy_notional * eligible.PRICE_EOM
        benchmark_type = 'OUTSTANDING_X_PRICE_PROXY_NOT_AGG'
    if not np.isfinite(eligible.benchmark_weight.to_numpy(float)).all():
        raise ValueError('Nonfinite benchmark weights after staging')
    if eligible.benchmark_weight.sum() <= 0:
        raise ValueError('No positive benchmark weights match the eligible forecast bonds')
    eligible['benchmark_weight'] /= eligible.benchmark_weight.sum()
    # Explicitly report that missing membership or skipped securities reduce coverage.
    benchmark_matched = int(eligible.benchmark_weight.gt(0).sum())

    tenor_coef, sigma_f, factor_moves = curve_factor_model(
        bundle['curve_params'], asof, lookback
    )
    loadings = bond_loadings(eligible, tenor_coef)
    for j, name in enumerate(FACTOR_NAMES):
        eligible[f'B_{name}'] = loadings[:, j]
    eligible['psi'] = residual_variances(
        bundle['history'], eligible, tenor_coef, factor_moves, lookback
    )

    if prior_portfolio is None:
        # First investment: the investor starts with $100,000 in uninvested
        # funding, NOT with thousands of AGG constituent positions.
        # Stage 1 uses normalized bond weights. The first trade purchases
        # 100% of capital in at most K bonds; transaction costs are objective
        # penalties and are reported as fees paid separately from capital.
        # A cash budget and self-financing constraint are Stage 2 additions.
        eligible['current_weight'] = 0.0
        initial_cash_weight = 1.0
        forced_outside = pd.DataFrame(columns=['cusip', 'current_weight'])
        holdings_type = 'initial_cash_investment_zero_bond_holdings'
    else:
        prior = read_weight_file(prior_portfolio)
        old_snapshot = now.set_index('cusip')
        unknown = set(prior.cusip) - set(old_snapshot.index)
        if unknown:
            raise ValueError(f'Prior holdings not in as-of bond data (cannot drift): {sorted(unknown)[:8]}')
        r = prior.cusip.map(old_snapshot.RET_EOM)
        if r.isna().any() or (1+r <= 0).any():
            raise ValueError('Missing/invalid realized as-of-month returns for prior portfolio drift')
        prior['current_weight'] = prior.weight * (1 + r.to_numpy())
        prior['current_weight'] /= prior.current_weight.sum()
        eligible['current_weight'] = eligible.cusip.map(
            prior.set_index('cusip').current_weight
        ).fillna(0.)
        forced_outside = prior.loc[~prior.cusip.isin(eligible.cusip),
                                  ['cusip','current_weight']].copy()
        holdings_type = 'previous_posttrade_weights_drifted_by_current_month_returns'
        initial_cash_weight = 0.0

    liquid = eligible.loc[eligible.T_DVolume.notna()].sort_values(
        ['T_DVolume','cusip'], ascending=[False,True]
    )
    # Retain currently held bonds wherever possible before choosing the
    # remainder by liquidity; K applies to post-trade, not starting holdings.
    retained = liquid.loc[liquid.current_weight.gt(0)].head(candidate_count)
    remainder = liquid.loc[~liquid.cusip.isin(retained.cusip)]
    candidates = pd.concat([retained, remainder.head(max(0, candidate_count-len(retained)))])
    if len(candidates) < 2:
        raise ValueError('Candidate universe has fewer than two bonds')
    eligible_idx = eligible.set_index('cusip')
    candidates = eligible_idx.loc[candidates.cusip].reset_index()
    benchmark_b = eligible.benchmark_weight.to_numpy(float)
    bload = loadings.T @ benchmark_b
    bmu = float(eligible.mu.to_numpy(float) @ benchmark_b)
    residual_benchmark_constant = float(
        (eligible.psi.to_numpy(float) * np.square(benchmark_b)).sum() -
        (candidates.psi.to_numpy(float) *
         np.square(candidates.benchmark_weight.to_numpy(float))).sum()
    )
    # Only actual bond positions omitted from the candidate set must be
    # liquidated. Uninvested initial funding is NOT an outside bond holding.
    outside_current = float(
        eligible.current_weight.sum() + forced_outside.current_weight.sum()
        - candidates.current_weight.sum()
    )
    if not np.isclose(
        eligible.current_weight.sum() + forced_outside.current_weight.sum()
        + initial_cash_weight, 1.0,
        atol=1e-8
    ):
        raise ValueError('Initial/pre-trade holdings are not fully invested')
    if outside_current < -1e-8:
        raise ValueError('Initial weights in candidates exceed the portfolio budget')
    out = {
        'asof': str(asof), 'target_month': str(asof+1),
        'benchmark_type': benchmark_type, 'holdings_type': holdings_type,
        'initial_capital': float(initial_capital),
        'pretrade_cash_weight': float(initial_cash_weight),
        'eligible': eligible, 'candidates': candidates,
        'sigma_f': sigma_f, 'tenor_factor_betas': tenor_coef,
        'benchmark_factor': bload, 'benchmark_mu': bmu,
        'benchmark_duration': float(eligible.duration @ eligible.benchmark_weight),
        'outside_benchmark_psi_constant': max(0., residual_benchmark_constant),
        'outside_pretrade_weight': max(0., outside_current),
        'factor_months': [str(x) for x in factor_moves.index],
        'lookback_months': lookback,
        'benchmark_positive_weights': benchmark_matched,
        'forced_outside_holdings': forced_outside
    }
    Q, H, g, offset = reduced_tracking_objective(out)
    out['candidate_covariance'] = Q
    out['quadratic_matrix'] = H
    out['quadratic_linear'] = g
    out['quadratic_constant'] = offset
    return out


def show_stage1_input_summary(stage, preview=5):
    """Print Section 5.2's five inputs, without a massive N-by-N covariance."""
    e = stage['eligible']
    c = stage['candidates']
    capital = float(stage['initial_capital'])
    b = e.benchmark_weight.to_numpy(float)
    current = e.current_weight.to_numpy(float)
    factor_cov = np.asarray(stage['sigma_f'], dtype=float)
    candidate_q = np.asarray(stage['candidate_covariance'], dtype=float)

    print('\n' + '=' * 76)
    print('STAGE 1: FIVE REQUIRED INPUTS (DELIVERABLE 3, SECTION 5.2)')
    print('=' * 76)
    print(f'Decision month: {stage["asof"]}   Forecast target: {stage["target_month"]}')
    print(f'Capital: ${capital:,.2f}   Eligible bonds: {len(e):,}   '
          f'Tradable candidates: {len(c):,}')
    print('1. EXPECTED RETURNS mu-hat_t: '
          'mu_i = YIELD_i/12 - D_i * forecast_delta_y_i (decimal units)')
    print(f'   Shape: ({len(e):,},)   Range: {e.mu.min():+.6f} to {e.mu.max():+.6f} '
          f'per month; weighted benchmark mu = {stage["benchmark_mu"]:+.6f}')
    print('2. RETURN COVARIANCE Q_t = B_t Sigma_f,t B_t.T + Psi_t')
    print(f'   Full {len(e):,} x {len(e):,} Q is represented by '
          '3 factor loadings per bond plus diagonal residual variances.')
    print(f'   Materialized candidate covariance Q: {candidate_q.shape}')
    print('   Monthly Treasury factor covariance Sigma_f (3 x 3):')
    print(pd.DataFrame(factor_cov, index=FACTOR_NAMES,
                       columns=FACTOR_NAMES).to_string(float_format=lambda x: f'{x:.7g}'))
    print('3. BENCHMARK WEIGHTS b_t: full eligible universe')
    print(f'   {stage["benchmark_type"]}; sum = {b.sum():.6f}; '
          f'nonzero constituents = {(b > 0).sum():,}')
    print('4. PRE-TRADE HOLDINGS w_tilde_t: actual holdings BEFORE rebalancing')
    print(f'   {stage["holdings_type"]}; bond weights sum = {current.sum():.6f}; '
          f'cash/funding = {stage["pretrade_cash_weight"]:.6f}')
    print(f'   Pre-trade bond value: ${capital * current.sum():,.2f}; '
          f'available uninvested funding: '
          f'${capital * stage["pretrade_cash_weight"]:,.2f}')
    print('5. DURATION D_t: current WRDS bond duration observations')
    print(f'   Individual durations: {e.duration.min():.2f} to '
          f'{e.duration.max():.2f} years; '
          f'benchmark duration = {stage["benchmark_duration"]:.4f} years')
    show = e[['cusip','mu','benchmark_weight','current_weight','duration']].head(preview).copy()
    show['benchmark_dollars'] = show.benchmark_weight * capital
    show['pretrade_dollars'] = show.current_weight * capital
    print(f'\nSample of first {len(show)} eligible bonds '
          '(all rows saved in eligible_inputs.csv):')
    print(show.to_string(index=False, float_format=lambda v: f'{v:,.6f}'))
    print('\nPortfolio convention: optimizer invests 100% of the initial '
          'capital in <= K bonds.')
    print('Stage 1 transaction fees are charged in the objective and reported '
          'separately; they are not deducted from the invested $100,000.')
    if stage['benchmark_type'] != 'filtered_AGG_original_weights':
        print('WARNING: This benchmark is an OUTSTANDING x PRICE PROXY, '
              'not actual AGG ETF holdings.')
    print('=' * 76)


def forecast_bundle_from_elastic_net(asof='latest', fixed=None):
    """Build H=1 inputs using the existing elastic-net and processing functions.

    The standard elastic-net Run button prints forecasts but saves no pickle.
    This function reuses its modeling functions without changing either script.
    """
    print('No saved elastic-net bundle supplied; constructing H=1 forecasts '
          'with the existing forecasting functions.')
    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve_params = macro_features(
        history_for_features(data, 'curve'),
        history_for_features(data, 'fred'),
    )
    dates = data['bonds'][1].DATE.dt.to_period('M')
    if dates.empty:
        raise ValueError('No prepared bond observations')
    calendar = pd.period_range(dates.min(), dates.max(), freq='M')
    macro = macro.reindex(calendar)
    curve_params = curve_params.reindex(calendar)
    panel = load_panel(data['bonds'][1], macro, curve_params)
    del data
    complete = panel[FEATURES].notna().all(axis=1)
    if not complete.any():
        raise ValueError('No bonds have the elastic-net predictor set')
    if asof == 'latest':
        coverage = monthly_quote_coverage(panel)
        realized = panel.loc[complete & panel.dy.notna(), 'month'].unique()
        good_months = [m for m in sorted(realized)
                       if trustworthy_target_month(coverage, m + 1)]
        if not good_months:
            raise ValueError('No month has complete predictors and credible '
                             'next-month quotes; pass --asof YYYY-MM explicitly')
        chosen = good_months[-1]
    else:
        chosen = pd.Period(asof, freq='M')
    prior = training_rows(panel, chosen, FEATURES)
    if prior.empty:
        raise ValueError(f'No historical training observations for {chosen}')
    if fixed is None:
        alpha, penalty, rmse, folds, _ = tune_elastic_net(
            prior, FEATURES, CV_MIXING_GRID, CV_PENALTY_GRID
        )
        print(f'Chronological CV: alpha={alpha:g}, lambda={penalty:g}, '
              f'validation RMSE={rmse:.2f} bp (folds: {folds})')
    else:
        alpha, penalty = fixed
        print(f'Using supplied elastic-net alpha={alpha:g}, lambda={penalty:g}')
    now, _, count = fit_forecast(panel, chosen, FEATURES, alpha, penalty)
    print(f'Elastic net: {count:,} training rows; {len(now):,} bond forecasts '
          f'for {chosen} -> {chosen + 1}')
    return {
        'asof': str(chosen),
        'target_month': str(chosen + 1),
        'snapshot': now,
        'history': panel.loc[panel.month.between(chosen - 35, chosen)].copy(),
        'errors': pd.DataFrame(),
        'curve_params': curve_params,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('forecast_file', nargs='?', type=Path, default=None,
                        help='Optional existing elastic_net_outputs.pkl')
    benchmark = parser.add_mutually_exclusive_group()
    benchmark.add_argument('--benchmark-weights', type=Path,
                           help='AGG holdings CSV with DATE, cusip, weight')
    benchmark.add_argument('--benchmark-proxy', action='store_true',
                           help='Demonstration benchmark: outstanding amount times price')
    parser.add_argument('--prior-portfolio', type=Path,
                        help='Previous post-trade holdings (cusip,weight)')
    parser.add_argument('--asof', default='latest',
                        help='Forecast month if no bundle was supplied')
    parser.add_argument('--no-tune', action='store_true',
                        help='Use fixed elastic-net parameters instead of CV if rebuilding forecasts')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda', dest='penalty_lambda', type=float, default=0.001)
    parser.add_argument('--candidates', type=int, default=20,
                        help='Number of candidate bonds (VS Code default: 20)')
    parser.add_argument('--lookback', type=int, default=RISK_WINDOW_MONTHS)
    parser.add_argument('--capital', type=float, default=100_000.0,
                        help='Initial investment in dollars (default $100,000)')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_STAGE_DIR,
                        help='Output folder, default: stage1_outputs/inputs')
    args = parser.parse_args()
    if args.candidates < 2 or args.lookback < MIN_FACTOR_OBSERVATIONS + 1:
        parser.error('Require candidates >= 2 and lookback >= 13')
    if not np.isfinite(args.capital) or args.capital <= 0:
        parser.error('Require positive --capital')
    if args.no_tune and not (0 < args.alpha < 1 and args.penalty_lambda > 0):
        parser.error('Require 0 < alpha < 1 and lambda > 0')

    if args.forecast_file is not None:
        if not args.forecast_file.is_file():
            parser.error(f'Elastic-net bundle not found: {args.forecast_file}')
        print(f'Reading elastic-net bundle: {args.forecast_file.resolve()}')
        source = pd.read_pickle(args.forecast_file)
    else:
        source = forecast_bundle_from_elastic_net(
            args.asof,
            fixed=(args.alpha, args.penalty_lambda) if args.no_tune else None,
        )

    # No original constituent weights in the checked-in WRDS bond file.
    # A no-argument VS Code run explicitly uses the demonstration proxy.
    proxy = args.benchmark_proxy or args.benchmark_weights is None
    if proxy:
        print('BENCHMARK: DEMONSTRATION PROXY (outstanding amount × price); '
              'NOT original AGG holdings.')
    stage = build_inputs(
        source, benchmark_weights=args.benchmark_weights,
        benchmark_proxy=proxy,
        prior_portfolio=args.prior_portfolio,
        candidate_count=args.candidates, lookback=args.lookback,
        initial_capital=args.capital
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'stage1_inputs.pkl').open('wb') as handle:
        pickle.dump(stage, handle)
    stage['eligible'].to_csv(args.output_dir / 'eligible_inputs.csv', index=False)
    stage['candidates'].to_csv(args.output_dir / 'candidate_inputs.csv', index=False)
    labels = stage['candidates'].cusip.tolist()
    pd.DataFrame(stage['candidate_covariance'], index=labels, columns=labels).to_csv(
        args.output_dir / 'candidate_covariance.csv'
    )
    pd.DataFrame(stage['sigma_f'], index=FACTOR_NAMES, columns=FACTOR_NAMES).to_csv(
        args.output_dir / 'factor_covariance.csv'
    )
    report = {
        'asof': stage['asof'], 'target_month': stage['target_month'],
        'benchmark': stage['benchmark_type'],
        'pretrade_holdings': stage['holdings_type'],
        'initial_capital_dollars': stage['initial_capital'],
        'initial_uninvested_funding_dollars':
            stage['initial_capital'] * stage['pretrade_cash_weight'],
        'pretrade_bond_value_dollars':
            stage['initial_capital'] * stage['eligible'].current_weight.sum(),
        'eligible_bonds': len(stage['eligible']),
        'benchmark_constituents_with_positive_weight': stage['benchmark_positive_weights'],
        'tradable_candidates': len(stage['candidates']),
        'benchmark_duration': stage['benchmark_duration'],
        'benchmark_expected_return': stage['benchmark_mu'],
        'outside_pretrade_weight_forced_sale': stage['outside_pretrade_weight'],
        'risk_factor_months': stage['factor_months'],
    }
    (args.output_dir / 'stage1_report.json').write_text(
        json.dumps(report, indent=2) + '\n'
    )
    print('\nSTAGE 1 INITIALIZATION (Deliverable 3, Section 5.2)')
    print(json.dumps(report, indent=2))
    print(f'\nSUCCESS: Stage 1 inputs saved to '
          f'{(args.output_dir / "stage1_inputs.pkl").resolve()}')
    print('Run robust_mvo_one_period.py next using the VS Code Run arrow.')
    if proxy:
        print('WARNING: proxy benchmark is not the actual eligible AGG benchmark.')
    show_stage1_input_summary(stage)


if __name__ == '__main__':
    main()
