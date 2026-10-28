"""Stage 1 inputs for GoalPath-Bond (Deliverable 3, Section 5.2).

Construct the eligible bond universe, active-tracking return and factor-risk
inputs, benchmark weights, pre-trade holdings and monthly duration exposures.
No optimizer or forecasts are fitted here.

Usage:
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

from elastic_net_one_period import sven_yield

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


def build_inputs(bundle, benchmark_weights=None, benchmark_proxy=False,
                 prior_portfolio=None, candidate_count=60,
                 lookback=RISK_WINDOW_MONTHS):
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
        eligible['benchmark_weight'] = eligible.AMOUNT_OUTSTANDING * eligible.PRICE_EOM
        benchmark_type = 'OUTSTANDING_X_PRICE_PROXY_NOT_AGG'
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
        # Only initial demonstration: a hypothetical portfolio equal to the
        # current eligible benchmark before the first rebalance.
        eligible['current_weight'] = eligible.benchmark_weight
        forced_outside = pd.DataFrame(columns=['cusip', 'current_weight'])
        holdings_type = 'initial_hypothetical_benchmark'
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
    outside_current = float(
        1. - candidates.current_weight.sum()
    )
    if not np.isclose(
        eligible.current_weight.sum() + forced_outside.current_weight.sum(), 1.0,
        atol=1e-8
    ):
        raise ValueError('Initial/pre-trade holdings are not fully invested')
    if outside_current < -1e-8:
        raise ValueError('Initial weights in candidates exceed the portfolio budget')
    out = {
        'asof': str(asof), 'target_month': str(asof+1),
        'benchmark_type': benchmark_type, 'holdings_type': holdings_type,
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
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('forecast_file', type=Path,
                        help='elastic_net_outputs.pkl from an H=1 forecasting backtest')
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument('--benchmark-weights', type=Path,
                           help='AGG holdings CSV with DATE, cusip, weight (decimal values)')
    benchmark.add_argument('--benchmark-proxy', action='store_true',
                           help='DEMO ONLY: approximate weights by outstanding x price')
    parser.add_argument('--prior-portfolio', type=Path,
                        help='Prior-month posttrade holdings CSV (cusip, weight); drift using t returns')
    parser.add_argument('--candidates', type=int, default=60)
    parser.add_argument('--lookback', type=int, default=RISK_WINDOW_MONTHS)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.candidates < 2 or args.lookback < MIN_FACTOR_OBSERVATIONS + 1:
        parser.error('Require candidates >= 2 and lookback >= 13')
    source = pd.read_pickle(args.forecast_file)
    stage = build_inputs(
        source, benchmark_weights=args.benchmark_weights,
        benchmark_proxy=args.benchmark_proxy,
        prior_portfolio=args.prior_portfolio,
        candidate_count=args.candidates, lookback=args.lookback
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'stage1_inputs.pkl').open('wb') as handle:
        pickle.dump(stage, handle)
    stage['eligible'].to_csv(args.output_dir / 'eligible_inputs.csv', index=False)
    stage['candidates'].to_csv(args.output_dir / 'candidate_inputs.csv', index=False)
    report = {
        'asof': stage['asof'], 'target_month': stage['target_month'],
        'benchmark': stage['benchmark_type'],
        'pretrade_holdings': stage['holdings_type'],
        'eligible_bonds': len(stage['eligible']),
        'benchmark_constituents_with_positive_weight': stage['benchmark_positive_weights'],
        'tradable_candidates': len(stage['candidates']),
        'benchmark_duration': stage['benchmark_duration'],
        'benchmark_expected_return': stage['benchmark_mu'],
        'outside_pretrade_weight_forced_sale': stage['outside_pretrade_weight'],
        'risk_factor_months': stage['factor_months'],
    }
    (args.output_dir / 'stage1_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('STAGE 1 INITIALIZATION (Deliverable 3, Section 5.2)')
    print(json.dumps(report, indent=2))
    print('Saved: stage1_inputs.pkl, eligible_inputs.csv, candidate_inputs.csv')
    if args.benchmark_proxy:
        print('WARNING: outstanding x price is NOT the original eligible AGG benchmark.')


if __name__ == '__main__':
    main()
