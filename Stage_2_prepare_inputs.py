"""Prepare multi-period optimizer inputs using the existing Stage 1 data and risk model.
Only as-of-date forecasts and historically realized forecast errors enter the plan.
Run Stage_2_elastic_net_multi_period.py first to refresh the forecast history."""

import gzip
import pickle
import numpy as np
import pandas as pd

from elastic_net_one_period import (PROJECT_ROOT, DEFAULT_FORECAST_BUNDLE, BOND_COLUMNS, process_all_data, history_for_features, macro_features, load_panel, sven_yield)
from prepare_stage1_inputs import build_inputs, FACTOR_NAMES

# Settings for one decision month; the Stage 2 backtest can call the functions below.
ASOF = '2025-09'  # Three planned months: October–December 2025.
HORIZON = 3
FORECAST_MODEL = 'direct'  # 'constant' (H=1 forecast repeated) or 'direct' as alternative model
CANDIDATE_COUNT = 150
INITIAL_CAPITAL = 100_000.0
MIN_ERROR_OBSERVATIONS = 6
KEY_RATE_TENORS = np.array([2.0, 5.0, 10.0, 30.0])  # Years
KEY_RATE_BUMP = 0.0001  # One basis point in decimal yield units

FORECAST_FILE = PROJECT_ROOT / 'Stage_2_outputs' / 'forecasts' / 'Stage_2_bond_forecast_history.csv.gz'
OUTPUT_FILE = PROJECT_ROOT / 'Stage_2_outputs' / 'inputs' / 'Stage_2_inputs.pkl'



def _key_rate_basis(times):
    """Piecewise-linear Treasury-curve shocks; endpoint shocks extend outward."""
    eye = np.eye(len(KEY_RATE_TENORS))
    return np.stack([
        np.interp(times, KEY_RATE_TENORS, eye[:, j],
                  left=eye[0, j], right=eye[-1, j])
        for j in range(len(KEY_RATE_TENORS))
    ], axis=-1)


def _bond_convexity_krd(bonds, valuation_date, curve_parameters,
                        spread_reference=None, krd_scale_reference=None,
                        elapsed_years=0.0):
    """Cash-flow convexity (RSM430 Lecture 5) and spread-held key-rate bumps.

    Standard fixed-coupon/zero-coupon cash flows are reconstructed from existing
    fields. Other issues retain their full benchmark weight via an explicitly
    labeled duration-based approximation. All outputs are on annual-yield units.
    """
    n = len(bonds)
    convexity = np.zeros(n, dtype=float)
    krd = np.zeros((n, len(KEY_RATE_TENORS)), dtype=float)
    method = np.full(n, 'duration_proxy', dtype=object)
    spreads = np.full(n, np.nan)
    krd_scales = np.full(n, np.nan)
    maturity = pd.to_datetime(bonds.MATURITY, errors='coerce')
    next_coupon = pd.to_datetime(bonds.nextcoup, errors='coerce')
    ttm = ((maturity - valuation_date).dt.total_seconds() / 86400 / 365.25).to_numpy()
    first = ((next_coupon - valuation_date).dt.total_seconds() / 86400 / 365.25).to_numpy()
    duration = pd.to_numeric(bonds.duration, errors='coerce').to_numpy(float)
    coupon = pd.to_numeric(bonds.COUPON, errors='coerce').to_numpy(float)
    frequency = pd.to_numeric(bonds.NCOUPS, errors='coerce').to_numpy(float)
    ytm = pd.to_numeric(bonds.yield_for_risk, errors='coerce').to_numpy(float)
    clean = pd.to_numeric(bonds.PRICE_EOM, errors='coerce').to_numpy(float)
    accrued = pd.to_numeric(bonds.COUPACC, errors='coerce').to_numpy(float)
    # When accrued interest is absent, use clean price and flag the approximation.
    dirty = clean + np.where(np.isfinite(accrued), accrued, 0.0)

    safe_duration = np.where(
        np.isfinite(duration), np.maximum(duration - elapsed_years, 0.0), 0.0)
    missing_maturity = ~np.isfinite(ttm)
    # Do not silently assign zero risk to benchmark bonds with missing dates.
    # Approximate their tenor from duration but never invent a coupon schedule.
    alive = (np.isfinite(ttm) & (ttm > 0)) | (missing_maturity & (safe_duration > 0))
    method[~alive] = 'matured_or_invalid_maturity'
    method[missing_maturity & alive] = 'duration_proxy_missing_maturity'
    ttm = np.where(missing_maturity, safe_duration, ttm)
    safe_yield = np.where(np.isfinite(ytm), ytm, 0.0)
    proxy_denom = np.maximum(1.0 + safe_yield / 2.0, 0.1)
    # An option-free zero-coupon surrogate is a documented fallback, NOT a
    # claim that unsupported bonds have known fixed cash flows.
    convexity[alive] = (
        safe_duration[alive] ** 2 +
        safe_duration[alive] / (2.0 * proxy_denom[alive])
    )
    krd[alive] = safe_duration[alive, None] * _key_rate_basis(ttm[alive])

    freq = np.where(np.isfinite(frequency), np.rint(frequency), 0).astype(int)
    freq = np.where(coupon == 0.0, 2, freq)  # Zero-coupon bonds need no coupon schedule.
    # Roll an observed next-coupon date forward using the contractual frequency
    # when planning future months. Never consult future coupon observations.
    for m in (1, 2, 4, 12):
        expired_coupon = (coupon > 0) & (freq == m) & np.isfinite(first) & (first <= 0)
        first[expired_coupon] += (np.floor(-first[expired_coupon] * m) + 1) / m
    supported = (
        alive & ~missing_maturity & np.isfinite(duration) & (duration > 0) &
        np.isfinite(ytm) & np.isfinite(coupon) & (coupon >= 0) &
        np.isfinite(dirty) & (dirty > 0) &
        np.isin(freq, [1, 2, 4, 12]) &
        ((coupon == 0) | (np.isfinite(first) & (first > 0) &
                          (first <= ttm + 7.0 / 365.25)))
    )
    params = np.asarray(curve_parameters, float)
    if params.shape != (6,) or not np.isfinite(params).all():
        raise ValueError('Missing as-of Svensson zero-curve parameters for Stage 2 risk.')

    for m in (1, 2, 4, 12):
        ids = np.flatnonzero(supported & (freq == m) & (1 + ytm / m > 0.05))
        if not len(ids):
            continue
        years = ttm[ids]
        start = np.where(coupon[ids] == 0, 1.0 / m, first[ids])
        # Actual first coupon date anchors the schedule; remaining periods
        # use standard coupon frequency. Principal is paid at exact maturity.
        nperiods = max(1, int(np.ceil(np.max(years) * m)) + 2)
        times = start[:, None] + np.arange(nperiods)[None, :] / m
        flows = (coupon[ids, None] / m) * (
            times <= years[:, None] + 14.0 / 365.25
        )
        q = m * times
        y_per = (1.0 + ytm[ids] / m)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            disc = y_per[:, None] ** (-q)
            pv_coupon = flows * disc
            pv_principal = 100.0 * y_per ** (-m * years)
            ytm_price = pv_coupon.sum(axis=1) + pv_principal
            c_num = (
                (pv_coupon * q * (q + 1)).sum(axis=1) +
                pv_principal * (m * years) * (m * years + 1)
            )
            cv = c_num / (ytm_price * m ** 2 * y_per ** 2)

        # Use continuously compounded Svensson ZERO rates (percentage-point
        # input converted to decimal). Calibrate a constant spread to the
        # observed dirty price, then freeze that spread under each key bump.
        zc = sven_yield(np.maximum(times, 1e-6), params) / 100.0
        zp = sven_yield(np.maximum(years, 1e-6), params) / 100.0
        cp0 = flows * np.exp(-zc * times)
        pp0 = 100.0 * np.exp(-zp * years)
        if spread_reference is None:
            spread = np.clip(ytm[ids] - zp, -0.5, 0.5)
            for _ in range(15):
                cp = cp0 * np.exp(-spread[:, None] * times)
                pp = pp0 * np.exp(-spread * years)
                modeled = cp.sum(axis=1) + pp
                d_dollar = (times * cp).sum(axis=1) + years * pp
                spread = np.clip(
                    spread + (modeled - dirty[ids]) / np.maximum(d_dollar, 1e-8),
                    -0.5, 0.5
                )
        else:
            # Later periods use only the spread observed at the as-of date.
            spread = np.asarray(spread_reference, float)[ids]
        safe_spread = np.where(np.isfinite(spread), spread, 0.0)
        cp = cp0 * np.exp(-safe_spread[:, None] * times)
        pp = pp0 * np.exp(-safe_spread * years)
        modeled = cp.sum(axis=1) + pp
        k_raw = np.empty((len(ids), len(KEY_RATE_TENORS)))
        for j in range(len(KEY_RATE_TENORS)):
            eye = np.eye(len(KEY_RATE_TENORS))[:, j]
            basis_cash = np.interp(
                times, KEY_RATE_TENORS, eye, left=eye[0], right=eye[-1]
            )
            basis_principal = np.interp(
                years, KEY_RATE_TENORS, eye, left=eye[0], right=eye[-1]
            )
            # Equivalent to (P_- - P_+) / (2 P * one_basis_point).
            k_raw[:, j] = (
                (cp * np.sinh(KEY_RATE_BUMP * times * basis_cash)).sum(axis=1) +
                pp * np.sinh(KEY_RATE_BUMP * years * basis_principal)
            ) / (modeled * KEY_RATE_BUMP)
        k_sum = k_raw.sum(axis=1)
        if krd_scale_reference is None:
            # At the decision month, tie sum of modeled KRDs to WRDS duration.
            scale = duration[ids] / np.maximum(k_sum, 1e-12)
        else:
            # The same calibration is retained as cash flows age; KRDs are
            # allowed to decline instead of being reset to old duration.
            scale = np.asarray(krd_scale_reference, float)[ids]
        good = (
            np.isfinite(cv) & (cv > 0) & np.isfinite(k_raw).all(axis=1) &
            np.isfinite(modeled) & (modeled > 0) &
            np.isfinite(spread) & np.isfinite(scale) &
            (scale >= 0.5) & (scale <= 2.0)
        )
        if spread_reference is None:
            good &= (np.abs(modeled - dirty[ids]) / dirty[ids] < 1e-3)
        chosen = ids[good]
        convexity[chosen] = cv[good]
        krd[chosen] = k_raw[good] * scale[good, None]
        spreads[chosen] = spread[good]
        krd_scales[chosen] = scale[good]
        method[chosen] = np.where(
            np.isfinite(accrued[chosen]), 'cash_flows', 'cash_flows_no_accrued'
        )

    if not np.isfinite(convexity).all() or not np.isfinite(krd).all():
        raise ValueError('Non-finite convexity/KRD values in Stage 2 inputs.')
    return convexity, krd, method, spreads, krd_scales


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

    # Full benchmark risk, cached on the shared decision-month bundle so all
    # forecast/K strategies reuse the same ~8,000-bond calculations.
    risk_cache = bundle.setdefault('_stage2_benchmark_risk_cache', {})
    benchmark_index = pd.Index(benchmark.cusip)
    candidate_indices = benchmark_index.get_indexer(candidates.cusip)
    if (candidate_indices < 0).any():
        raise ValueError('Stage 2 candidates must be constituents of the benchmark.')
    curve_columns = ['BETA0', 'BETA1', 'BETA2', 'BETA3', 'TAU1', 'TAU2']
    curve_parameters = bundle['curve_params'].loc[asof, curve_columns].to_numpy(float)
    benchmark_weights = benchmark.benchmark_weight.to_numpy(float)
    decision_date = asof.to_timestamp(how='end').normalize()

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
    stage['key_rate_tenors'] = KEY_RATE_TENORS.copy()
    maturity_month = pd.to_datetime(candidates.MATURITY).dt.to_period('M')

    for h in required_horizons:
        fh = forecasts.loc[forecasts.horizon.eq(h)].set_index('cusip')
        if h not in risk_cache:
            # Period h is planned at asof + h - 1, not with future realized rates.
            valuation_date = (asof + h - 1).to_timestamp(how='end').normalize()
            baseline = risk_cache.get(1)
            risk_cache[h] = _bond_convexity_krd(
                benchmark, valuation_date, curve_parameters,
                spread_reference=None if baseline is None else baseline[3],
                krd_scale_reference=None if baseline is None else baseline[4],
                elapsed_years=(valuation_date - decision_date).days / 365.25)
        full_convexity, full_krd, methods, _, _ = risk_cache[h]
        candidate_convexity = full_convexity[candidate_indices]
        candidate_krd = full_krd[candidate_indices]
        # Cumulative direct horizon forecasts imply an incremental monthly
        # yield change; constant forecasts repeat the H=1 monthly change.
        if FORECAST_MODEL == 'direct':
            prev = (forecasts.loc[forecasts.horizon.eq(h - 1)]
                    .set_index('cusip').predicted_dy_pp if h > 1 else None)
            dy = fh.predicted_dy_pp if prev is None else fh.predicted_dy_pp.sub(prev)
        else:
            dy = h1.predicted_dy_pp
        candidate_dy = candidates.cusip.map(dy).to_numpy(float) / 100.0
        benchmark_dy = benchmark.cusip.map(dy).fillna(0.0).to_numpy(float) / 100.0
        # RSM430: Delta P/P = -D Delta y + 1/2 C (Delta y)^2.
        # Only the second-order correction is added here; the saved bond
        # forecast already includes carry and the first-order duration term.
        mu = (candidates.cusip.map(fh[return_col]).to_numpy(float) +
              0.5 * candidate_convexity * candidate_dy ** 2)
        benchmark_returns = benchmark.cusip.map(fh[return_col])
        # Keep every benchmark constituent even without a horizon forecast.
        benchmark_returns = benchmark_returns.fillna(benchmark.yield_for_risk / 12)
        benchmark_mu = (benchmark_returns.to_numpy(float) +
                        0.5 * full_convexity * benchmark_dy ** 2)
        cashflow_mask = np.isin(methods, ['cash_flows', 'cash_flows_no_accrued'])
        coverage = float(benchmark_weights @ cashflow_mask.astype(float))
        stage['periods'].append({
            'month': str(asof + h),
            'mu': mu,
            'benchmark_mu': float(benchmark_weights @ benchmark_mu),
            'convexity': candidate_convexity,
            'benchmark_convexity': float(benchmark_weights @ full_convexity),
            'krd': candidate_krd,
            'benchmark_krd': benchmark_weights @ full_krd,
            'cashflow_risk_coverage': coverage,
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
        # Recreate the as-of snapshot using the same Stage 1 data functions.
        # The saved Stage 1 bundle may still correspond to October; do not
        # overwrite it or modify Stage 1 just to prepare September inputs.
        print(f'Stage 1 bundle is dated {bundle["asof"]}; rebuilding the {ASOF} snapshot.')
        data = process_all_data(bond_columns=BOND_COLUMNS)
        macro, curve = macro_features(
            history_for_features(data, 'curve'), history_for_features(data, 'fred'))
        panel = load_panel(data['bonds'][1], macro, curve)
        asof = pd.Period(ASOF, freq='M')
        snapshot = panel.loc[panel.month.eq(asof)].copy()
        if snapshot.empty:
            raise ValueError(f'No bond snapshot available for {ASOF}')
        history = panel.loc[panel.month.between(asof - 35, asof),
                            ['cusip', 'month', 'RET_EOM', 'DURATION',
                             'time_to_maturity', 'RATING_NUM']].copy()
        bundle = {'asof': ASOF, 'snapshot': snapshot,
                  'benchmark_snapshot': snapshot.copy(),
                  'history': history, 'curve_params': curve}
        del data, panel

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
    print('   - Direct: mu_i[h] = forecast_start_yield_i[h]/12 - D_i,t * forecast_monthly_dy_i[h]/100.')
    print('   - Monthly dy[h] = cumulative forecast dy[h] - cumulative forecast dy[h-1] (percentage points).')
    print('   - Constant alternative: repeat the H=1 return forecast at each planned month.')
    print(f'   Forecast method: {FORECAST_MODEL}; H={HORIZON} monthly return vectors, '
          f'each with {len(stage["candidates"])} candidates.')
    print('2. FORECAST UNCERTAINTY rho[h]: standard deviation of past out-of-sample '
          'monthly return errors for each candidate and horizon.')
    print('   - Error e_i,s,h = realized monthly return_i,s+h - forecast monthly return_i,s+h|s.')
    print('   - rho_i[h] = sample_std(e_i,s,h), computed separately for each bond and horizon.')
    print(f'   At least {MIN_ERROR_OBSERVATIONS} errors per bond; otherwise use the '
          f'horizon-pooled estimate. Only errors realized by {ASOF} are used.')
    print('3. TRACKING-RISK TERMS: active variance = x.T Q x + risk_linear.T x + risk_constant.')
    print('   - Candidate covariance: Q = B * Sigma_f * B.T + diag(psi).')
    print('   - risk_linear = -2 * (B * Sigma_f * benchmark_factor + psi * candidate_benchmark_weight).')
    print('   - risk_constant = benchmark_factor.T * Sigma_f * benchmark_factor + sum(psi_B * b_B^2).')
    print(f'   Q shape: {stage["candidate_covariance"].shape}; '
          f'linear shape: {stage["risk_linear"].shape}; '
          f'constant: {stage["risk_constant"]:.8f}.')
    print('   Q and full benchmark exposures are reused from Stage 1 and held fixed in the plan.')
    print('4. BOND RISK: cash-flow convexity and calibrated 2/5/10/30Y key-rate durations.')
    print('   Full benchmark exposures are market-value weighted; unsupported bonds use a duration-based proxy.')
    for period in stage['periods']:
        print(f'   {period["month"]}: convexity={period["benchmark_convexity"]:.3f}, '
              f'KRD={np.round(period["benchmark_krd"], 3)}, '
              f'cash-flow coverage={period["cashflow_risk_coverage"]:.1%}')
    print('5. CASH / FUNDING: pre-trade cash plus forced sales; cash earns '
          f'{stage["cash_monthly_return"]:.2%} per month.')
    print('   - Starting weights: candidate holdings + cash + forced outside sales = 1.')
    print('   - Planned self-financing: cash_after = cash_before + sales - buys - transaction_costs.')
    print('   - Maturity repayments will flow into cash in the optimizer (not applied by this script).')
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
