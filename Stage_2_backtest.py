"""Stage 2 monthly receding-horizon backtest.
Reuse Stage 1's data/realized-return handling and Stage 2's input builder/solver.
Plan H months at each decision date, but execute ONLY the first month's trades."""

from time import perf_counter

import numpy as np
import pandas as pd

from elastic_net_one_period import (PROJECT_ROOT, BOND_COLUMNS, process_all_data, history_for_features,macro_features, load_panel)
import Stage_2_prepare_inputs as stage2_inputs
from Stage_2_prepare_inputs import FORECAST_FILE, CANDIDATE_COUNT, INITIAL_CAPITAL
import Stage_2_multi_period_optimizer as optimizer
from Stage_2_multi_period_optimizer import (TRANSACTION_COST, GAMMA,
    UNCERTAINTY_SCALE, DURATION_TOLERANCE, CONVEXITY_TOLERANCE, KRD_TOLERANCE,
    TURNOVER_MULTIPLIER, TERMINAL_TE_CAP, MAX_CASH_WEIGHT, MAX_WEIGHT)
from backtest_stage1 import evaluate_month

# Final calibration Part D: investor-adaptive selection from Parts B and C.
START_MONTH = '2023-12'
END_MONTH = '2025-09'      # Common realized window: Jan 2024–Oct 2025.
CALIBRATION_END = '2024-12'  # Realized month; 2025 is retrospective validation.
K_VALUES = (10, 25, 50)
GAMMA_VALUES = (0.1, 1, 10, 100)
REFERENCE_K, REFERENCE_GAMMA = 25, 1.0
CALIBRATION_DIR = PROJECT_ROOT / 'Stage_2_outputs' / 'backtest_final_calibration'
OUTPUT_DIR = CALIBRATION_DIR / 'D_investor_adaptive_selection'

BASE_PARAMETERS = {
    'duration_tolerance': DURATION_TOLERANCE,
    'convexity_tolerance': CONVEXITY_TOLERANCE,
    'krd_tolerance': KRD_TOLERANCE,
    'turnover_multiplier': TURNOVER_MULTIPLIER,
    'uncertainty_scale': UNCERTAINTY_SCALE,
    'terminal_te_cap': TERMINAL_TE_CAP,
}
# Four global Pareto members plus the forecast-free H3 challenger.
# Preserve the exact B/C parameter ordering for matching archived strategy IDs.
SHORTLIST = (
    ('random_walk', 1, (('turnover_multiplier', 10),)),
    ('direct', 1, (('turnover_multiplier', 5), ('krd_tolerance', 0.5))),
    ('constant', 3, (('turnover_multiplier', 10), ('krd_tolerance', 0.5))),
    ('constant', 3, (('uncertainty_scale', 0.5), ('turnover_multiplier', 5))),
    ('random_walk', 3, (('turnover_multiplier', 10), ('krd_tolerance', 0.5))),
)


def comparisons():
    """Five calibrated models for each of the 12 investor profiles."""
    return [(model, h, k, gamma, changes)
            for k in K_VALUES for gamma in GAMMA_VALUES
            for model, h, changes in SHORTLIST]


def model_id(model, horizon, changes):
    suffix = '_'.join(f'{key}_{value:g}' for key, value in changes)
    return f'{model}_H{horizon}_{suffix}'


def archived_name(model, horizon, k, changes):
    suffix = '_'.join(f'{key}_{value:g}' for key, value in changes)
    return f'{model}_heuristic_H{horizon}_K{k}_N{CANDIDATE_COUNT}_{suffix}'


def strategy_name(model, horizon, k, gamma, changes):
    return f'{archived_name(model, horizon, k, changes)}_gamma_{gamma:g}'


def strategy_parameters(gamma, changes):
    # GAMMA is temporarily applied to the optimizer, not merely the score.
    return {**BASE_PARAMETERS, **dict(changes), 'gamma': gamma}


def is_reference(config):
    return config[2] == REFERENCE_K and config[3] == REFERENCE_GAMMA


def reference_folder(changes):
    return 'B_parameter_sensitivity' if len(changes) == 1 else 'C_parameter_interactions'


def config_metadata(model, h, k, gamma, changes):
    return {
        'model_id': model_id(model, h, changes),
        'reused_reference': is_reference((model, h, k, gamma, changes)),
        'source_phase': ('B' if len(changes) == 1 else 'C')
                        if is_reference((model, h, k, gamma, changes)) else 'D',
    }


def variation_labels(changes):
    """Keep the existing CSV metadata columns; full settings are also exported."""
    return {
        'varied_parameter': '+'.join(key for key, _ in changes),
        'parameter_value': '+'.join(f'{value:g}' for _, value in changes),
    }


def run_configured_solver(stage, k, settings):
    """Apply temporary settings to both solves and solution validation."""
    previous = {key: getattr(optimizer, key.upper()) for key in settings}
    try:
        for key, value in settings.items():
            setattr(optimizer, key.upper(), value)
        return optimizer.run_solver(stage, k=k)
    finally:
        for key, value in previous.items():
            setattr(optimizer, key.upper(), value)


def load_forecast_history(max_horizon):
    """Read the large saved forecast history once for both planning and past errors."""
    cols = ['cusip', 'month', 'target_month', 'horizon', 'dy', 'predicted_dy_pp',
            'predicted_return', 'constant_predicted_return',
            'return_error', 'constant_return_error',
            'random_walk_predicted_return', 'random_walk_return_error']
    origins, errors = [], []
    for part in pd.read_csv(FORECAST_FILE, usecols=cols, dtype={'cusip': str},
                            chunksize=150_000):
        part = part.loc[part.horizon.between(1, max_horizon)]
        current = part.loc[part.month.between(START_MONTH, END_MONTH),
                           ['cusip', 'month', 'horizon', 'dy', 'predicted_dy_pp',
                            'predicted_return', 'constant_predicted_return',
                             'random_walk_predicted_return']]
        past = part.loc[part.target_month.le(END_MONTH),
                        ['cusip', 'horizon', 'target_month',
                         'return_error', 'constant_return_error', 'random_walk_return_error']]
        if not current.empty:
            origins.append(current.copy())
        if not past.empty:
            errors.append(past.copy())
    if not origins or not errors:
        raise ValueError('Missing Stage 2 forecast origins or historical errors.')
    forecasts = pd.concat(origins, ignore_index=True)
    available = set(forecasts.horizon.unique())
    missing = sorted(set(range(1, max_horizon + 1)) - available)
    if missing:
        raise ValueError(
            f'Forecast history is missing horizons {missing}. '
            f'Regenerate Stage_2_bond_forecast_history.csv.gz through H={max_horizon} '
            'before running the Part D backtest.')
    # Ensure every decision has forecasts through the longest horizon before any solving.
    for asof in pd.period_range(START_MONTH, END_MONTH, freq='M'):
        present = set(forecasts.loc[forecasts.month.eq(str(asof)), 'horizon'])
        if not set(range(1, max_horizon + 1)).issubset(present):
            raise ValueError(f'Incomplete H={max_horizon} forecasts at decision {asof}')
    if forecasts.duplicated(['cusip', 'month', 'horizon']).any():
        raise ValueError('Duplicate CUSIP/origin/horizon forecasts in saved history.')
    return forecasts, pd.concat(errors, ignore_index=True)


def prepare_strategy_inputs(bundle, latest, errors, pretrade_state, capital, model, horizon):
    """Reuse Stage 2's input builders with per-strategy settings, then restore defaults."""
    previous_h, previous_model = stage2_inputs.HORIZON, stage2_inputs.FORECAST_MODEL
    try:
        # These two settings are read by prepare_inputs and add_forecast_uncertainty.
        # Calls are sequential: no settings leak into the next strategy.
        stage2_inputs.HORIZON = horizon
        # Reuse direct input path for zero yield changes and carry-only returns.
        # Copy so other strategies retain their original forecasts.
        stage2_inputs.FORECAST_MODEL = 'direct' if model == 'random_walk' else model
        if model == 'random_walk':
            latest = latest.copy()
            latest['predicted_dy_pp'] = 0.0
            latest['predicted_return'] = latest['random_walk_predicted_return']
        stage = stage2_inputs.prepare_inputs(bundle, latest, pretrade_state=pretrade_state)
        stage['forecast_model'] = model
        stage['initial_capital'] = capital
        historical = errors.loc[
            errors.target_month.le(bundle['asof']) &
            errors.cusip.isin(stage['candidates'].cusip)]
        if model == 'random_walk':
            historical = historical.copy()
            historical['return_error'] = historical['random_walk_return_error']
        stage2_inputs.add_forecast_uncertainty(stage, historical)
    finally:
        stage2_inputs.HORIZON = previous_h
        stage2_inputs.FORECAST_MODEL = previous_model
    if any(not np.isfinite(p['rho']).all() for p in stage['periods']):
        raise ValueError(f'Not enough historical forecast errors for {model} H={horizon}')
    return stage


def forecast_accuracy_by_horizon(forecasts):
    """Average monthly yield-change RMSE/MAE (bp) by horizon and forecast model."""
    first = forecasts.loc[forecasts.horizon.eq(1),['cusip', 'month', 'predicted_dy_pp']].rename(
        columns={'predicted_dy_pp': 'h1_dy_pp'})
    scored = forecasts.merge(first, on=['cusip', 'month'], how='left', validate='many_to_one')
    rows = []
    for h, group in scored.groupby('horizon', sort=True):
        actual = group.dy.to_numpy(float)
        forecasts_h = {
            'direct': group.predicted_dy_pp.to_numpy(float),
            'constant': h * group.h1_dy_pp.to_numpy(float),
            'random_walk': np.zeros(len(group)),
        }
        # All three methods are compared on exactly the same realized targets.
        valid = np.isfinite(actual)
        for predicted in forecasts_h.values():
            valid &= np.isfinite(predicted)
        for model, predicted in forecasts_h.items():
            errors = pd.DataFrame({
                'origin': group.loc[valid, 'month'].to_numpy(),
                'error_bp': (actual[valid] - predicted[valid]) * 100,
            })
            by_month = errors.groupby('origin').error_bp.agg(
                rmse=lambda e: float(np.sqrt(np.mean(e ** 2))),
                mae=lambda e: float(np.mean(np.abs(e))))
            rows.append({
                'horizon': int(h), 'model': model,
                'avg_monthly_rmse_bp': float(by_month.rmse.mean()),
                'avg_monthly_mae_bp': float(by_month.mae.mean()),
            })
    return pd.DataFrame(rows)


def summarize(monthly):
    """Preserve the original detailed Stage 2 summary CSV."""
    results = []
    for name, rows in monthly.groupby('strategy', sort=False):
        rows = rows.sort_values('target_month')
        active = rows.active_return.to_numpy(float)
        net_active = rows.net_active_return.to_numpy(float)
        results.append({
            'strategy': name,
            'forecast': rows.forecast.iloc[0],
            'solver': 'relax_and_fix',
            'H': int(rows.H.iloc[0]),
            'K': int(rows.K.iloc[0]),
            'candidate_count': int(rows.candidate_count.iloc[0]),
            'months': len(rows),
            'annualized_tracking_error': float(np.std(active, ddof=1) * np.sqrt(12))
                                         if len(active) > 1 else np.nan,
            'annualized_mean_active_return': float(active.mean() * 12),
            'annualized_net_tracking_error': float(np.std(net_active, ddof=1) * np.sqrt(12))
                                             if len(net_active) > 1 else np.nan,
            'annualized_mean_net_active_return': float(net_active.mean() * 12),
            'annualized_net_active_rmse': float(np.sqrt(12 * np.mean(net_active ** 2))),
            'cumulative_gross_return': float((1 + rows.portfolio_return).prod() - 1),
            'cumulative_net_return': float((1 + rows.net_portfolio_return).prod() - 1),
            'cumulative_benchmark_return': float((1 + rows.benchmark_return).prod() - 1),
            'total_transaction_fees_dollars': float(rows.transaction_cost_dollars.sum()),
            'average_monthly_one_way_turnover': float(rows.one_way_turnover.mean()),
            'average_holdings': float(rows.holdings_count.mean()),
            'max_holdings': int(rows.holdings_count.max()),
            'average_expected_monthly_tracking_error': float(rows.expected_monthly_tracking_error.mean()),
            'average_cash_weight': float(rows.cash_weight_after_trade.mean()),
            'min_benchmark_return_coverage': float(rows.benchmark_return_coverage.min()),
            'max_individual_bond_weight': float(rows.max_individual_bond_weight.max()),
            'average_effective_holdings': float(rows.effective_holdings.mean()),
            'total_runtime_seconds': float(rows.runtime_seconds.sum()),
            'optimal_months': int(rows.optimal.sum()),
            'gamma': float(rows.gamma.iloc[0]),
            'model_id': rows.model_id.iloc[0],
            'reused_reference': bool(rows.reused_reference.iloc[0]),
            'source_phase': rows.source_phase.iloc[0],
            'varied_parameter': rows.varied_parameter.iloc[0],
            'parameter_value': rows.parameter_value.iloc[0],
            **{key: rows[key].iloc[0] for key in BASE_PARAMETERS},
            'max_cash_weight': MAX_CASH_WEIGHT,
            'max_bond_weight': MAX_WEIGHT, 'transaction_cost_rate': TRANSACTION_COST,
            'initial_capital': INITIAL_CAPITAL, 'solver_time_limit_seconds': 120,
        })
    return pd.DataFrame(results)


def settle_with_cash(held, solution, stage, bond_return, fee_fraction):
    """Settle realized returns after Stage 2's cash-funded trades and fees."""
    weights = held.weight.to_numpy(float)
    next_returns = held.next_return.to_numpy(float)
    if not np.isfinite(next_returns).all() or (next_returns < -1).any():
        raise ValueError('Missing or invalid realized returns for held bonds.')
    if not np.isclose(weights.sum(), solution['weights'].sum(), atol=1e-5):
        raise ValueError('Realized holdings do not reconcile to optimized bond weights.')

    cash = float(solution['cash_weight'])
    cash_return = float(stage['cash_monthly_return'])
    if not np.isclose(weights.sum() + cash + fee_fraction, 1.0, atol=1e-5):
        raise ValueError('Post-trade holdings, cash and fees do not reconcile to capital.')

    # The optimizer has already paid fees out of cash: do not subtract them again.
    end_bond_values = weights * (1.0 + next_returns)
    end_cash = cash * (1.0 + cash_return)
    gross_return = float(bond_return + cash * cash_return)  # Before fees.
    net_growth = float(end_bond_values.sum() + end_cash)    # After fees.
    if net_growth <= 0 or not np.isfinite(net_growth):
        raise ValueError('Nonpositive or nonfinite portfolio value after settlement.')
    if not np.isclose(net_growth - 1, gross_return - fee_fraction, atol=1e-5):
        raise ValueError('Gross return and transaction-fee reconciliation failed.')

    # Redemption dates are used only AFTER optimization, as in Stage 1.
    target = pd.Period(stage['target_month'], freq='M')
    matured = (pd.to_datetime(held.MATURITY).dt.to_period('M') <= target).to_numpy()
    redeemed = held.was_redeemed.to_numpy(bool) | matured
    redeemed_cash = float(end_bond_values[redeemed].sum())
    survivors = held.loc[~redeemed, ['cusip']].copy().reset_index(drop=True)
    survivors['weight'] = end_bond_values[~redeemed] / net_growth
    next_state = {
        'holdings': survivors,
        'cash_weight': float((end_cash + redeemed_cash) / net_growth),
    }
    if not np.isclose(next_state['holdings'].weight.sum()
                      + next_state['cash_weight'], 1.0, atol=1e-5):
        raise ValueError('Settled bond and cash weights do not sum to one.')
    return next_state, gross_return, net_growth - 1, redeemed_cash


def save_progress(monthly, positions, plans, failures):
    """Save completed monthly work after each decision month (not a resume checkpoint)."""
    frames = {
        'Stage_2_monthly_results.csv': pd.DataFrame(monthly),
        'Stage_2_holdings.csv': pd.concat(positions, ignore_index=True) if positions else pd.DataFrame(),
        'Stage_2_planned_monthly.csv': pd.DataFrame(plans),
        'Stage_2_failed_strategies.csv': pd.DataFrame(failures, columns=[
            'strategy', 'decision_month', 'reason', 'varied_parameter', 'parameter_value',
             'model_id', 'K', 'gamma', 'reused_reference', 'source_phase']),
    }
    for filename, frame in frames.items():
        path = OUTPUT_DIR / filename
        temporary = path.with_suffix('.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(path)


def load_reference_paths(configs, months):
    """Reuse complete B/C histories, with the same dates and full settings."""
    monthly, positions, plans = [], [], []
    cache = {}
    decision_months = {str(m) for m in months}
    target_months = {str(m + 1) for m in months}
    for model, h, k, gamma, changes in configs:
        folder = reference_folder(changes)
        if folder not in cache:
            directory = CALIBRATION_DIR / folder
            cache[folder] = {
                key: pd.read_csv(directory / filename, dtype={'cusip': str} if key == 'holdings' else None)
                for key, filename in (
                    ('summary', 'Stage_2_summary.csv'),
                    ('monthly', 'Stage_2_monthly_results.csv'),
                    ('holdings', 'Stage_2_holdings.csv'),
                    ('plans', 'Stage_2_planned_monthly.csv'),
                )
            }
        archive = cache[folder]
        old_name = archived_name(model, h, k, changes)
        new_name = strategy_name(model, h, k, gamma, changes)
        settings = strategy_parameters(gamma, changes)
        metadata = config_metadata(model, h, k, gamma, changes)
        reference = archive['summary'].loc[archive['summary'].strategy.eq(old_name)]
        expected = {**settings, 'K': k, 'H': h, 'months': len(months),
                    'initial_capital': INITIAL_CAPITAL, 'candidate_count': CANDIDATE_COUNT,
                    'transaction_cost_rate': TRANSACTION_COST, 'max_cash_weight': MAX_CASH_WEIGHT,
                    'max_bond_weight': MAX_WEIGHT, 'solver_time_limit_seconds': 120}
        if len(reference) != 1 or any(not np.isclose(float(reference.iloc[0][key]), value)
                                      for key, value in expected.items()):
            raise ValueError(f'Archived reference settings do not match: {old_name}')
        m = archive['monthly'].loc[archive['monthly'].strategy.eq(old_name)].copy()
        p = archive['plans'].loc[archive['plans'].strategy.eq(old_name)].copy()
        held = archive['holdings'].loc[archive['holdings'].strategy.eq(old_name)].copy()
        if (len(m) != len(months) or set(m.decision_month) != decision_months
                or set(m.target_month) != target_months
                or len(p) != len(months) * h
                or set(zip(p.decision_month, p.horizon_step)) !=
                   {(month, step) for month in decision_months for step in range(1, h + 1)}
                or set(held.decision_month) != decision_months
                or not held.groupby('decision_month').size().reindex(m.decision_month)
                           .eq(m.holdings_count.to_numpy()).all()):
            raise ValueError(f'Incomplete archived reference history: {old_name}')
        for frame in (m, p, held):
            frame['strategy'] = new_name
            frame['K'] = k
            frame['gamma'] = gamma
            for key, value in metadata.items():
                frame[key] = value
        monthly.extend(m.to_dict('records'))
        positions.append(held)
        plans.extend(p.to_dict('records'))
    return monthly, positions, plans


def evaluate_profiles(summary, manifest):
    """Rank complete strategies within each investor profile; retain failed rows."""
    metrics = ['strategy', 'months', 'annualized_net_tracking_error',
               'annualized_mean_net_active_return', 'annualized_net_active_rmse',
               'cumulative_net_return', 'cumulative_benchmark_return',
               'average_monthly_one_way_turnover', 'total_transaction_fees_dollars',
               'average_holdings', 'max_holdings', 'average_effective_holdings',
               'average_expected_monthly_tracking_error', 'optimal_months',
               'total_runtime_seconds']
    ranked = manifest.merge(summary.reindex(columns=metrics), on='strategy',
                            how='left', validate='one_to_one')
    ranked['complete'] = ranked.months.notna()
    # Returns in decimal units. Multiplying monthly utility by 12 preserves rankings.
    ranked['annualized_net_utility'] = (ranked.annualized_mean_net_active_return
                                       - ranked.gamma * ranked.annualized_net_tracking_error ** 2)
    ranked = ranked.sort_values(
        ['K', 'gamma', 'annualized_net_utility', 'annualized_net_active_rmse',
         'average_monthly_one_way_turnover', 'model_id'],
        ascending=[True, True, False, True, True, True], na_position='last')
    ranked['utility_rank'] = ranked.groupby(['K', 'gamma']).cumcount().add(1).where(ranked.complete)
    ranked['pareto_efficient'] = False
    for _, group in ranked.loc[ranked.complete].groupby(['K', 'gamma']):
        for idx, row in group.iterrows():
            te = group.annualized_net_tracking_error
            ret = group.annualized_mean_net_active_return
            dominated = ((te <= row.annualized_net_tracking_error)
                         & (ret >= row.annualized_mean_net_active_return)
                         & ((te < row.annualized_net_tracking_error)
                            | (ret > row.annualized_mean_net_active_return))).any()
            ranked.loc[idx, 'pareto_efficient'] = not dominated
    winners = ranked.loc[ranked.utility_rank.eq(1)].copy()
    # Compare a universal model with the best available model per profile.
    ranked = ranked.merge(winners[['K', 'gamma', 'annualized_net_utility']].rename(
        columns={'annualized_net_utility': 'best_profile_utility'}),
        on=['K', 'gamma'], how='left', validate='many_to_one')
    ranked['utility_regret'] = ranked.best_profile_utility - ranked.annualized_net_utility
    universal = ranked.groupby('model_id', as_index=False).agg(
        complete_profiles=('complete', 'sum'),
        mean_utility_regret=('utility_regret', 'mean'),
        max_utility_regret=('utility_regret', 'max'))
    universal['eligible_universal'] = universal.complete_profiles.eq(len(K_VALUES) * len(GAMMA_VALUES))
    universal = universal.sort_values(['eligible_universal', 'mean_utility_regret', 'model_id'],
                                      ascending=[False, True, True])
    # Include profiles with no completed strategy instead of silently dropping them.
    profiles = manifest[['K', 'gamma']].drop_duplicates()
    winners = profiles.merge(winners, on=['K', 'gamma'], how='left', validate='one_to_one')
    ranked.to_csv(OUTPUT_DIR / 'Stage_2_profile_rankings.csv', index=False)
    winners.to_csv(OUTPUT_DIR / 'Stage_2_profile_winners.csv', index=False)
    universal.to_csv(OUTPUT_DIR / 'Stage_2_universal_comparison.csv', index=False)


def main():
    all_configs = comparisons()
    reference_configs = [config for config in all_configs if is_reference(config)]
    configs = [config for config in all_configs if not is_reference(config)]
    longest_horizon = max(h for _, h, _, _, _ in all_configs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame([
        {'strategy': strategy_name(model, h, k, gamma, changes),
         'forecast': model, 'H': h, 'K': k, 'candidate_count': CANDIDATE_COUNT,
         **variation_labels(changes), **strategy_parameters(gamma, changes),
         **config_metadata(model, h, k, gamma, changes)}
        for model, h, k, gamma, changes in all_configs])
    if manifest.strategy.duplicated().any():
        raise ValueError('Duplicate Part D strategy IDs.')
    manifest.to_csv(OUTPUT_DIR / 'Stage_2_configurations.csv', index=False)
    # Clear final summaries so an interrupted rerun cannot leave a stale ranking.
    for filename in ('Stage_2_summary.csv', 'Stage_2_split_summary.csv',
                     'Stage_2_profile_rankings.csv', 'Stage_2_profile_winners.csv',
                     'Stage_2_universal_comparison.csv'):
        (OUTPUT_DIR / filename).unlink(missing_ok=True)
    months = pd.period_range(START_MONTH, END_MONTH, freq='M')
    monthly, positions, plans = load_reference_paths(reference_configs, months)
    failures = []
    forecasts, errors = load_forecast_history(longest_horizon)
    accuracy = forecast_accuracy_by_horizon(forecasts)
    by_origin = {month: rows for month, rows in forecasts.groupby('month', sort=False)}

    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve = macro_features(
        history_for_features(data, 'curve'), history_for_features(data, 'fred'))
    panel = load_panel(data['bonds'][1], macro, curve)
    del data

    states = {strategy_name(*config): {'pretrade_state': None, 'capital': INITIAL_CAPITAL,
              'gross_capital': INITIAL_CAPITAL, 'failed': False} for config in configs}
    print('\nSTAGE 2 | FINAL CALIBRATION PART D: INVESTOR-ADAPTIVE SELECTION')
    print(f'Decision months: {START_MONTH} through {END_MONTH}; '
          f'realized returns: {pd.Period(START_MONTH, freq="M") + 1} through '
          f'{pd.Period(END_MONTH, freq="M") + 1}')
    print(f'Strategies: {len(all_configs)} total; {len(configs)} new; '
          f'{len(reference_configs)} reused; initial capital: ${INITIAL_CAPITAL:,.2f}')
    print(f'\nSTRATEGY GRID ({len(configs)} new independent portfolios):')
    print(manifest[['forecast', 'H', 'K', 'gamma', 'varied_parameter', 'parameter_value', 'reused_reference']]
          .to_string(index=False))
    print('Execute only the FIRST planned trades; all later planned trades are discarded.')
    print('Fees are funded from portfolio cash (unlike the Stage 1 fee convention).')

    for asof in months:
        label = str(asof)
        latest = by_origin.get(label)
        if latest is None:
            raise ValueError(f'No precomputed H={longest_horizon} forecasts for {label}.')
        snapshot = panel.loc[panel.month.eq(asof)].copy()
        history = panel.loc[panel.month.between(asof - 35, asof),['cusip', 'month', 'RET_EOM', 'DURATION', 'time_to_maturity', 'RATING_NUM']].copy()
        bundle = {
            'asof': label, 'snapshot': snapshot,
            'benchmark_snapshot': snapshot.copy(),
            'history': history, 'curve_params': curve,
        }

        successes = 0
        coverages = []
        for model, horizon, k, gamma, changes in configs:
            strategy = strategy_name(model, horizon, k, gamma, changes)
            settings = strategy_parameters(gamma, changes)
            metadata = config_metadata(model, horizon, k, gamma, changes)
            labels = variation_labels(changes)
            state = states[strategy]
            if state['failed']:
                continue
            capital_before = float(state['capital'])
            started = perf_counter()
            try:
                stage = prepare_strategy_inputs(
                    bundle, latest, errors, state['pretrade_state'],
                    capital_before, model, horizon)

                solution, metrics = run_configured_solver(stage, k, settings)
                held, bond_return, benchmark_return, coverage, benchmark = evaluate_month(stage, solution)
                fee = float(metrics['first_trade_cost_dollars'])
                fee_fraction = fee / capital_before
                next_state, gross_return, net_return, redeemed_cash = settle_with_cash(
                    held, solution, stage, bond_return, fee_fraction)
            except (RuntimeError, ValueError, KeyError, IndexError, optimizer.gp.GurobiError) as exc:
                failures.append({'strategy': strategy, 'decision_month': label,
                                 'reason': str(exc), **labels, **metadata,
                                 'K': k, 'gamma': gamma})
                state['failed'] = True
                print(f'{label} | {strategy}: not evaluable ({exc})', flush=True)
                continue

            # Stage 2 net value already includes cash-paid transaction fees.
            state['capital'] = capital_before * (1.0 + net_return)
            state['gross_capital'] *= 1.0 + gross_return
            state['pretrade_state'] = next_state
            gross_traded = float(solution['buys'].sum() + solution['sells'].sum() + stage['outside_pretrade_weight'])
            first_plan = metrics['monthly_plan'][0]
            # Preserve the full H-period plan for audit; only h=1 executes.
            for h, planned in enumerate(metrics['monthly_plan'], start=1):
                plans.append({
                    'strategy': strategy, 'decision_month': label, 'K': k,
                    **labels, **metadata,
                    **settings,
                    'horizon_step': h, 'planned_month': planned['month'],
                    'executed': h == 1,
                    'planned_holdings': planned['holdings'],
                    'planned_cash_weight': planned['cash_weight'],
                    'planned_tracking_error': planned['expected_tracking_error'],
                    'planned_expected_return': planned['expected_return'],
                    'benchmark_expected_return': planned['benchmark_expected_return'],
                    'planned_duration_gap': planned['duration_gap'],
                    'planned_convexity_gap': planned['convexity_gap'],
                    'planned_krd_2y_gap': planned['krd_gaps'][0],
                    'planned_krd_5y_gap': planned['krd_gaps'][1],
                    'planned_krd_10y_gap': planned['krd_gaps'][2],
                    'planned_krd_30y_gap': planned['krd_gaps'][3],
                    'planned_max_abs_krd_gap': max(abs(x) for x in planned['krd_gaps']),
                    'cashflow_risk_coverage': planned['cashflow_risk_coverage'],
                })
            monthly.append({
                'strategy': strategy, 'forecast': model, 'H': horizon,
                'K': k, 'decision_month': label, 'target_month': str(asof + 1),
                **labels, **metadata,
                **settings,
                'candidate_count': len(stage['candidates']),
                'holdings_count': len(held),
                'max_individual_bond_weight': float(held.weight.max()),
                'effective_holdings': float(held.weight.sum() ** 2 / np.sum(held.weight.to_numpy(float) ** 2)),
                'runtime_seconds': perf_counter() - started,
                'portfolio_return': gross_return,
                'net_portfolio_return': net_return,
                'benchmark_return': benchmark_return,
                'active_return': gross_return - benchmark_return,
                'net_active_return': net_return - benchmark_return,
                'expected_monthly_tracking_error': first_plan['expected_tracking_error'],
                'portfolio_duration': float(stage['candidates'].duration.to_numpy(float)
                                             @ solution['weights']),
                'benchmark_duration': float(stage['benchmark_duration']),
                'cash_weight_after_trade': float(solution['cash_weight']),
                'cash_weight_after_settlement': next_state['cash_weight'],
                'one_way_turnover': gross_traded / 2,
                'transaction_cost_dollars': fee,
                'benchmark_return_coverage': coverage,
                'redeemed_holdings_count': int(held.was_redeemed.sum()),
                'redemption_cash_dollars': redeemed_cash * capital_before,
                'estimated_redemption_returns': int(
                    held.return_source.eq('estimated_redemption_30_360').sum()),
                'benchmark_estimated_redemption_returns': int(
                    benchmark.return_source.eq('estimated_redemption_30_360').sum()),
                'unverified_redemption_return_fallbacks': int(
                    held.return_source.eq('observed_redemption_month_unverified').sum()),
                'benchmark_unresolved_redemptions': int(
                    benchmark.return_source.eq('unresolved_redemption').sum()),
                'gross_capital': state['gross_capital'],
                'net_capital': state['capital'],
                'optimal': metrics['optimal'],
                'solver_status': metrics['solver_status'],
                'objective_value': metrics['objective_value'],
                'terminal_tracking_error': metrics['terminal_tracking_error'],
            })
            held['strategy'] = strategy
            held['K'] = k
            held['gamma'] = gamma
            for key, value in metadata.items():
                held[key] = value
            held['decision_month'] = label
            held['target_month'] = str(asof + 1)
            held['capital_allocated'] = held.weight * capital_before
            held['expected_return'] = held.cusip.map(
                pd.Series(stage['periods'][0]['mu'], index=stage['candidates'].cusip))
            positions.append(held[[
                'strategy', 'decision_month', 'target_month', 'cusip',
                'K', 'gamma', 'model_id', 'reused_reference', 'source_phase',
                'weight', 'capital_allocated', 'expected_return', 'duration',
                'next_return', 'was_redeemed', 'redemption_date',
                'return_source', 'return_note',
            ]])
            successes += 1
            coverages.append(coverage)
            print(f'{label} -> {asof + 1} | {strategy}: '
                  f'portfolio {gross_return:+.2%}, net {net_return:+.2%}, '
                  f'benchmark {benchmark_return:+.2%}, '
                  f'cash {solution["cash_weight"]:.2%}, '
                  f'coverage {coverage:.2%}', flush=True)
        coverage_text = (f'{min(coverages):.1%}–{max(coverages):.1%}'
                         if coverages else 'n/a')
        print(f'{label} -> {asof + 1} | {successes}/{len(configs)} strategies evaluated; '
              f'benchmark coverage {coverage_text}', flush=True)
        save_progress(monthly, positions, plans, failures)

    results = pd.DataFrame(monthly)
    failures = pd.DataFrame(failures, columns=[
        'strategy', 'decision_month', 'reason', 'varied_parameter', 'parameter_value',
             'model_id', 'K', 'gamma', 'reused_reference', 'source_phase'])
    completed = (results.groupby('strategy').filter(
                     lambda x: len(x) == len(months)
                     and set(x.target_month) == {str(m + 1) for m in months})
                 if not results.empty else results)
    summary = summarize(completed) if not completed.empty else pd.DataFrame()
    if not summary.empty:
        summary = summary.merge(accuracy, left_on=['H', 'forecast'],
                                right_on=['horizon', 'model'], how='left',
                                validate='many_to_one').drop(columns=['horizon', 'model'])
        summary = summary.sort_values('annualized_net_tracking_error')
    # Evaluate realized subperiods without resetting portfolio state.
    split_rows = []
    if not completed.empty:
        for period, mask in [('calibration', completed.target_month.le(CALIBRATION_END)),
                             ('retrospective_validation', completed.target_month.gt(CALIBRATION_END))]:
            split = summarize(completed.loc[mask])
            split.insert(0, 'evaluation_period', period)
            split_rows.append(split)
    split_summary = pd.concat(split_rows, ignore_index=True) if split_rows else pd.DataFrame()
    accuracy.to_csv(OUTPUT_DIR / 'Stage_2_forecast_accuracy.csv', index=False)
    split_summary.to_csv(OUTPUT_DIR / 'Stage_2_split_summary.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'Stage_2_summary.csv', index=False)
    evaluate_profiles(summary, manifest)

    print('\nSTAGE 2 | PART D INVESTOR-ADAPTIVE RESULTS (complete strategies only)')
    if summary.empty:
        print('No complete strategies; inspect Stage_2_failed_strategies.csv.')
    else:
        # Five headline portfolio metrics; all other diagnostics remain in the CSVs.
        headline = {
            'strategy': 'Strategy',
            'K': 'K', 'gamma': 'Gamma',
            'annualized_net_tracking_error': 'Net TE (annual)',
            'cumulative_net_return': 'Net return (period)',
            'annualized_mean_net_active_return': 'Net active return (annual)',
            'average_monthly_one_way_turnover': 'Monthly turnover',
            'total_transaction_fees_dollars': 'Transaction costs ($)',
        }
        display = summary[list(headline)].rename(columns=headline).copy()
        for col in ['Net TE (annual)', 'Net return (period)',
                    'Net active return (annual)', 'Monthly turnover']:
            display[col] = display[col].map(lambda x: f'{x:.2%}')
        display['Transaction costs ($)'] = display['Transaction costs ($)'].map(
            lambda x: f'{x:,.2f}')
        print(display.to_string(index=False))
    if not failures.empty:
        print(f'{len(failures)} unsuccessful decisions; see Stage_2_failed_strategies.csv.')
    print(f'\nSaved monthly performance, summary, holdings, planned paths and failures to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
