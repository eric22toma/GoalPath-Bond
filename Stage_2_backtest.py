"""Stage 2 monthly receding-horizon backtest.
Reuse Stage 1's data/realized-return handling and Stage 2's input builder/solver.
Plan H months at each decision date, but execute ONLY the first month's trades.
"""

import numpy as np
import pandas as pd

from elastic_net_one_period import (
    PROJECT_ROOT, BOND_COLUMNS, process_all_data, history_for_features,
    macro_features, load_panel,
)
from Stage_2_prepare_inputs import (
    FORECAST_FILE, HORIZON, FORECAST_MODEL, CANDIDATE_COUNT, INITIAL_CAPITAL,
    prepare_inputs, add_forecast_uncertainty,
)
from Stage_2_multi_period_optimizer import run_solver, TRANSACTION_COST
from backtest_stage1 import evaluate_month

# Common initial comparison with Stage 1; extend K_VALUES after validating K=10.
START_MONTH = '2023-12'
END_MONTH = '2025-09'      # H=3 plans through December 2025.
K_VALUES = (10,)
OUTPUT_DIR = PROJECT_ROOT / 'Stage_2_outputs' / 'backtest'


def load_forecast_history():
    """Read the large saved forecast history once for both planning and past errors."""
    cols = ['cusip', 'month', 'target_month', 'horizon', 'predicted_dy_pp',
            'predicted_return', 'constant_predicted_return',
            'return_error', 'constant_return_error']
    origins, errors = [], []
    for part in pd.read_csv(FORECAST_FILE, usecols=cols, dtype={'cusip': str},
                            chunksize=150_000):
        part = part.loc[part.horizon.between(1, HORIZON)]
        current = part.loc[part.month.between(START_MONTH, END_MONTH),
                           ['cusip', 'month', 'horizon', 'predicted_dy_pp',
                            'predicted_return', 'constant_predicted_return']]
        past = part.loc[part.target_month.le(END_MONTH),
                        ['cusip', 'horizon', 'target_month',
                         'return_error', 'constant_return_error']]
        if not current.empty:
            origins.append(current.copy())
        if not past.empty:
            errors.append(past.copy())
    if not origins or not errors:
        raise ValueError('Missing Stage 2 forecast origins or historical errors.')
    return pd.concat(origins, ignore_index=True), pd.concat(errors, ignore_index=True)


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


def summarize(monthly):
    """Same realized performance definitions as Stage 1; fees included in net capital."""
    results = []
    for name, rows in monthly.groupby('strategy', sort=False):
        active = rows.active_return.to_numpy(float)
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
            'cumulative_gross_return': float(rows.gross_capital.iloc[-1] / INITIAL_CAPITAL - 1),
            'cumulative_net_return': float(rows.net_capital.iloc[-1] / INITIAL_CAPITAL - 1),
            'total_transaction_fees_dollars': float(rows.transaction_cost_dollars.sum()),
            'average_monthly_one_way_turnover': float(rows.one_way_turnover.mean()),
            'average_holdings': float(rows.holdings_count.mean()),
            'average_cash_weight': float(rows.cash_weight_after_trade.mean()),
            'min_benchmark_return_coverage': float(rows.benchmark_return_coverage.min()),
        })
    return pd.DataFrame(results)


def main():
    forecasts, errors = load_forecast_history()
    by_origin = {month: rows for month, rows in forecasts.groupby('month', sort=False)}

    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve = macro_features(
        history_for_features(data, 'curve'), history_for_features(data, 'fred'))
    panel = load_panel(data['bonds'][1], macro, curve)
    del data

    months = pd.period_range(START_MONTH, END_MONTH, freq='M')
    states = {k: {'pretrade_state': None, 'capital': INITIAL_CAPITAL,
                  'gross_capital': INITIAL_CAPITAL, 'failed': False}
              for k in K_VALUES}
    monthly, positions, failures = [], [], []

    print('\nSTAGE 2 | OUT-OF-SAMPLE RECEDING-HORIZON BENCHMARK TRACKING')
    print(f'Decisions: {START_MONTH} through {END_MONTH}; H={HORIZON}, '
          f'forecast={FORECAST_MODEL}, K={K_VALUES}, candidates={CANDIDATE_COUNT}')
    print('Execute only the FIRST planned trades; all later planned trades are discarded.')
    print('Fees are funded from portfolio cash (unlike the Stage 1 fee convention).')

    for asof in months:
        label = str(asof)
        latest = by_origin.get(label)
        if latest is None:
            raise ValueError(f'No precomputed H={HORIZON} forecasts for {label}.')
        snapshot = panel.loc[panel.month.eq(asof)].copy()
        history = panel.loc[
            panel.month.between(asof - 35, asof),
            ['cusip', 'month', 'RET_EOM', 'DURATION', 'time_to_maturity', 'RATING_NUM']
        ].copy()
        bundle = {
            'asof': label, 'snapshot': snapshot,
            'benchmark_snapshot': snapshot.copy(),
            'history': history, 'curve_params': curve,
        }

        successes = 0
        for k, state in states.items():
            if state['failed']:
                continue
            strategy = f'{FORECAST_MODEL}_heuristic_H{HORIZON}_K{k}_N{CANDIDATE_COUNT}'
            capital_before = float(state['capital'])
            try:
                stage = prepare_inputs(bundle, latest, pretrade_state=state['pretrade_state'])
                stage['initial_capital'] = capital_before
                # Only historical errors already observed by this decision month.
                eligible_errors = errors.loc[
                    errors.target_month.le(label) &
                    errors.cusip.isin(stage['candidates'].cusip)
                ]
                stage = add_forecast_uncertainty(stage, eligible_errors)
                if any(not np.isfinite(p['rho']).all() for p in stage['periods']):
                    raise ValueError(f'Not enough historical forecast errors at {label}.')

                solution, metrics = run_solver(stage, k=k)
                held, bond_return, benchmark_return, coverage, benchmark = evaluate_month(stage, solution)
                fee = float(metrics['first_trade_cost_dollars'])
                fee_fraction = fee / capital_before
                next_state, gross_return, net_return, redeemed_cash = settle_with_cash(
                    held, solution, stage, bond_return, fee_fraction)
            except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                failures.append({'strategy': strategy, 'decision_month': label,
                                 'reason': str(exc)})
                state['failed'] = True
                print(f'{label} | {strategy}: not evaluable ({exc})', flush=True)
                continue

            # Stage 2 net value already includes cash-paid transaction fees.
            state['capital'] = capital_before * (1.0 + net_return)
            state['gross_capital'] *= 1.0 + gross_return
            state['pretrade_state'] = next_state
            gross_traded = float(solution['buys'].sum() + solution['sells'].sum()
                                 + stage['outside_pretrade_weight'])
            first_plan = metrics['monthly_plan'][0]
            monthly.append({
                'strategy': strategy, 'forecast': FORECAST_MODEL, 'H': HORIZON,
                'K': k, 'decision_month': label, 'target_month': str(asof + 1),
                'candidate_count': len(stage['candidates']),
                'holdings_count': len(held),
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
            held['decision_month'] = label
            held['target_month'] = str(asof + 1)
            held['capital_allocated'] = held.weight * capital_before
            held['expected_return'] = held.cusip.map(
                pd.Series(stage['periods'][0]['mu'], index=stage['candidates'].cusip))
            positions.append(held[[
                'strategy', 'decision_month', 'target_month', 'cusip',
                'weight', 'capital_allocated', 'expected_return', 'duration',
                'next_return', 'was_redeemed', 'redemption_date',
                'return_source', 'return_note',
            ]])
            successes += 1
            print(f'{label} -> {asof + 1} | K={k}: '
                  f'portfolio {gross_return:+.2%}, net {net_return:+.2%}, '
                  f'benchmark {benchmark_return:+.2%}, '
                  f'cash {solution["cash_weight"]:.2%}, '
                  f'coverage {coverage:.2%}', flush=True)
        if not successes:
            print(f'{label}: no strategies completed this month.', flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(monthly)
    failures = pd.DataFrame(failures, columns=['strategy', 'decision_month', 'reason'])
    completed = (results.groupby('strategy').filter(lambda x: len(x) == len(months))
                 if not results.empty else results)
    summary = summarize(completed) if not completed.empty else pd.DataFrame(
        columns=['strategy', 'forecast', 'H', 'K', 'months',
                 'annualized_tracking_error', 'annualized_mean_active_return',
                 'cumulative_net_return', 'average_monthly_one_way_turnover'])
    results.to_csv(OUTPUT_DIR / 'Stage_2_monthly_results.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'Stage_2_summary.csv', index=False)
    (pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()).to_csv(
        OUTPUT_DIR / 'Stage_2_holdings.csv', index=False)
    failures.to_csv(OUTPUT_DIR / 'Stage_2_failed_strategies.csv', index=False)

    print('\nSTAGE 2 | FULL-BACKTEST STRATEGY SUMMARY')
    if summary.empty:
        print('No strategy completed the entire backtest; inspect Stage_2_failed_strategies.csv.')
    else:
        view = summary.sort_values('annualized_tracking_error')[
            ['forecast', 'H', 'K', 'months', 'annualized_tracking_error',
             'annualized_mean_active_return', 'cumulative_net_return',
             'average_monthly_one_way_turnover', 'total_transaction_fees_dollars']]
        print(view.to_string(index=False, float_format=lambda x: f'{x:,.5f}'))
    print(f'\nSaved monthly performance, summary, holdings and failures to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
