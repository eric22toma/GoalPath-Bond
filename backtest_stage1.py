"""Monthly out-of-sample Stage 1 bond benchmark tracking backtest.
Reuses the existing forecasting, Stage 1 input preparation, and Gurobi solver.
Transaction fees are reported separately from fully invested bond capital."""

import numpy as np
import pandas as pd
from process_investable_universe import process_all_data, history_for_features, PROJECT_ROOT
from elastic_net_one_period import (BOND_COLUMNS, FEATURES, CV_PENALTY_GRID, macro_features, load_panel, forecast_month, training_rows, limit_tuning_rows)
from prepare_stage1_inputs import build_inputs
from robust_mvo_one_period import run_solver, TRANSACTION_COST, DURATION_TOLERANCE, TIME_LIMIT

# Decision months: December 2023 produces the first January 2024 return.
START_MONTH = '2023-12'
END_MONTH = '2025-10'  # The latest confirmed target is November 2025.
INITIAL_CAPITAL = 100_000.0
CANDIDATE_COUNT = 150
K_VALUES = (10, 25, 50, 100)
OUTPUT_DIR = PROJECT_ROOT / 'stage1_outputs' / 'backtest'

def comparisons():
    """Four report comparisons: forecast contribution, forecast accuracy, exact vs. heuristic, and tracking/costs across K. 
    Forecast accuracy is calculated separately; only elastic net and the zero-return-forecast control require portfolio optimizations."""
    # Main 150-candidate backtest: elastic net vs. zero forecasts, across K.
    configs = [(model, k, solver, CANDIDATE_COUNT) for model in ('elastic_net', 'no_forecast')for k in K_VALUES for solver in ('heuristic', 'exact')]
    return configs


def forecast_baselines(panel, asof, now, penalties=None):
    """Evaluate chronologically tuned ridge and lasso yield forecasts."""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    known = training_rows(panel, asof, FEATURES)
    models = {'ridge': Ridge, 'lasso': Lasso}
    if penalties is None:
        sample = limit_tuning_rows(known)
        first = pd.Period(f'{asof.year - 1}-01', 'M')
        train = sample.loc[sample.target_month < first]
        valid = sample.loc[sample.target_month.between(first, first + 11)]
        penalties = {}
        for name, model in models.items():
            scores = {}
            for alpha in CV_PENALTY_GRID:
                fit = make_pipeline(StandardScaler(), model(alpha=alpha))
                fit.fit(train[FEATURES], train.dy)
                error = valid.dy - fit.predict(valid[FEATURES])
                scores[alpha] = float(np.mean(error ** 2))
            penalties[name] = min(scores, key=scores.get)

    predictions = {}
    for name, model in models.items():
        fit = make_pipeline(StandardScaler(), model(alpha=penalties[name]))
        fit.fit(known[FEATURES], known.dy)
        predictions[name] = fit.predict(now[FEATURES])
    return predictions, penalties


def evaluate_month(stage, solution):
    """Calculate realized next-month returns using post-trade weights at t."""
    c = stage['candidates']
    held = c.loc[solution['weights'] > 1e-6].copy()
    held['weight'] = solution['weights'][solution['weights'] > 1e-6]
    if held.next_return.isna().any():
        missing = held.loc[held.next_return.isna(), 'cusip'].tolist()
    portfolio_return = float(held.weight @ held.next_return)

    # Evaluate against the entire (unfiltered) benchmark, not portfolio-eligible bonds.
    benchmark = stage['benchmark']
    observed = benchmark.next_return.notna()
    coverage = float(benchmark.loc[observed, 'benchmark_weight'].sum())
    benchmark_return = float(benchmark.loc[observed, 'benchmark_weight'] @ benchmark.loc[observed, 'next_return'] / coverage)
    return held, portfolio_return, benchmark_return, coverage


def summarize(monthly):
    """Summarize realized tracking, turnover, fees, and portfolio performance."""
    results = []
    for name, rows in monthly.groupby('strategy', sort=False):
        active = rows.active_return.to_numpy(float)
        annual_te = float(np.std(active, ddof=1) * np.sqrt(12)) if len(active) > 1 else np.nan
        results.append({
            'strategy': name,
            'forecast': rows.forecast.iloc[0],
            'solver': rows.solver.iloc[0],
            'K': int(rows.K.iloc[0]),
            'candidate_count': int(rows.candidate_count.iloc[0]),
            'months': len(rows),
            'annualized_tracking_error': annual_te,
            'annualized_mean_active_return': float(active.mean() * 12),
            'cumulative_gross_return': float(rows.gross_capital.iloc[-1] / INITIAL_CAPITAL - 1),
            'cumulative_net_return': float(rows.net_capital.iloc[-1] / INITIAL_CAPITAL - 1),
            'total_transaction_fees_dollars': float(rows.transaction_cost_dollars.sum()),
            'average_monthly_one_way_turnover': float(rows.one_way_turnover.mean()),
            'average_holdings': float(rows.holdings_count.mean()),
            'min_benchmark_return_coverage': float(rows.benchmark_return_coverage.min()),})
    return pd.DataFrame(results)


def main():
    """Compare monthly Stage 1 tracking strategies"""
    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve_params = macro_features(history_for_features(data, 'curve'), history_for_features(data, 'fred'))
    panel = load_panel(data['bonds'][1], macro, curve_params)
    del data

    configs = comparisons()
    print('\nSTAGE 1 | OUT-OF-SAMPLE BENCHMARK TRACKING COMPARISON')
    print(f'Decision months: {START_MONTH} through {END_MONTH}; '
          f'realized returns: {pd.Period(START_MONTH, freq="M") + 1} through '
          f'{pd.Period(END_MONTH, freq="M") + 1}')
    print('Benchmark: monthly outstanding amount x price, normalized across benchmark constituents')
    print('Evaluation: realized tracking error, active return, turnover, fees, and holdings')
    print(f'\nSTRATEGY GRID ({len(configs)} independent portfolios):')
    print(pd.DataFrame(configs, columns=['forecast', 'K', 'solver', 'candidates']).to_string(index=False))
    print('\nYield forecast accuracy: elastic net, ridge, lasso, random walk')
    states = {
        f'{model}_{method}_K{k}_N{n}': {'prior': None, 'capital': INITIAL_CAPITAL, 'fees': 0.0}
        for model, k, method, n in configs
    }
    monthly, positions, forecasting, failures = [], [], [], []
    elastic_params, baseline_penalties, tuned_year = None, None, None

    for asof in pd.period_range(START_MONTH, END_MONTH, freq='M'):
        if asof.year != tuned_year:
            elastic_params, baseline_penalties, tuned_year = None, None, asof.year

        now, _, _, elastic_params = forecast_month(panel, asof, params=elastic_params)
        if now.empty:
            raise ValueError(f'No forecastable bonds for {asof}')

        valid = now.loc[now.dy.notna()].copy()
        if not valid.empty:
            error = (valid.dy - valid.predicted_dy_pp) * 100
            row = {
                'target_month': str(asof + 1),
                'forecast_bonds': len(valid),
                'elastic_net_rmse_bp': float(np.sqrt(np.mean(error ** 2))),
                'random_walk_rmse_bp': float(np.sqrt(np.mean((valid.dy * 100) ** 2))),
            }
            row['elastic_net_mae_bp'] = float(np.mean(np.abs(error)))
            row['random_walk_mae_bp'] = float(np.mean(np.abs(valid.dy.to_numpy(float) * 100)))
            preds, baseline_penalties = forecast_baselines(
                panel, asof, now, baseline_penalties
            )
            for name, estimate in preds.items():
                difference = (now.dy.to_numpy(float) - estimate) * 100
                difference = difference[np.isfinite(difference)]
                row[f'{name}_rmse_bp'] = float(np.sqrt(np.mean(difference ** 2))) if len(difference) else np.nan
                row[f'{name}_mae_bp'] = float(np.mean(np.abs(difference))) if len(difference) else np.nan
            forecasting.append(row)

        history = panel.loc[panel.month.between(asof - 35, asof),
                            ['cusip', 'month', 'RET_EOM', 'DURATION',
                             'time_to_maturity', 'RATING_NUM']].copy()
        bundle = {
            'asof': str(asof),
            'snapshot': now,
            'benchmark_snapshot': panel.loc[panel.month.eq(asof)].copy(),
            'history': history,
            'curve_params': curve_params,
        }

        successes = 0
        coverages = []
        for model, k, method, n in configs:
            name = f'{model}_{method}_K{k}_N{n}'
            state = states[name]
            if state.get('failed'):
                continue
            try:
                stage = build_inputs(
                    bundle, prior_portfolio=state['prior'],
                    candidate_count=n, initial_capital=state['capital'],
                    use_forecasts=(model == 'elastic_net')
                )
                answer, metrics = run_solver(stage, method=method, k=k)[method]
                held, portfolio_return, benchmark_return, coverage = evaluate_month(stage, answer)
            except (RuntimeError, ValueError) as exc:
                failures.append({'strategy': name, 'decision_month': str(asof),
                                 'reason': str(exc)})
                state['failed'] = True
                print(f'{asof} | {name}: not evaluable ({exc})')
                continue

            successes += 1
            coverages.append(coverage)
            # Fees are paid separately: bond capital stays fully invested.
            fee = float(metrics['transaction_cost_dollars'])
            state['fees'] += fee
            state['capital'] *= 1 + portfolio_return
            state['prior'] = held[['cusip', 'weight']].copy()

            gross_traded = float(answer['buys'].sum() + answer['sells'].sum()
                                 + stage['outside_pretrade_weight'])
            monthly.append({
                'strategy': name,
                'forecast': model,
                'solver': method,
                'K': k,
                'decision_month': str(asof),
                'target_month': str(asof + 1),
                'holdings_count': len(held),
                'candidate_count': len(stage['candidates']),
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'active_return': portfolio_return - benchmark_return,
                'expected_monthly_tracking_error': metrics['expected_monthly_tracking_error'],
                'one_way_turnover': gross_traded / 2,
                'transaction_cost_dollars': fee,
                'benchmark_return_coverage': coverage,
                'portfolio_duration': metrics['portfolio_duration'],
                'benchmark_duration': metrics['benchmark_duration'],
                'gross_capital': state['capital'],
                'net_capital': state['capital'] - state['fees'],
                'optimal': metrics['optimal'],
                'objective_value': metrics['objective_value'],
            })
            held['strategy'] = name
            held['decision_month'] = str(asof)
            held['target_month'] = str(asof + 1)
            held['capital_allocated'] = held.weight * float(stage['initial_capital'])
            positions.append(held[['strategy', 'decision_month', 'target_month',
                                   'cusip', 'weight', 'capital_allocated',
                                   'mu', 'duration', 'next_return']])
        coverage_text = f'{min(coverages):.1%}–{max(coverages):.1%}' if coverages else 'n/a'
        print(f'{asof} -> {asof + 1} | {successes}/{len(configs)} strategies evaluated; '
              f'benchmark coverage {coverage_text}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame(monthly)
    target_months = len(pd.period_range(START_MONTH, END_MONTH, freq='M'))
    completed = (monthly.groupby('strategy').filter(lambda rows: len(rows) == target_months)
                 if not monthly.empty else monthly)
    summary = (summarize(completed) if not completed.empty else
               pd.DataFrame(columns=['forecast', 'solver', 'K', 'candidate_count',
                                     'annualized_tracking_error', 'annualized_mean_active_return',
                                     'cumulative_net_return', 'average_monthly_one_way_turnover',
                                     'total_transaction_fees_dollars']))
    files = [OUTPUT_DIR / name for name in (
        'monthly_results.csv', 'summary.csv', 'holdings.csv',
        'forecast_accuracy.csv', 'failed_strategies.csv'
    )]
    monthly.to_csv(files[0], index=False)
    summary.to_csv(files[1], index=False)
    (pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()).to_csv(files[2], index=False)
    pd.DataFrame(forecasting).to_csv(files[3], index=False)
    pd.DataFrame(failures, columns=['strategy', 'decision_month', 'reason']).to_csv(files[4], index=False)

    print('\nFINAL STRATEGY COMPARISON (complete backtests only; sorted by tracking error)')
    summary = summary.sort_values('annualized_tracking_error')
    display = summary[['forecast', 'solver', 'K', 'candidate_count',
                       'annualized_tracking_error', 'annualized_mean_active_return',
                       'cumulative_net_return', 'average_monthly_one_way_turnover',
                       'total_transaction_fees_dollars']].copy()
    for col in ('annualized_tracking_error', 'annualized_mean_active_return',
                'cumulative_net_return', 'average_monthly_one_way_turnover'):
        display[col] = display[col].map(lambda x: f'{x:.2%}')
    print(display.to_string(index=False))
    if failures:
        print(f'\n{len(failures)} strategies could not complete the full backtest; '
              'details saved in failed_strategies.csv.')

    if forecasting:
        scores = pd.DataFrame(forecasting)
        score_columns = [col for col in scores if col.endswith('_rmse_bp')]
        print('\nFORECAST ACCURACY COMPARISON (average monthly RMSE, basis points)')
        print(scores[score_columns].mean().sort_values().to_string(float_format=lambda x: f'{x:.2f}'))
        mae_columns = [col for col in scores if col.endswith('_mae_bp')]
        print('\nFORECAST ACCURACY COMPARISON (average monthly MAE, basis points)')
        print(scores[mae_columns].mean().sort_values().to_string(float_format=lambda x: f'{x:.2f}'))

    print(f'\nFive comparison files saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()