"""Monthly out-of-sample Stage 1 bond benchmark tracking backtest.

Reuses the existing forecasting, Stage 1 input preparation, and Gurobi solver.
Transaction fees are reported separately from fully invested bond capital.
"""

import numpy as np
import pandas as pd

from process_investable_universe import process_all_data, history_for_features, PROJECT_ROOT
from elastic_net_one_period import (
    BOND_COLUMNS, FEATURES, CV_PENALTY_GRID, macro_features, load_panel,
    forecast_month, training_rows, limit_tuning_rows
)
from prepare_stage1_inputs import build_inputs
from robust_mvo_one_period import run_solver, TRANSACTION_COST, DURATION_TOLERANCE, TIME_LIMIT

# Decision months: December 2023 produces the first January 2024 return.
START_MONTH = '2023-12'
END_MONTH = '2025-10'  # The latest confirmed target is November 2025.
INITIAL_CAPITAL = 100_000.0
CANDIDATE_COUNT = 150
K = 10
METHOD = 'heuristic'             # 'heuristic' or 'exact'
RUN_COMPARISONS = False          # Add no-forecast, exact/heuristic, and K comparisons
RUN_FORECAST_BASELINES = False   # Also compare yield forecasts with ridge and lasso
OUTPUT_DIR = PROJECT_ROOT / 'stage1_outputs' / 'backtest'


def comparisons():
    """Return independently rebalanced tracking strategies."""
    configs = [('elastic_net', K, METHOD)]
    if RUN_COMPARISONS:
        other = 'exact' if METHOD == 'heuristic' else 'heuristic'
        configs += [('no_forecast', K, METHOD), ('elastic_net', K, other)]
        configs += [('elastic_net', k, 'heuristic') for k in (25, 50, 100) if k <= CANDIDATE_COUNT]
    return list(dict.fromkeys(configs))


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
        raise ValueError(f"Missing realized returns for held bonds at {stage['target_month']}: {missing}")
    portfolio_return = float(held.weight @ held.next_return)

    # Report observed benchmark-weight coverage; missing returns are not set to zero.
    eligible = stage['eligible']
    observed = eligible.next_return.notna()
    coverage = float(eligible.loc[observed, 'benchmark_weight'].sum())
    if coverage <= 0:
        raise ValueError(f"No benchmark returns observed for {stage['target_month']}")
    benchmark_return = float(
        eligible.loc[observed, 'benchmark_weight'] @ eligible.loc[observed, 'next_return']
        / coverage
    )
    return held, portfolio_return, benchmark_return, coverage


def summarize(monthly):
    """Summarize realized tracking, turnover, fees, and portfolio performance."""
    results = []
    for name, rows in monthly.groupby('strategy', sort=False):
        active = rows.active_return.to_numpy(float)
        annual_te = float(np.std(active, ddof=1) * np.sqrt(12)) if len(active) > 1 else np.nan
        results.append({
            'strategy': name,
            'months': len(rows),
            'annualized_tracking_error': annual_te,
            'annualized_mean_active_return': float(active.mean() * 12),
            'cumulative_gross_return': float(rows.gross_capital.iloc[-1] / INITIAL_CAPITAL - 1),
            'cumulative_net_return': float(rows.net_capital.iloc[-1] / INITIAL_CAPITAL - 1),
            'total_transaction_fees_dollars': float(rows.transaction_cost_dollars.sum()),
            'average_monthly_one_way_turnover': float(rows.one_way_turnover.mean()),
            'average_holdings': float(rows.holdings_count.mean()),
            'min_benchmark_return_coverage': float(rows.benchmark_return_coverage.min()),
        })
    return pd.DataFrame(results)


def main():
    """Run monthly rebalancing and save all comparisons locally."""
    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve_params = macro_features(
        history_for_features(data, 'curve'), history_for_features(data, 'fred')
    )
    panel = load_panel(data['bonds'][1], macro, curve_params)
    del data

    configs = comparisons()
    states = {
        f'{model}_{method}_K{k}': {'prior': None, 'capital': INITIAL_CAPITAL, 'fees': 0.0}
        for model, k, method in configs
    }
    monthly, positions, forecasting = [], [], []
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
                'elastic_net_mae_bp': float(np.mean(np.abs(error))),
                'random_walk_rmse_bp': float(np.sqrt(np.mean((valid.dy * 100) ** 2))),
            }
            if RUN_FORECAST_BASELINES:
                preds, baseline_penalties = forecast_baselines(
                    panel, asof, now, baseline_penalties
                )
                for name, estimate in preds.items():
                    difference = (now.dy.to_numpy(float) - estimate) * 100
                    row[f'{name}_rmse_bp'] = float(np.sqrt(np.nanmean(difference ** 2)))
            forecasting.append(row)

        history = panel.loc[panel.month.between(asof - 35, asof),
                            ['cusip', 'month', 'RET_EOM', 'DURATION',
                             'time_to_maturity', 'RATING_NUM']].copy()
        bundle = {
            'asof': str(asof),
            'snapshot': now,
            'history': history,
            'curve_params': curve_params,
        }

        for model, k, method in configs:
            name = f'{model}_{method}_K{k}'
            state = states[name]
            stage = build_inputs(
                bundle, prior_portfolio=state['prior'],
                candidate_count=CANDIDATE_COUNT, initial_capital=state['capital'],
                use_forecasts=(model == 'elastic_net')
            )
            answer, metrics = run_solver(stage, method=method, k=k)[method]
            held, portfolio_return, benchmark_return, coverage = evaluate_month(stage, answer)

            # Fees are paid separately: bond capital stays fully invested.
            fee = float(metrics['transaction_cost_dollars'])
            state['fees'] += fee
            state['capital'] *= 1 + portfolio_return
            state['prior'] = held[['cusip', 'weight']].copy()

            gross_traded = float(answer['buys'].sum() + answer['sells'].sum()
                                 + stage['outside_pretrade_weight'])
            monthly.append({
                'strategy': name,
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
            })
            held['strategy'] = name
            held['decision_month'] = str(asof)
            held['target_month'] = str(asof + 1)
            held['capital_allocated'] = held.weight * float(stage['initial_capital'])
            positions.append(held[['strategy', 'decision_month', 'target_month',
                                   'cusip', 'weight', 'capital_allocated',
                                   'mu', 'duration', 'next_return']])
            print(f'{asof} | {name}: {portfolio_return:+.3%} vs '
                  f'benchmark {benchmark_return:+.3%}; '
                  f'coverage {coverage:.1%}; {len(held)} holdings')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame(monthly)
    summary = summarize(monthly)
    monthly.to_csv(OUTPUT_DIR / 'monthly_results.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'summary.csv', index=False)
    pd.concat(positions, ignore_index=True).to_csv(OUTPUT_DIR / 'holdings.csv', index=False)
    pd.DataFrame(forecasting).to_csv(OUTPUT_DIR / 'forecast_accuracy.csv', index=False)
    print('\nSTAGE 1 BACKTEST SUMMARY')
    print(summary.to_string(index=False))
    print(f'\nSaved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
