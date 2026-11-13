"""Monthly out-of-sample Stage 1 bond benchmark tracking backtest.
Reuses the existing forecasting, Stage 1 input preparation, and Gurobi solver.
Transaction fees are reported separately from fully invested bond capital."""

import numpy as np
import pandas as pd
from process_investable_universe import process_all_data, history_for_features, PROJECT_ROOT
from elastic_net_one_period import (BOND_COLUMNS, FEATURES, CV_PENALTY_GRID, macro_features, load_panel, forecast_month, training_rows, limit_tuning_rows)
from prepare_stage1_inputs import build_inputs
from scan_early_redemptions import EARLY_REDEMPTIONS
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


def days_30_360(start, end):
    """Simple 30/360 coupon accrual approximation (year fraction)."""
    return (360 * (end.year - start.year) + 30 * (end.month - start.month)+ min(end.day, 30) - min(start.day, 30)) / 360.0


def terminal_redemption_return(bond, redemption_date, principal_price):
    """Estimate the completed month's redemption return, in decimal units.
    Prices, coupon payments and accrued interest are per $100 face. Uses the
    decision-month dirty price, interim coupons and redemption proceeds.
    Redemption timing/price are used only AFTER portfolio optimization."""
    
    asof = pd.Timestamp(bond['DATE'])
    event = pd.Timestamp(redemption_date)
    clean = float(bond['PRICE_EOM'])
    coupon = float(bond['COUPON'])
    frequency = float(bond['NCOUPS']) if pd.notna(bond['NCOUPS']) else np.nan
    principal = float(principal_price)
    interim_coupons, accrued_redemption, estimated_accrued_start = 0.0, 0.0, 0.0
    if coupon > 0:
        next_coupon = pd.to_datetime(bond['nextcoup'], errors='coerce')
        step = pd.DateOffset(months=12 // int(frequency))
        previous_coupon = next_coupon - step
        estimated_accrued_start = coupon * days_30_360(previous_coupon, asof)
        due = next_coupon
        while due <= event:
            interim_coupons += coupon / frequency
            previous_coupon = due
            due += step
        accrued_redemption = coupon * days_30_360(previous_coupon, event)

    recorded_accrued = pd.to_numeric(bond['COUPACC'], errors='coerce')
    accrued_start = (float(recorded_accrued) if pd.notna(recorded_accrued) else estimated_accrued_start)
    dirty_start = clean + accrued_start
    total_return = (principal + accrued_redemption + interim_coupons) / dirty_start - 1.0
    return float(total_return)


def resolve_next_month_returns(bonds, target_month):
    """Apply actual redemptions only in realized-return evaluation, never in MVO.
    Prefer cash-flow reconstruction for redemption-month returns. An existing
    source RET_EOM is retained, but explicitly flagged, if reconstruction fails.
    Missing returns remain missing rather than being silently set to zero. """
    target = pd.Period(target_month, freq='M')
    events = {
        cusip: (pd.Timestamp(date), price)
        for cusip, (date, price) in EARLY_REDEMPTIONS.items()
        if pd.notna(pd.to_datetime(date, errors='coerce'))
        and pd.Timestamp(date).to_period('M') == target}
    result = bonds.copy()
    result['was_redeemed'] = False
    result['redemption_date'] = ''
    result['return_source'] = np.where(result.next_return.notna(), 'observed_RET_EOM', 'missing')
    result['return_note'] = ''
    for idx in result.index[result.cusip.isin(events)]:
        cusip = result.at[idx, 'cusip']
        event_date, principal = events[cusip]
        result.at[idx, 'was_redeemed'] = True
        result.at[idx, 'redemption_date'] = event_date.strftime('%Y-%m-%d')
        try:
            result.at[idx, 'next_return'] = terminal_redemption_return(
                result.loc[idx], event_date, principal
            )
            result.at[idx, 'return_source'] = 'estimated_redemption_30_360'
        except (ValueError, TypeError, OverflowError) as exc:
            result.at[idx, 'return_note'] = str(exc)
            result.at[idx, 'return_source'] = ('observed_redemption_month_unverified' if pd.notna(result.at[idx, 'next_return']) else 'unresolved_redemption')
    return result


def settle_portfolio(held, portfolio_return):
    """Create next month-end holdings and redemption cash after realized returns."""
    values = held.weight.to_numpy(float) * (1.0 + held.next_return.to_numpy(float))
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0 or not np.isclose(
        total, 1.0 + portfolio_return, atol=1e-5, rtol=0
    ):
        raise ValueError('Settled portfolio value does not match realized return')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid settled holding values')
    redeemed = held.was_redeemed.to_numpy(bool)
    surviving = held.loc[~redeemed, ['cusip']].copy().reset_index(drop=True)
    surviving['weight'] = values[~redeemed] / total
    return {
        'holdings': surviving,
        'cash_weight': float(values[redeemed].sum() / total),
    }


def evaluate_month(stage, solution):
    """Evaluate completed target-month returns, including actual redemptions."""
    candidates = stage['candidates']
    held = candidates.loc[solution['weights'] > 1e-6].copy()
    held['weight'] = solution['weights'][solution['weights'] > 1e-6]
    held = resolve_next_month_returns(held, stage['target_month'])
    if held.next_return.isna().any():
        missing = held.loc[held.next_return.isna(), 'cusip'].tolist()
        raise ValueError(f"Missing {stage['target_month']} realized returns for holdings: {missing}")
    portfolio_return = float(held.weight @ held.next_return)
    if not np.isfinite(portfolio_return):
        raise ValueError('Non-finite realized portfolio return')

    # Preserve the full unscreened benchmark; repair its known redemption
    # months before applying the pre-existing observed-return coverage rule.
    benchmark = resolve_next_month_returns(stage['benchmark'], stage['target_month'])
    observed = benchmark.next_return.notna()
    coverage = float(benchmark.loc[observed, 'benchmark_weight'].sum())
    if not np.isfinite(coverage) or coverage <= 0:
        raise ValueError(f"No observable benchmark return for {stage['target_month']}")
    benchmark_return = float(
        benchmark.loc[observed, 'benchmark_weight']
        @ benchmark.loc[observed, 'next_return'] / coverage
    )
    return held, portfolio_return, benchmark_return, coverage, benchmark


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
    print(f'\nDecision months: {START_MONTH} through {END_MONTH}; '
          f'realized returns: {pd.Period(START_MONTH, freq="M") + 1} through {pd.Period(END_MONTH, freq="M") + 1}')
    print('\nEVALUATION FORMULAS (realized monthly returns):')
    print('  Active return: A_t = R_P,t - R_B,t')
    print('  Annualized tracking error: TE = sqrt(12) * std(A_t, ddof=1)')
    print('  Annualized mean active return: AR = 12 * mean(A_t)')
    print('  Monthly one-way turnover: TO_t = (sum(buys_t) + sum(sells_t) + outside_t) / 2')
    print('  Transaction fees ($): F_t = capital_t * fee_rate * (sum(buys_t) + sum(sells_t) + outside_t)')
    print('  Holdings: N_t = sum_i 1(weight_i,t > 1e-6)')
    print('  Cumulative net return: R_net = (gross_capital_T - cumulative_fees_T) / initial_capital - 1')
    print('  R_P,t / R_B,t: portfolio / benchmark return; outside_t: forced sales outside candidates')
    print(f'\nProposed backtest comparisons grid ({len(configs)} independent portfolios):')
    print(pd.DataFrame(configs, columns=['forecast', 'K', 'solver', 'candidates']).to_string(index=False))
    states = {f'{model}_{method}_K{k}_N{n}': {'pretrade_state': None, 'capital': INITIAL_CAPITAL, 'fees': 0.0} for model, k, method, n in configs}
    monthly, positions, forecasting, failures = [], [], [], []
    elastic_params, baseline_penalties, tuned_year = None, None, None

    for asof in pd.period_range(START_MONTH, END_MONTH, freq='M'):
        # Hyperparameter retruning at beginning of year
        if asof.year != tuned_year:
            elastic_params, baseline_penalties, tuned_year = None, None, asof.year

        now, _, _, elastic_params = forecast_month(panel, asof, params=elastic_params)

        valid = now.loc[now.dy.notna()].copy()
        if not valid.empty:
            error = (valid.dy - valid.predicted_dy_pp) * 100
            row = {'target_month': str(asof + 1),
                'forecast_bonds': len(valid),
                'elastic_net_rmse_bp': float(np.sqrt(np.mean(error ** 2))),
                'random_walk_rmse_bp': float(np.sqrt(np.mean((valid.dy * 100) ** 2))),}
            row['elastic_net_mae_bp'] = float(np.mean(np.abs(error)))
            row['random_walk_mae_bp'] = float(np.mean(np.abs(valid.dy.to_numpy(float) * 100)))
            preds, baseline_penalties = forecast_baselines(panel, asof, now, baseline_penalties)
            for name, estimate in preds.items():
                difference = (now.dy.to_numpy(float) - estimate) * 100
                difference = difference[np.isfinite(difference)]
                row[f'{name}_rmse_bp'] = float(np.sqrt(np.mean(difference ** 2))) if len(difference) else np.nan
                row[f'{name}_mae_bp'] = float(np.mean(np.abs(difference))) if len(difference) else np.nan
            forecasting.append(row)

        history = panel.loc[panel.month.between(asof - 35, asof), ['cusip', 'month', 'RET_EOM', 'DURATION', 'time_to_maturity', 'RATING_NUM']].copy()
        bundle = {
            'asof': str(asof),
            'snapshot': now,
            'benchmark_snapshot': panel.loc[panel.month.eq(asof)].copy(),
            'history': history,
            'curve_params': curve_params,}

        successes = 0
        coverages = []
        for model, k, method, n in configs:
            name = f'{model}_{method}_K{k}_N{n}'
            state = states[name]
            if state.get('failed'):
                continue
            try:
                stage = build_inputs(bundle, pretrade_state=state['pretrade_state'],candidate_count=n, initial_capital=state['capital'],use_forecasts=(model == 'elastic_net'))
                answer, metrics = run_solver(stage, method=method, k=k)[method]
                held, portfolio_return, benchmark_return, coverage, benchmark = evaluate_month(stage, answer)
                next_state = settle_portfolio(held, portfolio_return)
            except (RuntimeError, ValueError) as exc:
                failures.append({'strategy': name, 'decision_month': str(asof),'reason': str(exc)})
                state['failed'] = True
                print(f'{asof} | {name}: not evaluable ({exc})')
                continue

            successes += 1
            coverages.append(coverage)
            # Fees are paid separately: bond capital stays fully invested.
            fee = float(metrics['transaction_cost_dollars'])
            state['fees'] += fee
            state['capital'] *= 1 + portfolio_return
            state['pretrade_state'] = next_state

            gross_traded = float(answer['buys'].sum() + answer['sells'].sum()+ stage['outside_pretrade_weight'])
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
                'redeemed_holdings_count': int(held.was_redeemed.sum()),
                'redemption_cash_weight': next_state['cash_weight'],
                'estimated_redemption_returns': int(
                    held.return_source.eq('estimated_redemption_30_360').sum()),
                'benchmark_estimated_redemption_returns': int(
                    benchmark.return_source.eq('estimated_redemption_30_360').sum()),
                'unverified_redemption_return_fallbacks': int(
                    held.return_source.eq('observed_redemption_month_unverified').sum()),
                'benchmark_unverified_redemption_return_fallbacks': int(
                    benchmark.return_source.eq('observed_redemption_month_unverified').sum()),
                'benchmark_unresolved_redemptions': int(
                    benchmark.return_source.eq('unresolved_redemption').sum()),
                'portfolio_duration': metrics['portfolio_duration'],
                'benchmark_duration': metrics['benchmark_duration'],
                'gross_capital': state['capital'],
                'net_capital': state['capital'] - state['fees'],
                'optimal': metrics['optimal'],
                'objective_value': metrics['objective_value'],})
            held['strategy'] = name
            held['decision_month'] = str(asof)
            held['target_month'] = str(asof + 1)
            held['capital_allocated'] = held.weight * float(stage['initial_capital'])
            positions.append(held[['strategy', 'decision_month', 'target_month', 'cusip', 'weight', 'capital_allocated', 'mu', 'duration', 'next_return', 'was_redeemed', 'redemption_date', 'return_source', 'return_note']])
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
    files = [OUTPUT_DIR / name for name in ('monthly_results.csv', 'summary.csv', 'holdings.csv', 'forecast_accuracy.csv', 'failed_strategies.csv')]
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