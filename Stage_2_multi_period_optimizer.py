"""Stage 2 multi-period benchmark tracking (Deliverable 3, Section 5.5).
Reuse Stage 1 candidate/risk inputs and its Gurobi relax-and-fix structure.
Plan H months; only the FIRST trade is executed in a future rolling backtest."""

import json
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from elastic_net_one_period import PROJECT_ROOT
from Stage_2_prepare_inputs import OUTPUT_FILE
from robust_mvo_one_period import (TRANSACTION_COST, DURATION_TOLERANCE, TIME_LIMIT, MAX_WEIGHT,)

# Initial H=3 demonstration; refine preference parameters after testing.
K = 10
GAMMA = 1.0                    # Weight on benchmark tracking-error variance
UNCERTAINTY_SCALE = 1.0        # Weight on rho.T @ |x - benchmark|
TERMINAL_TE_CAP = 0.025        # Maximum terminal expected MONTHLY tracking error
RESULTS_DIR = PROJECT_ROOT / 'Stage_2_outputs' / 'results'


def solve_quadratic(stage, k=K, allowed=None):
    """Continuous H-period optimization; optionally restrict the common bond universe."""
    c = stage['candidates']
    n, H = len(c), len(stage['periods'])
    Q = np.asarray(stage['candidate_covariance'], float)
    g = np.asarray(stage['risk_linear'], float)
    constant = float(stage['risk_constant'])
    b = c.benchmark_weight.to_numpy(float)
    duration = c.duration.to_numpy(float)
    previous = c.current_weight.to_numpy(float)
    outside = float(stage['forced_sale_weight'])
    cash_start = float(stage['cash_weight']) + (1 - TRANSACTION_COST) * outside
    maturity = [np.asarray(p['matures_in_month'], bool) for p in stage['periods']]
    # Market-value weights become principal cash at maturity (par = $100).
    principal_ratio = 100.0 / c.PRICE_EOM.to_numpy(float)
    m = gp.Model('GoalPath_Bond_Multi_Period')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = TIME_LIMIT
    w = [m.addMVar(n, lb=0, ub=MAX_WEIGHT, name=f'weights_{h}') for h in range(H)]
    buy = [m.addMVar(n, lb=0, name=f'buys_{h}') for h in range(H)]
    sell = [m.addMVar(n, lb=0, name=f'sells_{h}') for h in range(H)]
    cash = [m.addVar(lb=0, name=f'cash_{h}') for h in range(H)]
    active = [m.addMVar(n, lb=0, name=f'abs_active_{h}') for h in range(H)]
    objective = TRANSACTION_COST * outside
    excluded = sorted(set(range(n)) - set(allowed)) if allowed is not None else []
    already_matured = np.zeros(n, dtype=bool)

    for h, period in enumerate(stage['periods']):
        if h:
            already_matured |= maturity[h - 1]
        prior_bonds = previous if h == 0 else (1.0 - maturity[h - 1]) * w[h - 1]
        prior_cash = cash_start if h == 0 else (cash[h - 1] + (maturity[h - 1] * principal_ratio) @ w[h - 1])
        m.addConstr(w[h] - prior_bonds == buy[h] - sell[h], name=f'trade_balance_{h}')
        trades = buy[h].sum() + sell[h].sum()
        m.addConstr(cash[h] == prior_cash + sell[h].sum() - buy[h].sum() - TRANSACTION_COST * trades, name=f'cash_balance_{h}')
        m.addConstr(duration @ w[h] <= stage['benchmark_duration'] + DURATION_TOLERANCE)
        m.addConstr(duration @ w[h] >= stage['benchmark_duration'] - DURATION_TOLERANCE)
        if excluded:
            m.addConstr(w[h][excluded] == 0, name=f'fixed_universe_{h}')
        expired = np.flatnonzero(already_matured)
        if len(expired):
            m.addConstr(w[h][expired] == 0, name=f'expired_bonds_{h}')

        m.addConstr(active[h] >= w[h] - b, name=f'active_positive_{h}')
        m.addConstr(active[h] >= b - w[h], name=f'active_negative_{h}')
        mu = np.asarray(period['mu'], float)
        rho = np.asarray(period['rho'], float)
        risk = w[h] @ Q @ w[h] + g @ w[h] + constant
        objective += (GAMMA * risk - mu @ w[h] - stage['cash_monthly_return'] * cash[h] + UNCERTAINTY_SCALE * (rho @ active[h]) + TRANSACTION_COST * trades)

    # The terminal position is the final post-trade portfolio after any H-th-month maturities.
    survive = (~maturity[-1]).astype(float)
    terminal_q = Q * survive[:, None] * survive[None, :]
    terminal_g = g * survive
    terminal_risk = w[-1] @ terminal_q @ w[-1] + terminal_g @ w[-1] + constant
    m.addConstr(terminal_risk <= TERMINAL_TE_CAP**2, name='terminal_tracking_cap')
    m.setObjective(objective, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f'No feasible Stage 2 solution (Gurobi status {m.Status})')

    weights = np.vstack([x.X for x in w])
    buys = np.vstack([x.X for x in buy])
    sells = np.vstack([x.X for x in sell])
    cash_weights = np.array([x.X for x in cash])
    return {
        'weights': weights[0], 'buys': buys[0], 'sells': sells[0],
        'cash_weight': cash_weights[0],
        'planned_weights': weights, 'planned_buys': buys,
        'planned_sells': sells, 'planned_cash': cash_weights,
        'solver_status': int(m.Status), 'optimal': m.Status == GRB.OPTIMAL,
        'objective_value': float(m.ObjVal),}


def report_solution(stage, solution, k=K):
    """Planned holdings, duration, terminal tracking and first-month self-financing."""
    c = stage['candidates']
    w = solution['planned_weights']
    cash = solution['planned_cash']
    Q, g, offset = stage['candidate_covariance'], stage['risk_linear'], stage['risk_constant']
    duration = c.duration.to_numpy(float)
    monthly = []
    for h, period in enumerate(stage['periods']):
        x = w[h]
        holdings = int((x > 1e-6).sum())
        duration_gap = float(duration @ x - stage['benchmark_duration'])
        if holdings > k or abs(duration_gap) > DURATION_TOLERANCE + 1e-5 or cash[h] < -1e-6:
            raise ValueError(f'Invalid period {h+1}: holdings={holdings}, duration gap={duration_gap}')
        variance = max(0.0, float(x @ Q @ x + g @ x + offset))
        monthly.append({
            'month': period['month'], 'holdings': holdings,
            'cash_weight': float(cash[h]), 'duration_gap': duration_gap,
            'expected_return': float(np.asarray(period['mu']) @ x),
            'benchmark_expected_return': float(period['benchmark_mu']),
            'expected_tracking_error': float(np.sqrt(variance)),
            'transaction_cost_fraction': float(TRANSACTION_COST * (solution['planned_buys'][h].sum() + solution['planned_sells'][h].sum() + (stage['forced_sale_weight'] if h == 0 else 0))),
            })

    survive = ~np.asarray(stage['periods'][-1]['matures_in_month'], bool)
    terminal_w = w[-1] * survive
    terminal_var = max(0.0, float(terminal_w @ Q @ terminal_w + g @ terminal_w + offset))
    first_cost = monthly[0]['transaction_cost_fraction']
    return {
        'asof': stage['asof'], 'H': stage['horizon'],
        'K': k, 'forecast_model': stage['forecast_model'],
        'candidate_count': len(c), 'gamma': GAMMA,
        'uncertainty_scale': UNCERTAINTY_SCALE,
        'first_trade_cost_dollars': first_cost * stage['initial_capital'],
        'first_trade_cash_dollars': float(cash[0] * stage['initial_capital']),
        'terminal_tracking_error': float(np.sqrt(terminal_var)),
        'terminal_te_cap': TERMINAL_TE_CAP,
        'solver_status': solution['solver_status'],
        'optimal': solution['optimal'],
        'objective_value': solution['objective_value'],
        'monthly_plan': monthly,
    }


def run_solver(stage, k=K):
    """Stage 1-style relax-and-fix: continuous solve, select K, restricted re-solve."""
    relaxed = solve_quadratic(stage, k=k)
    # One selected universe for every planned period, ranked by total planned weight.
    top = set(np.argsort(-relaxed['planned_weights'].sum(axis=0), kind='stable')[:k])
    answer = solve_quadratic(stage, k=k, allowed=top)
    return answer, report_solution(stage, answer, k=k)


def main():
    stage = pd.read_pickle(OUTPUT_FILE)
    answer, metrics = run_solver(stage)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bonds = stage['candidates'].copy()
    bonds['optimized_weight'] = answer['weights']
    bonds['buy'] = answer['buys']
    bonds['sell'] = answer['sells']
    bonds['capital_allocated'] = bonds.optimized_weight * stage['initial_capital']
    # Match the bond-level percentage/basis-point columns in Stage 1.
    bonds['current_yield_pct'] = bonds.YIELD * 100
    bonds['expected_return_pct'] = np.asarray(stage['periods'][0]['mu'], float) * 100
    bonds['forecast_yield_change_bp'] = bonds.predicted_dy_pp * 100
    bonds.to_csv(RESULTS_DIR / 'Stage_2_portfolio.csv', index=False)

    planned = []
    # Direct forecasts use the predicted yield at the start of each planned month;
    # constant forecasts repeat the as-of H=1 change each month.
    forecast_start_yield = bonds.YIELD.to_numpy(float).copy()
    duration = bonds.duration.to_numpy(float)
    h1_change_bp = bonds.predicted_dy_pp.to_numpy(float) * 100
    for h, period in enumerate(stage['periods']):
        if stage['forecast_model'] == 'direct':
            period_change_bp = (forecast_start_yield / 12 - np.asarray(period['mu'], float)) * 10000 / duration
            forecast_start_yield += period_change_bp / 10000
        else:
            period_change_bp = h1_change_bp
        rows = pd.DataFrame({
            'month': period['month'], 'cusip': bonds.cusip,
            'weight': answer['planned_weights'][h],
            'buy': answer['planned_buys'][h], 'sell': answer['planned_sells'][h],
            'expected_return': period['mu'], 'uncertainty': period['rho'],
            'current_yield_pct': bonds['current_yield_pct'],
            'expected_return_pct': np.asarray(period['mu'], float) * 100,
            'forecast_yield_change_bp': period_change_bp,
        })
        planned.append(rows)
    pd.concat(planned, ignore_index=True).to_csv(
        RESULTS_DIR / 'Stage_2_planned_trades.csv', index=False)
    (RESULTS_DIR / 'Stage_2_optimization_report.json').write_text(
        json.dumps(metrics, indent=2) + '\n')

    print(f'\nSTAGE 2 | asof={metrics["asof"]}, H={metrics["H"]}, '
          f'K={K}, forecast={metrics["forecast_model"]}')
    print(f'Gurobi status={metrics["solver_status"]}; optimal={metrics["optimal"]}')
    for row in metrics['monthly_plan']:
        print(f'{row["month"]} | bonds={row["holdings"]} | cash={row["cash_weight"]:.2%} '
              f'| expected TE={row["expected_tracking_error"]:.3%} '
              f'| duration gap={row["duration_gap"]:+.3f} years')
    print(f'Terminal TE: {metrics["terminal_tracking_error"]:.3%} '
          f'(cap={TERMINAL_TE_CAP:.3%})')
    print(f'First-month fees (USD): {metrics["first_trade_cost_dollars"]:,.2f}; '
          f'remaining cash (USD): {metrics["first_trade_cash_dollars"]:,.2f}')
    selected = bonds.loc[bonds.optimized_weight > 1e-6,
                         ['cusip', 'optimized_weight', 'buy', 'sell', 'duration', 'capital_allocated',
                          'current_yield_pct', 'expected_return_pct', 'forecast_yield_change_bp']]
    print('\nFIRST-MONTH TRADES (only these execute in a backtest):')
    print(selected.to_string(index=False, float_format=lambda x: f'{x:,.6f}'))
    print(f'\nResults saved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
