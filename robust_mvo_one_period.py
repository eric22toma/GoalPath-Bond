"""One-period benchmark tracking using the saved GoalPath-Bond Stage 1 inputs.
All forecasts, benchmark weights, factor exposures, and quadratic objective
terms are prepared by prepare_stage1_inputs.py and stored in stage1_inputs.pkl."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / 'stage1_outputs' / 'inputs' / 'stage1_inputs.pkl'
RESULTS_DIR = ROOT / 'stage1_outputs' / 'results'

# Optimization settings (to be changed here - fixed for initial one-period testing).
METHOD = 'exact'                  # 'exact', 'heuristic', or 'both'
K = 5                             # Maximum number of selected bonds
MAX_WEIGHT = 1.0                  # Maximum weight in any single bond
TRANSACTION_COST = 0.001          # 0.1% of traded notional
DURATION_TOLERANCE = 0.5          # Years, relative to benchmark duration
TIME_LIMIT = 60                   # Seconds per Gurobi solve
FACTORS = ('level', 'slope', 'curvature')


def load_stage(path=INPUT_FILE):
    stage = pd.read_pickle(path)
    c = stage['candidates']
    n = len(c)
    needed = ['cusip', 'mu', 'psi', 'duration', 'benchmark_weight', 'current_weight']
    needed += [f'B_{factor}' for factor in FACTORS]
    return stage


def solve_quadratic(stage, k, max_weight, transaction_cost, duration_tolerance, time_limit, allowed=None, exact=False):
    """Optimize staged quadratic tracking error with optional cardinality."""
    c = stage['candidates']
    n = len(c)
    H = np.asarray(stage['quadratic_matrix'], dtype=float)
    g = np.asarray(stage['quadratic_linear'], dtype=float)
    constant = float(stage['quadratic_constant'])
    previous = c.current_weight.to_numpy(float)
    duration = c.duration.to_numpy(float)
    benchmark_duration = float(stage['benchmark_duration'])

    m = gp.Model('GoalPath_Bond_One_Period')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.001
    w = m.addMVar(n, lb=0, ub=max_weight, name='weight')
    buy = m.addMVar(n, lb=0, name='buy')
    sell = m.addMVar(n, lb=0, name='sell')

    m.addConstr(w.sum() == 1, name='fully_invested')
    m.addConstr(w - previous == buy - sell, name='trade_balance')
    m.addConstr(duration @ w <= benchmark_duration + duration_tolerance)
    m.addConstr(duration @ w >= benchmark_duration - duration_tolerance)

    if allowed is not None:
        excluded = [i for i in range(n) if i not in allowed]
        if excluded:
            m.addConstr(w[excluded] == 0, name='excluded_candidates')
    if exact:
        held = m.addMVar(n, vtype=GRB.BINARY, name='held')
        m.addConstr(w <= max_weight * held)
        m.addConstr(held.sum() <= k, name='cardinality')

    trading = buy.sum() + sell.sum() + stage['outside_pretrade_weight']
    m.setObjective(w @ H @ w + g @ w + constant + transaction_cost * trading,
                   GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        if m.Status == GRB.INFEASIBLE:
            raise RuntimeError('No portfolio satisfies the investment and duration constraints.')
        raise RuntimeError(f'Gurobi returned no portfolio (status {m.Status}).')

    return {
        'weights': w.X,
        'buys': buy.X,
        'sells': sell.X,
        'solver_status': int(m.Status),
        'optimal': m.Status == GRB.OPTIMAL,
        'mip_gap': float(m.MIPGap) if exact else None,
    }


def report_solution(stage, solution, k, transaction_cost, method, duration_tolerance):
    """Measure the solution against the complete staged benchmark."""
    c = stage['candidates']
    w = solution['weights']
    b = c.benchmark_weight.to_numpy(float)
    mu = c.mu.to_numpy(float)
    psi = c.psi.to_numpy(float)
    B = c[[f'B_{name}' for name in FACTORS]].to_numpy(float)

    active_factor = B.T @ w - stage['benchmark_factor']
    active_mean = float(mu @ w - stage['benchmark_mu'])
    factor_variance = float(active_factor @ stage['sigma_f'] @ active_factor)
    residual_variance = float(psi @ (w - b)**2 + stage['outside_benchmark_psi_constant'])
    tracking_variance = max(0.0, factor_variance + residual_variance)
    squared_tracking = tracking_variance + active_mean**2

    buys = float(solution['buys'].sum())
    sells = float(solution['sells'].sum() + stage['outside_pretrade_weight'])
    cost = transaction_cost * (buys + sells)
    capital = float(stage['initial_capital'])
    portfolio_duration = float(c.duration @ w)
    holdings = int((w > 1e-6).sum())
    if holdings > k or not np.isclose(w.sum(), 1.0, atol=1e-5):
        raise ValueError('Portfolio violates holdings or investment constraint.')
    if abs(portfolio_duration - stage['benchmark_duration']) > duration_tolerance + 1e-5:
        raise ValueError('Portfolio violates benchmark duration constraint.')

    return {
        'method': method,
        'asof': stage['asof'],
        'target_month': stage['target_month'],
        'benchmark_type': stage['benchmark_type'],
        'candidate_count': len(c),
        'holdings_count': holdings,
        'K': k,
        'portfolio_expected_return': float(mu @ w),
        'benchmark_expected_return': float(stage['benchmark_mu']),
        'expected_active_return': active_mean,
        'active_factor_variance': factor_variance,
        'active_residual_variance': residual_variance,
        'tracking_variance': tracking_variance,
        'expected_squared_active_return': squared_tracking,
        'expected_monthly_tracking_error': float(np.sqrt(squared_tracking)),
        'transaction_cost_fraction': cost,
        'transaction_cost_dollars': cost * capital,
        'portfolio_capital_dollars': capital,
        'buy_notional_dollars': capital * buys,
        'sell_notional_dollars': capital * sells,
        'benchmark_duration': float(stage['benchmark_duration']),
        'portfolio_duration': portfolio_duration,
        'duration_gap': portfolio_duration - stage['benchmark_duration'],
        'objective_value': squared_tracking + cost,
        'solver_status': solution['solver_status'],
        'optimal': solution['optimal'],
        'mip_gap': solution['mip_gap'],
    }


def run_solver(stage, method=METHOD, k=K, max_weight=MAX_WEIGHT,
               transaction_cost=TRANSACTION_COST,
               duration_tolerance=DURATION_TOLERANCE, time_limit=TIME_LIMIT):
    """Solve exactly, by relax-and-fix heuristic, or both for comparison."""
    if method not in ('exact', 'heuristic', 'both') or k < 1 or k > len(stage['candidates']):
        raise ValueError('Invalid solver method or holdings limit.')
    if k * max_weight < 1 or transaction_cost < 0 or duration_tolerance < 0:
        raise ValueError('Infeasible weight limit or invalid cost/duration settings.')

    def solve(allowed=None, exact=False):
        return solve_quadratic(stage, k, max_weight, transaction_cost,
                               duration_tolerance, time_limit, allowed, exact)

    results = {}
    if method in ('heuristic', 'both'):
        relaxed = solve()
        top = set(np.argsort(-relaxed['weights'], kind='stable')[:k])
        restricted = solve(allowed=top)
        results['heuristic'] = (
            restricted, report_solution(stage, restricted, k, transaction_cost,
                                        'relax_and_fix', duration_tolerance)
        )
    if method in ('exact', 'both'):
        answer = solve(exact=True)
        results['exact'] = (
            answer, report_solution(stage, answer, k, transaction_cost,
                                    'exact_miqp', duration_tolerance)
        )
    return results


def main():
    """Run optimization using only saved Stage 1 inputs."""
    stage = load_stage()
    results = run_solver(stage)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, (solution, metrics) in results.items():
        portfolio = stage['candidates'].copy()
        portfolio['optimized_weight'] = solution['weights']
        portfolio['buy'] = solution['buys']
        portfolio['sell'] = solution['sells']
        portfolio.to_csv(RESULTS_DIR / f'portfolio_{name}.csv', index=False)
        report[name] = metrics
        selected = portfolio.loc[portfolio.optimized_weight > 1e-6,
                                 ['cusip', 'optimized_weight', 'mu', 'duration']]
        print(f'\n{name.upper()}: {len(selected)} bonds, '
              f'tracking error {metrics["expected_monthly_tracking_error"]:.4%}, '
              f'fees ${metrics["transaction_cost_dollars"]:,.2f}')
        print(selected.to_string(index=False))

    if 'heuristic' in report and 'exact' in report:
        report['comparison'] = {
            'heuristic_minus_exact_objective':
                report['heuristic']['objective_value'] - report['exact']['objective_value']
        }
    (RESULTS_DIR / 'stage1_optimization_report.json').write_text(
        json.dumps(report, indent=2) + '\n'
    )
    print(f'\nResults saved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
