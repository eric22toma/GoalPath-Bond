"""Stage 1 H=1 bond benchmark tracking (Deliverable 3, Sections 5.2–5.3).

Reads ONLY the prepared stage1_inputs.pkl. All bond, forecast, benchmark
and factor-covariance preparation belongs in prepare_stage1_inputs.py.

Minimize expected squared active return plus 0.1% of traded value:
   (mu' (w-b))^2 + (w-b)' (B Sigma_f B' + Psi) (w-b)
      + kappa * sum(buys + sells).
The full eligible benchmark is retained in the benchmark exposure offsets,
even when the tradable candidate universe is reduced for a test solve.

Methods:
  heuristic: relax cardinality, select K largest weights, convex re-solve
  exact:     binary cardinality via MIQP on the reduced candidate universe
  both:      compare identical inputs with both methods
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_stage1_inputs import DEFAULT_STAGE_DIR, DEFAULT_RESULTS

FACTOR_NAMES = ('level', 'slope', 'curvature')


def validate_stage(stage, k):
    """Validate the staged model inputs before opening a solver."""
    if pd.Period(stage['target_month'], 'M') != pd.Period(stage['asof'], 'M') + 1:
        raise ValueError('Stage 1 accepts H=1 only')
    eligible = stage['eligible']
    c = stage['candidates']
    needed = ['cusip', 'mu', 'psi', 'duration', 'benchmark_weight',
              'current_weight'] + [f'B_{x}' for x in FACTOR_NAMES]
    for name, table in [('eligible', eligible), ('candidates', c)]:
        missing = set(needed) - set(table.columns)
        if missing:
            raise ValueError(f'{name}: missing Stage 1 fields {sorted(missing)}')
        if table.cusip.duplicated().any():
            raise ValueError(f'{name}: duplicate CUSIPs')
        if not np.isfinite(table[needed[1:]].to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: nonfinite model inputs')
    if k < 1 or k > len(c):
        raise ValueError('K must be between 1 and number of candidates')
    if not np.isclose(eligible.benchmark_weight.sum(), 1.0, atol=1e-8):
        raise ValueError('Full eligible benchmark weights must total 1')
    sigma = np.asarray(stage['sigma_f'], dtype=float)
    if sigma.shape != (3,3) or not np.isfinite(sigma).all():
        raise ValueError('Factor covariance must be finite and 3x3')
    if np.linalg.eigvalsh((sigma+sigma.T)/2).min() < -1e-8:
        raise ValueError('Factor covariance not positive semidefinite')
    if (c.psi <= 0).any() or (eligible.psi <= 0).any():
        raise ValueError('Residual covariance diagonal must be positive')
    cash = float(stage.get('pretrade_cash_weight', 0.0))
    if cash < -1e-8 or cash > 1+1e-8:
        raise ValueError('Pre-trade funding weight must be between 0 and 1')
    if stage['outside_pretrade_weight'] < -1e-8:
        raise ValueError('Negative forced sale weight')
    if not np.isclose(
        c.current_weight.sum() + stage['outside_pretrade_weight'] + cash,
        1.0, atol=1e-8
    ):
        raise ValueError('Pre-trade bond holdings plus initial funding must '
                         'sum to one')
    if cash > 1-1e-8 and (c.current_weight.abs().max() > 1e-8 or
                           stage['outside_pretrade_weight'] > 1e-8):
        raise ValueError('New investment must start with zero existing bonds')


def objective_matrices(stage):
    """Use the covariance and objective assembled by prepare_stage1_inputs.py."""
    n = len(stage['candidates'])
    required = ['candidate_covariance', 'quadratic_matrix',
                'quadratic_linear', 'quadratic_constant']
    absent = [key for key in required if key not in stage]
    if absent:
        raise ValueError(f'Incomplete Stage 1 initialization: {absent}')
    covariance = np.asarray(stage['candidate_covariance'], dtype=float)
    A = np.asarray(stage['quadratic_matrix'], dtype=float)
    linear = np.asarray(stage['quadratic_linear'], dtype=float)
    constant = float(stage['quadratic_constant'])
    if (covariance.shape != (n,n) or A.shape != (n,n)
            or linear.shape != (n,) or not np.isfinite(constant)
            or not np.isfinite(A).all() or not np.isfinite(linear).all()):
        raise ValueError('Invalid staged candidate covariance or quadratic terms')
    if np.linalg.eigvalsh((A+A.T)/2).min() < -1e-7:
        raise ValueError('Staged objective matrix must be positive semidefinite')
    return A, linear, constant


def solve_quadratic(stage, allowed, k, max_weight, transaction_cost,
                    time_limit, exact):
    """Solve a convex quadratic tracking problem, optionally with cardinality."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError('Install gurobipy and configure a Gurobi license') from exc
    c = stage['candidates']
    n = len(c)
    if allowed is not None and len(allowed)*max_weight < 1-1e-9:
        raise ValueError('Allowed support and max weight cannot sum to one')
    if exact and k*max_weight < 1-1e-9:
        raise ValueError('K times max_weight must be >= 1')
    A, lin, const = objective_matrices(stage)
    previous = c.current_weight.to_numpy(dtype=float)
    durations = c.duration.to_numpy(dtype=float)

    m = gp.Model('GoalPath_Stage1_H1_Active_Tracking')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = 0.001
    nvars = range(n)
    w = m.addVars(nvars, lb=0, ub=max_weight, name='w')
    buy = m.addVars(nvars, lb=0, name='buy')
    sell = m.addVars(nvars, lb=0, name='sell')
    m.addConstr(gp.quicksum(w[i] for i in nvars) == 1, 'full_investment')
    for i in nvars:
        m.addConstr(w[i] - previous[i] == buy[i] - sell[i], f'trade_balance_{i}')
        if allowed is not None and i not in allowed:
            m.addConstr(w[i] == 0, f'fixed_zero_{i}')
    if exact:
        z = m.addVars(nvars, vtype=GRB.BINARY, name='held')
        for i in nvars:
            m.addConstr(w[i] <= max_weight * z[i], f'cardinality_link_{i}')
        m.addConstr(gp.quicksum(z[i] for i in nvars) <= k, 'cardinality')
    dbenchmark = float(stage['benchmark_duration'])
    exposure = gp.quicksum(durations[i]*w[i] for i in nvars)
    eps = float(stage['duration_tolerance'])
    m.addConstr(exposure <= dbenchmark + eps, 'duration_upper')
    m.addConstr(exposure >= dbenchmark - eps, 'duration_lower')

    quadratic = gp.QuadExpr()
    for i in nvars:
        quadratic.add(A[i,i] * w[i] * w[i])
        for j in range(i+1,n):
            if abs(A[i,j]) > 1e-17:
                quadratic.add(2*A[i,j]*w[i]*w[j])
    linear = gp.quicksum(lin[i]*w[i] for i in nvars)
    traded = gp.quicksum(buy[i] + sell[i] for i in nvars)
    m.setObjective(quadratic + linear + const
                   + transaction_cost*(traded+stage['outside_pretrade_weight']),
                   GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        status = str(m.Status)
        if m.Status == GRB.INFEASIBLE:
            m.computeIIS()
            bad = [q.ConstrName for q in m.getConstrs() if q.IISConstr]
            raise RuntimeError(f'Infeasible Stage 1 optimization; IIS: {bad[:15]}')
        raise RuntimeError(f'Gurobi found no feasible portfolio; status={status}')
    weights = np.array([w[i].X for i in nvars])
    buys = np.array([buy[i].X for i in nvars])
    sells = np.array([sell[i].X for i in nvars])
    return {
        'weights': weights, 'buys': buys, 'sells': sells,
        'solver_status': int(m.Status),
        'optimal': bool(m.Status == GRB.OPTIMAL),
        'mip_gap': float(m.MIPGap) if exact else None
    }


def report_solution(stage, answer, k, costs, method):
    """Recalculate the full-benchmark objective directly for auditability."""
    c = stage['candidates']
    w = answer['weights']
    b = c.benchmark_weight.to_numpy(float)
    mu = c.mu.to_numpy(float)
    psi = c.psi.to_numpy(float)
    B = c[[f'B_{name}' for name in FACTOR_NAMES]].to_numpy(float)
    fac = B.T @ w - stage['benchmark_factor']
    active_mean = float(mu @ w - stage['benchmark_mu'])
    factor_variance = float(fac @ stage['sigma_f'] @ fac)
    residual_variance = float(psi @ np.square(w-b)
                              + stage['outside_benchmark_psi_constant'])
    tracking_variance = max(0., factor_variance + residual_variance)
    buy_total = float(sum(answer['buys']))
    sell_total = float(sum(answer['sells']) + stage['outside_pretrade_weight'])
    traded_fraction = buy_total + sell_total
    tc = float(costs * traded_fraction)
    capital = float(stage.get('initial_capital', 100_000.0))
    expected_sq = tracking_variance + active_mean**2
    actual_duration = float(c.duration @ w)
    held = int((w > 1e-6).sum())
    if held > k or abs(w.sum()-1) > 1e-5:
        raise AssertionError('Portfolio violates cardinality or investment constraints')
    if abs(actual_duration-stage['benchmark_duration']) > stage['duration_tolerance']+1e-5:
        raise AssertionError('Duration band violated')
    if not np.allclose(
        w - c.current_weight.to_numpy(float),
        answer['buys'] - answer['sells'], atol=1e-5
    ):
        raise AssertionError('Trade definition violated')
    return {
        'method': method, 'asof': stage['asof'],
        'target_month': stage['target_month'],
        'benchmark_type': stage['benchmark_type'],
        'initial_holdings_type': stage['holdings_type'],
        'candidate_count': len(c), 'holdings_count': held, 'K': k,
        'portfolio_expected_return': float(mu @ w),
        'benchmark_expected_return': float(stage['benchmark_mu']),
        'expected_active_return': active_mean,
        'active_factor_variance': factor_variance,
        'active_residual_variance': residual_variance,
        'tracking_variance': tracking_variance,
        'expected_squared_active_return': expected_sq,
        'expected_monthly_tracking_error': float(np.sqrt(expected_sq)),
        'transaction_cost_fraction': tc,
        'transaction_cost_dollars': tc * capital,
        'portfolio_capital_dollars': capital,
        'initial_uninvested_cash_dollars':
            capital * float(stage.get('pretrade_cash_weight', 0.0)),
        'buy_notional_dollars': capital * buy_total,
        'sell_notional_dollars': capital * sell_total,
        'gross_traded_fraction': traded_fraction,
        'transaction_cost_budget_convention':
            'Stage 1 fee penalty; fees funded separately from invested capital',
        'objective_value': expected_sq + tc,
        'one_way_turnover': float(0.5*traded_fraction),
        'benchmark_duration': float(stage['benchmark_duration']),
        'portfolio_duration': actual_duration,
        'duration_gap': actual_duration-stage['benchmark_duration'],
        'forced_sale_outside_candidates': float(stage['outside_pretrade_weight']),
        'solver_status': answer['solver_status'],
        'optimal': answer['optimal'],
        'mip_gap': answer['mip_gap']
    }


def run_solver(stage, method='heuristic', k=10, max_weight=1.0,
               transaction_cost=0.001, duration_tolerance=0.5,
               time_limit=60):
    if method not in ('heuristic','exact','both'):
        raise ValueError('method must be heuristic, exact, or both')
    if not 0 < max_weight <= 1 or transaction_cost < 0 or duration_tolerance < 0:
        raise ValueError('Invalid weight, transaction-cost, or duration parameters')
    stage = dict(stage)
    stage['duration_tolerance'] = duration_tolerance
    validate_stage(stage, k)
    if k*max_weight < 1-1e-9:
        raise ValueError('K * max_weight must be >= 1')
    results = {}
    if method in ('heuristic','both'):
        relaxed = solve_quadratic(
            stage, None, k, max_weight, transaction_cost, time_limit, exact=False
        )
        top = np.argsort(-relaxed['weights'], kind='stable')[:k]
        restricted = solve_quadratic(
            stage, set(top.tolist()), k, max_weight, transaction_cost,
            time_limit, exact=False
        )
        results['heuristic'] = (restricted, report_solution(
            stage, restricted, k, transaction_cost, 'relax_and_fix'))
    if method in ('exact','both'):
        exact = solve_quadratic(
            stage, None, k, max_weight, transaction_cost, time_limit, exact=True
        )
        results['exact'] = (exact, report_solution(
            stage, exact, k, transaction_cost, 'exact_miqp'))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage_file', nargs='?', type=Path,
                        default=DEFAULT_STAGE_DIR / 'stage1_inputs.pkl',
                        help='Optional staging pickle (default: stage1_outputs/inputs/stage1_inputs.pkl)')
    parser.add_argument('--output', type=Path, default=DEFAULT_RESULTS / 'results',
                        help='Results folder (default: stage1_outputs/results)')
    parser.add_argument('--method', choices=['heuristic','exact','both'], default='heuristic')
    parser.add_argument('--max-holdings', type=int, default=5,
                        help='Stage 1 K; VS Code demonstration default: 5')
    parser.add_argument('--max-weight', type=float, default=1.0)
    parser.add_argument('--trade-cost', type=float, default=0.001,
                        help='0.1%% of traded portfolio value by default')
    parser.add_argument('--duration-tolerance', type=float, default=0.5)
    parser.add_argument('--time-limit', type=float, default=60)
    args = parser.parse_args()
    if not args.stage_file.is_file():
        parser.error(
            f'Stage 1 inputs not found: {args.stage_file}. '
            'Click Run on prepare_stage1_inputs.py first, then run this file.'
        )
    print(f'Loading staged inputs: {args.stage_file.resolve()}')
    stage = pd.read_pickle(args.stage_file)
    results = run_solver(
        stage, method=args.method, k=args.max_holdings,
        max_weight=args.max_weight, transaction_cost=args.trade_cost,
        duration_tolerance=args.duration_tolerance, time_limit=args.time_limit
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for method, (answer, metrics) in results.items():
        candidates = stage['candidates'].copy()
        candidates['optimized_weight'] = answer['weights']
        candidates['buy'] = answer['buys']
        candidates['sell'] = answer['sells']
        candidates.to_csv(args.output / f'portfolio_{method}.csv', index=False)
        report[method] = metrics
        print(f'\n{method.upper()} (Stage 1, H=1)')
        print(json.dumps(metrics, indent=2))
        selected = candidates.loc[candidates.optimized_weight > 1e-6,
                                  ['cusip','optimized_weight','mu','duration',
                                   'benchmark_weight','current_weight']]
        print(selected.to_string(index=False))
    if 'exact' in report and 'heuristic' in report:
        report['comparison'] = {
            'heuristic_minus_exact_objective': (
                report['heuristic']['objective_value'] -
                report['exact']['objective_value']
            )
        }
    (args.output/'stage1_optimization_report.json').write_text(
        json.dumps(report,indent=2)+'\n'
    )
    print('\nONE-PERIOD INVESTMENT SUMMARY')
    print(f'Initial capital: ${float(stage.get("initial_capital", 100_000.0)):,.2f}')
    print(f'Pretrade cash funding: '
          f'${float(stage.get("initial_capital", 100_000.0)) * float(stage.get("pretrade_cash_weight", 0.0)):,.2f}')
    print('Benchmark: full eligible universe (proxy unless original AGG '
          'holding weights supplied)')
    for label, metrics in report.items():
        if label == 'comparison':
            continue
        print(f'{label}: {metrics["holdings_count"]} selected bonds; '
              f'capital invested ${metrics["portfolio_capital_dollars"]:,.2f}; '
              f'buys ${metrics["buy_notional_dollars"]:,.2f}; '
              f'sells ${metrics["sell_notional_dollars"]:,.2f}; '
              f'transaction fees ${metrics["transaction_cost_dollars"]:,.2f}')
    print('Stage 1 fees are not deducted from invested capital; '
          'explicit cash/self-financing is deferred to Stage 2.')
    print(f'\nSUCCESS: Results saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
