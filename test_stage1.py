"""Checks for Deliverable 3 Section 5 portfolio initialization and optimization."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_stage1_inputs import (
    read_weight_file, bond_loadings, curve_factor_model, build_inputs,
    reduced_tracking_objective,
)
from robust_mvo_one_period import (
    objective_matrices, report_solution, run_solver, validate_stage,
)


def sample_stage():
    cusips = ['A','B','C','D','E']
    mu = np.array([0.004,0.005,0.003,0.006,0.002])
    duration = np.array([4.,5.,6.,7.,8.])
    benchmark = np.array([.2,.2,.2,.2,.2])
    factor = np.array([
        [-.04,.01,.02],[-.05,.02,.01],[-.06,.02,.02],
        [-.07,.02,.01],[-.08,.03,.02]
    ])
    data = pd.DataFrame({
        'cusip':cusips, 'mu':mu, 'duration':duration,
        'psi':np.full(5,.00001),
        'benchmark_weight':benchmark,
        'current_weight':benchmark,
        'B_level':factor[:,0], 'B_slope':factor[:,1],
        'B_curvature':factor[:,2],
    })
    sigma=np.array([[.1,.01,0],[.01,.08,0],[0,0,.07]])
    c=data.iloc[:4].copy()
    stage=dict(asof='2025-10',target_month='2025-11',
               eligible=data, candidates=c, benchmark_type='test',
               holdings_type='initial_hypothetical_benchmark',
               sigma_f=sigma, benchmark_factor=factor.T@benchmark,
               benchmark_mu=float(mu@benchmark),
               benchmark_duration=float(duration@benchmark),
               outside_benchmark_psi_constant=float(data.psi.iloc[4]*.2**2),
               outside_pretrade_weight=.2)
    Q, H, g, c0 = reduced_tracking_objective(stage)
    stage.update(candidate_covariance=Q, quadratic_matrix=H,
                 quadratic_linear=g, quadratic_constant=c0)
    return stage


class Stage1Tests(unittest.TestCase):
    def test_weight_csv_is_month_specific(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'weights.csv'
            p.write_text('DATE,cusip,weight\n2025-10-31,A,0.3\n'
                         '2025-10-31,B,0.7\n2025-09-30,A,1.0\n')
            weights=read_weight_file(p, asof=pd.Period('2025-10','M'))
            self.assertEqual(len(weights),2)
            self.assertAlmostEqual(weights.weight.sum(),1)

    def test_expected_squared_tracking_matches_full_covariance(self):
        stage=sample_stage()
        validate_stage(stage,2)
        A, linear, constant=objective_matrices(stage)
        w=np.array([0,0.5,0.5,0])
        qp=float(w@A@w+linear@w+constant)
        universe=stage['eligible']
        z=np.append(w,0) - universe.benchmark_weight.to_numpy(float)
        b=universe[['B_level','B_slope','B_curvature']].to_numpy(float)
        expected=float(
            (z@universe.mu.to_numpy(float))**2+
            (z@b)@stage['sigma_f']@(b.T@z)+
            np.sum(universe.psi.to_numpy(float)*z*z)
        )
        self.assertAlmostEqual(qp,expected,places=11)

    def test_bond_sensitivity_scales_by_duration(self):
        bonds=pd.DataFrame({'time_to_maturity':[2.,5.,10.],
                            'DURATION':[4.,5.,6.]})
        coefs=np.column_stack((np.ones(30),np.zeros(30),np.zeros(30)))
        B=bond_loadings(bonds,coefs)
        self.assertTrue(np.allclose(B[:,0],[-.04,-.05,-.06]))
        self.assertTrue(np.allclose(B[:,1:],0))

    def test_gurobi_support_and_duration(self):
        try:
            import gurobipy
        except ImportError:
            self.skipTest('Gurobi not installed')
        stage=sample_stage()
        # Covers the relax-and-fix procedure with full-benchmark risk offsets.
        sols=run_solver(stage,method='heuristic',k=2,
                        duration_tolerance=1.0,time_limit=15)
        answer,report=sols['heuristic']
        self.assertLessEqual(report['holdings_count'],2)
        self.assertAlmostEqual(float(sum(answer['weights'])),1.0,places=5)
        self.assertLessEqual(abs(report['duration_gap']),1.0+1e-6)
        self.assertGreater(report['expected_squared_active_return'],-1e-10)


if __name__ == '__main__':
    unittest.main()
