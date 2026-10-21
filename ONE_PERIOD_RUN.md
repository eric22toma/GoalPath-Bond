# Elastic net and one optimization period — configuration 1

Completed an executed historical prototype at 2024-12 month-end for January
2025, using the supplied three CSVs. Source files and the TA reference are intact.

## Sources and model

Optimization follows **MIE479 Deliverable 2**, section 4.1, printed pages 9–11
(file supplied by Bill: `/Users/billjiang/Desktop/MSF_Yr4_Fall/Capstone/MIE479___Deliverable__2.pdf`).

Predictors follow the list in [Prof Kwon Questions for Team Meeting #1](https://docs.google.com/document/d/1kionj5vFHl9naPaflLXFkYv77wMp4Y6VtIPjFYGGeF4/edit).

Implemented objective:

`gamma (w-b)' Q (w-b) - (mu'w - epsilon_mu sqrt(sum(theta_i w_i^2))) + kappa'(buy+sell)`

Constraints: weights sum to one; weights minus current weights equal buys minus
sells; nonnegative trades; long-only; binary selection/cardinality; duration,
convexity and each key-rate duration within benchmark bands; robust return at
least `(G/Wt)^(1/(T-t))-1`. The SOC epigraph uses a nonnegative eta. All exposures
are actually included, not represented by zeros. Transaction costs remain an
objective penalty, as written in the report. The extra realized wealth number
explicitly illustrates an ex-post cost deduction; this does not settle the
project's open self-financing convention.

## Forecast configuration

- Target: `YIELD(i,t+1) - YIELD(i,t)`, in percentage points, exact adjacent months.
- Pooled shared slopes and common intercept; bond-specific intercepts are not
  included in this first configuration.
- Predictors: SVENY10; 10Y minus 2Y slope; 2*5Y-2Y-10Y curvature;
  lagged level change per meeting formula; within-month daily 10Y-change standard
  deviation; lagged Fed funds change; VIX; maturity in years; own yield minus
  Svensson fitted zero yield at its maturity.
- Missing predictor: **core CPI inflation (CPILFESL)**. The supplied CPI series
  is not silently substituted. `--core-cpi` accepts a Date,CPILFESL CSV and adds
  12-month inflation lagged one month. No complete ten-predictor fit is claimed.
- Fed funds uses `(FF[t-1]-FF[t-2])` to add a release-delay assumption; this
  intentionally differs from the contemporaneous meeting formula.
- Svensson zero-yield formula is checked against supplied SVENY10 to 0.002
  percentage points. No outside dataset is used to replace the source curve.
- `sklearn alpha=0.001` (report's lambda), `l1_ratio=0.5` (report's alpha).
- One prespecified setting, not a claimed tuned optimum. Expanding-year
  validation targets 2021, 2022, 2023, 2024. Every scaler fits only its training
  observations; all bonds in a validation month remain together. Training target
  dates strictly precede validation targets. Final model uses target outcomes
  only through December 2024. January 2025 is held out.
- Ridge (alpha=1), lasso (alpha=.001), and no-yield-change comparators included.

Validation RMSE, basis points of monthly yield change:

| Year | Elastic net | Ridge | Lasso | No change |
|---|---:|---:|---:|---:|
|2021|104.08|104.36|103.87|101.13|
|2022|144.30|144.32|144.29|141.51|
|2023|158.36|158.47|158.26|155.39|
|2024|145.47|145.49|145.45|145.49|

Elastic net does not consistently improve on no-change. These files contain
extreme yield changes (including moves exceeding 100 percentage points), which
have NOT been clipped or removed. Verify field units and distressed/security
records before changing the training sample. Many bond observations share the
same monthly macro features: 600,000 panel rows are not 600,000 independent
macroeconomic observations.

## Optimization settings and input construction

- W0=Wt=$100,000; annual goal 3%; Y=1 year; T=12 months; t=0.
  Required monthly robust return = 0.246627%.
- Candidates: up to 40 highest as-of dollar-trading-volume bonds with complete
  trailing 36 monthly RET_EOM observations and valid current valuation inputs.
  Maturity must exceed one month; DEFAULTED=N; CONV=0. These screens are only
  views used for this small run. They do not modify the source universe.
- No future return availability is used to choose candidates.
- **Proxy benchmark**: amount outstanding times clean price, normalized across
  the candidate subset. This is NOT historical AGG membership/weights.
  Equal units across AMOUNT_OUTSTANDING observations are assumed.
- Illustrative existing weights equal proxy benchmark weights, rather than cash.
- At most 10 bonds; gamma=10; epsilon_mu=.5; buy/sell rates each .001.
  Thus costs = .002 times the reported half-L1 turnover, NOT .001 per turnover.
- Duration tolerance=.5 years; convexity tolerance=10 years squared;
  each KRD tolerance=.5 years, at keys 1,2,5,10,20,30 years.
- Q: Ledoit–Wolf covariance of trailing 36 supplied monthly total returns.
- Theta: per-bond variance of earlier validation return errors; pooled variance
  fallback when fewer than 12 bond errors exist; positive floor 1e-12.
  This is an illustrative uncertainty calibration, not validated epsilon_mu.
- Expected return: coupon/dirty-price/12 - estimated D*dy + .5*estimated C*dy^2,
  with dy converted from percentage points to decimal yields.
- D, C and KRD estimated by finite-difference repricing regular coupon cash
  flows on the Svensson curve plus a fitted constant spread. Coupon schedule
  approximated backwards from maturity; actual day-count and irregular coupons
  not modeled. KRD uses triangular interpolation of 1bp key-node bumps.
  These exposures are estimates, not source-verified risk analytics.
- CONV=0 does not prove absence of calls/options; conventional fixed-rate
  eligibility remains unverified. The 40 candidates are not presented as an
  approved investable set under the final scope.

## Executed result

Gurobi status 2 (optimal), reported MIP gap 0.0; 10 positive-weight holdings.

| Quantity | Result |
|---|---:|
|Forecast nominal monthly return|3.3324%|
|Robust monthly return|1.8608%|
|Monthly tracking-error standard deviation vs proxy|1.8034%|
|Transaction cost / portfolio value|0.14737%|
|Half-L1 turnover|73.6847%|
|Observed January 2025 portfolio return, before cost|0.88181%|
|Observed January 2025 proxy return|0.76705%|
|Observed active return|0.11476 percentage points|
|Illustrative realized wealth after cost|$100,733.14|

Largest weight is approximately 51.6% in CUSIP 71654QCC4. The optimistic forecast
is strongly affected by estimated yield mean reversion on a high-yield bond.
No issuer/concentration cap is part of Deliverable 2, so none was silently added.
This result is not evidence of a reliable strategy or successful AGG tracking.

Solver checks: budget residual zero; trade residual below 4e-18; robust goal
slack positive; duration deviation .1911; convexity deviation .3757; maximum
KRD deviation .5000. Constraint tolerances are independently checked in code.

## Run and outputs

```
python -m pip install -r requirements_one_period.txt
python elastic_net_one_period.py --output outputs/my_new_run
python -m unittest test_elastic_net_one_period.py
```

A working Gurobi license is required. Relative data paths are resolved from the
terminal's working directory; flags can supply alternative paths. Output must
be a NEW directory to prevent overwriting previous results.

Original outputs are preserved in `outputs/elastic_net_one_period/`. Refactored outputs are in `outputs/elastic_net_shared_processing/`:
report.json (settings, source SHA256, limitations and solve), portfolio.csv (all
40 candidates and weights/trades), coefficients.csv, forecast_validation.csv,
monthly_predictors.csv, historical_oos_errors.csv and risk_inputs.npz (Q,
Theta and identifier ordering). Original source CSVs stay unchanged.

## Verification boundaries and next steps

The run implements the Deliverable 2 optimization structure, with estimated
risk analytics, unverified security eligibility, and an illustrative benchmark.
It is NOT completion of the production model. Monthly labels do not establish
release-time availability. Source files are revised snapshots, not historical
vintages; adding simple lags does not remove revision bias. VIX last-trading-day
semantics are assumed from the supplied file, not independently verified.

Two focused synthetic tests passed: changing future labels cannot change the
final forecast/coefficients; key-rate sensitivities sum to parallel duration on
a simple coupon bond. Actual solve constraints were also checked. One
realization does not establish terminal goal probability; no terminal wealth
distribution or goal-achievement probability is claimed.

Next priorities: obtain CPILFESL with documented release/vintage timing;
verify yield outliers and bond eligibility; load genuine benchmark weights;
verify price, coupon and risk units; then tune a small grid and reevaluate
against no-change before running a rolling portfolio backtest.


## Updated chronological workflow: shared processing first

1. `process_investable_universe.py` prepares **all three CSVs** through
   `process_all_data()`: bonds, FRED monthly data, and daily Federal Reserve
   curve. Raw/prepared/report triples retain all original source rows/columns.
2. `elastic_net_one_period.py` imports this function, calls it first, and passes
   prepared DataFrames into `macro_features()` and `load_panel()`. It has no
   duplicate CSV parsers. Numeric/date parsing belongs to the processing module;
   economic predictors and exact one-month targets belong to the forecast module.
3. Elastic net standardizes training predictors, fits shared beta coefficients,
   and exports coefficients, intercepts, validation errors, and as-of forecasts.
4. Optional: continue to the unchanged Deliverable 2 optimization model.

The TA's `data_processing.py` remains intact. Keep both updated modules in the
same folder so Python resolves the import. You do not have to run the preparation
script separately: the elastic-net entry point calls preparation before fitting.

### Preparation only, optional inspection/export

```
python process_investable_universe.py --output-dir processed_all_three
```

This reads all three default CSVs and saves raw/prepared tables and a preparation
report. It refuses an existing destination. Exporting the bond copies takes
substantial disk space; the forecast can use prepared tables directly in memory.

### Prepare -> predictors -> beta coefficients, without optimization

```
python elastic_net_one_period.py --forecast-only --output outputs/beta_run_1
```

This needs pandas, NumPy, SciPy and scikit-learn. Gurobi is imported only if
optimization is requested. The source filenames remain configurable with
`--bonds`, `--fred`, and `--curve`. All three input files must be present locally.

### Understanding the exported beta coefficients

`coefficients.csv` includes each feature's coefficient for standardized inputs,
its coefficient in original predictor units, its training mean and scale, and
both intercept conventions. `intercept_original_inputs` is repeated per row for
convenience; use it ONCE when reconstructing a forecast:

`predicted_dy_pp = intercept_original_inputs + sum(beta_original_j * feature_j)`

For standardized coefficients, use each input `(x_j-mean_j)/scale_j` and the
standardized-input intercept. Do not mix the two intercept conventions.
Predicted changes are percentage points; divide by 100 before return mapping.

Canonical prepared units: bond YIELD and RET_* are decimal fractions; source
FRED and Federal Reserve rates retain percentage-point units. Coupon remains
percent of face per year. The feature code explicitly converts bond yield to
percentage points when constructing rich/cheap and the target. The prepared
source YIELD column itself remains decimal throughout.

Additional outputs: `preparation_report.json` and
`asof_predictors_and_forecasts.csv`. The latter lists actual predictor values
and predicted yield changes at the decision month; it can be regenerated rather
than committed as a large derived table. Core CPI is still missing and remains
optional, with no replacement by headline CPI.

Refactor verification: 3 focused tests passed, including an end-to-end synthetic
three-file preparation/unit-conversion test; the full 830,630-row run completed;
source hashes and fitted original-unit slopes match the original run. All three
source files have zero unexpected parsing failures. The numerical portfolio
result is unchanged to floating-point precision. Original run artifacts remain
available; updated outputs are saved separately.
