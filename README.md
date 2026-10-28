# GoalPath-Bond

Prototype bond-level yield forecasting and one-period robust portfolio optimization.

## Local setup

Clone the repository and install the forecasting dependencies:

```bash
python3 -m pip install pandas numpy scipy scikit-learn
```

The `01_Source_Data` directory contains the FRED and Treasury CSV files and the
compressed `Investable_universe_AGG.csv.zip`. The processing script reads the
ZIP directly when the extracted CSV is absent. **Do not commit the extracted
CSV**; it exceeds GitHub's file-size limit.

## Prepare and inspect source data

```bash
python3 process_investable_universe.py
```

Preparation reports the 2016–2025 data coverage and first three rows of each
dataset. Optional `--output-dir` exports tables to a NEW directory. Prefer
an output directory outside the repository so large data files are not added
to Git.

## Elastic-net yield forecast (H = 1 month only)

```bash
python3 elastic_net_one_period.py --asof latest
```

This builds a predictor row per bond-month and trains a pooled elastic net to
forecast the next calendar month's **change in bond yield**, not its total
return. The default selects the latest forecast origin for which the **following month's**
quoted prices pass a basic cross-sectional movement check. The raw December
2025 prices appear largely unchanged from November, so that month is flagged
as unsuitable for performance assessment; by default, the latest verified
comparison is October 2025 → November 2025. An explicit origin such as
`--asof 2025-11` is allowed but will display the data-quality warning.

The script tests 25 mixing-alpha/penalty-lambda combinations using chronological
annual validation (up to three recent folds), or sequential monthly validation
while the sample is too short for complete annual folds. Hyperparameter search samples up
to 150 bond observations per month for tractable runtime; the final fit uses
all complete historical training observations. The program reports the top
settings, fitted coefficients, three illustrative bond forecasts with actual
yield and quoted-price changes (selected retrospectively *for display only*),
full-sample error metrics for credible target months, and predictor contributions.

Forward-rate inputs are optional (`--include-forward`). Override tuning with
`--no-tune --alpha 0.5 --lambda 0.001` when debugging.

To evaluate monthly expanding-window forecasts and save output tables:

```bash
python3 elastic_net_one_period.py --asof latest --backtest \
  --output-dir ~/GoalPath-H1-Results
```

The backtest starts by default at the **earliest possible forecast target**
after at least 12 observed training target months, instead of a hard-coded
2019 start. Use `--backtest-start YYYY-MM` to override. Earlier observations
remain in-sample training history. The full multi-year backtest can be
computationally expensive because the model refits each month.

The output directory must not already exist. With `--backtest`, the output
includes `elastic_net_outputs.pkl`, which can be passed directly to
`prepare_stage1_inputs.py`; otherwise staging invokes the existing
elastic-net functions itself.

## Stage 1 tracking initialization and one-month optimization

### Run directly in VS Code

With the repository open and Python/Gurobi dependencies installed, use the
**Run Python File** arrow in this order:

1. `process_investable_universe.py` — reports data loading.
2. `elastic_net_one_period.py` — displays the one-month forecasts (unchanged).
3. `prepare_stage1_inputs.py` — **no arguments required**. Because the
   elastic-net Run button doesn't save a pickle by default, staging calls the
   existing elastic-net functions to rebuild its own H=1 forecast and compute
   portfolio inputs. It automatically uses an **explicitly labeled proxy**
   benchmark if no AGG constituent-weight file has been supplied.
4. `robust_mvo_one_period.py` — **no arguments required**. Reads the staging
   file written by Step 3 and solves the Stage 1 portfolio.

Files are written under `stage1_outputs/inputs/` and
`stage1_outputs/results/` (ignored by Git). Re-running either script updates
its own generated files, so you don't need to create new directories.
For a fast demonstration, defaults are 20 candidate bonds, K=5, 0.1% trading
costs, and a ±0.5-year duration band, using the relax-and-fix method.
Both Python scripts resolve paths from the repository location rather than
the current VS Code working directory.

This automatic staging re-fits the elastic net because the unchanged
forecasting script prints but does not save a bundle on an ordinary Run.
When a saved bundle exists, optionally provide its file path to avoid
re-fitting. To see the source data's first rows, run the processing script
independently (no saved preprocessing files are required).

Section 5 of Deliverable 3 defines the H=1 objective as expected **squared
active return**, not a separate reward for exceeding expected benchmark return:

```text
(w-b)' Q (w-b) + (mu' (w-b))^2 + 0.001 * total traded weight
Q = B Sigma_f B' + Psi
mu = YIELD / 12 - WRDS_Duration * (forecast_yield_change_pp / 100)
```

The new `prepare_stage1_inputs.py` computes all five inputs independently:
the expected-return vector, Treasury level/slope/curvature factor covariance,
eligible benchmark weights, pre-trade portfolio weights, and WRDS duration.
It estimates residual Psi from historical returns, using rating/maturity bucket
fallbacks for short bond histories. Securities with remaining maturity below
one year are excluded from the tradable universe. The initial portfolio is
**explicitly** set to the same weights as the current benchmark for a first
one-period illustration; supply `--prior-portfolio previous.csv` (CUSIP and
posttrade weight) at later rebalances to drift holdings using the current
month's observed bond returns.

**Important benchmark-data limitation:** The checked-in bond-level CSV has
`AMOUNT_OUTSTANDING`, but NO original AGG ETF constituent-weight field.
Section 3.2 calls for renormalized original AGG holdings weights at each
month. A separate monthly file with `DATE,cusip,weight` (decimal weights)
is required for that benchmark:

```bash
python3 prepare_stage1_inputs.py ~/GoalPath-H1-Results/elastic_net_outputs.pkl \
    --benchmark-weights eligible_AGG_holdings.csv \
    --candidates 60 --output-dir ~/GoalPath-Stage1-Inputs
```

For **mechanics testing only**, the no-argument VS Code run defaults to the
explicitly labeled proxy; the `--benchmark-proxy` option is also available.
This normalizes outstanding amount times price, not actual AGG holdings:

```bash
python3 prepare_stage1_inputs.py ~/GoalPath-H1-Results/elastic_net_outputs.pkl \
    --benchmark-proxy --candidates 60 \
    --output-dir ~/GoalPath-Stage1-Inputs-Demo
```

Then solve Stage 1 for one month:

```bash
python3 -m pip install gurobipy
python3 robust_mvo_one_period.py \
    ~/GoalPath-Stage1-Inputs-Demo/stage1_inputs.pkl \
    --method heuristic --max-holdings 10 --trade-cost 0.001 \
    --duration-tolerance 0.5 --output ~/GoalPath-Stage1-Solution
```

`--method both` also runs the exact cardinality MIQP on the *same reduced*
candidate universe for comparison. Heuristic uses continuous optimization,
retains the largest K positions and resolves with that support. The objective
preserves aggregate factor/return exposures of **all eligible benchmark
constituents**, including those outside the candidate subset, instead of
renormalizing the benchmark to only the selected candidates.

Stage 2 items—robust forecast-error penalties, convexity, KRD, cash account
and self-financing—are intentionally excluded. The Gurobi installation and
license are required to solve the MIQP. The current run is **one period only**;
rolling H=1 portfolio re-solving and full AGG-weight backtesting remain to do.

The temporary `audit_bond_yields.py` script was removed after its useful
cross-month quote-consistency checks were retained in CI.

## Automated checks

GitHub Actions additionally checks the two no-argument VS Code entry points
with real source data and a Gurobi Stage 1 solve. It also runs
`python -m py_compile`, the H=1 regression suite
(`python -m unittest -v test_goalpath_h1`), and an elastic-net forecasting
smoke test against the repository's real source files. Successful execution
does not establish data quality, predictive validity, or absence of
publication-timing look-ahead in all macro features.
