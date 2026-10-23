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
return. The default uses the most recent origin with complete predictors and
observed next-month yields for comparison. An explicit date can be supplied
with `--asof 2024-12`.

The script tests 25 mixing-alpha/penalty-lambda combinations using chronological
annual validation (up to three recent folds). Hyperparameter search samples up
to 150 bond observations per month for tractable runtime; the final fit uses
all complete historical training observations. The program reports the top
settings, fitted coefficients, three bond-specific forecasts, observed yield
changes if available, and a breakdown of predictor contributions.

Forward-rate inputs are optional (`--include-forward`). Override tuning with
`--no-tune --alpha 0.5 --lambda 0.001` when debugging.

To evaluate monthly expanding-window forecasts and save output tables:

```bash
python3 elastic_net_one_period.py --asof latest --backtest \
  --output-dir ~/GoalPath-H1-Results
```

The output directory must not already exist. With `--backtest`, the output
includes `elastic_net_outputs.pkl`, which is the input format expected by
`robust_mvo_one_period.py`.

## Portfolio optimization

The robust one-period optimizer is a separate prototype and requires an
installed, licensed Gurobi environment:

```bash
python3 robust_mvo_one_period.py ~/GoalPath-H1-Results/elastic_net_outputs.pkl \
  --output ~/GoalPath-MVO-Results
```

It currently tracks a candidate-subset proxy rather than the full AGG
benchmark and does not yet implement the full investment-goal formulation.

## Automated checks

GitHub Actions runs `python -m py_compile`, the H=1 regression suite
(`python -m unittest -v test_goalpath_h1`), and an elastic-net forecasting
smoke test against the repository's real source files. Successful execution
does not establish data quality, predictive validity, or absence of
publication-timing look-ahead in all macro features.
