# GoalPath-Bond — three-script workflow

1. `process_investable_universe.py`: validated shared preparation, UNCHANGED.
2. `elastic_net_one_period.py`: coefficients and next-month yield forecasts.
3. `robust_mvo_one_period.py`: one-period mixed-integer robust conic allocation.

In the organized local folder, scripts are in `02_Python_Code/`; in GitHub's
flat layout they are in the root. Install dependencies once:

```
python3 -m pip install pandas numpy scipy scikit-learn gurobipy
```

From a flat GitHub checkout:

```
python3 elastic_net_one_period.py --cusips 00037BAC6 00037BAF9 001055AF9
```

For the local organized folder, prefix script paths with `02_Python_Code/`.
All source CSVs must exist locally; preparation is called automatically by
elastic net. Each destination must be NEW to protect previous results.

Elastic net is currently CONSOLE ONLY. It prints all original-unit slopes
and the intercept, then displays actual observed RET_EOM returns, forecast
next-month yield changes, and the ten input predictor values for each selected
CUSIP. `--cusips` controls display only; pooled training still uses eligible
historical observations across bonds. Without it, display defaults to the three
highest current dollar-volume bonds. No output data file is created.

Predictors are ONLY the agreed ten: Treasury level, slope, curvature, lagged
level change, within-month rate volatility, lagged Fed funds change, VIX,
time to maturity, rich/cheap spread, and lagged supplied CPI inflation.
Next-month observed return is labeled ex-post and is not a predictor or a
forecast of total return. Missing observations print as unavailable.

RMVO remains paused while inputs are being agreed. Its existing script requires
a previously generated handoff file; the current console-only elastic-net
script does not generate a new handoff. Do not run RMVO against old outputs and
interpret it as a fresh result from this console run.

There is NO annual-return requirement, terminal wealth goal, or goal constraint.
Objective: tracking variance minus robust expected return plus transaction cost.
Constraints: full investment, trade balance, nonnegative trades, long-only,
binary cardinality, maximum bond weight, duration/convexity/key-rate bands,
and conic forecast-error uncertainty. Tracking variance uses a convex epigraph
with a linear objective, giving a mixed-integer SOCP-representable formulation.

Demonstration settings: 40 candidates, max 10 holdings, max 20% per bond,
gamma=10, uncertainty radius=.5, each buy/sell cost=.001, duration band=.5,
convexity band=10, each KRD band=.5. Flags can override them; e.g.
`--max-weight 1` removes the extra concentration restriction.

The benchmark remains a candidate-subset market-value proxy, not AGG.
Risk analytics/coupon schedules are estimates; security eligibility and release
vintages remain unverified. Printed returns are forecasts, not realized profits.
Costs remain an objective penalty; no wealth update is performed.

Superseded supporting files are archived, not deleted. Historical run reports
record their original settings and may still mention goals; those are not the
current model. Source data and the validated processing script are unchanged.
