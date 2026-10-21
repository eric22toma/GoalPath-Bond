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
python3 elastic_net_one_period.py --output results/elastic_net_outputs.pkl
python3 robust_mvo_one_period.py results/elastic_net_outputs.pkl --output results/rmvo
```

For the local organized folder, prefix script paths with `02_Python_Code/`.
All source CSVs must exist locally; preparation is called automatically by
elastic net. Each destination must be NEW to protect previous results.

Elastic net prints all original-unit slopes and the intercept, then prints
yield forecasts for the top THREE bonds by current dollar trading volume
(with CUSIP tie-breaking). These are examples, not the three best investments.
It writes ONE `.pkl` handoff file containing all forecasts and the dated
historical inputs the optimizer needs. Load only the handoff generated locally
by this project; it is a Python serialized data file.

RMVO requires a Gurobi license. Its code contains the optimization, not the
elastic-net script. It consumes the saved data without refitting. It prints
selected CUSIPs, weights/trades, nominal/robust forecast monthly return,
tracking error, costs and turnover; saves portfolio.csv and report.json.

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
