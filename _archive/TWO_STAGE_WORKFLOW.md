# Paper sequence: yield-change forecasts, then one-term robust MVO

Stage 1 estimates shared elastic-net beta coefficients and applies them to each
bond's predictor vector to produce a final one-month change-in-yield forecast:

`delta_y_hat = intercept + x @ beta`

Run from the organized local workspace:

```
python3 02_Python_Code/elastic_net_one_period.py --forecast-only --output 03_Results/my_yield_forecasts
```

In the flat GitHub checkout, omit `02_Python_Code/` from script paths.
Forecast-only is now the default, even without the explicit flag. To retain the
old single-command behavior, use `--optimize`.

Open `yield_change_forecasts.csv`. It includes CUSIP, decision month, target
month, current yield, forecast yield change in decimal/percentage-point/basis-
point units, and forecast next-period yield. A predicted change of -0.10
percentage points equals -10 basis points or -0.001 decimal yield. It is not a
bond return. `coefficients.csv` remains a supporting output.

The script also saves a dated optimizer bundle: current bond snapshot, trailing
36 months of realized return history, earlier out-of-sample return errors,
current Svensson parameters, and a SHA256 manifest. Future realized bond yields
and returns are excluded from the current snapshot and return-history bundle.
The full bundle must stay together locally. Large derived files need not be
uploaded to GitHub; regenerate them by running stage 1 with the source CSVs.

Stage 2 consumes that saved output without rerunning elastic net:

```
python3 02_Python_Code/robust_mvo_one_period.py 03_Results/my_yield_forecasts --output 03_Results/my_robust_mvo
```

A Gurobi installation/license is required for stage 2. It checks bundle hashes,
identifier alignment, decision/target dates, and uncertainty-sample dates.
It explicitly reads the final yield forecast file, not coefficients to refit.

Following Deliverable 2, section 4.1, map decimal yield changes into expected
returns using carry - duration*dy + .5*convexity*dy^2, then solve the robust
tracking/return/cost objective with full investment, trade balance, binary
cardinality, duration/convexity/KRD bands and robust required return. It reuses
the existing `solve_period` formulation, so there are not two competing solvers.
The model has a convex quadratic tracking objective and second-order-cone robust
return uncertainty, plus binary selection (a mixed-integer conic formulation).

Stage 2 saves `portfolio.csv`, `risk_inputs.npz` and `report.json`. The one-term
goal constraint is the dynamic monthly requirement implied by the investor goal,
not a requirement to earn the entire annual goal in one month. Defaults are
inherited from the forecast manifest; flags can override the goal, max holdings,
gamma, costs and exposure tolerances. Selected bonds' final forecasts and trades
are printed in the terminal.

The data-quality, proxy-benchmark, estimated-exposure and availability caveats
remain unchanged. A mathematically feasible robust return is conditional on
these inputs and the chosen uncertainty radius; it is not a guaranteed realized
return. This separate solver intentionally does not calculate ex-post returns
or a multi-period goal probability.
