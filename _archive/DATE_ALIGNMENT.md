# Date alignment and inflation

The modeling window is the overlap of the source date ranges. With the current
three files it is January 2016 through December 2025. Matching uses calendar
months, not row numbers. A January 29 Treasury trading-day observation therefore
belongs to the same January month as January 31 bond/FRED observations.

Original CSVs and raw-cell copies remain intact. Prepared modeling tables exclude
irrelevant observations outside this overlap. Separate `_warmup` tables keep up
to 13 prior months for lag and inflation calculations; earlier history, including
1961 Treasury observations, never enters this model view.

Treasury data remain daily until feature construction: use the final trading-day
row for yield-curve levels and within-month daily changes for rate volatility.
Do not simply discard all pre-2016 Treasury/FRED observations before constructing
features. Warm-up history is not an additional bond training sample.

The supplied FRED `Inflation_YoY` column is already year-over-year inflation in
percent units. It matches `100*(CPI/CPI.shift(12)-1)` in the current source.
Elastic net uses `Inflation_YoY.shift(1)`, not a second year-over-year calculation.
If that column is absent, CPI-based calculation remains a fallback. Where both
are present, overlapping values must agree within numerical tolerance. Missing
values in an existing Inflation_YoY column are not silently filled.

The supplied file starts in January 2015. Its YoY inflation is blank during 2015,
so January 2016's one-month-lagged inflation is unavailable. February 2016 is the
first complete inflation-feature month. To include January 2016 with this lag,
provide December 2015 YoY inflation (or December 2014 CPI as well as December
2015 CPI). Do not borrow January 2016 inflation for a December 2015 lag.

The bond record for January 2016 stays in the processed file. Training excludes
rows whose required features or next-month target are missing. Missing calendar
months are reported rather than compressed together or forward-filled.

`preparation_report.json` includes each source's model range, retained/excluded
row counts and missing model months. The forecast additionally writes
`predictor_month_coverage.csv`, identifying missing predictors month by month.
Warm-up months are excluded from exported monthly model predictors and training.

Run preparation using the existing command with a NEW output directory. The
raw/prepared first-record preview now compares the same selected source record,
even when earlier source observations were excluded from the model view.

Matching dates alone does not verify economic release timing or historical
revision vintages; those limitations of the prototype remain.
