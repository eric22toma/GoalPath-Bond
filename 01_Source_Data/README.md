# Source data

Original source files copied from Bill's local `01_Source_Data` folder on October 21, 2026. The two smaller CSVs are unchanged. `Investable_universe_AGG.csv` is stored as a lossless ZIP because the original exceeds GitHub's 100 MiB file limit; the ZIP is about 38.4 MiB.

Extract the AGG CSV into this folder before running scripts that require the uncompressed file:

```bash
unzip 01_Source_Data/Investable_universe_AGG.csv.zip -d 01_Source_Data
```

Files are provided as supplied; this upload does not independently verify their source provenance or contents.

## SHA-256 checksums of original CSVs

```text
89b7ab1c916b17199364c3ff816cbefbbc1aa09dc8d1faad603b55aabd925b46  FRED_Monthly_2015_2025.csv
87660f570a4fe299dbfe64ffa07264040db087fc01728ca0bcdec1dd95e435fe  Investable_universe_AGG.csv
ea60ddd7211a439087415d5e56a3d891c732c1a80e85eaeb26756978da916be2  federal_reserve_nominal_yield_curve (1).csv
```
