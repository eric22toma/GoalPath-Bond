"""Load the supplied bond-month universe without filtering its rows or columns.

Requires pandas: pip install pandas
Run: python process_investable_universe.py Investable_universe_AGG.csv
Optional exports: append --output-dir processed_universe (a NEW directory).

The supplied CSV and TA reference script are never modified. This adapts the
TA's load/slice/pivot approach to long-form bond observations. It does not
calculate returns from prices: the CSV already supplies RET_* observations.
No missing-value imputation, eligibility screen, deduplication, or full-history
survival filter is applied. No optimizer, forecasts, or benchmark weights are
constructed here. The dataset name alone does not verify AGG membership.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DATE_COLUMNS = (
    'DATE', 'OFFERING_DATE', 'MATURITY', 'DATED_DATE', 'FIRST_INTEREST_DATE',
    'LAST_INTEREST_DATE', 'T_DATE', 'nextcoup', 'DEFAULT_DATE', 'REINSTATED_DATE',
)
# Keep CUSIP, ISSUE_ID, ISIN, and TREASURY_MATURITY as text identifiers.
NUMERIC_COLUMNS = (
    'CONV', 'OFFERING_AMT', 'OFFERING_PRICE', 'PRINCIPAL_AMT', 'COUPON',
    'NCOUPS', 'AMOUNT_OUTSTANDING', 'N_SP', 'N_MR', 'N_FR', 'RATING_NUM',
    'T_Volume', 'T_DVolume', 'T_Yld_Pt', 'PRICE_EOM', 'PRICE_LDM',
    'PRICE_L5M', 'GAP', 'coupmonth', 'COUPAMT', 'COUPACC', 'MULTICOUPS',
    'TMT', 'REMCOUPS', 'DURATION',
)
PERCENT_COLUMNS = ('T_Spread', 'YIELD', 'RET_EOM', 'RET_LDM', 'RET_L5M')


def load_data(csv_path='Investable_universe_AGG.csv'):
    """Return (raw, prepared, report); both tables retain source row/column order.

    raw contains original CSV cell strings (including blanks and NA tokens).
    prepared converts recognized columns; failed conversions become NaN/NaT
    and are counted in report. Original values remain available in raw.
    Percent columns MUST use a percent sign for nonmissing values; e.g.
    '2.78%' becomes 0.0278. Bare values are flagged rather than guessed.
    COUPON and T_Yld_Pt retain source numerical units (not divided by 100).
    Missing tokens in converted columns: blank, NA, N/A, NULL, NAN, NONE, '-'.
    """
    path = Path(csv_path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False,
                      encoding='utf-8-sig', skip_blank_lines=False)
    required = {'cusip', 'DATE'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing required columns: {sorted(required - set(raw.columns))}')
    prepared = raw.copy()
    conversion_failures = {}
    missing = {}
    tokens = {'', 'NA', 'N/A', 'NULL', 'NAN', 'NONE', '-'}
    for column in raw.columns:
        text = raw[column].str.strip()
        absent = text.str.upper().isin(tokens)
        missing[column] = int(absent.sum())
        if column not in DATE_COLUMNS + NUMERIC_COLUMNS + PERCENT_COLUMNS:
            continue
        values = text.mask(absent)
        if column in DATE_COLUMNS:
            converted = pd.to_datetime(values, format='%Y-%m-%d', errors='coerce')
        elif column in PERCENT_COLUMNS:
            # Require the explicit percent sign to avoid a 100x unit error.
            marked = values.str.endswith('%', na=False)
            converted = pd.to_numeric(values.where(marked).str[:-1], errors='coerce') / 100
        else:
            values = values.str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            converted = pd.to_numeric(values, errors='coerce')
        prepared[column] = converted
        conversion_failures[column] = int((~absent & converted.isna()).sum())
    dates = prepared['DATE'].dropna()
    report = {
        'source': str(path.resolve()),
        'rows': len(raw), 'columns': len(raw.columns),
        'unique_nonblank_cusips': int(raw.loc[raw.cusip.str.strip().ne(''), 'cusip'].nunique()),
        'first_date': str(dates.min().date()) if len(dates) else None,
        'last_date': str(dates.max().date()) if len(dates) else None,
        'duplicate_cusip_date_rows': int(raw.duplicated(['cusip', 'DATE'], keep=False).sum()),
        'missing_counts': missing,
        'conversion_failure_counts': conversion_failures,
        'percent_columns_in_decimal_units': list(PERCENT_COLUMNS),
        'notes': [
            'All source rows and columns retained; raw preserves original cell text.',
            'Unknown fields remain strings. Conversion failures remain in raw.',
            'COUPON and T_Yld_Pt retain source units; confirm definitions before return mapping.',
            'Missing returns remain missing, not zero. No eligibility filters applied.',
            'DATE is an observation label, not proof of information availability.',
            'Verify release timing, vintage, and membership before a rolling backtest.',
            'No source verification of benchmark membership or return definitions performed.',
        ],
    }
    return raw, prepared, report


def get_period_data(data, start_date, end_date):
    """Inclusive observation-date slice of prepared data; no fill or asset removal.

    This does not establish point-in-time availability of the other fields.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError('Provide valid start_date <= end_date')
    return data.loc[data.DATE.between(start, end)].copy()


def get_panel(data, value_column='RET_EOM'):
    """Optional date x CUSIP view, preserving missing observations as NaN.

    Raises on blank keys, invalid dates, or duplicate keys instead of silently
    dropping/aggregating them. Use the long-form table for lossless storage.
    RET_* decimal values are supplied returns, not inferred price returns.
    """
    if data.DATE.isna().any() or data.cusip.str.strip().eq('').any():
        raise ValueError('Panel requires valid DATE and nonblank CUSIP on every row')
    if data.duplicated(['DATE', 'cusip']).any():
        raise ValueError('Duplicate DATE/CUSIP keys; resolve explicitly before pivoting')
    return data.pivot(index='DATE', columns='cusip', values=value_column).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', nargs='?', default='Investable_universe_AGG.csv')
    parser.add_argument('--output-dir', type=Path,
                        help='New directory for raw_cells.csv, prepared.csv, and report.json')
    args = parser.parse_args()
    raw, prepared, report = load_data(args.csv_path)
    print(json.dumps(report, indent=2))
    if args.output_dir is not None:
        # Refuse existing directories so previous files cannot be overwritten.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        raw.to_csv(args.output_dir / 'raw_cells.csv', index=False)
        prepared.to_csv(args.output_dir / 'prepared.csv', index=False)
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
