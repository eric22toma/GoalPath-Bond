"""Load, prepare, and check GoalPath-Bond datasets.

Reads AGG bonds, FRED macroeconomic data, and Federal Reserve yield curves.
Preserves the raw data, prepares aligned modeling tables, and displays loading
status plus the first three observations from each dataset in 2016-2025.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


CODE_FOLDER = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_FOLDER.parent if CODE_FOLDER.name == "02_Python_Code" else CODE_FOLDER
SOURCE_DATA = PROJECT_ROOT / "01_Source_Data" if (PROJECT_ROOT / "01_Source_Data").is_dir() else PROJECT_ROOT
DEFAULT_BONDS = SOURCE_DATA / "Investable_universe_AGG.csv"
DEFAULT_FRED = SOURCE_DATA / "FRED_Monthly_2015_2025.csv"
DEFAULT_CURVE = SOURCE_DATA / "federal_reserve_nominal_yield_curve (1).csv"

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


def load_data(csv_path=DEFAULT_BONDS):
    """Load raw bond cells and a type-converted copy.

    Percent-marked yields and returns become decimals. Other numeric units
    are preserved, and conversion failures are counted in the report.
    """
    path = Path(csv_path)
    zipped = Path(str(path) + '.zip')
    if path.suffix == '.csv' and not path.is_file() and zipped.is_file():
        path = zipped
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
            'Raw cells are preserved; prepared values may contain missing data.',
            'Yield/return fields with percent signs are converted to decimals.',
            'No eligibility screening or return reconstruction is applied.',
        ],
    }
    return raw, prepared, report


def load_numeric_table(csv_path, sentinel=None):
    """Prepare a dated numeric source, retaining every source row/column.

    FRED/curve rates retain source percentage-point units. Missing curve
    sentinel -999.99 is mapped to NaN only in the prepared table.
    """
    path = Path(csv_path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False,
                      encoding='utf-8-sig', skip_blank_lines=False)
    if 'Date' not in raw:
        raise ValueError(f'{path}: required Date column missing')
    prepared = raw.copy()
    failures, missing = {}, {}
    for column in raw:
        text = raw[column].str.strip()
        absent = text.str.upper().isin({'', 'NA', 'N/A', 'NULL', 'NAN', 'NONE', '-'})
        values = text.mask(absent)
        if column == 'Date':
            converted = pd.to_datetime(values, format='%Y-%m-%d', errors='coerce')
        else:
            converted = pd.to_numeric(values, errors='coerce')
            if sentinel is not None:
                absent = absent | converted.eq(sentinel)
                converted = converted.mask(converted.eq(sentinel))
        prepared[column] = converted
        failures[column] = int((~absent & converted.isna()).sum())
        missing[column] = int(absent.sum())
    report = {'source': str(path.resolve()), 'rows': len(raw), 'columns': len(raw.columns),
              'conversion_failure_counts': failures, 'missing_counts': missing,
              'units': 'Source numerical units preserved; rates in percentage points',
              'sentinel_treated_as_missing': sentinel}
    return raw, prepared, report


def process_all_data(bond_path=DEFAULT_BONDS,
                     fred_path=DEFAULT_FRED,
                     curve_path=DEFAULT_CURVE,
                     core_path=None):
    """Load datasets and align prepared tables over overlapping months.

    Raw data stays unchanged. Historical warm-up observations are kept
    separately for downstream feature generation.
    """
    data = {'bonds': load_data(bond_path),
            'fred': load_numeric_table(fred_path),
            'curve': load_numeric_table(curve_path, sentinel=-999.99)}
    if core_path is not None:
        data['core_cpi'] = load_numeric_table(core_path)
        if 'CPILFESL' not in data['core_cpi'][1]:
            raise ValueError('Core CPI input requires Date and CPILFESL columns')
    return synchronize_data(data)


def synchronize_data(data):
    """Align prepared tables by month and retain 13 warm-up months.

    Treasury observations remain daily to support volatility calculations.
    """
    sources = list(data)
    ranges = {}
    for name in sources:
        table = data[name][1]
        date_column = 'DATE' if name == 'bonds' else 'Date'
        if table[date_column].isna().any() or table.empty:
            raise ValueError(f'{name}: empty data or invalid dates; inspect source before alignment')
        months = table[date_column].dt.to_period('M')
        ranges[name] = (months.min(), months.max())
    start = max(bounds[0] for bounds in ranges.values())
    end = min(bounds[1] for bounds in ranges.values())
    if start > end:
        raise ValueError('Input files have no overlapping calendar-month range')
    aligned = {}
    for name in sources:
        raw, prepared, report = data[name]
        date_column = 'DATE' if name == 'bonds' else 'Date'
        months = prepared[date_column].dt.to_period('M')
        mask = months.between(start, end)
        selected = prepared.loc[mask].copy()
        available_months = set(months[mask])
        report = dict(report)
        report['alignment'] = {
            'model_start_month': str(start), 'model_end_month': str(end),
            'source_rows': len(raw), 'prepared_rows': len(selected),
            'excluded_from_model_view': int((~mask).sum()),
            'missing_model_months': [str(m) for m in pd.period_range(start, end, freq='M')
                                     if m not in available_months],
            'note': 'Calendar-month alignment; Treasury rows remain daily, not month-end labels. Raw source records are preserved; warm-up is exported separately.',
        }
        aligned[name] = (raw, selected, report)
        if name != 'bonds':
            warm_mask = months.between(start-13, start-1)
            warm = prepared.loc[warm_mask].copy()
            warm_report = {
                'source': report['source'], 'rows': len(warm), 'columns': len(warm.columns),
                'purpose': 'Predictor warm-up only; not extra bond observations or training targets',
                'requested_start_month': str(start-13),
                'end_month': str(start-1),
            }
            aligned[name+'_warmup'] = (raw.loc[warm_mask].copy(), warm, warm_report)
    return aligned


def history_for_features(data, name):
    """Supply only needed warm-up plus aligned observations to feature building."""
    current = data[name][1]
    warm = data.get(name+'_warmup')
    return pd.concat([warm[1], current], axis=0) if warm is not None else current


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


PREVIEW_START = pd.Timestamp('2016-01-01')
PREVIEW_END = pd.Timestamp('2025-12-31')


def preview_period(prepared, name):
    """Select 2016–2025 rows for display or export without altering the data."""
    date_column = 'DATE' if name == 'bonds' else 'Date'
    return prepared.loc[prepared[date_column].between(PREVIEW_START, PREVIEW_END)]


def print_first_rows(data):
    """Display source status, data-quality issues, and three 2016–2025 rows."""
    print('\nDATA LOAD STATUS | 2016-01-01 through 2025-12-31 (120 months)')
    all_loaded = True
    any_warnings = False

    for name in ('bonds', 'fred', 'curve', 'core_cpi'):
        if name not in data:
            if name == 'core_cpi':
                print('\nCORE_CPI — SKIPPED (optional file not supplied)')
            else:
                print(f'\n{name.upper()} — ERROR: Required dataset was not loaded.')
                all_loaded = False
            continue

        raw, prepared, report = data[name]
        period_data = preview_period(prepared, name)
        date_column = 'DATE' if name == 'bonds' else 'Date'
        errors = []
        warnings = []

        if raw.empty:
            errors.append('Source file contains no data rows.')
        if prepared.empty:
            errors.append('No rows remain after monthly alignment.')
        if period_data.empty:
            errors.append('No prepared observations dated 2016-01-01 to 2025-12-31.')
        elif len(period_data) < 3:
            warnings.append(f'Only {len(period_data)} period rows available; fewer than three can be displayed.')

        months_present = set(period_data[date_column].dt.to_period('M'))
        missing_months = [str(m) for m in pd.period_range('2016-01', '2025-12', freq='M')
                          if m not in months_present]
        if missing_months:
            warnings.append(f'Missing {len(missing_months)} of 120 calendar months: '
                            + ', '.join(missing_months[:24])
                            + (' ... (full list in the preparation report)' if len(missing_months) > 24 else ''))

        failures = {k: v for k, v in report.get('conversion_failure_counts', {}).items() if v}
        if failures:
            warnings.append('Unconverted nonmissing source cells: '
                            + ', '.join(f'{key}={count}' for key, count in failures.items()))
        duplicate_count = report.get('duplicate_cusip_date_rows', 0) if name == 'bonds' else 0
        if duplicate_count:
            warnings.append(f'Duplicate CUSIP/date source rows: {duplicate_count}')

        status = 'ERROR' if errors else 'LOADED WITH WARNINGS' if warnings else 'SUCCESS'
        report['preview_2016_2025'] = {
            'status': status,
            'rows': int(len(period_data)),
            'first_date': (str(period_data[date_column].min().date())
                           if not period_data.empty else None),
            'last_date': (str(period_data[date_column].max().date())
                          if not period_data.empty else None),
            'missing_months': missing_months,
            'warnings': warnings,
            'errors': errors,
        }
        all_loaded &= not errors
        any_warnings |= bool(warnings)
        print(f"\n{name.upper()} — {status} — {Path(report['source']).name}")
        print(f'  Source rows: {len(raw):,} | Aligned rows: {len(prepared):,} '
              f'| 2016–2025 rows: {len(period_data):,}')
        if not period_data.empty:
            print(f"  Dates in period: {period_data[date_column].min():%Y-%m-%d} "
                  f"to {period_data[date_column].max():%Y-%m-%d}")
        for issue in errors:
            print(f'  ERROR: {issue}')
        for issue in warnings:
            print(f'  WARNING: {issue}')
        if not period_data.empty:
            print('  First three prepared rows in 2016–2025 (all columns):')
            print(period_data.head(3).to_string(index=False, max_cols=None, max_colwidth=28))

    overall = 'ERROR' if not all_loaded else 'SUCCESS WITH WARNINGS' if any_warnings else 'SUCCESS'
    print(f'\nOVERALL: {overall} — data loaded locally; no files uploaded to GitHub.')
    return all_loaded

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', nargs='?', default=str(DEFAULT_BONDS))
    parser.add_argument('--fred', default=str(DEFAULT_FRED))
    parser.add_argument('--curve', default=str(DEFAULT_CURVE))
    parser.add_argument('--core-cpi')
    parser.add_argument('--output-dir', type=Path,
                        help='New directory for prepared/raw tables, reports, and three-row previews')
    args = parser.parse_args()
    try:
        data = process_all_data(args.csv_path, args.fred, args.curve, args.core_cpi)
        if not print_first_rows(data):
            parser.exit(1, '\nERROR: One or more datasets failed validation; see the messages above.\n')
        if args.output_dir is not None:
            reports = {name: tables[2] for name, tables in data.items()}
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for name, (raw, prepared, report) in data.items():
                raw.to_csv(args.output_dir / f'{name}_raw_cells.csv', index=False)
                prepared.to_csv(args.output_dir / f'{name}_prepared.csv', index=False)
                if name in ('bonds', 'fred', 'curve', 'core_cpi'):
                    preview_period(prepared, name).head(3).to_csv(
                        args.output_dir / f'{name}_preview_2016_2025.csv', index=False)
            (args.output_dir / 'preparation_report.json').write_text(
                json.dumps(reports, indent=2) + '\n')
            print(f'\nSUCCESS: Prepared tables, source copies, reports, and 2016–2025 previews saved to: {args.output_dir.resolve()}')
    except Exception as exc:
        parser.exit(1, f'\nERROR: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
