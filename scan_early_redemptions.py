"""One-time audit of bonds that disappear before contractual maturity.

Run from VS Code (Run Python File) or: python scan_early_redemptions.py
Reads the existing zipped AGG history and the small WRDS CUSIP reference CSV.
Writes a review-only CSV; DOES NOT modify the backtest, returns, or eligibility.

A missing future observation + a callable provision is NOT proof of redemption.
The WRDS NEXT_CALL_PRICE is a contractual schedule price, NOT an actual payoff.
"""

from pathlib import Path
import argparse
import pandas as pd

from process_investable_universe import DEFAULT_BONDS, PROJECT_ROOT

REFERENCE = PROJECT_ROOT / '01_Source_Data' / 'early_redemption_reference.csv'
OUTPUT = PROJECT_ROOT / 'stage1_outputs' / 'early_redemption_candidates.csv'
START_MONTH = '2023-12'  # Match the current backtest decision-month window.
END_MONTH = '2025-10'
TAPESTRY = {'876030AD9', '876030AE7', '876030AF4'}


def excel_date(series):
    """WRDS-exported date cells were stored as Excel serial numbers."""
    numeric = pd.to_numeric(series, errors='coerce')
    from_serial = pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')
    from_text = pd.to_datetime(series.where(numeric.isna()), errors='coerce')
    return from_serial.fillna(from_text)


def read_wrds(path):
    wrds = pd.read_csv(path, dtype=str, keep_default_na=False)
    needed = {'COMPLETE_CUSIP', 'CALLABLE', 'MAKE_WHOLE', 'MATURITY'}
    if not needed.issubset(wrds):
        raise ValueError(f'WRDS reference missing: {sorted(needed - set(wrds))}')
    wrds['cusip'] = wrds.COMPLETE_CUSIP.str.strip().str.upper()
    wrds = wrds.loc[wrds.cusip.ne('')].drop_duplicates('cusip').copy()
    for col in ('MATURITY', 'NEXT_CALL_DATE', 'CALL_DATE'):
        if col in wrds:
            wrds[col] = excel_date(wrds[col])
    return wrds.set_index('cusip')


def read_last_bond_observations(path):
    """Read the 40 MB zipped source in chunks; retain only one row per CUSIP."""
    columns = {'cusip', 'DATE', 'MATURITY', 'PRICE_EOM', 'YIELD'}
    path = Path(path)
    if not path.exists() and Path(str(path) + '.zip').exists():
        path = Path(str(path) + '.zip')
    if not path.exists():
        raise FileNotFoundError(f'Bond source not found: {path}')

    latest = {}
    sample_end = None
    source_rows = 0
    for chunk in pd.read_csv(path, usecols=lambda col: col in columns,
                             dtype=str, chunksize=200_000,
                             encoding='utf-8-sig', low_memory=False):
        if not {'cusip', 'DATE', 'MATURITY'}.issubset(chunk):
            raise ValueError('AGG source requires cusip, DATE, MATURITY')
        source_rows += len(chunk)
        chunk['cusip'] = chunk.cusip.fillna('').str.strip().str.upper()
        chunk['DATE'] = pd.to_datetime(chunk.DATE, errors='coerce')
        chunk['MATURITY'] = pd.to_datetime(chunk.MATURITY, errors='coerce')
        chunk = chunk.loc[chunk.cusip.ne('') & chunk.DATE.notna()]
        if chunk.empty:
            continue
        newest = chunk.DATE.max()
        sample_end = newest if sample_end is None else max(sample_end, newest)
        # Source is not assumed to be in date order.
        last_in_chunk = chunk.sort_values('DATE').drop_duplicates('cusip', keep='last')
        for rec in last_in_chunk.to_dict('records'):
            cusip = rec['cusip']
            if cusip not in latest or rec['DATE'] > latest[cusip]['DATE']:
                latest[cusip] = rec

    if sample_end is None:
        raise ValueError('No dated bond observations found')
    return pd.DataFrame(latest.values()), sample_end.to_period('M'), source_rows


def scan(bonds, wrds, sample_end, start, end, all_history=False):
    bonds = bonds.copy()
    # The source maturity takes priority; WRDS can fill any missing source date.
    joined = bonds.join(wrds, on='cusip', how='left', rsuffix='_wrds')
    joined['maturity_for_scan'] = joined.MATURITY.fillna(joined.MATURITY_wrds)
    joined['last_month'] = joined.DATE.dt.to_period('M')
    joined['first_missing_month'] = joined.last_month + 1
    joined['maturity_month'] = joined.maturity_for_scan.dt.to_period('M')
    # Right-censored sample-end rows and scheduled maturities aren't early exits.
    joined = joined.loc[
        joined.first_missing_month.le(sample_end)
        & joined.maturity_month.gt(joined.first_missing_month)
    ].copy()
    history_count = len(joined)
    if not all_history:
        joined = joined.loc[joined.last_month.between(start, end)].copy()

    joined['months_until_maturity'] = (
        joined.maturity_month.astype('int64')
        - joined.first_missing_month.astype('int64')
    )
    joined['tapestry_like'] = (
        joined.CALLABLE.eq('Y') & joined.MAKE_WHOLE.eq('Y')
        & joined.NEXT_CALL_DATE.gt(joined.DATE)
        & pd.to_numeric(joined.NEXT_CALL_PRICE, errors='coerce').eq(100)
    )
    joined['review_priority'] = '3 - Other premature exit'
    joined.loc[joined.CALLABLE.eq('Y'), 'review_priority'] = '2 - Callable premature exit'
    joined.loc[joined.tapestry_like, 'review_priority'] = '1 - Tapestry-like call terms'
    joined['maturity_match'] = (
        joined.MATURITY.dt.to_period('M')
        .eq(joined.MATURITY_wrds.dt.to_period('M'))
    )
    joined['maturity_match'] = joined.maturity_match.astype('boolean')
    joined.loc[joined.MATURITY_wrds.isna() | joined.MATURITY.isna(), 'maturity_match'] = pd.NA

    result = pd.DataFrame({
        'review_priority': joined.review_priority,
        'cusip': joined.cusip,
        'issuer': joined.PROSPECTUS_ISSUER_NAME,
        'last_observed_month': joined.last_month.astype(str),
        'first_missing_month': joined.first_missing_month.astype(str),
        'contractual_maturity': joined.maturity_for_scan.dt.strftime('%Y-%m-%d'),
        'months_until_maturity': joined.months_until_maturity,
        'callable': joined.CALLABLE,
        'make_whole': joined.MAKE_WHOLE,
        'next_scheduled_call_date': joined.NEXT_CALL_DATE.dt.strftime('%Y-%m-%d'),
        'next_scheduled_call_price_NOT_actual': joined.NEXT_CALL_PRICE,
        'wrds_maturity_matches': joined.maturity_match,
        'last_recorded_clean_price': joined.get('PRICE_EOM'),
        'last_recorded_yield': joined.get('YIELD'),
        'verification_status': 'POTENTIAL EXIT - NOT VERIFIED',
    })
    result = result.sort_values(
        ['review_priority', 'last_observed_month', 'cusip'],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    return result, history_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bonds', type=Path, default=DEFAULT_BONDS)
    parser.add_argument('--wrds', type=Path, default=REFERENCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--start', default=START_MONTH)
    parser.add_argument('--end', default=END_MONTH)
    parser.add_argument('--all-history', action='store_true',
                        help='Include suspected exits before the backtest window')
    args = parser.parse_args()

    wrds = read_wrds(args.wrds)
    bonds, sample_end, source_rows = read_last_bond_observations(args.bonds)
    result, all_exits = scan(bonds, wrds, sample_end,
                             pd.Period(args.start, 'M'), pd.Period(args.end, 'M'),
                             args.all_history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print('\nEARLY REDEMPTION REVIEW (NOT CONFIRMED REDEMPTIONS)')
    print(f'AGG observations: {source_rows:,} | Unique bonds: {len(bonds):,}')
    print(f'WRDS CUSIPs: {len(wrds):,} | Last source month: {sample_end}')
    print(f'Potential premature exits across full history: {all_exits:,}')
    print(f'Candidates in review window {args.start} through {args.end}: {len(result):,}')
    if not result.empty:
        print(result.review_priority.value_counts().sort_index().to_string())
        print('\nALL FLAGGED BONDS FOR REVIEW:')
        print(result[['review_priority', 'cusip', 'issuer', 'last_observed_month',
                      'contractual_maturity']].to_string(index=False, max_rows=None))
    for cusip in sorted(TAPESTRY):
        hit = result.loc[result.cusip.eq(cusip)]
        print(f'Tapestry {cusip}: ' + (f"FLAGGED ({hit.iloc[0]['last_observed_month']})"
                                        if not hit.empty else 'not flagged (inspect history)'))
    print(f'\nSaved: {args.output}\n')
    print('WARNING: A premature exit can be a redemption OR data/index coverage loss.')
    print('Do not use this report for point-in-time investment eligibility.')


if __name__ == '__main__':
    main()
