"""One-time AGG premature-disappearance audit with manually reviewed exceptions.

Run from VS Code or: python scan_early_redemptions.py
Cross-reference zipped AGG observations against WRDS callable-bond data.
Exceptions were individually checked against SEC 8-K/10-K issuer filings,
issuer notices, and SEC bond holdings. EARLY_REDEMPTIONS stores full cash
redemptions with known dates and prices (None when price is unverified);
REVIEWED_NON_REDEMPTIONS stores continuing issues, partial tenders and earlier
exchanges, not full redemptions during their flagged months.
Both groups are excluded from future exception reports. Unresolved flags remain.

Disappearance from the filtered AGG source is NOT proof of a redemption.
WRDS NEXT_CALL_PRICE is not necessarily a realized redemption price.
The script does not change any model, backtest or raw bond returns.

Primary documents reviewed:
Tapestry: https://tapestry.gcs-web.com/static-files/46cd29c5-330c-443d-a130-8b77d86ae650
Kroger: https://www.sec.gov/Archives/edgar/data/56873/000110465926037723/kr-20260131x10k.htm
Micron: https://www.sec.gov/Archives/edgar/data/723125/000072312526000023/mu-20260903.htm
Vulcan: https://www.sec.gov/Archives/edgar/data/1396009/000114036124008449/ef20021903_8k.htm
Verizon: https://www.verizon.com/about/news/verizon-announces-accepted-amounts
EQM: https://ir.eqt.com/investor-relations/news/news-release-details/2024/Early-Results-and-Upsizing-of-EQM-Midstream-Partners-LPs-Tender-Offer-and-Consent-Solicitation-for-Certain-of-its-Senior-Notes/default.aspx
"""

from pathlib import Path
import argparse
import pandas as pd

from process_investable_universe import DEFAULT_BONDS, PROJECT_ROOT

REFERENCE = PROJECT_ROOT / '01_Source_Data' / 'early_redemption_reference.csv'
OUTPUT = PROJECT_ROOT / 'stage1_outputs' / 'early_redemption_candidates.csv'
START_MONTH = '2023-12'  # Match the current backtest decision-month window.
END_MONTH = '2025-10'
# Verified FULL cash redemption (date, percent of par EXCLUDING accrued interest).
# None = exact payoff not yet established; NEVER infer it from NEXT_CALL_PRICE.
EARLY_REDEMPTIONS = {
    '876030AD9': ('2024-11-25', 101.0),   # Tapestry
    '876030AE7': ('2024-11-25', 101.0),
    '876030AF4': ('2024-11-25', 101.0),
    '501044DS7': ('2024-12-18', 101.0),   # Kroger
    '501044DT5': ('2024-12-18', 101.0),
    '501044DU2': ('2024-12-18', 101.0),
    '76132FAC1': ('2025-02-22', None),    # Retail Opportunity, make-whole
    '595112BP7': ('2025-05-27', None),    # Micron 2027
    '46590XAZ9': ('2025-07-03', 102.478), # JBS
    '929160BA6': ('2024-03-01', 100.0),   # Vulcan, redeemed at par
    '750236AX9': ('2024-03-19', 100.396), # Radian
    '91911TAP8': ('2024-07-25', None),    # Vale make-whole
    '595112BV4': ('2025-10-24', None),    # Micron 2029 B
    '595112CA9': ('2025-10-24', None),    # Micron 2028
}

# Reviewed through issuer/SEC materials: continuing bonds, partial tenders or
# previous exchanges; not evidence of a full cash redemption at disappearance.
# Cases with no adequate evidence are NOT added here and remain flagged.
REVIEWED_NON_REDEMPTIONS = {
    '25470DAT6', '92343VDC5',
    '91087BAY6', '91087BAZ3', '91087BBA7',
    '91087BBB5', '91087BBC3', '91087BBD1',
    '559080AL0', '559080AM8', '559080AN6', '559080AP1', '559080AQ9',
    '035242AM8', '035242AN6', '29336TAE0',
    '20605PAJ0', '84861TAG3', '03349MAB1',
    '92343VDU5', '92343VDV3', '26885BAE0',
    '485170AQ7', '485170BC7', '87612BBL5', '837004CK4',
    '11134LAH2', '12673PAJ4', '84861TAJ7', '264414AX1',
    '92340LAC3', '928563AE5',
}



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

    # Already audited; remove from the list of NEW candidates for review.
    reviewed = set(EARLY_REDEMPTIONS) | REVIEWED_NON_REDEMPTIONS
    joined = joined.loc[~joined.cusip.isin(reviewed)].copy()

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
    print(f'Potential premature exits before prior-review exclusions: {all_exits:,}')
    print(f'Previously confirmed redemptions excluded: {len(EARLY_REDEMPTIONS):,}')
    print(f'Other previously reviewed CUSIPs excluded: {len(REVIEWED_NON_REDEMPTIONS):,}')
    print(f'Candidates in review window {args.start} through {args.end}: {len(result):,}')
    if not result.empty:
        print(result.review_priority.value_counts().sort_index().to_string())
        print('\nALL FLAGGED BONDS FOR REVIEW:')
        print(result[['review_priority', 'cusip', 'issuer', 'last_observed_month',
                      'contractual_maturity']].to_string(index=False, max_rows=None))
    verified_path = args.output.with_name('verified_early_redemptions.csv')
    pd.DataFrame([
        {'cusip': cusip, 'redemption_date': date,
         'redemption_price_pct_par_ex_accrued': price,
         'status': ('PAYOFF VERIFIED' if price is not None
                    else 'REDEMPTION VERIFIED - PRICE UNVERIFIED')}
        for cusip, (date, price) in sorted(EARLY_REDEMPTIONS.items())
    ]).to_csv(verified_path, index=False)
    print(f'Verified redemption list saved: {verified_path}')
    print(f'\\nSaved: {args.output}\\n')
    print('WARNING: A premature exit can be a redemption OR data/index coverage loss.')
    print('Do not use this report for point-in-time investment eligibility.')


if __name__ == '__main__':
    main()
