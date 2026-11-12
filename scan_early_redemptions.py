"""One-time AGG premature data disappearance audit for select bonds.
This file cross-references zipped AGG observations against WRDS callable-bond data
to flag bonds that may have been redeemed early. Flagged bonds 
were reviewed against SEC filings and the FINRA fixed income site.

Confirmed early redemptions are grouped in EARLY_REDEMPTIONS. Other reviewed
bonds are grouped in NOT_REDEEMED_EARLY. Both groups are excluded from future
scans and recorded in all_reviewed_exclusions.csv; unresolved cases remain.

Terminal returns are estimated using the final pre-redemption month's bond
price, accrued interest, redemption proceeds, and interim coupon payments:

    R_terminal = (P_redemption + AI_redemption + C_interim) / (P_t + AI_t) - 1

Returns are calculated using a 30/360 interest approximation and saved
in verified_early_redemptions.csv. Bonds without an exact redemption date
(including records with only a month, year, or an unknown date) are excluded from the return calculation.
"""

from pathlib import Path
import argparse
import pandas as pd

from process_investable_universe import DEFAULT_BONDS, PROJECT_ROOT

REFERENCE = PROJECT_ROOT / '01_Source_Data' / 'early_redemption_reference.csv'
OUTPUT = PROJECT_ROOT / 'stage1_outputs' / 'early_redemption_candidates.csv'
START_MONTH = '2023-12'  # Match the current backtest decision-month window.
END_MONTH = '2025-10'
# Early redemptions: CUSIP -> (redemption date, price per $100 face).
# YYYY-MM or YYYY means only the month or year is known; '' means
# the exact redemption date has not yet been recorded. Returns remain blank
# for these entries until the date is verified.
EARLY_REDEMPTIONS = {
    '876030AD9': ('2024-11-25', 101.0),
    '876030AE7': ('2024-11-25', 101.0),
    '876030AF4': ('2024-11-25', 101.0),
    '501044DS7': ('2024-12-18', 101.0),
    '501044DT5': ('2024-12-18', 101.0),
    '501044DU2': ('2024-12-18', 101.0),
    '76132FAC1': ('2025-02-22', 100.0),
    '595112BP7': ('2025-05-27', 100.0),
    '46590XAZ9': ('2025-07-03', 102.478),
    '929160BA6': ('2024-03-01', 100.0),
    '750236AX9': ('2024-03-19', 100.396),
    '91911TAP8': ('2024-07-25', 100.0),
    '595112BV4': ('2025-10-24', 100.0),
    '595112CA9': ('2025-10-24', 100.0),
    '12673PAJ4': ('2025-10-23', 100.0),
    '11134LAH2': ('2025-10-23', 100.0),
    '928563AE5': ('2025-10-03', 100.0),
    '87612BBL5': ('2025-07-15', 100.0),
    '256677AE5': ('2025-09-30', 100.0),
    '23317HAF9': ('2024-08-21', 100.0),
    '81618TAE0': ('2024-03', 100.0),
    '04273WAD3': ('2024-04', 100.0),
    '16412XAC9': ('2024-04', 100.0),
    '14162VAB2': ('2025-07-31', 100.0),
    '29250NBQ7': ('2024-03-08', 100.0),
    '462613AM2': ('2024-04', 100.0),
    '14448CBB9': ('2024-06', 100.0),
    '29364WBK3': ('2024-08', 100.0),
    '581557BS3': ('2024', 100.0),  # McKesson — 2024 redemption; exact month/day unavailable
    '42824CBM0': ('2024-10', 100.0),  # Hewlett Packard Enterprise
    '71429MAB1': ('2024-10-02', 100.0),
    '444859BW1': ('2024-11', 100.0),
    '559080AK2': ('2025', 100.0),  # Magellan — 2025 redemption; exact month/day unavailable
    '844741BJ6': ('2024-12', 100.0),
    '92556HAA5': ('2024-12-27', 100.0),
    '92940PAB0': ('2024-12-06', 100.0),
    '370334CS1': ('2025', 100.0),  # General Mills — fiscal Q3 FY2025; exact month/day unavailable
    '361841AJ8': ('2025-03-03', 100.0),
    '436106AA6': ('2025-02-16', 100.0),
    '53522KAB9': ('2025-02', 100.0),
    '55903VBG7': ('2025-02', 100.0),
    '595112BM4': ('2025-02', 100.0),
    '636180BQ3': ('2025-03-06', 100.0),
    '876030AB3': ('2024-11-25', 101.0),  # Tapestry 2025 notes
    '876030AC1': ('2024-11-25', 101.0),  # Tapestry 2026 notes
    '501044DR9': ('2024-12-18', 101.0),  # Kroger 2026 notes
    '24703TAC0': ('', 100.0),  # Dell International / EMC; redemption date not yet recorded
    '95001DCG3': ('', 100.0),  # Wells Fargo; redemption date not yet recorded
    '00206RMP4': ('', 100.0),  # AT&T; redemption date not yet recorded
    '31572UAG1': ('', 100.0),  # Fibria Overseas; redemption date not yet recorded
    '097023CT0': ('', 100.0),  # Boeing; redemption date not yet recorded
    '025816CQ0': ('', 100.0),  # American Express; redemption date not yet recorded
    '035240AL4': ('2025-05-30', 100),  # Anheuser-Busch InBev Worldwide
    '866677AK3': ('2025-05-10', 106.153),  # Sun Communities 2029
    '866677AJ6': ('2025-05-10', 108.588),  # Sun Communities 2033
    '74348TAU6': ('2025-06-18', 100),  # Prospect Capital
    '026874DH7': ('2025-06-26', 100),  # AIG
    '21036PBN7': ('2025-06-12', 100),  # Constellation Brands
    '52107QAH8': ('2025-09-18', 100),  # Lazard
    '55336VAJ9': ('2025-04-09', 100),  # MPLX
    '674599EE1': ('2025', 100),  # Occidental; Q1 2025, exact date unknown
    '674599EB7': ('2025', 100),  # Occidental; Apr-Jun 2025, exact date unknown
    '674599DC6': ('2025', 100),  # Occidental; Apr-Jun 2025, exact date unknown
    '92343VDD3': ('2025-04-23', 100),  # Verizon
    '559222AX2': ('2025-05-23', 100),  # Magna
    '65163LAQ2': ('2025-02-07', 100),  # Newmont/Newcrest
    '125523CR9': ('2025', 100),  # Cigna; Q1 2025, exact date unknown
    '302491AW5': ('2025', 100),  # FMC; May-Jun 2025, exact date unknown
    '651229AW6': ('2025-05', 100),  # Newell Rubbermaid
    '00751YAH9': ('2025', 100),  # Advance Auto; fiscal Q3 2025, exact date unknown
    '031162DN7': ('2025', 100),  # Amgen; 2025 date unknown
    '95000U2K8': ('', 100.0),  # Wells Fargo; redemption date not yet recorded
    '198280AF6': ('', 100.0),  # Columbia Pipeline; redemption date not yet recorded
    '26885BAC4': ('', 100.0),  # EQM Midstream; redemption date not yet recorded
    '025816CY3': ('', 100.0),  # American Express; redemption date not yet recorded
    '29336UAF4': ('', 100.0),  # EnLink Midstream Partners; redemption date not yet recorded
    '81180WAR2': ('', 100.0),  # Seagate; redemption date not yet recorded
    '37045VAV2': ('', 100.0),  # General Motors; redemption date not yet recorded
    '403949AB6': ('', 100.0),  # HF Sinclair; redemption date not yet recorded
    '403949AP5': ('', 100.0),  # HF Sinclair; redemption date not yet recorded
    '44106MAV4': ('', 100.0),  # Hospitality Properties; redemption date not yet recorded
    '87264AAT2': ('', 100.0),  # T-Mobile USA; redemption date not yet recorded
    '98978VAU7': ('', 100.0),  # Zoetis; redemption date not yet recorded
    '023135BN5': ('', 100.0),  # Amazon; redemption date not yet recorded
    '28622HAA9': ('', 100.0),  # Elevance Health; redemption date not yet recorded
    '361841AH2': ('', 100.0),  # GLP Capital; redemption date not yet recorded
    '674599CH6': ('', 100.0),  # Occidental Petroleum; redemption date not yet recorded
    '723787AV9': ('', 100.0),  # Pioneer Natural Resources; redemption date not yet recorded
}

# Other reviewed bonds not confirmed as early redemptions.
NOT_REDEEMED_EARLY = {
    '25470DAT6': None,
    '92343VDC5': None,
    '91087BAY6': None,
    '91087BAZ3': None,
    '91087BBA7': None,
    '91087BBB5': None,
    '91087BBC3': None,
    '91087BBD1': None,
    '559080AL0': None,
    '559080AM8': None,
    '559080AN6': None,
    '559080AP1': None,
    '559080AQ9': None,
    '035242AM8': None,
    '035242AN6': None,
    '29336TAE0': None,
    '20605PAJ0': None,
    '84861TAG3': None,
    '03349MAB1': None,
    '92343VDU5': None,
    '92343VDV3': None,
    '26885BAE0': None,
    '485170AQ7': None,
    '485170BC7': None,
    '84861TAJ7': None,
    '264414AX1': None,
    '92340LAC3': None,
    '09857LAR9': "Booking 2030 security observed in 2025 SEC holdings",
    '161175CM4': "Charter 2026 issue still outstanding in June 2025",
    '837004CK4': "2019 partial tender; CUSIP in September 2026 holdings",
    '517834AG2': "Scheduled maturity 2024-08-08; preceding data gap",
    '78349AAC7': "RWJ Barnabas Health — set aside for this audit",
    '29336TAA8': "EnLink Midstream — later holdings; set aside",
    '29336UAG2': "EnLink Midstream Partners — partial tender / continuing balance",
    '103730BF6': "BP Capital Markets America — later outstanding debt",
    '53227JAD6': "Life Storage — debt exchange / successor security",
    '31572UAF3': "Fibria Overseas — debt exchange / successor security",
    '00217GAA1': "Aptiv — set aside per latest review; prior evidence of early redemption, not rejected",
    '20030NDZ1': "Comcast — announced redemption; settlement not established in this review",
    '10373QBF6': "BP Capital Markets — continuing issue, excluded from redemption review",
    '032511BN6': "Anadarko Petroleum — full early redemption not established",
    '92340LAA7': "VEREIT Operating Partnership — legacy debt exchange",
    '92340LAD1': "VEREIT Operating Partnership — legacy debt exchange / maturity",
    '92343VDR2': "Verizon Communications — subsequent holdings of the 2039 issue",
    '92857WBJ8': "Vodafone - not callable"
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
    reviewed = set(EARLY_REDEMPTIONS) | set(NOT_REDEEMED_EARLY)
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


def number(value):
    """Read cash amounts, allowing source $, commas, and percent signs."""
    if pd.isna(value):
        return float('nan')
    clean = str(value).strip().replace('$', '').replace(',', '').rstrip('%')
    return pd.to_numeric(clean, errors='coerce')


def days_30_360(start, end):
    """Approximate 30/360 year fraction used for conventional coupon bonds."""
    return (360 * (end.year - start.year)+ 30 * (end.month - start.month)+ min(end.day, 30) - min(start.day, 30)) / 360.0


def coupon_cashflows(row, event_date):
    """Return coupon cash paid, accrued interest at redemption, estimated AI at t.
    All monetary amounts per $100 face; COUPON is the annual coupon in %.
    Requires a conventional fixed-rate coupon calendar. No guessed schedule."""
    rate = number(row['COUPON'])
    freq_raw = number(row['NCOUPS'])
    if not pd.notna(rate) or not pd.notna(freq_raw) or rate < 0:
        raise ValueError('missing or invalid coupon or frequency')
    if rate == 0:
        return 0.0, 0.0, 0.0
    if freq_raw not in (1, 2, 4, 12):
        raise ValueError('unsupported coupon frequency')
    frequency = int(freq_raw)
    next_coupon = pd.to_datetime(row['nextcoup'], errors='coerce')
    asof_date = row['DATE']
    if pd.isna(next_coupon) or not (asof_date < next_coupon):
        raise ValueError('missing or stale next coupon date')

    # Start from the last scheduled coupon preceding the valuation date.
    step_months = 12 // frequency
    previous_coupon = next_coupon - pd.DateOffset(months=step_months)
    coupons_paid = 0.0
    due = next_coupon
    while due <= event_date:
        coupons_paid += rate / frequency
        previous_coupon = due
        due = due + pd.DateOffset(months=step_months)

    ai_at_purchase_estimate = rate * days_30_360(
        next_coupon - pd.DateOffset(months=step_months), asof_date
    )
    ai_at_redemption = rate * days_30_360(previous_coupon, event_date)
    if ai_at_purchase_estimate < -1e-9 or ai_at_redemption < -1e-9:
        raise ValueError('invalid accrued-interest dates')
    return coupons_paid, ai_at_redemption, ai_at_purchase_estimate


def compute_terminal_returns(bond_path):
    """Estimate t -> redemption-month returns for the verified event registry.

    Uses month t immediately BEFORE the event month, even if the raw dataset
    contains an observation in the redemption month. Existing RET_EOM is
    included only for comparison, never overwritten by this scanner.
    """
    path = Path(bond_path)
    if not path.exists() and Path(str(path) + '.zip').exists():
        path = Path(str(path) + '.zip')
    needed = {'cusip', 'DATE', 'PRICE_EOM', 'COUPACC', 'COUPON',
              'NCOUPS', 'nextcoup', 'RET_EOM'}
    subsets = []
    for chunk in pd.read_csv(
        path, usecols=lambda c: c in needed, dtype=str,
        chunksize=200_000, encoding='utf-8-sig', low_memory=False
    ):
        if not needed.issubset(chunk):
            raise ValueError(f'Bond history missing terminal cashflow columns: {needed - set(chunk)}')
        chunk['cusip'] = chunk.cusip.fillna('').str.strip().str.upper()
        small = chunk.loc[chunk.cusip.isin(EARLY_REDEMPTIONS)].copy()
        if not small.empty:
            small['DATE'] = pd.to_datetime(small.DATE, errors='coerce')
            subsets.append(small)
    history = (pd.concat(subsets, ignore_index=True)
               if subsets else pd.DataFrame(columns=list(needed)))
    history['month'] = history.DATE.dt.to_period('M')

    output = []
    for cusip, (date, principal_payout) in sorted(EARLY_REDEMPTIONS.items()):
        if len(date) != 10:  # Exact redemption day unknown.
            continue
        event_date = pd.Timestamp(date)
        event_month = event_date.to_period('M')
        decision_month = event_month - 1
        own = history.loc[history.cusip.eq(cusip)]
        prior = own.loc[own.month.eq(decision_month)].sort_values('DATE')
        actual = own.loc[own.month.eq(event_month)].sort_values('DATE')
        info = {
            'cusip': cusip, 'decision_month': str(decision_month),
            'redemption_month': str(event_month),
            'R_terminal': float('nan'), 'R_terminal_pct': float('nan'),
            'purchase_clean_price_per_100': float('nan'),
            'purchase_accrued_interest_per_100': float('nan'),
            'redemption_accrued_interest_per_100': float('nan'),
            'interim_coupons_per_100': float('nan'),
            'recorded_event_month_RET_EOM': float('nan'),
            'calculation_status': 'MISSING PRIOR-MONTH DATA',
        }
        if not actual.empty:
            source_return = str(actual.iloc[-1]['RET_EOM']).strip()
            # The source expresses realized returns with an explicit percent sign.
            if source_return.endswith('%'):
                info['recorded_event_month_RET_EOM'] = (
                    number(source_return) / 100.0
                )
        if prior.empty:
            output.append(info)
            continue

        row = prior.iloc[-1]
        price = number(row['PRICE_EOM'])
        if pd.isna(price) or price <= 0:
            info['calculation_status'] = 'INVALID PRIOR-MONTH PRICE'
            output.append(info)
            continue
        try:
            paid_coupons, final_ai, estimated_purchase_ai = coupon_cashflows(
                row, event_date
            )
            purchase_ai = number(row['COUPACC'])
            if pd.isna(purchase_ai):
                purchase_ai = estimated_purchase_ai
            dirty_purchase = price + purchase_ai
            if not (pd.notna(dirty_purchase) and dirty_purchase > 0):
                raise ValueError('invalid purchase dirty price')
            if row['DATE'] >= event_date:
                raise ValueError('valuation on/after redemption date')
            r_terminal = (
                (principal_payout + final_ai + paid_coupons)
                / dirty_purchase - 1.0
            )
            info.update({
                'R_terminal': r_terminal,
                'R_terminal_pct': 100.0 * r_terminal,
                'purchase_clean_price_per_100': price,
                'purchase_accrued_interest_per_100': purchase_ai,
                'redemption_accrued_interest_per_100': final_ai,
                'interim_coupons_per_100': paid_coupons,
                'calculation_status': 'ESTIMATED - 30/360 INTEREST',
            })
        except (ValueError, TypeError, OverflowError) as exc:
            info['calculation_status'] = f'UNRESOLVED CASH FLOWS: {exc}'
        output.append(info)
    return pd.DataFrame(output)

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
    print(f'Confirmed early redemptions excluded: {len(EARLY_REDEMPTIONS):,}')
    print(f'Other reviewed bonds excluded: {len(NOT_REDEEMED_EARLY):,}')
    print(f'Candidates in review window {args.start} through {args.end}: {len(result):,}')
    if not result.empty:
        print(result.review_priority.value_counts().sort_index().to_string())
        print('\nALL FLAGGED BONDS FOR REVIEW:')
        print(result[['review_priority', 'cusip', 'issuer', 'last_observed_month',
                      'contractual_maturity']].to_string(index=False, max_rows=None))
    # Store all previously reviewed bonds in a single exclusion ledger.
    exclusions = pd.DataFrame(
        [{
            'cusip': cusip,
            'classification': 'EARLY_REDEMPTION',
            'redemption_date': date if len(date) == 10 else '',
            'redemption_month': date[:7],
            'principal_price_per_100': price,
            'note': 'Redemption date not recorded' if not date else (
                'Exact redemption day unavailable' if len(date) != 10 else ''
            ),
        } for cusip, (date, price) in sorted(EARLY_REDEMPTIONS.items())]
        + [{
            'cusip': cusip,
            'classification': 'NOT_REDEEMED_EARLY',
            'redemption_date': '',
            'redemption_month': '',
            'principal_price_per_100': '',
            'note': note or '',
        } for cusip, note in sorted(NOT_REDEEMED_EARLY.items())]
    )
    exclusion_path = args.output.with_name('all_reviewed_exclusions.csv')
    exclusions.to_csv(exclusion_path, index=False)
    print(f'Full reviewed/excluded list ({len(exclusions):,} CUSIPs): {exclusion_path}')

    verified_path = args.output.with_name('verified_early_redemptions.csv')
    verified = pd.DataFrame([
        {'cusip': cusip, 'redemption_date': date,
         'redemption_price_pct_par_ex_accrued': price,
         'status': 'EARLY REDEMPTION'}
        for cusip, (date, price) in sorted(EARLY_REDEMPTIONS.items())
    ]).merge(
        compute_terminal_returns(args.bonds), on='cusip',
        how='left', validate='one_to_one'
    )
    no_day = verified.redemption_date.str.len().ne(10)
    verified.loc[no_day, 'calculation_status'] = 'EXACT REDEMPTION DAY NOT VERIFIED'
    verified.loc[no_day, 'redemption_month'] = verified.loc[no_day, 'redemption_date']
    verified.loc[no_day, 'redemption_date'] = ''
    verified.to_csv(verified_path, index=False)
    print(f'Verified redemptions and R_terminal saved: {verified_path}')
    print(verified[['cusip', 'redemption_date', 'R_terminal_pct',
                    'calculation_status']].to_string(index=False))
    print(f'Saved: {args.output}')
    print('WARNING: A premature exit can be a redemption OR data/index coverage loss.')
    print('Do not use this report for point-in-time investment eligibility.')



if __name__ == '__main__':
    main()
