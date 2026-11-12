"""One-time AGG premature data disappearance audit for select bonds.
This file cross-references zipped AGG observations against WRDS callable-bond data
to flag bonds that may have been redeemed early. Flagged bonds 
were reviewed against SEC filings and the FINRA fixed income site.

The confirmed full redemption events live in EARLY_REDEMPTIONS; reviewed
non-redemptions, debt exchanges and partial tenders are excluded separately.
These and the separately recorded month-verified redemptions and regular
maturities are excluded from future candidate outputs; unresolved cases remain.
An all_reviewed_exclusions.csv ledger lists EVERY excluded CUSIP and reason.

A bond disappearing from the filtered AGG source is NOT proof of redemption.
WRDS NEXT_CALL_PRICE is a contractual schedule term, NOT a realized payoff.
When a full redemption is confirmed but its make-whole premium is unknown,
the listed 100.0 is a CONSERVATIVE principal-only assumption, not an exact price.

Terminal holding-period return (all dollar figures per $100 face value):
    R_terminal = (P_redemption + AI_redemption + C_interim) / (P_t + AI_t) - 1

where:
  P_t            last observed clean purchase-date/month-end bond price;
  AI_t           accrued interest at that valuation date;
  P_redemption   verified or clearly labeled assumed redemption principal
                 payoff, as a percentage of $100 face value;
  AI_redemption  accrued interest payable on the redemption date, excluding
                 interest separately paid as an interim coupon; and
  C_interim      sum of separate coupon payments between valuation and exit.

The scan calculates an indicative return for the month preceding the event,
with a 30/360 coupon approximation, and reports R_terminal as a decimal and
R_terminal_pct in percent. A coupon paid on the redemption date is counted
only once; do not double-count it as both accrued interest and an interim coupon.
Compare to observed RET_EOM before substituting it in any backtest. The scan
does NOT change the portfolio, raw returns, eligibility or benchmark.
"""

from pathlib import Path
import argparse
import pandas as pd

from process_investable_universe import DEFAULT_BONDS, PROJECT_ROOT

REFERENCE = PROJECT_ROOT / '01_Source_Data' / 'early_redemption_reference.csv'
OUTPUT = PROJECT_ROOT / 'stage1_outputs' / 'early_redemption_candidates.csv'
START_MONTH = '2023-12'  # Match the current backtest decision-month window.
END_MONTH = '2025-10'
# Verified FULL redemption (event date, principal payout per $100 face).
# Excludes accrued interest and any separately payable coupon. For issues with
# unverified make-whole premiums, assume 100.0 (conservative PAR-ONLY proxy);
# those CUSIPs are labeled in ASSUMED_PAR_REDEMPTIONS below.
# NEVER infer an actual executed redemption price from WRDS NEXT_CALL_PRICE.
EARLY_REDEMPTIONS = {
    '876030AD9': ('2024-11-25', 101.0),   # Tapestry
    '876030AE7': ('2024-11-25', 101.0),
    '876030AF4': ('2024-11-25', 101.0),
    '501044DS7': ('2024-12-18', 101.0),   # Kroger
    '501044DT5': ('2024-12-18', 101.0),
    '501044DU2': ('2024-12-18', 101.0),
    '76132FAC1': ('2025-02-22', 100.0),    # Retail Opportunity, make-whole
    '595112BP7': ('2025-05-27', 100.0),    # Micron 2027
    '46590XAZ9': ('2025-07-03', 102.478), # JBS
    '929160BA6': ('2024-03-01', 100.0),   # Vulcan, redeemed at par
    '750236AX9': ('2024-03-19', 100.396), # Radian
    '91911TAP8': ('2024-07-25', 100.0),    # Vale make-whole
    '595112BV4': ('2025-10-24', 100.0),    # Micron 2029 B
    '595112CA9': ('2025-10-24', 100.0),    # Micron 2028
    '12673PAJ4': ('2025-10-23', 100.0),    # CA Inc 4.700% 2027: 100% base + make-whole
    '11134LAH2': ('2025-10-23', 100.0),    # Broadcom Corp 3.875% 2027
    '928563AE5': ('2025-10-03', 100.0),    # VMware 4.650% 2027
    '87612BBL5': ('2025-07-15', 100.0),    # Targa 6.500% 2027: full redemption at par
    '256677AE5': ('2025-09-30', 100.0),    # Dollar General, par recovery assumption
    '23317HAF9': ('2024-08-21', 100.0),    # SITE Centers / DDR, par recovery assumption
}

# Conservative 100%-of-face principal proxies where make-whole cash premium
# was not verified. These are *not* confirmed clean redemption prices.
ASSUMED_PAR_REDEMPTIONS = {
    '76132FAC1', '595112BP7', '91911TAP8', '595112BV4',
    '595112CA9', '12673PAJ4', '11134LAH2', '928563AE5',
    '256677AE5', '23317HAF9',
}

# Full redemptions evidenced by issuer/SEC disclosures, but exact day unknown.
# These cannot receive a calculated R_terminal until an actual date is verified.
MONTH_ONLY_REDEMPTIONS = {
    '81618TAE0': ('2024-03', 100.0),  # Office Properties, at par
    '04273WAD3': ('2024-04', 100.0),  # Arrow 6.125% 2026, at par
    '16412XAC9': ('2024-04', 100.0),  # Cheniere 5.875% 2025, par-only proxy
}

# Not an early redemption: the bond's stated due date was August 8, 2024.
# Its June-to-July missing return remains a DATA-COVERAGE issue; ordinary
# maturity in August does not supply an observed July return.
REVIEWED_SCHEDULED_MATURITIES = {
    '517834AG2': ('2024-08-08', 'Las Vegas Sands 3.2% contractual maturity'),
}

# Evidence trail / reasons for the newest decisions.
REVIEW_NOTES = {
    '09857LAR9': 'Booking 2030 CUSIP appears in 2025 SEC fund holdings',
    '161175CM4': 'Charter 2026 CUSIP still outstanding on June 2025 issuer debt schedule',
    '256677AE5': 'Dollar General fully repaid in September 2025; insurer transaction says September 30',
    '23317HAF9': 'SITE Centers redeemed entire 4.700% 2027 issue on August 21 2024',
    '81618TAE0': 'Office Properties redeemed all 4.25% 2024 notes at par in March; day not verified',
    '04273WAD3': 'Arrow called 6.125% 2026 notes at par in April; day not verified',
    '16412XAC9': 'Cheniere fully redeemed 2025 notes in April 2024; pre-Oct par call, premium unknown',
    '517834AG2': 'Due Aug 8 2024; absent from July 2024 AGG data: unresolved preceding-month coverage gap, NOT an established early call',
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
    '485170AQ7', '485170BC7',
    '84861TAJ7', '264414AX1',
    '92340LAC3',
    '09857LAR9', '161175CM4',  # SEC/issuer still outstanding past flagged month
    '837004CK4',  # 2019 partial tender; same CUSIP in Sep 2026 fund holdings
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
    reviewed = (set(EARLY_REDEMPTIONS) | set(MONTH_ONLY_REDEMPTIONS)
                | REVIEWED_NON_REDEMPTIONS | set(REVIEWED_SCHEDULED_MATURITIES))
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
    return (
        360 * (end.year - start.year)
        + 30 * (end.month - start.month)
        + min(end.day, 30) - min(start.day, 30)
    ) / 360.0


def coupon_cashflows(row, event_date):
    """Return coupon cash paid, accrued interest at redemption, estimated AI at t.

    All monetary amounts per $100 face; COUPON is the annual coupon in %.
    Requires a conventional fixed-rate coupon calendar. No guessed schedule.
    """
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
                'calculation_status': (
                    'ESTIMATED - PAR PAYOUT ASSUMED, 30/360 INTEREST'
                    if cusip in ASSUMED_PAR_REDEMPTIONS
                    else 'ESTIMATED - DISCLOSED PAYOUT, 30/360 INTEREST'
                ),
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
    print(f'Previously confirmed redemptions excluded: {len(EARLY_REDEMPTIONS):,}')
    print(f'Other previously reviewed CUSIPs excluded: {len(REVIEWED_NON_REDEMPTIONS):,}')
    print(f'Confirmed redemption, exact day still unknown: {len(MONTH_ONLY_REDEMPTIONS):,}')
    print(f'Reviewed ordinary maturities: {len(REVIEWED_SCHEDULED_MATURITIES):,}')
    print(f'Candidates in review window {args.start} through {args.end}: {len(result):,}')
    if not result.empty:
        print(result.review_priority.value_counts().sort_index().to_string())
        print('\nALL FLAGGED BONDS FOR REVIEW:')
        print(result[['review_priority', 'cusip', 'issuer', 'last_observed_month',
                      'contractual_maturity']].to_string(index=False, max_rows=None))
    # Audit trail for every excluded CUSIP: verified events, ordinary
    # maturities, continuing issues, partial tenders and exchanges.
    excluded_rows = []
    for cusip, (date, price) in sorted(EARLY_REDEMPTIONS.items()):
        excluded_rows.append({
            'cusip': cusip, 'review_status': 'FULL_REDEMPTION_DATED',
            'event_date': date, 'event_month': date[:7],
            'principal_price_per_100': price,
            'price_basis': ('ASSUMED_PAR' if cusip in ASSUMED_PAR_REDEMPTIONS
                            else 'DISCLOSED_OR_VERIFIED'),
            'review_note': REVIEW_NOTES.get(
                cusip, 'Reviewed full redemption (SEC/issuer filings)'),
        })
    for cusip, (month, price) in sorted(MONTH_ONLY_REDEMPTIONS.items()):
        excluded_rows.append({
            'cusip': cusip, 'review_status': 'FULL_REDEMPTION_DAY_PENDING',
            'event_date': '', 'event_month': month,
            'principal_price_per_100': price,
            'price_basis': ('ASSUMED_PAR' if cusip == '16412XAC9'
                            else 'PAR_CONFIRMED'),
            'review_note': REVIEW_NOTES.get(
                cusip, 'Full redemption established; exact payment day unknown'),
        })
    for cusip in sorted(REVIEWED_NON_REDEMPTIONS):
        excluded_rows.append({
            'cusip': cusip, 'review_status': 'NOT_FULL_REDEMPTION_AT_FLAGGED_DATE',
            'event_date': '', 'event_month': '',
            'principal_price_per_100': '', 'price_basis': 'NOT_APPLICABLE',
            'review_note': REVIEW_NOTES.get(
                cusip, 'Previously reviewed: continuing issue, partial tender or exchange'),
        })
    for cusip, (date, description) in sorted(REVIEWED_SCHEDULED_MATURITIES.items()):
        excluded_rows.append({
            'cusip': cusip, 'review_status': 'REGULAR_MATURITY_WITH_DATA_GAP',
            'event_date': date, 'event_month': date[:7],
            'principal_price_per_100': 100.0, 'price_basis': 'PAR_AT_MATURITY',
            'review_note': REVIEW_NOTES.get(cusip, description),
        })
    exclusions = pd.DataFrame(excluded_rows).sort_values(
        ['review_status', 'cusip']
    )
    exclusion_path = args.output.with_name('all_reviewed_exclusions.csv')
    exclusions.to_csv(exclusion_path, index=False)
    print(f'Full reviewed/excluded list ({len(exclusions):,} CUSIPs): {exclusion_path}')

    verified_path = args.output.with_name('verified_early_redemptions.csv')
    verified = pd.DataFrame([
        {'cusip': cusip, 'redemption_date': date,
         'redemption_price_pct_par_ex_accrued': price,
         'status': ('ASSUMED PAR - MAKE-WHOLE PREMIUM UNVERIFIED'
                    if cusip in ASSUMED_PAR_REDEMPTIONS
                    else 'DISCLOSED REDEMPTION PRICE')}
        for cusip, (date, price) in sorted(EARLY_REDEMPTIONS.items())
    ]).merge(
        compute_terminal_returns(args.bonds), on='cusip',
        how='left', validate='one_to_one'
    )
    # Date-incomplete events are included in the verified event report,
    # but cannot have an R_terminal calculated without inventing cash timing.
    undated = pd.DataFrame([
        {'cusip': cusip, 'redemption_date': '',
         'redemption_month': month,
         'redemption_price_pct_par_ex_accrued': price,
         'status': ('ASSUMED PAR - EXACT DAY UNVERIFIED' if cusip == '16412XAC9'
                    else 'PAR REDEMPTION - EXACT DAY UNVERIFIED'),
         'R_terminal': float('nan'), 'R_terminal_pct': float('nan'),
         'calculation_status': 'EXACT REDEMPTION DAY NOT VERIFIED'}
        for cusip, (month, price) in sorted(MONTH_ONLY_REDEMPTIONS.items())
    ])
    verified = pd.concat([verified, undated], ignore_index=True)
    verified.to_csv(verified_path, index=False)
    print(f'Verified redemptions and R_terminal saved: {verified_path}')
    print(verified[['cusip', 'redemption_date', 'R_terminal_pct',
                    'calculation_status']].to_string(index=False))
    print(f'Saved: {args.output}')
    print('WARNING: A premature exit can be a redemption OR data/index coverage loss.')
    print('Do not use this report for point-in-time investment eligibility.')



if __name__ == '__main__':
    main()
