"""Inspect whether monthly AGG YIELD behaves like a changing market yield."""
import pandas as pd
import numpy as np
from process_investable_universe import load_data

FIELDS = ['cusip', 'DATE', 'YIELD', 'PRICE_EOM', 'RET_EOM', 'T_Yld_Pt', 'COUPON']
raw, bonds, info = load_data(usecols=FIELDS)
bonds['month'] = bonds.DATE.dt.to_period('M')
bonds = bonds.sort_values(['cusip', 'month'])
previous = bonds[['cusip','month','YIELD','PRICE_EOM']].copy()
previous['month'] += 1
previous = previous.rename(columns={'YIELD':'prev_yield','PRICE_EOM':'prev_price'})
pairs = bonds.merge(previous, on=['cusip','month'], how='inner',validate='one_to_one')
pairs['dy_bp'] = (pairs.YIELD - pairs.prev_yield) * 10000
pairs['dp'] = pairs.PRICE_EOM - pairs.prev_price
pairs['yield_static'] = pairs.dy_bp.abs() < 0.0001
pairs['price_changed'] = pairs.dp.abs() > .00001
pairs['year'] = pairs.month.dt.year
pairs = pairs.dropna(subset=['dy_bp'])
print('BOND YIELD FIELD AUDIT: consecutive-month same-CUSIP observations')
print('Source observations:',len(bonds),'consecutive pairs with yields:',len(pairs))
print('YIELD nonmissing:', bonds.YIELD.notna().sum())
print('T_Yld_Pt nonmissing:', bonds.T_Yld_Pt.notna().sum())
print('RET_EOM nonmissing:',bonds.RET_EOM.notna().sum())
print('Overall unchanged-yield share:',float(pairs.yield_static.mean()))
print('Unchanged yield AND changing quoted price:',int((pairs.yield_static & pairs.price_changed).sum()))
print('Of pairs with both monthly prices and yields, unchanged yield, changed price:',
      float((pairs.yield_static & pairs.price_changed).mean()))
yearly=pairs.groupby('year').agg(pairs=('dy_bp','size'),zero_change_share=('yield_static','mean'),
    changed_price_share=('price_changed','mean'),median_absolute_change_bp=('dy_bp',lambda x:float(x.abs().median())),
    p99_abs_change_bp=('dy_bp',lambda x:float(x.abs().quantile(.99))))
print(yearly.to_string())
nov=pairs.loc[pairs.month==pd.Period('2025-12','M')]
print('2025-12 pairs:',len(nov),'unchanged share:',float(nov.yield_static.mean()) if len(nov) else 'n/a')
print('First 10 observed nonzero yields in 2025-12:')
print(nov.loc[~nov.yield_static,['cusip','month','prev_yield','YIELD','dy_bp','prev_price','PRICE_EOM']].head(10).to_string(index=False))
print('First 10 bonds with unchanged yields but moving prices in 2025-12:')
print(nov.loc[nov.yield_static & nov.price_changed,
    ['cusip','month','prev_yield','YIELD','prev_price','PRICE_EOM','RET_EOM','COUPON']].head(10).to_string(index=False))
