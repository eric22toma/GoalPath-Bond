"""Pooled elastic net forecasting next-month bond yield changes.
Each bond-month has its own predictors; the model is trained across bonds.
Forecasted changes are in percentage points, with basis-point output. """

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from process_investable_universe import (process_all_data, history_for_features, PROJECT_ROOT,DEFAULT_BONDS, DEFAULT_FRED, DEFAULT_CURVE)

FEATURES = ['level', 'slope', 'curvature', 'lag_level_change', 'rate_volatility',
            'fed_funds_change', 'vix', 'time_to_maturity', 'rich_cheap', 'cpi_inflation_lag1']
OPTIONAL_FEATURES = ['forward_5y']
BOND_COLUMNS = ['cusip', 'DATE', 'YIELD', 'MATURITY', 'RET_EOM', 'DURATION', 'COUPON',
    'PRICE_EOM', 'COUPACC', 'AMOUNT_OUTSTANDING', 'T_DVolume',
    'NCOUPS', 'nextcoup', 'DEFAULTED', 'CONV', 'BOND_TYPE', 'RATING_NUM']

MAX_CV_BONDS_PER_MONTH = 150
MIN_INITIAL_TRAIN_MONTHS = 12
CV_MIXING_GRID = (0.20, 0.40, 0.60, 0.80, 0.95)
CV_PENALTY_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
FIRST_OOS_YEAR = 2019
LAST_OOS_YEAR = 2025
# Experimental training-only screen; set to None to reproduce the original model.
# dy is stored in percentage points, so 500 bp corresponds to 5.0 pp.
MAX_TRAIN_YIELD_CHANGE_BP = 500.0

# Save file location for prepare_stage1_inputs.
DEFAULT_FORECAST_BUNDLE = (PROJECT_ROOT / 'stage1_outputs' / 'forecasts'/ 'elastic_net_outputs.pkl.gz')


def sven_yield(tau, params):
    """Svensson continuously compounded zero yield, percentage-point units."""
    t = np.asarray(tau, dtype=float)
    b0,b1,b2,b3,t1,t2 = params
    x, z = t/t1, t/t2
    a, c = -np.expm1(-x)/x, -np.expm1(-z)/z
    return b0 + b1*a + b2*(a-np.exp(-x)) + b3*(c-np.exp(-z))


def macro_features(curve, fred, core=None):
    """Build monthly Treasury-curve and macroeconomic predictors."""
    daily = curve.sort_values('Date').set_index('Date')
    # Last trading day for each month.
    monthly = daily.groupby(daily.index.to_period('M')).tail(1).copy()
    if monthly.isna().all(axis=1).any():
        monthly = daily.dropna(how='all').groupby(daily.dropna(how='all').index.to_period('M')).tail(1).copy()
    monthly['month'] = monthly.index.to_period('M')
    monthly = monthly.set_index('month')
    monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq='M'))
    f = pd.DataFrame(index=monthly.index)
    f['level'] = monthly.SVENY10
    f['slope'] = monthly.SVENY10-monthly.SVENY02
    f['curvature'] = 2*monthly.SVENY05-monthly.SVENY02-monthly.SVENY10
    if 'SVENF05' in monthly:
        f['forward_5y'] = monthly.SVENF05
    f['lag_level_change'] = f.level.shift(1)-f.level.shift(2)
    # Daily changes include the prior month's final trading day.
    f['rate_volatility'] = daily.SVENY10.diff().groupby(daily.index.to_period('M')).std()
    macro = fred.sort_values('Date').set_index('Date')
    if macro.index.isna().any(): 
        raise ValueError('FRED contains invalid dates')
    macro.index = macro.index.to_period('M')
    if macro.index.has_duplicates:
        raise ValueError('Duplicate monthly FRED dates')
    macro = macro.reindex(pd.period_range(macro.index.min(), macro.index.max(), freq='M'))
    f = f.reindex(pd.period_range(f.index.min(), f.index.max(), freq='M'))
    # Lag monthly Fed funds changes for availability.
    f['fed_funds_change'] = macro.Fed_Funds_Rate.diff().shift(1)
    f['vix'] = macro.VIX
    # Check supplied YoY inflation against the CPI series.
    computed_yoy = 100*(macro.CPI/macro.CPI.shift(12)-1)
    inflation_column = next((c for c in macro if c.lower() == 'inflation_yoy'), None)
    if inflation_column is not None:
        supplied_yoy = macro[inflation_column]
        comparable = supplied_yoy.notna() & computed_yoy.notna()
        inflation = supplied_yoy
    else:
        inflation = computed_yoy
    # At month t, use the latest value recorded by t-1, at most two months old.
    # Filling occurs before the lag, so this never uses a future CPI release.
    f['cpi_inflation_lag1'] = inflation.ffill(limit=2).shift(1)
    if core is not None:
        core = core.sort_values('Date').set_index('Date')
        core.index = core.index.to_period('M')
        core = core.reindex(pd.period_range(core.index.min(), core.index.max(), freq='M'))
        f['core_inflation_lag1'] = (core.CPILFESL/core.CPILFESL.shift(12)-1).shift(1)*100
    params = monthly[['BETA0','BETA1','BETA2','BETA3','TAU1','TAU2']]
    # Verify Svensson parameters against the observed 10-year yield.
    check = params.dropna().loc['2016':'2025']
    error = [abs(sven_yield([10], row.to_numpy())[0]-monthly.loc[m,'SVENY10'])
             for m,row in check.iterrows()]
    if error and max(error) > 0.002:
        raise ValueError('Svensson formula does not reproduce source SVENY10')
    return f, params


def load_panel(prepared_bonds, macro, params):
    cols = BOND_COLUMNS
    # Input yields are decimal fractions; yield changes use percentage points.
    p = prepared_bonds.loc[:, cols].copy()
    p['month'] = p.DATE.dt.to_period('M')
    p = p.merge(macro, left_on='month', right_index=True, how='left', validate='many_to_one')
    p['time_to_maturity'] = (p.MATURITY-p.DATE).dt.days/365.25
    p['rich_cheap'] = np.nan
    for month, rows in p.groupby('month').groups.items():
        if month not in params.index or params.loc[month].isna().any(): continue
        ix = p.loc[rows].index[p.loc[rows,'time_to_maturity']>0]
        p.loc[ix,'rich_cheap'] = 100*p.loc[ix,'YIELD']-sven_yield(p.loc[ix,'time_to_maturity'],params.loc[month])
    # Require the immediately following calendar month for realized targets.
    future = p[['cusip','month','YIELD','RET_EOM','PRICE_EOM']].copy()
    future['month'] = future.month-1
    future = future.rename(columns={'YIELD':'next_yield','RET_EOM':'next_return','PRICE_EOM':'next_price'})
    p = p.merge(future,on=['cusip','month'],how='left',validate='one_to_one')
    p['target_month'] = p.month+1
    p['dy'] = 100*(p.next_yield-p.YIELD)  # realized one-month yield change (pp)
    return p


def estimator(alpha, l1_ratio):
    return make_pipeline(StandardScaler(),ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, tol=1e-7))

def training_rows(panel, asof, features, max_yield_change_bp=MAX_TRAIN_YIELD_CHANGE_BP):
    """Use only realized historical targets; optionally exclude extreme changes from training."""
    known = panel.loc[(panel.month >= pd.Period('2016-01', 'M')) &
                      (panel.target_month <= asof)].dropna(subset=features + ['dy'])
    if max_yield_change_bp is not None:
        known = known.loc[known.dy.abs() * 100 <= max_yield_change_bp]
    return known.copy()


def limit_tuning_rows(known, limit=MAX_CV_BONDS_PER_MONTH):
    """Deterministically cap bonds per month for tractable hyperparameter tuning.
    The final fitted model still uses every complete historical bond-month."""
    if limit <= 0:
        return known
    counts = known.groupby('month')['cusip'].transform('size')
    if not (counts > limit).any():
        return known
    sample = known.copy()
    ids = sample[['cusip', 'month']].astype(str)
    sample['_cv_hash'] = pd.util.hash_pandas_object(ids, index=False).to_numpy()
    sample['_cv_rank'] = sample.groupby('month')['_cv_hash'].rank(method='first')
    return sample.loc[sample['_cv_rank'] <= limit].drop(
        columns=['_cv_hash', '_cv_rank'])


def tune_elastic_net(known, features, alphas=CV_MIXING_GRID, lambdas=CV_PENALTY_GRID):
    """Tune elastic net through expanding, chronological annual validation.
    Use up to three recent folds; each fold trains on earlier target months."""
    if known.empty:
        raise ValueError('No historical yield-change observations for tuning')
    original_rows = len(known)
    known = limit_tuning_rows(known)
    years = sorted(set(known.target_month.dt.year))
    folds = []
    for year in years[1:]:
        first = pd.Period(f'{year}-01', 'M')
        train = known.target_month < first
        valid = known.target_month.between(first, pd.Period(f'{year}-12', 'M'))
        if known.loc[train, 'target_month'].nunique() >= 12 and valid.any():
            folds.append((train.to_numpy(), valid.to_numpy(), year))
    folds = folds[-3:]
    if not folds:
        # Early history: validate the last three known months, one at a time.
        # At least eight distinct prior targets precede each validation month.
        months = sorted(known.target_month.unique())
        for month in months[-3:]:
            train = known.target_month < month
            valid = known.target_month == month
            if known.loc[train, 'target_month'].nunique() >= 8 and valid.any():
                folds.append((train.to_numpy(), valid.to_numpy(), str(month)))
    grid = []
    for mixing in alphas:
        for penalty in lambdas:
            rmses = []
            for train, valid, _ in folds:
                m = estimator(penalty, mixing).fit(known.loc[train, features], known.loc[train, 'dy'])
                pred = m.predict(known.loc[valid, features])
                delta = known.loc[valid, 'dy'].to_numpy() - pred
                rmses.append(float(np.sqrt(np.mean(delta ** 2)) * 100))
            grid.append({'alpha_mixing': float(mixing),'lambda_penalty': float(penalty),'mean_validation_rmse_bp': float(np.mean(rmses)),})
    scores = pd.DataFrame(grid).sort_values(['mean_validation_rmse_bp', 'alpha_mixing', 'lambda_penalty']).reset_index(drop=True)
    best = scores.iloc[0]
    return (float(best.alpha_mixing), float(best.lambda_penalty),float(best.mean_validation_rmse_bp),[year for _, _, year in folds], scores)


def fit_forecast(panel, asof, features, mixing, penalty):
    """Refit on an expanding window and forecast bond yields from asof."""
    known = training_rows(panel, asof, features)
    if known.empty:
        raise ValueError(f'No training targets available by {asof}')
    selection = panel.loc[panel.month == asof]
    now = selection.dropna(subset=features).copy()
    model = estimator(penalty, mixing).fit(known[features], known.dy)
    now['predicted_dy_pp'] = model.predict(now[features])
    original_intercept = float(
        model[-1].intercept_ -
        np.dot(model[-1].coef_ / model[0].scale_, model[0].mean_)
    )
    coef = pd.DataFrame({
        'feature': features,
        'coefficient_per_standard_deviation': model[-1].coef_,
        'coefficient_original_units': model[-1].coef_ / model[0].scale_,
        'training_mean': model[0].mean_,
        'training_scale': model[0].scale_,
    })
    coef['intercept_standardized_inputs'] = float(model[-1].intercept_)
    coef['intercept_original_inputs'] = original_intercept
    return now, coef, len(known)


def forecast_month(panel, asof, features=FEATURES, params=None):
    """Generate H=1 yield forecasts for a chosen historical month."""
    asof = pd.Period(asof, freq='M')
    if params is None:
        known = training_rows(panel, asof, features)
        mixing, penalty, _, _, _ = tune_elastic_net(known, features)
    else:
        mixing, penalty = params
    now, coefficients, training_count = fit_forecast(panel, asof, features, mixing, penalty)
    return now, coefficients, training_count, (mixing, penalty)


def rolling_backtest(panel, features, final_asof, start_target, alphas, lambdas, fixed=None):
    """Evaluate one-month-ahead forecasts with expanding monthly refits.
    Retune once per year and use only training targets observed by each as-of
    date. Stop before the final as-of date for optimizer-compatible errors."""
    first_asof = start_target - 1
    last_asof = final_asof - 1
    results = []
    selected_params = None
    last_tuning_year = None
    for asof in pd.period_range(first_asof, last_asof, freq='M'):
        candidates = panel.loc[panel.month == asof]
        if candidates.empty or not candidates[features].notna().all(axis=1).any():
            continue
        known = training_rows(panel, asof, features)
        if known.target_month.nunique() < MIN_INITIAL_TRAIN_MONTHS:
            continue
        if fixed is not None:
            mixing, penalty = fixed
        else:
            if selected_params is None or asof.year != last_tuning_year:
                mixing, penalty, _, _, _ = tune_elastic_net(known, features, alphas, lambdas)
                selected_params = (mixing, penalty)
                last_tuning_year = asof.year
            mixing, penalty = selected_params
        now, _, _ = fit_forecast(panel, asof, features, mixing, penalty)
        valid = now.dropna(subset=['dy']).copy()
        if valid.empty:
            continue
        valid['forecast_error_bp'] = (valid.dy - valid.predicted_dy_pp) * 100
        valid['alpha_mixing'] = mixing
        valid['lambda_penalty'] = penalty
        # These return errors are used by the single-period optimizer.
        valid['predicted_return'] = (valid.COUPON / valid.PRICE_EOM / 12 -valid.DURATION * valid.predicted_dy_pp / 100 )
        valid['return_error'] = valid.next_return - valid.predicted_return
        results.append(valid)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def rolling_metrics(backtest):
    if backtest.empty:
        return pd.DataFrame()
    result = backtest.copy()
    result['year'] = result.target_month.dt.year
    return result.groupby(['year'], as_index=False).agg(
        observations=('forecast_error_bp', 'size'),
        rmse_yield_bp=('forecast_error_bp',lambda x: float(np.sqrt(np.mean(x ** 2)))),
        mae_yield_bp=('forecast_error_bp',lambda x: float(np.mean(np.abs(x)))),)


def show_selected_bonds(now, coef, asof, cusips=None):
    """Show bond predictors, next-month forecasts, and realized yield changes."""
    if cusips:
        requested = list(dict.fromkeys(cusips))
        missing = [c for c in requested if c not in set(now.cusip)]
        selected = now.set_index('cusip').loc[requested].reset_index()
    else:
        # Illustrative liquid bonds, chosen without examining future outcomes.
        selected = now.sort_values(
            ['T_DVolume','cusip'], ascending=[False,True]
        ).head(3).copy()

    feature_names = coef['feature'].tolist()
    beta = coef.set_index('feature').loc[feature_names, 'coefficient_original_units']
    intercept = float(coef.intercept_original_inputs.iloc[0])
    target_month = asof + 1
    print(f'\nONE-MONTH YIELD CHANGE: {asof} -> {target_month}')
    print(f'Bonds with complete predictors: {len(now):,}; displayed: {len(selected)}.')

    for row in selected.itertuples(index=False):
        start_yield_pct = row.YIELD * 100
        forecast_change_bp = row.predicted_dy_pp * 100
        forecast_yield_pct = start_yield_pct + row.predicted_dy_pp
        actual_yield_pct = row.next_yield * 100 if pd.notna(row.next_yield) else np.nan
        actual_change_bp = row.dy * 100 if pd.notna(row.dy) else np.nan
        error_bp = actual_change_bp - forecast_change_bp if pd.notna(actual_change_bp) else np.nan

        print(f'\nCUSIP: {row.cusip}')
        print(f'  Observed yield ({asof}):        {start_yield_pct:.4f}%')
        print(f'  Forecast yield change:          {forecast_change_bp:+.2f} bp')
        print(f'  Forecast yield ({target_month}):  {forecast_yield_pct:.4f}%')
        if pd.notna(actual_yield_pct):
            np.testing.assert_allclose(actual_change_bp, (actual_yield_pct - start_yield_pct) * 100, atol=1e-8)
            print(f'  Actual yield ({target_month}):    {actual_yield_pct:.4f}% [evaluation only]')
            print(f'  Actual yield change:            {actual_change_bp:+.2f} bp [evaluation only]')
            print(f'  Forecast error (actual - pred): {error_bp:+.2f} bp')
        else:
            print(f'  Actual yield ({target_month}):    not available')

        values = selected.loc[selected.cusip.eq(row.cusip), feature_names].iloc[0].astype(float)
        table = pd.DataFrame({'predictor': feature_names, 'value': values.to_numpy(), 'beta_original_units': beta.to_numpy(), 'contribution_bp': values.to_numpy() * beta.to_numpy() * 100,})
        print('Predictors and forecast contributions:')
        print(table.to_string(index=False))
        print(f'  Intercept contribution: {intercept * 100:+.2f} bp')
        np.testing.assert_allclose(
            intercept + values.to_numpy() @ beta.to_numpy(),
            row.predicted_dy_pp, atol=1e-10)
        
    print('Realized future yields are evaluation data, never model inputs.')


def first_backtest_target(panel, features, final_asof):
    """Find earliest month with 12 observed training targets and forecast inputs."""
    available = panel[features].notna().all(axis=1)
    targets = panel.loc[available & panel.dy.notna(), 'target_month']
    unique_targets = sorted(targets.unique())
    earliest_origin = unique_targets[MIN_INITIAL_TRAIN_MONTHS-1]
    origins = panel.loc[available & panel.month.ge(earliest_origin) & panel.month.lt(final_asof), 'month']
    return origins.min() + 1


def main():
    """Run one H=1 forecast and save the bundle consumed by Stage 1."""
    # Change these two settings when needed; no command-line arguments required.
    forecast_month = '2025-10'  # Forecast October -> November 2025.
    run_backtest = False       # Optional expanding-window historical evaluation.
    features = FEATURES.copy()

    # Prepare the original source datasets and construct bond-month predictors.
    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve_params = macro_features(
        history_for_features(data, 'curve'),
        history_for_features(data, 'fred')
    )
    months = data['bonds'][1].DATE.dt.to_period('M')
    if months.empty:
        raise ValueError('No aligned bond observations')
    model_months = pd.period_range(months.min(), months.max(), freq='M')
    macro = macro.reindex(model_months)
    curve_params = curve_params.reindex(model_months)
    panel = load_panel(data['bonds'][1], macro, curve_params)
    del data

    asof = pd.Period(forecast_month, freq='M')
    print(f'Bond data range: {months.min()} through {months.max()}')
    print(f'Forecast horizon: H=1 month ({asof} -> {asof + 1})')
    print(f'Predictors ({len(features)}): {", ".join(features)}')

    # Tune the regularization parameters using only historically known targets.
    known = training_rows(panel, asof, features)
    if known.empty:
        raise ValueError(f'No historical training observations available by {asof}')
    original_training_count = len(training_rows(panel, asof, features, max_yield_change_bp=None))
    filter_label = (f'{MAX_TRAIN_YIELD_CHANGE_BP:g} bp' if MAX_TRAIN_YIELD_CHANGE_BP is not None
                    else 'disabled')
    print(f'Training yield-change filter: {filter_label}; '
          f'excluded {original_training_count - len(known):,} of '
          f'{original_training_count:,} historical bond-month observations.')
    mixing, penalty, score, folds, cv_scores = tune_elastic_net(
        known, features, CV_MIXING_GRID, CV_PENALTY_GRID
    )
    print(f'Chronological CV: {len(CV_MIXING_GRID)} alpha values × '
          f'{len(CV_PENALTY_GRID)} lambda values = '
          f'{len(CV_MIXING_GRID) * len(CV_PENALTY_GRID)} settings')
    print('Five best elastic-net settings:')
    print(cv_scores.head(5).to_string(index=False))

    # Refit on all available training observations and forecast next-month yields.
    now, coefficients, training_count = fit_forecast(
        panel, asof, features, mixing, penalty
    )
    print(f'\nELASTIC NET: {asof} -> {asof + 1}')
    print(f'\nTraining bond-month rows: {training_count:,}')
    print(f'\nSelected alpha (mixing): {mixing:.2f}; lambda (penalty): {penalty:g}')
    print(f'\nValidation RMSE: {score:.3f} bp; years: {folds}')
    print('\nElastic-net coefficients (original predictor units):')
    print(coefficients[['feature', 'coefficient_original_units']].to_string(index=False))
    show_selected_bonds(now, coefficients, asof)

    # Diagnostic evaluation; realized future outcomes are never model inputs.
    evaluated = now.loc[now.dy.notna()]
    if not evaluated.empty:
        flat = int(evaluated.dy.abs().le(1e-10).sum())
        residual_bp = (evaluated.dy - evaluated.predicted_dy_pp) * 100
        print(f'ALL-BOND EVALUATION (unscreened): '
              f'{len(evaluated):,} realized yield changes; '
              f'{flat:,} unchanged ({flat/len(evaluated):.1%}).')
        print(f'  Full-month RMSE: {np.sqrt(np.mean(residual_bp ** 2)):.2f} bp; '
              f'MAE: {np.mean(np.abs(residual_bp)):.2f} bp.')
        print('  NOTE: Evaluation data have not been screened for stale prices.')
        if MAX_TRAIN_YIELD_CHANGE_BP is not None:
            # Additional diagnostic only: primary evaluation above remains unfiltered.
            ordinary = evaluated.loc[evaluated.dy.abs() * 100 <= MAX_TRAIN_YIELD_CHANGE_BP]
            if not ordinary.empty:
                ordinary_errors = (ordinary.dy - ordinary.predicted_dy_pp) * 100
                print(f'  Diagnostic (realized changes <= {MAX_TRAIN_YIELD_CHANGE_BP:g} bp): '
                      f'{len(ordinary):,} bonds, '
                      f'RMSE {np.sqrt(np.mean(ordinary_errors ** 2)):.2f} bp, '
                      f'MAE {np.mean(np.abs(ordinary_errors)):.2f} bp.')

    # Optional historical backtest; disabled for the normal four-script workflow.
    history_errors = pd.DataFrame()
    if run_backtest:
        backtest_start = first_backtest_target(panel, features, asof)
        history_errors = rolling_backtest(
            panel, features, asof, backtest_start,
            CV_MIXING_GRID, CV_PENALTY_GRID
        )
        if history_errors.empty:
            print('No historical one-month backtest observations available.')
        else:
            print('\nONE-MONTH EXPANDING-WINDOW BACKTEST')
            print(rolling_metrics(history_errors).to_string(index=False))
            earliest = history_errors.sort_values(
                ['target_month', 'cusip']
            ).head(3).copy()
            earliest['predicted_change_bp'] = earliest.predicted_dy_pp * 100
            earliest['actual_change_bp'] = earliest.dy * 100
            print('First three out-of-sample bond yield-change forecasts:')
            print(earliest[['cusip', 'month', 'target_month',
                            'predicted_change_bp', 'actual_change_bp',
                            'forecast_error_bp']].to_string(index=False))

    # Save the same shared H=1 bundle expected by prepare_stage1_inputs.py.
    import gzip
    import pickle
    history_cols = ['cusip', 'month', 'RET_EOM', 'DURATION',
                    'time_to_maturity', 'RATING_NUM']
    history = panel.loc[
        panel.month.between(asof - 35, asof), history_cols
    ].copy()
    errors = (history_errors.dropna(
        subset=['return_error', 'predicted_return', 'next_return']
    ) if not history_errors.empty else pd.DataFrame())
    bundle = {
        'asof': str(asof),
        'target_month': str(asof + 1),
        'snapshot': now,
        'history': history,
        'errors': errors,
        'curve_params': curve_params,
        'features': features,
        'mixing_alpha': mixing,
        'penalty_lambda': penalty,
    }
    DEFAULT_FORECAST_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = DEFAULT_FORECAST_BUNDLE.with_suffix('.pkl.tmp')
    with gzip.open(temporary_file, 'wb', compresslevel=6) as output:
        pickle.dump(bundle, output, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_file.replace(DEFAULT_FORECAST_BUNDLE)
    print(f'\nStage 1 forecasts saved to {DEFAULT_FORECAST_BUNDLE.resolve()}')
    print('Next: run prepare_stage1_inputs.py (no forecast re-fitting).')


if __name__ == '__main__':
    main()
