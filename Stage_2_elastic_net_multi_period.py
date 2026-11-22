"""Stage 2.1: direct multi-horizon elastic-net forecasts for GoalPath-Bond.
Reuses the implementaiton made in Stage 1 including preparation, fitting, regularization 
and H=1 tuning. Each horizon predicts cumulative yield changes from a common decision month;
differences of these forecasts give the monthly return inputs for Boyd's MPO.
No optimizer, future benchmark data, or realized future predictors are used. """

import argparse
import numpy as np
import pandas as pd

from elastic_net_one_period import (
    PROJECT_ROOT, BOND_COLUMNS, FEATURES,
    CV_MIXING_GRID, CV_PENALTY_GRID, MIN_INITIAL_TRAIN_MONTHS,
    process_all_data, history_for_features, macro_features, load_panel,
    training_rows, limit_tuning_rows, estimator, tune_elastic_net, fit_forecast,
)

HORIZON = 3
FINAL_ASOF = '2025-10'
PORTFOLIO_START = '2023-12'
OUTPUT_DIR = PROJECT_ROOT / 'Stage_2_outputs' / 'forecasts'
HISTORY_FILE = OUTPUT_DIR / 'Stage_2_bond_forecast_history.csv.gz'
ACCURACY_FILE = OUTPUT_DIR / 'Stage_2_forecast_accuracy.csv'
KEYS = ['cusip', 'month']
BASE_COLUMNS = list(dict.fromkeys(KEYS + FEATURES + ['YIELD', 'DURATION', 'RATING_NUM', 'RET_EOM','next_yield', 'next_return', 'target_month', 'dy']))


def build_horizon_panel(panel, h):
    """Use exact calendar matches, not row shifts; dy is cumulative in pp.
    next_return is the realized ONE-month return ending at t+h,
    not the compounded return over h months. """
    if h == 1:
        return panel
    base = panel.drop(columns=['next_yield', 'next_return', 'target_month', 'dy'])
    future = panel[['cusip', 'month', 'YIELD', 'RET_EOM']].copy()
    future['month'] = future.month - h
    future = future.rename(columns={'YIELD': 'next_yield', 'RET_EOM': 'next_return'})
    result = base.merge(future, on=KEYS, how='left', validate='one_to_one', sort=False)
    result['target_month'] = result.month + h
    result['dy'] = 100 * (result.next_yield - result.YIELD)
    return result


def tune_horizon(known, h):
    """Tune elastic-net hyperparameters for each forecasting horizon using
    chronological cross-validation, excluding training targets not yet
    observable at the validation date. Retain the 500 bp training filter."""
    if h == 1:
        mixing, penalty, _, _, _ = tune_elastic_net(known, FEATURES, CV_MIXING_GRID, CV_PENALTY_GRID)
        return mixing, penalty

    sampled = limit_tuning_rows(known)
    years = sorted(sampled.target_month.dt.year.unique())
    folds = []
    for year in years[1:]:
        valid = sampled.target_month.dt.year.eq(year)
        if not valid.any():
            continue
        cutoff = sampled.loc[valid, 'month'].min()
        train = sampled.target_month.le(cutoff)
        if sampled.loc[train, 'target_month'].nunique() >= 12:
            folds.append((train, valid))
    folds = folds[-3:]

    if not folds:
        # Same early-history principle as Stage 1, with horizon-safe labels.
        for target in sorted(sampled.target_month.unique())[-3:]:
            valid = sampled.target_month.eq(target)
            cutoff = sampled.loc[valid, 'month'].min()
            train = sampled.target_month.le(cutoff)
            if sampled.loc[train, 'target_month'].nunique() >= 8:
                folds.append((train, valid))
    if not folds:
        raise ValueError(f'No leakage-safe validation folds for h={h}')

    best = None
    for mixing in CV_MIXING_GRID:
        for penalty in CV_PENALTY_GRID:
            scores = []
            for train, valid in folds:
                model = estimator(penalty, mixing).fit(
                    sampled.loc[train, FEATURES], sampled.loc[train, 'dy'])
                pred = model.predict(sampled.loc[valid, FEATURES])
                error = sampled.loc[valid, 'dy'].to_numpy() - pred
                scores.append(float(np.sqrt(np.mean(error**2)) * 100))
            candidate = (float(np.mean(scores)), float(mixing), float(penalty))
            if best is None or candidate < best:
                best = candidate
    return best[1], best[2]


def rolling_horizon(panel, h, final_asof, start_origin=None):
    """Expanding-window forecasts; retune at the first origin of each year.

    Forecasts are retained even when realized targets are missing. The
    last origin may be forecast without any subsequent observations.
    """
    known = training_rows(panel, final_asof, FEATURES)
    months = sorted(known.target_month.unique())
    # Additional history for purged early validation folds at longer horizons.
    minimum = MIN_INITIAL_TRAIN_MONTHS + h - 1
    if len(months) < minimum:
        raise ValueError(f'Insufficient observed targets for h={h}')
    first = months[minimum - 1]
    if start_origin is not None:
        first = max(first, start_origin)
    if first > final_asof:
        raise ValueError(f'No eligible forecast origins for h={h}')

    results = []
    params, tuned_year = None, None
    forecast_cols = KEYS + [
        'target_month', 'YIELD', 'DURATION', 'RATING_NUM', 'time_to_maturity',
        'predicted_dy_pp', 'dy', 'next_return'
    ]
    for asof in pd.period_range(first, final_asof, freq='M'):
        if not panel.loc[panel.month.eq(asof), FEATURES].notna().all(axis=1).any():
            continue
        if params is None or asof.year != tuned_year:
            known = training_rows(panel, asof, FEATURES)
            if known.target_month.nunique() < minimum:
                continue
            params = tune_horizon(known, h)
            tuned_year = asof.year
            print(f'h={h} | {asof} | alpha={params[0]:g}, lambda={params[1]:g}', flush=True)
        # Stage 1 fit enforces target_month <= asof and fits its own scaler.
        now, _, _ = fit_forecast(panel, asof, FEATURES, *params)
        result = now[forecast_cols].copy()
        result['horizon'] = h
        result['alpha_mixing'], result['lambda_penalty'] = params
        results.append(result)
    if not results:
        raise ValueError(f'No forecasts generated for h={h}')
    return pd.concat(results, ignore_index=True)


def map_monthly_returns(forecasts):
    """Align horizons by CUSIP and origin, not by future bond observations.

    Yield changes are in percentage points. YIELD and returns are decimals.
    Constant duration is the preliminary Stage 2 approximation.
    """
    output = []
    first = forecasts[1][KEYS + ['predicted_dy_pp']].rename(
        columns={'predicted_dy_pp': 'first_dy_pp'})
    for h, frame in sorted(forecasts.items()):
        result = frame.merge(first, on=KEYS, how='left', validate='one_to_one', sort=False)
        if h == 1:
            result['previous_dy_pp'] = 0.0
        else:
            previous = forecasts[h - 1][KEYS + ['predicted_dy_pp']].rename(
                columns={'predicted_dy_pp': 'previous_dy_pp'})
            result = result.merge(previous, on=KEYS, how='left',
                                  validate='one_to_one', sort=False)
        result['predicted_period_dy_pp'] = result.predicted_dy_pp - result.previous_dy_pp
        result['predicted_start_yield'] = result.YIELD + result.previous_dy_pp / 100
        result['predicted_return'] = (
            result.predicted_start_yield / 12
            - result.DURATION * result.predicted_period_dy_pp / 100
        ).where(result.DURATION.gt(0))

        # Boyd section 5.4: constant monthly expected-return baseline.
        result['constant_predicted_dy_pp'] = h * result.first_dy_pp
        result['constant_predicted_return'] = (
            result.YIELD / 12 - result.DURATION * result.first_dy_pp / 100
        ).where(result.DURATION.gt(0))
        # Random walk in yield implies zero change and yield-based carry.
        result['random_walk_predicted_dy_pp'] = 0.0
        result['random_walk_predicted_return'] = (result.YIELD / 12).where(
            result.DURATION.gt(0))

        result['forecast_error_bp'] = (result.dy - result.predicted_dy_pp) * 100
        for name, col in [('return_error', 'predicted_return'),
                          ('constant_return_error', 'constant_predicted_return'),
                          ('random_walk_return_error', 'random_walk_predicted_return')]:
            result[name] = result.next_return - result[col]
        output.append(result.drop(columns=['previous_dy_pp', 'first_dy_pp']))
    history = pd.concat(output, ignore_index=True)
    if history.duplicated(KEYS + ['horizon']).any():
        raise ValueError('Duplicate CUSIP/origin/horizon forecasts')
    if not history.target_month.eq(history.month + history.horizon).all():
        raise ValueError('Target dates do not match forecast origins and horizons')
    return history.sort_values(['month', 'cusip', 'horizon'], ignore_index=True)


def forecast_accuracy(history):
    """Compare forecasts on common realized observations, without tail imputation."""
    rows = []
    windows = {'all_history': history,
               'portfolio_backtest': history.loc[
                   history.month.ge(pd.Period(PORTFOLIO_START, freq='M'))]}
    for window, sample in windows.items():
        for h, group in sample.groupby('horizon'):
            for unit, actual, predictions, scale in [
                ('yield_bp', 'dy', ['predicted_dy_pp', 'constant_predicted_dy_pp',
                                    'random_walk_predicted_dy_pp'], 100),
                ('monthly_return', 'next_return', [
                    'predicted_return', 'constant_predicted_return',
                    'random_walk_predicted_return'], 1),
            ]:
                mask = np.isfinite(group[[actual] + predictions]).all(axis=1)
                common = group.loc[mask]
                for model, pred in zip(['direct', 'constant', 'random_walk'], predictions):
                    errors = (common[actual] - common[pred]) * scale
                    rows.append({'window': window, 'horizon': h, 'model': model,
                                 'unit': unit, 'observations': len(errors),
                                 'rmse': float(np.sqrt(np.mean(errors**2))),
                                 'mae': float(np.mean(np.abs(errors)))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon', type=int, default=HORIZON)
    parser.add_argument('--asof', default=FINAL_ASOF, help='Last forecast origin (YYYY-MM)')
    parser.add_argument('--start-origin', default=None, help='Optional first forecast origin')
    parser.add_argument('--no-save', action='store_true', help='Diagnostic without overwriting outputs')
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error('--horizon must be positive')
    asof = pd.Period(args.asof, freq='M')
    start = pd.Period(args.start_origin, freq='M') if args.start_origin else None
    if start is not None and start > asof:
        parser.error('--start-origin must not follow --asof')

    # Same data and predictors as Stage 1; prepare once for all horizons.
    data = process_all_data(bond_columns=BOND_COLUMNS)
    macro, curve = macro_features(history_for_features(data, 'curve'),
                                  history_for_features(data, 'fred'))
    months = data['bonds'][1].DATE.dt.to_period('M')
    model_months = pd.period_range(months.min(), months.max(), freq='M')
    panel = load_panel(data['bonds'][1], macro.reindex(model_months),
                       curve.reindex(model_months))[BASE_COLUMNS].copy()
    del data
    if not panel.month.eq(asof).any():
        raise ValueError(f'No bond observations for final origin {asof}')

    print(f'STAGE 2.1 | H={args.horizon} | last origin={asof}', flush=True)
    forecasts = {}
    for h in range(1, args.horizon + 1):
        horizon_panel = build_horizon_panel(panel, h)
        forecasts[h] = rolling_horizon(horizon_panel, h, asof, start)
        del horizon_panel

    history = map_monthly_returns(forecasts)
    del forecasts
    if not history.loc[history.month.eq(asof), 'horizon'].nunique() == args.horizon:
        raise ValueError('One or more horizons missing for the final decision month')
    scores = forecast_accuracy(history)
    print('\nMatched forecast accuracy (yield in bp; returns in decimal units):')
    print(scores.to_string(index=False))
    preview = history.loc[history.month.eq(asof)]
    complete = preview.groupby('cusip').horizon.transform('nunique').eq(args.horizon)
    preview = preview.loc[complete]
    selected = preview.cusip.drop_duplicates().head(3)
    print('\nLatest forecast paths (first three bonds with complete horizons):')
    print(preview.loc[preview.cusip.isin(selected), [
        'cusip', 'month', 'target_month', 'horizon', 'predicted_dy_pp',
        'predicted_period_dy_pp', 'predicted_return', 'next_return'
    ]].to_string(index=False))
    print(f'\nSaved-history candidate rows: {len(history):,}')
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        history.to_csv(HISTORY_FILE, index=False, compression='gzip')
        scores.to_csv(ACCURACY_FILE, index=False)
        print(f'Forecast history: {HISTORY_FILE}')
        print(f'Forecast evaluation: {ACCURACY_FILE}')
        print('For Stage 2 uncertainty, use only rows with target_month <= decision month.')


if __name__ == '__main__':
    main()
