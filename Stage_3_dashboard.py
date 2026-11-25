"""GoalPath-Bond | Stage 3 investor implementation.

Run: python -m streamlit run Stage_3_dashboard.py
Historical-market demonstration: fresh monthly optimization, not archived
portfolio outcomes or future-market scenario generation.
"""
import math
from threading import RLock

import numpy as np
import pandas as pd
import streamlit as st

MODELS = {
    "S1": ("random_walk", 1, (("turnover_multiplier", 10),)),
    "S2": ("direct", 1, (("turnover_multiplier", 5), ("krd_tolerance", 0.5))),
    "S3": ("constant", 3, (("turnover_multiplier", 10), ("krd_tolerance", 0.5))),
    "S4": ("constant", 3, (("uncertainty_scale", 0.5), ("turnover_multiplier", 5))),
    "S5": ("random_walk", 3, (("turnover_multiplier", 10), ("krd_tolerance", 0.5))),
}
MODEL_NAMES = {
    "S1": "Single-period | Random walk",
    "S2": "Single-period | Direct elastic net",
    "S3": "Multi-period | Balanced constant forecast",
    "S4": "Multi-period | Return-oriented constant forecast",
    "S5": "Multi-period | Random walk fallback",
}
K_VALUES = (10, 25, 50)
GAMMA_VALUES = (0.1, 1, 10, 100)
SELECTION = {
    0.1: {10: "S4", 25: "S4", 50: "S4"},
    1: {10: "S3", 25: "S4", 50: "S4"},
    10: {10: "S4", 25: "S4", 50: "S4"},
    100: {10: "S2", 25: "S3", 50: "S1"},
}

# January 2016 onward is the training history. Allow two years of warm-up.
FIRST_DECISION = pd.Period("2017-12", freq="M")
RISK_LABELS = {"Low": 0.1, "Medium": 1.0, "High": 10.0, "Very high": 100.0}
ERROR_COLS = ["return_error", "constant_return_error", "random_walk_return_error"]
FORECAST_COLS = ["predicted_dy_pp", "predicted_return",
                 "constant_predicted_return", "random_walk_predicted_return"]


class OptimizationFailure(RuntimeError):
    """The calibrated solver failed to produce a valid portfolio."""


@st.cache_resource(show_spinner=False)
def solver_lock():
    # Stage 2 wrappers temporarily change module-level parameters.
    return RLock()


def reference_k(k):
    # Bucket only for model selection; the solver receives the actual hard cap.
    return 50 if k == 50 else 25 if k >= 25 else 10


def settings_for(sid, gamma):
    return {**{"duration_tolerance": 0.5, "convexity_tolerance": 25.0,
               "krd_tolerance": 2.0, "turnover_multiplier": 1.0,
               "uncertainty_scale": 1.0, "terminal_te_cap": 0.025},
            **dict(MODELS[sid][2]), "gamma": float(gamma)}


@st.cache_resource(show_spinner=False)
def load_engine_data():
    """Load point-in-time forecasts, not the Stage 2 backtest's output portfolios."""
    import Stage_2_backtest as bt

    cols = ["cusip", "month", "target_month", "horizon", "dy",
            *FORECAST_COLS, *ERROR_COLS]
    chunks = []
    for part in pd.read_csv(bt.FORECAST_FILE, usecols=cols,
                            dtype={"cusip": str}, chunksize=150_000):
        chunks.append(part.loc[part.horizon.between(1, 3)].copy())
    if not chunks:
        raise ValueError("Missing Stage 2 forecast history.")
    raw = pd.concat(chunks, ignore_index=True)
    del chunks
    if raw.duplicated(["cusip", "month", "horizon"]).any():
        raise ValueError("Duplicate bond/origin/horizon forecasts.")
    forecasts = raw[["cusip", "month", "horizon", "dy", *FORECAST_COLS]].copy()
    errors = raw[["cusip", "horizon", "target_month", *ERROR_COLS]].copy()
    del raw
    errors[ERROR_COLS] = errors[ERROR_COLS].replace([np.inf, -np.inf], np.nan)

    data = bt.process_all_data(bond_columns=bt.BOND_COLUMNS)
    macro, curve = bt.macro_features(bt.history_for_features(data, "curve"),
                                     bt.history_for_features(data, "fred"))
    panel = bt.load_panel(data["bonds"][1], macro, curve)
    del data
    return panel, curve, forecasts, errors


def investment_window(data):
    """Common contiguous decision window with H=1–3 inputs and settled returns.

    Start no earlier than Dec 2017. Require some mature forecast errors for
    each model/horizon (six pooled observations, not six months per bond).
    The Stage 2 builder still checks actual candidate-level uncertainty.
    """
    panel, curve, forecasts, errors = data
    valid = forecasts.loc[np.isfinite(forecasts[FORECAST_COLS]).all(axis=1)]
    complete = valid.groupby(["month", "cusip"]).horizon.nunique().eq(3)
    forecast_months = set(pd.PeriodIndex(
        complete[complete].index.get_level_values("month").unique(), freq="M"))
    snapshots = set(panel.month.unique())
    realized = set(panel.loc[np.isfinite(panel.RET_EOM), "month"].unique())
    curve_cols = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    curve_months = set(curve.index[np.isfinite(curve[curve_cols]).all(axis=1)])
    counts = errors.groupby(["target_month", "horizon"])[ERROR_COLS].count()
    counts = counts.unstack("horizon", fill_value=0).sort_index().cumsum()
    expected = pd.MultiIndex.from_product([ERROR_COLS, [1, 2, 3]])
    counts = counts.reindex(columns=expected, fill_value=0)
    ready = counts.index[counts.ge(6).all(axis=1)]
    if len(ready) == 0:
        raise ValueError("Forecast history lacks mature errors for H=1–3.")
    earliest = max(FIRST_DECISION, pd.Period(ready[0], freq="M"))
    usable = sorted(m for m in forecast_months if m >= earliest and
                    m in snapshots and m in curve_months and m + 1 in realized)
    if not usable:
        raise ValueError("No usable investment month after forecast warm-up.")
    # Never skip a missing month or silently shorten a requested investment.
    start = end = usable[0]
    for month in usable[1:]:
        if month != end + 1:
            break
        end = month
    return start, end


def prepare_and_solve(panel, curve, forecasts, errors, asof, sid, k, gamma, capital, state, remaining):
    """Call the real Stage 2 input builder, solver and solution validator."""
    import Stage_2_backtest as bt
    model, horizon, changes = MODELS[sid]
    horizon = min(horizon, remaining)
    month = pd.Period(asof, freq="M")
    snapshot = panel.loc[panel.month.eq(month)].copy()
    if snapshot.empty:
        raise ValueError(f"No bond snapshot at {asof}.")
    history = panel.loc[panel.month.between(month - 35, month),
                        ["cusip", "month", "RET_EOM", "DURATION", "time_to_maturity", "RATING_NUM"]].copy()
    latest = forecasts.loc[forecasts.month.eq(asof)]
    bundle = {"asof": asof, "snapshot": snapshot, "benchmark_snapshot": snapshot.copy(),
              "history": history, "curve_params": curve}
    stage = bt.prepare_strategy_inputs(bundle, latest, errors, state, capital, model, horizon)
    # The existing wrapper restores module settings even when Gurobi raises.
    settings = bt.strategy_parameters(gamma, changes)
    if settings != settings_for(sid, gamma):
        raise ValueError("Stage 2 defaults no longer match Stage 3 calibration.")
    if (bt.CANDIDATE_COUNT != 150 or bt.TRANSACTION_COST != 0.001
            or bt.MAX_CASH_WEIGHT != 0.01 or bt.MAX_WEIGHT != 1.0):
        raise ValueError("Stage 2 fixed settings no longer match calibration.")
    try:
        solution, metrics = bt.run_configured_solver(stage, k, settings)
    except (RuntimeError, ValueError, bt.optimizer.gp.GurobiError) as exc:
        raise OptimizationFailure(f"{sid} at {asof}: {exc}") from exc
    if not np.isclose(metrics["gamma"], gamma) or metrics["K"] != k:
        raise ValueError("Solver did not use the requested investor settings.")
    return stage, solution, metrics



def run_path(data, sid, k, gamma, capital, start, months, progress):
    """Invest from cash, optimize monthly, execute only the first trade."""
    import Stage_2_backtest as bt

    state, value, benchmark_value = None, capital, capital
    rows, positions = [], []
    for i, month in enumerate(pd.period_range(start, periods=months, freq="M")):
        progress.progress(i / months, text=f"{sid} | Optimizing {month + 1} ({i + 1}/{months})")
        stage, solution, metrics = prepare_and_solve(
            *data, str(month), sid, k, gamma, value, state, months - i)
        held, bond_return, benchmark_return, coverage, _ = bt.evaluate_month(stage, solution)
        fee = float(metrics["first_trade_cost_dollars"])
        state, _, net, _ = bt.settle_with_cash(held, solution, stage, bond_return, fee / value)
        weights = held.weight.to_numpy(float)
        if not 0 < len(weights) <= k or not np.isfinite(weights).all():
            raise OptimizationFailure(f"Invalid holdings at {month}.")
        if not np.isfinite([net, benchmark_return, fee]).all() or min(net, benchmark_return) <= -1:
            raise ValueError(f"Invalid realized returns at {month + 1}.")
        before = value
        value *= 1 + net
        benchmark_value *= 1 + benchmark_return
        if not np.isfinite([value, benchmark_value]).all():
            raise ValueError("Portfolio wealth is outside the supported numerical range.")
        rows.append({"month": str(month + 1), "net_return": net,
                     "benchmark_return": benchmark_return, "portfolio_value": value,
                     "benchmark_value": benchmark_value, "fees": fee,
                     "holdings": len(weights), "effective_holdings": weights.sum()**2 / (weights @ weights),
                     "optimal": bool(metrics["optimal"])})
        positions.append(held[["cusip", "weight", "next_return"]].assign(
            month=str(month + 1), allocation=weights * before))
    progress.progress(1.0, text=f"Completed all {months} investment months")
    return pd.DataFrame(rows), pd.concat(positions, ignore_index=True)


def evaluate_goal(k, gamma, capital, months, goal):
    target = capital * math.exp(math.log1p(goal) * months / 12)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Target wealth is outside the supported numerical range.")
    primary = SELECTION[gamma][reference_k(k)]
    progress = st.progress(0.0, text="Loading bond history and forecasts")
    try:
        with solver_lock():
            data = load_engine_data()
            start, end = investment_window(data)
            available = end.ordinal - start.ordinal + 1
            if months > available:
                raise ValueError(f"Only {available} consecutive investment months are available "
                                 f"({start + 1} through {end + 1}). Shorten the horizon.")
            sid, fallback = primary, None
            try:
                monthly, holdings = run_path(data, sid, k, gamma, capital, start, months, progress)
            except OptimizationFailure as exc:
                # Preserve the calibrated whole-path fallback; no mixed strategies.
                sid, fallback = "S5", str(exc)
                progress.progress(0.0, text="Primary solver failed; restarting from cash with S5")
                monthly, holdings = run_path(data, sid, k, gamma, capital, start, months, progress)
        return dict(monthly=monthly, holdings=holdings, sid=sid, primary=primary,
                    fallback=fallback, capital=capital, target=target, goal=goal,
                    k=k, gamma=gamma, months=months, start=str(start + 1))
    finally:
        progress.empty()


def money(x):
    return f"${x:,.2f}"


def show_results(result):
    m = result["monthly"]
    capital, target = result["capital"], result["target"]
    ending, benchmark = float(m.portfolio_value.iloc[-1]), float(m.benchmark_value.iloc[-1])
    r, b = ending / capital - 1, benchmark / capital - 1
    annualized = math.expm1(math.log(ending / capital) * 12 / result["months"])
    active = m.net_return - m.benchmark_return
    te = float(active.std(ddof=1) * np.sqrt(12))
    utility = 12 * active.mean() - result["gamma"] * te**2
    sid = result["sid"]
    st.caption("Automatically selected configuration")
    st.subheader(f"{sid} — {MODEL_NAMES[sid]}")
    st.caption(f'{result["start"]} – {m.month.iloc[-1]} | {result["months"]} months | '
               f'K ≤ {result["k"]} | γ = {result["gamma"]:g} | '
               f'Calibration reference K = {reference_k(result["k"])}')
    if result["fallback"]:
        st.warning(f'Primary {result["primary"]} failed; S5 completed the entire path. '
                   f'{result["fallback"]}')
    if (reference_k(result["k"]), result["gamma"]) in {(10, 1), (50, 1)}:
        st.caption("Provisional calibration: one competing reference strategy did not complete.")
    if not m.optimal.all():
        st.caption(f"{int((~m.optimal).sum())} monthly solutions were feasible but not certified optimal.")

    a, c = st.columns(2)
    a.metric("Net cumulative return", f"{r:.2%}", f"{100 * (r-b):+.2f} pp vs benchmark")
    c.metric("Annualized net tracking error", f"{te:.2%}" if np.isfinite(te) else "N/A")
    a, c = st.columns(2)
    a.metric("Ending portfolio value", money(ending))
    c.metric("Annualized net utility", f"{utility:.2%}" if np.isfinite(utility) else "N/A")

    st.subheader("Portfolio vs. benchmark")
    a, c = st.columns(2)
    a.metric("GoalPath-Bond", f"{r:.2%}", help="Net of transaction costs")
    c.metric("Benchmark proxy", f"{b:.2%}")
    initial = pd.DataFrame({"Portfolio": [capital], "Benchmark": [capital]},
                           index=[str(pd.Period(result["start"], freq="M") - 1)])
    wealth = m.set_index("month")[["portfolio_value", "benchmark_value"]]
    wealth.columns = initial.columns
    st.line_chart(pd.concat([initial, wealth]))
    a, c = st.columns(2)
    a.metric("Total transaction costs", money(m.fees.sum()))
    c.metric("Average effective holdings", f"{m.effective_holdings.mean():.2f}")

    st.subheader("Investment goal")
    a, c = st.columns(2)
    a.metric("Target terminal wealth", money(target))
    c.metric("Realized annualized return", f"{annualized:.2%}",
             f'{100 * (annualized-result["goal"]):+.2f} pp vs target')
    if ending >= target:
        st.success(f"Goal met in this historical investment: {money(ending - target)} above target.")
    else:
        st.warning(f"Goal not met: {money(target - ending)} below target. "
                   "Try a different risk aversion or bond limit, or revise the return target/horizon. "
                   "A change does not guarantee a better result.")
    with st.expander("Monthly portfolio holdings"):
        selected = st.selectbox("Investment month", m.month.tolist(), index=len(m)-1)
        held = result["holdings"].loc[result["holdings"].month.eq(selected),
                                      ["cusip", "weight", "allocation", "next_return"]].copy()
        held["weight"] = (held.weight * 100).round(3)
        held["allocation"] = held.allocation.round(2)
        held["next_return"] = (held.next_return * 100).round(3)
        st.dataframe(held.sort_values("weight", ascending=False).rename(columns={
            "cusip": "CUSIP", "weight": "Weight (%)", "allocation": "Allocation (USD)",
            "next_return": "Realized return (%)"}), hide_index=True)
    st.caption("Fresh monthly solves using point-in-time forecast inputs and historical returns. "
               "Results are a retrospective demonstration, not future returns or actual trades. "
               "Strategy selection was calibrated on later historical data; this is not an independent "
               "out-of-sample test. Bond amounts are continuous-dollar allocations, not executable lots.")


def dashboard():
    st.set_page_config(page_title="GoalPath-Bond", layout="centered")
    st.markdown("""<style>
        [data-testid="stMetric"] {border:1px solid #a0aeb133; border-radius:10px; padding:14px;}
        .gp-hero {background:#113345; color:white; padding:20px; border-radius:12px;}
        .gp-hero h1 {margin:0; font-size:2rem; color:white;}
        .gp-hero p {margin:4px 0 0; color:#bedbdc;}
        </style><div class="gp-hero"><h1>GoalPath-Bond</h1>
        <p>Investor Studio | Personalized fixed-income optimization</p></div>""",
        unsafe_allow_html=True)
    st.subheader("Configure your portfolio")
    with st.form("investor"):
        capital_text = st.text_input("Initial capital (USD)", "100000")
        k = st.slider("Maximum bond quantity", 1, 50, 25)
        risk = st.radio("Investor risk aversion", list(RISK_LABELS), index=1, horizontal=True)
        goal_pct = st.number_input("Desired annualized return (%)", value=6.0, step=0.5)
        months = st.number_input("Investment horizon (months)", min_value=1, value=36, step=1)
        st.caption("Historical data from 2016; investment begins after model warm-up (January 2018 or later).")
        submit = st.form_submit_button("Evaluate my goal", type="primary", use_container_width=True)
    if submit:
        st.session_state.pop("goal_result", None)
        try:
            capital, goal = float(capital_text.replace(",", "")), float(goal_pct) / 100
            if not math.isfinite(capital) or capital <= 0 or not math.isfinite(goal) or goal <= -1:
                raise ValueError("Enter positive finite capital and a finite annual return above -100%.")
            st.session_state["goal_result"] = evaluate_goal(
                k, RISK_LABELS[risk], capital, int(months), goal)
        except Exception as exc:
            st.error(f"Unable to complete this investment: {exc}")
            st.caption("No partial or archived result is substituted. Verify local data and Gurobi, "
                       "or try a different investor profile if the optimization failed.")
    if "goal_result" in st.session_state:
        st.divider()
        show_results(st.session_state["goal_result"])


if __name__ == "__main__":
    dashboard()
