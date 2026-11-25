"""GoalPath-Bond | Stage 3 investor dashboard.

Run from the repository root: python -m streamlit run Stage_3_dashboard.py
Requires the existing Stage 2 data/forecast files and a working Gurobi license
for fresh optimization. Archived replay needs only Part D CSVs.
"""
from threading import RLock

import numpy as np
import pandas as pd
import streamlit as st

# One authoritative selection grid. S5 is a documented whole-path fallback.
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
PROVISIONAL = {(10, 1), (50, 1)}
REFERENCE_CAPITAL = 100_000.0
BENCHMARK = "US aggregate bond universe | Outstanding amount × price proxy"
FIRST_DECISION, LAST_DECISION = "2023-12", "2025-09"
DECISIONS = [str(m) for m in pd.period_range(FIRST_DECISION, LAST_DECISION, freq="M")]

# Lazy imports keep archived replay usable without a Gurobi installation.
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "Stage_2_outputs" / "backtest_final_calibration" / "D_investor_adaptive_selection"


class OptimizationFailure(RuntimeError):
    """Only solver/solution-validation failures trigger a strategy fallback."""


@st.cache_resource(show_spinner=False)
def solver_lock():
    # Stage 2 temporarily changes module-level settings. Serialize fresh runs
    # across Streamlit sessions so one investor's gamma cannot leak into another.
    return RLock()


def model_id(sid):
    model, h, changes = MODELS[sid]
    suffix = "_".join(f"{key}_{value:g}" for key, value in changes)
    return f"{model}_H{h}_{suffix}"


def strategy_id(sid, k, gamma):
    model, h, changes = MODELS[sid]
    suffix = "_".join(f"{key}_{value:g}" for key, value in changes)
    return f"{model}_heuristic_H{h}_K{k}_N150_{suffix}_gamma_{gamma:g}"


def selected_model(k, gamma):
    if k not in K_VALUES or gamma not in GAMMA_VALUES:
        raise ValueError("Choose a calibrated K and gamma combination.")
    return SELECTION[gamma][k]


def settings_for(sid, gamma):
    # These are the exact Part A defaults + each shortlisted variation.
    return {**{
        "duration_tolerance": 0.5, "convexity_tolerance": 25.0,
        "krd_tolerance": 2.0, "turnover_multiplier": 1.0,
        "uncertainty_scale": 1.0, "terminal_te_cap": 0.025,
    }, **dict(MODELS[sid][2]), "gamma": float(gamma)}


def validate_capital(capital):
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Initial capital must be finite and positive.")


@st.cache_data(show_spinner=False)
def read_archive():
    files = {
        "summary": "Stage_2_summary.csv",
        "monthly": "Stage_2_monthly_results.csv",
        "holdings": "Stage_2_holdings.csv",
        "plans": "Stage_2_planned_monthly.csv",
        "rankings": "Stage_2_profile_rankings.csv",
    }
    return {key: pd.read_csv(ARCHIVE / file, dtype={"cusip": str} if key == "holdings" else None)
            for key, file in files.items()}


def archived_replay(k, gamma, capital):
    """Replay the complete calibrated path; scale dollars, never rescale returns."""
    validate_capital(capital)
    archive = read_archive()
    sid = selected_model(k, gamma)
    name = strategy_id(sid, k, gamma)
    summary = archive["summary"].loc[archive["summary"].strategy.eq(name)]
    if len(summary) != 1:
        raise ValueError(f"Missing complete archived path: {name}")
    expected = {**settings_for(sid, gamma), "K": k, "H": MODELS[sid][1],
                "initial_capital": REFERENCE_CAPITAL, "candidate_count": 150,
                "transaction_cost_rate": 0.001, "max_cash_weight": 0.01}
    if any(not np.isclose(float(summary.iloc[0][key]), value) for key, value in expected.items()):
        raise ValueError("Archived settings differ from the calibrated model.")
    monthly = archive["monthly"].loc[archive["monthly"].strategy.eq(name)].copy()
    holdings = archive["holdings"].loc[archive["holdings"].strategy.eq(name)].copy()
    plans = archive["plans"].loc[archive["plans"].strategy.eq(name)].copy()
    monthly = monthly.sort_values("decision_month").reset_index(drop=True)
    if (monthly.decision_month.tolist() != DECISIONS
            or len(plans) != len(DECISIONS) * MODELS[sid][1]
            or set(zip(plans.decision_month, plans.horizon_step)) !=
               {(m, h) for m in DECISIONS for h in range(1, MODELS[sid][1] + 1)}
            or not holdings.groupby("decision_month").size().reindex(DECISIONS)
                           .eq(monthly.holdings_count.to_numpy()).all()):
        raise ValueError("Archived monthly, holding or planned records are incomplete.")
    scale = capital / REFERENCE_CAPITAL
    for col in ("transaction_cost_dollars", "redemption_cash_dollars", "gross_capital", "net_capital"):
        if col in monthly:
            monthly[col] *= scale
    holdings["capital_allocated"] *= scale
    return {"kind": "history", "source": "Archived Part D replay", "sid": sid,
            "primary": sid, "k": k, "gamma": gamma, "capital": capital,
            "monthly": monthly, "holdings": holdings, "plans": plans,
            "fallback_reason": None}


@st.cache_resource(show_spinner=False)
def load_engine_data():
    """Read the existing Stage 2 data once; no forecast retraining or future prices."""
    import Stage_2_backtest as bt

    if (bt.START_MONTH, bt.END_MONTH) != (FIRST_DECISION, LAST_DECISION):
        raise ValueError("Stage 2 date window changed; refresh the calibrated Stage 3 grid.")
    forecasts, errors = bt.load_forecast_history(3)
    data = bt.process_all_data(bond_columns=bt.BOND_COLUMNS)
    macro, curve = bt.macro_features(bt.history_for_features(data, "curve"),
                                     bt.history_for_features(data, "fred"))
    panel = bt.load_panel(data["bonds"][1], macro, curve)
    return panel, curve, forecasts, errors


def prepare_and_solve(panel, curve, forecasts, errors, asof, sid, k, gamma, capital, state):
    """Call the real Stage 2 input builder, solver and solution validator."""
    import Stage_2_backtest as bt
    model, horizon, changes = MODELS[sid]
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


def plan_rows(asof, metrics):
    """Flatten the validated H-period plan using the same Stage 2 field names."""
    rows = []
    for h, p in enumerate(metrics["monthly_plan"], start=1):
        rows.append({
            "decision_month": asof, "horizon_step": h, "planned_month": p["month"],
            "executed": h == 1, "planned_holdings": p["holdings"],
            "planned_cash_weight": p["cash_weight"],
            "planned_tracking_error": p["expected_tracking_error"],
            "planned_expected_return": p["expected_return"],
            "benchmark_expected_return": p["benchmark_expected_return"],
            "planned_duration_gap": p["duration_gap"], "planned_convexity_gap": p["convexity_gap"],
            **{f"planned_krd_{t}y_gap": gap for t, gap in zip((2, 5, 10, 30), p["krd_gaps"])},
            "planned_max_abs_krd_gap": max(abs(x) for x in p["krd_gaps"]),
            "cashflow_risk_coverage": p["cashflow_risk_coverage"],
            "transaction_cost_fraction": p["transaction_cost_fraction"],
        })
    return rows


def fresh_history(data, sid, k, gamma, capital, start, end, progress):
    """Independent receding-horizon path starting from cash. Execute h=1 only."""
    import Stage_2_backtest as bt
    panel, curve, forecasts, errors = data
    state, value, gross_value = None, capital, capital
    monthly, positions, plans = [], [], []
    months = pd.period_range(start, end, freq="M")
    for i, month in enumerate(months):
        asof = str(month)
        progress.progress(i / len(months), text=f"{sid} | Optimizing {asof} ({i+1}/{len(months)})")
        stage, solution, metrics = prepare_and_solve(
            panel, curve, forecasts, errors, asof, sid, k, gamma, value, state)
        held, bond_return, benchmark_return, coverage, benchmark = bt.evaluate_month(stage, solution)
        fee = float(metrics["first_trade_cost_dollars"])
        state, gross, net, redeemed_cash = bt.settle_with_cash(
            held, solution, stage, bond_return, fee / value)
        before = value
        value *= 1 + net
        gross_value *= 1 + gross
        p = metrics["monthly_plan"][0]
        weights = held.weight.to_numpy(float)
        turnover = (solution["buys"].sum() + solution["sells"].sum()
                    + stage["outside_pretrade_weight"]) / 2
        monthly.append({
            "decision_month": asof, "target_month": str(month + 1),
            "portfolio_return": gross, "net_portfolio_return": net, "benchmark_return": benchmark_return,
            "active_return": gross - benchmark_return, "net_active_return": net - benchmark_return,
            "expected_monthly_tracking_error": p["expected_tracking_error"],
            "one_way_turnover": turnover, "transaction_cost_dollars": fee,
            "holdings_count": len(held), "effective_holdings": weights.sum()**2 / np.sum(weights**2),
            "max_individual_bond_weight": weights.max(),
            "cash_weight_after_trade": solution["cash_weight"],
            "cash_weight_after_settlement": state["cash_weight"],
            "redemption_cash_dollars": redeemed_cash * before,
            "benchmark_return_coverage": coverage, "gross_capital": gross_value, "net_capital": value,
            "terminal_tracking_error": metrics["terminal_tracking_error"],
            "solver_status": metrics["solver_status"], "optimal": metrics["optimal"],
        })
        held = held.copy()
        held["decision_month"], held["target_month"] = asof, str(month + 1)
        held["capital_allocated"] = held.weight * before
        held["expected_return"] = held.cusip.map(
            pd.Series(stage["periods"][0]["mu"], index=stage["candidates"].cusip))
        positions.append(held[["cusip", "decision_month", "target_month", "weight", "capital_allocated",
                               "expected_return", "duration", "next_return", "return_source"]])
        plans.extend(plan_rows(asof, metrics))
    progress.progress(1.0, text=f"{sid} | Completed {len(months)} months")
    return {"kind": "history", "source": "Fresh historical optimization", "sid": sid,
            "k": k, "gamma": gamma, "capital": capital,
            "monthly": pd.DataFrame(monthly), "holdings": pd.concat(positions, ignore_index=True),
            "plans": pd.DataFrame(plans)}


def fresh_allocation(data, sid, k, gamma, capital, asof, progress):
    """New investment from cash at a saved historical decision date; no realized returns."""
    progress.progress(0.2, text=f"{sid} | Preparing and solving {asof}")
    stage, solution, metrics = prepare_and_solve(*data, asof, sid, k, gamma, capital, None)
    w = np.asarray(solution["weights"], float)
    held = stage["candidates"].loc[w > 1e-6, ["cusip", "MATURITY", "COUPON", "YIELD", "PRICE_EOM", "duration"]].copy()
    held["weight"] = w[w > 1e-6]
    held["capital_allocated"] = held.weight * capital
    held["expected_return"] = np.asarray(stage["periods"][0]["mu"])[w > 1e-6]
    fee = float(metrics["first_trade_cost_dollars"])
    cash = float(metrics["first_trade_cash_dollars"])
    # Include tiny numerical positions in the accounting check, not the displayed table.
    if not np.isclose(w.sum() + cash / capital + fee / capital, 1, atol=1e-5):
        raise ValueError("Initial allocation, cash and fees do not reconcile.")
    progress.progress(1.0, text=f"{sid} | Allocation complete")
    return {"kind": "allocation", "source": "Fresh historical-date allocation", "sid": sid,
            "k": k, "gamma": gamma, "capital": capital, "asof": asof,
            "holdings": held, "plans": pd.DataFrame(plan_rows(asof, metrics)),
            "metrics": metrics, "cash": cash, "fee": fee,
            "invested": float(w.sum() * capital), "benchmark_type": stage["benchmark_type"]}


def run_fresh(mode, k, gamma, capital, start, end=None):
    """Retry the entire path with S5 after a primary solver failure; never splice histories."""
    validate_capital(capital)
    primary = selected_model(k, gamma)
    progress = st.progress(0.0, text="Loading existing forecasts and bond history")
    try:
        with solver_lock():
            data = load_engine_data()
            def execute(sid):
                if mode == "Fresh rolling backtest":
                    return fresh_history(data, sid, k, gamma, capital, start, end, progress)
                return fresh_allocation(data, sid, k, gamma, capital, start, progress)
            try:
                result = execute(primary)
                reason = None
            except OptimizationFailure as exc:
                reason = str(exc)
                progress.progress(0.0, text=f"Primary failed; restarting from cash with S5")
                result = execute("S5")
            result.update(primary=primary, fallback_reason=reason)
            return result
    finally:
        progress.empty()


def historical_metrics(monthly, gamma, capital):
    """Same realized utility and sample variance convention as Part D."""
    r = monthly.net_portfolio_return.to_numpy(float)
    b = monthly.benchmark_return.to_numpy(float)
    a = r - b
    if not np.isfinite(np.r_[r, b]).all():
        raise ValueError("Nonfinite historical returns.")
    te = np.std(a, ddof=1) * np.sqrt(12) if len(a) > 1 else np.nan
    active = 12 * a.mean()
    net, benchmark = np.prod(1 + r) - 1, np.prod(1 + b) - 1
    return {"net": net, "benchmark": benchmark, "te": te, "active": active,
            "rmse": np.sqrt(12 * np.mean(a**2)), "utility": active - gamma * te**2,
            "end": capital * (1 + net), "benchmark_end": capital * (1 + benchmark),
            "fees": monthly.transaction_cost_dollars.sum(), "turnover": monthly.one_way_turnover.mean(),
            "effective": monthly.effective_holdings.mean(),
            "expected_te": monthly.expected_monthly_tracking_error.mean()}


def pct(x):
    return f"{x:.2%}" if np.isfinite(x) else "N/A"


def money(x):
    return f"${x:,.2f}"


def download(frame, label, filename):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=filename)


def show_holdings(result, decision=None):
    held = result["holdings"].copy()
    if decision is not None:
        held = held.loc[held.decision_month.eq(decision)]
    held = held.sort_values("weight", ascending=False)
    st.subheader("Bond allocations")
    top = held.head(12).set_index("cusip")["weight"] * 100
    st.bar_chart(top, x_label="CUSIP", y_label="Allocation (% of pre-trade capital)")
    display = held.copy()
    display["weight (%)"] = display.weight * 100
    display["expected return (%)"] = display.expected_return * 100
    if "next_return" in display:
        display["realized bond return (%)"] = display.next_return * 100
    display = display.drop(columns=["weight", "expected_return", "next_return"], errors="ignore")
    cols = ["cusip", "weight (%)", "capital_allocated", "expected return (%)", "duration"]
    cols += [c for c in ("realized bond return (%)", "MATURITY", "COUPON", "YIELD", "PRICE_EOM",
                          "return_source") if c in display]
    st.dataframe(display[cols], hide_index=True, use_container_width=True,
                 column_config={"capital_allocated": st.column_config.NumberColumn("Allocation (USD)", format="$%.2f"),
                                "weight (%)": st.column_config.NumberColumn("Weight (%)", format="%.4f"),
                                "expected return (%)": st.column_config.NumberColumn("Expected monthly return (%)", format="%.3f")})
    st.caption("Weights are fractions of pre-trade capital, not renormalized after fees. Tiny numerical positions are omitted.")
    download(held, "Download displayed holdings", "goalpath_holdings.csv")


def show_risk(result, decision=None):
    plans = result["plans"].copy()
    if decision is not None:
        plans = plans.loc[plans.decision_month.eq(decision)]
    limits = settings_for(result["sid"], result["gamma"])
    st.subheader("Interest-rate exposure checks")
    checks = [
        ("Duration gap (years)", "planned_duration_gap", limits["duration_tolerance"]),
        ("Convexity gap (years²)", "planned_convexity_gap", limits["convexity_tolerance"]),
        ("KRD 2Y gap (years)", "planned_krd_2y_gap", limits["krd_tolerance"]),
        ("KRD 5Y gap (years)", "planned_krd_5y_gap", limits["krd_tolerance"]),
        ("KRD 10Y gap (years)", "planned_krd_10y_gap", limits["krd_tolerance"]),
        ("KRD 30Y gap (years)", "planned_krd_30y_gap", limits["krd_tolerance"]),
    ]
    audit = pd.DataFrame([{"Constraint": label, "Maximum absolute gap": plans[col].abs().max(),
                           "Limit": limit, "Within tolerance": plans[col].abs().max() <= limit + 1e-5}
                          for label, col, limit in checks])
    st.dataframe(audit, hide_index=True, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Max planned holdings", int(plans.planned_holdings.max()), f"Limit {result['k']}", delta_color="off")
    b.metric("Max planned cash", pct(plans.planned_cash_weight.max()), "Limit 1%", delta_color="off")
    c.metric("Min cash-flow risk coverage", pct(plans.cashflow_risk_coverage.min()))
    st.subheader("Planning horizon")
    cols = ["planned_month", "horizon_step", "planned_holdings", "planned_expected_return",
            "benchmark_expected_return", "planned_tracking_error", "planned_cash_weight"]
    table = plans[cols].copy()
    for col in cols[3:]:
        table[col] = table[col].map(pct)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Only the first planned allocation is implemented. Later periods are conditional plans, not realized returns.")
    download(plans, "Download risk and planning diagnostics", "goalpath_plan.csv")


def show_history(result):
    m = result["monthly"].sort_values("target_month").reset_index(drop=True)
    q = historical_metrics(m, result["gamma"], result["capital"])
    a, b, c, d = st.columns(4)
    a.metric("Net cumulative return", pct(q["net"]), f"{(q['net']-q['benchmark'])*100:+.2f} pp vs benchmark")
    b.metric("Annual net tracking error", pct(q["te"]))
    c.metric("Ending portfolio value", money(q["end"]))
    d.metric("Annualized net utility", pct(q["utility"]), "Higher is better within profile", delta_color="off")
    tabs = st.tabs(["Performance", "Portfolio", "Risk & constraints", "Model comparison"])
    with tabs[0]:
        wealth = pd.DataFrame({
            "Month": [m.decision_month.iloc[0]] + m.target_month.tolist(),
            "GoalPath-Bond (net)": [result["capital"]] + (result["capital"] * (1 + m.net_portfolio_return).cumprod()).tolist(),
            "Benchmark": [result["capital"]] + (result["capital"] * (1 + m.benchmark_return).cumprod()).tolist(),
        }).set_index("Month")
        st.subheader("Growth of invested capital")
        st.line_chart(wealth, color=["#008B83", "#8494AC"], y_label="Portfolio value (USD)")
        a, b, c = st.columns(3)
        a.metric("Benchmark return", pct(q["benchmark"]))
        b.metric("Annual mean net active return", pct(q["active"]))
        c.metric("Annual net active RMSE", pct(q["rmse"]))
        st.subheader("Monthly active return")
        st.bar_chart(m.set_index("target_month").net_active_return * 100,
                     y_label="Net active return (percentage points)", color="#008B83")
        a, b, c = st.columns(3)
        a.metric("Total transaction fees", money(q["fees"]))
        b.metric("Average monthly turnover", pct(q["turnover"]))
        c.metric("Mean predicted monthly TE", pct(q["expected_te"]))
        st.caption("Realized performance is historical, net of modeled trading fees, and is not a forecast.")
        download(m, "Download monthly performance", "goalpath_performance.csv")
    with tabs[1]:
        a, b, c = st.columns(3)
        a.metric("Average holdings", f"{m.holdings_count.mean():.1f}")
        b.metric("Effective holdings", f"{q['effective']:.2f}")
        c.metric("Maximum bond weight", pct(m.max_individual_bond_weight.max()))
        decision = st.selectbox("Inspect executed decision month", m.decision_month.tolist(),
                                index=len(m)-1, key="hold_month")
        show_holdings(result, decision)
    with tabs[2]:
        a, b, c = st.columns(3)
        a.metric("Minimum benchmark return coverage", pct(m.benchmark_return_coverage.min()))
        b.metric("Maximum terminal monthly TE", pct(m.terminal_tracking_error.max()))
        c.metric("Peak post-settlement cash", pct(m.cash_weight_after_settlement.max()))
        st.caption("Post-settlement cash may exceed 1% following redemption; the cap applies immediately after trading.")
        decision = st.selectbox("Inspect planned risk at decision", m.decision_month.tolist(),
                                index=len(m)-1, key="risk_month")
        show_risk(result, decision)
        st.caption(f"Restricted solves reported optimal: {int(m.optimal.sum())}/{len(m)}. This is not proof of global cardinality optimality.")
    with tabs[3]:
        show_comparison(result)


def show_comparison(result):
    st.subheader("Calibration comparison")
    st.caption("Part D reference results for the same K and γ, Jan 2024–Oct 2025. "
               "This table is not recalculated for a custom historical window.")
    try:
        r = read_archive()["rankings"]
    except (FileNotFoundError, OSError) as exc:
        st.info(f"Calibration comparison unavailable: {exc}")
        return
    r = r.loc[r.K.eq(result["k"]) & np.isclose(r.gamma, result["gamma"])].copy()
    r["Model"] = r.model_id.map({model_id(s): s for s in MODELS})
    r = r.sort_values("utility_rank", na_position="last")
    display = pd.DataFrame({"Model": r.Model, "Status": np.where(r.complete, "Complete", "Incomplete"),
                            "Net TE": r.annualized_net_tracking_error.map(pct),
                            "Net return": r.cumulative_net_return.map(pct),
                            "Annual net utility": r.annualized_net_utility.map(pct),
                            "Rank": r.utility_rank})
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("Utility = annual mean net active return − γ × annual net tracking error² (decimal returns). "
               "Ranking is retrospective; a near tie is not evidence of a statistically superior model.")


def show_allocation(result):
    m, p = result["metrics"], result["plans"].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("First-month expected net return", pct(p.planned_expected_return - result["fee"] / result["capital"]))
    b.metric("Benchmark expected return", pct(p.benchmark_expected_return))
    c.metric("Expected monthly TE", pct(p.planned_tracking_error))
    d.metric("First-trade fees", money(result["fee"]))
    a, b, c = st.columns(3)
    a.metric("Invested in bonds", money(result["invested"]))
    b.metric("Remaining cash", money(result["cash"]))
    c.metric("Number of holdings", len(result["holdings"]))
    st.caption("Model estimates for the saved decision date, not current prices or guaranteed future returns. "
               "The first-month estimate subtracts actual modeled trading fees once; cash return is assumed zero.")
    tabs = st.tabs(["Bond allocations", "Risk & planning", "Calibration comparison"])
    with tabs[0]:
        show_holdings(result)
    with tabs[1]:
        st.metric("Terminal expected monthly TE", pct(m["terminal_tracking_error"]))
        st.caption(f"Solver status {m['solver_status']} | restricted solve optimal: {m['optimal']}")
        show_risk(result)
    with tabs[2]:
        show_comparison(result)


def dashboard():
    st.set_page_config(page_title="GoalPath-Bond | Investor Studio", page_icon="📈", layout="wide")
    st.markdown("""
        <style>
        .block-container {max-width: 1380px; padding-top: 2rem;}
        .gp-hero {padding: 1.6rem 1.8rem; border-radius: 15px;
                  background: linear-gradient(110deg,#123447,#075950); color:#fff; margin-bottom:1.2rem;}
        .gp-hero h1 {margin:0; color:#fff; font-size:2rem; letter-spacing:-.035em;}
        .gp-hero p {margin:.45rem 0 0; color:#D2E9E5;}
        [data-testid="stMetric"] {border:1px solid rgba(128,145,160,.24); border-radius:12px; padding:12px;}
        </style>
        <div class="gp-hero"><h1>GoalPath-Bond</h1>
        <p>Investor Studio &nbsp;|&nbsp; Personalized fixed-income benchmark tracking</p></div>
        """, unsafe_allow_html=True)
    with st.sidebar:
        st.header("Investor profile")
        mode = st.radio("Analysis mode", ["Calibrated historical replay", "Fresh rolling backtest",
                                         "Fresh portfolio allocation"])
        with st.form("investor_inputs"):
            capital = st.number_input("Initial capital (USD)", min_value=1.0, value=100000.0,
                                      step=1000.0, format="%.2f")
            k = st.select_slider("Maximum bonds (K)", options=K_VALUES, value=25)
            gamma = st.select_slider("Risk aversion (γ)", options=GAMMA_VALUES, value=1,
                                     help="Higher γ puts more weight on predicted benchmark-relative variance.")
            start, end = FIRST_DECISION, LAST_DECISION
            if mode == "Fresh rolling backtest":
                start, end = st.select_slider("Decision window", options=DECISIONS, value=(FIRST_DECISION, LAST_DECISION))
            elif mode == "Fresh portfolio allocation":
                start = st.selectbox("Saved decision month", DECISIONS, index=len(DECISIONS)-1)
            submitted = st.form_submit_button("Run GoalPath-Bond", type="primary", use_container_width=True)
        st.divider()
        st.subheader("Benchmark")
        st.write(BENCHMARK)
        st.caption("Fixed calibrated benchmark. Other indices require new benchmark data and recalibration.")
        st.caption("The archive spans Jan 2024–Oct 2025 realized returns. Fresh runs use saved historical forecasts, not live markets.")
        if st.button("Clear results"):
            st.session_state.pop("gp_result", None)
        if st.button("Refresh local data caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("gp_result", None)

    if submitted:
        st.session_state.pop("gp_result", None)
        try:
            if mode == "Calibrated historical replay":
                with st.spinner("Loading calibrated portfolio history"):
                    result = archived_replay(k, gamma, capital)
            else:
                result = run_fresh(mode, k, gamma, capital, start, end)
            st.session_state.gp_result = result
        except Exception as exc:
            st.error(f"Unable to complete this analysis: {exc}")
            st.caption("Check the local data files and Gurobi license for fresh runs. No partial path is presented as a completed result.")

    result = st.session_state.get("gp_result")
    if result is None:
        st.subheader("Your portfolio, your risk preference")
        st.write("Choose a bond limit, risk aversion and starting capital in the sidebar. "
                 "GoalPath-Bond selects the calibrated model and reports its benchmark-relative performance.")
        a, b, c = st.columns(3)
        a.markdown("**Historical replay**\n\nInstantly explore the completed Part D portfolio histories.")
        b.markdown("**Fresh backtest**\n\nReoptimize month by month from cash over a chosen historical window.")
        c.markdown("**Portfolio allocation**\n\nSolve a new investment from cash at a saved decision date.")
        with st.expander("Calibrated model-selection grid", expanded=True):
            st.dataframe(pd.DataFrame(SELECTION).T.rename_axis("γ").rename(columns=lambda x: f"K = {x}"),
                         use_container_width=True)
            st.caption("S1–S4 are primary models; S5 is a solver-failure fallback. "
                       "The two γ=1 edge profiles are provisional due to missing competing backtests.")
        return

    st.caption(f"{result['source']}  •  K={result['k']}  •  γ={result['gamma']:g}  •  "
               f"Initial capital {money(result['capital'])}")
    st.subheader(f"{result['sid']} — {MODEL_NAMES[result['sid']]}")
    settings = settings_for(result["sid"], result["gamma"])
    st.caption(f"H={MODELS[result['sid']][1]} months  |  Turnover λ={settings['turnover_multiplier']:g}  |  "
               f"KRD tolerance={settings['krd_tolerance']:g} years  |  Uncertainty α={settings['uncertainty_scale']:g}")
    if result["fallback_reason"]:
        st.warning(f"Primary {result['primary']} failed. S5 was solved independently from cash. "
                   f"Reason: {result['fallback_reason']}")
    if (result["k"], result["gamma"]) in PROVISIONAL:
        st.info("This profile's calibrated primary selection is provisional: one competing Part D backtest did not complete.")
    if result["source"] == "Archived Part D replay":
        st.caption("Archived $100,000 path rescaled to your capital using continuous weights and proportional fees; no new solve.")
    if result["kind"] == "history":
        show_history(result)
    else:
        st.caption(f"Decision date: {result['asof']} month-end")
        show_allocation(result)
    with st.expander("Methodology and limitations"):
        st.write("Long-only relax-and-fix optimization; a maximum of 150 liquid candidates; "
                 "10 bp proportional transaction costs; duration, convexity, four key-rate duration bands "
                 "and a 1% post-trade cash cap. The terminal expected monthly TE cap is 2.5%. "
                 "The benchmark is a proxy, not official Bloomberg index returns.")
        st.write("The selection grid was calibrated on 22 months of historical data and is not independently validated. "
                 "Continuous-dollar allocations do not enforce bond denominations or minimum trading lots. "
                 "Risk estimates use reconstructed cash flows and proxy exposures; historical eligibility exclusions "
                 "and redemption estimates may affect results. This is an academic research tool, not live trading advice.")


if __name__ == "__main__":
    dashboard()
