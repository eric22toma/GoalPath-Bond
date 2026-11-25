"""GoalPath-Bond Stage 3 | local goal-planning server, no Streamlit.

Run: python Stage_3_server.py
The browser interface is Stage_3_dashboard.html. Instant replay requires only
Python's standard library (or open the HTML directly). Fresh exact-K solves
require the existing Stage 2 data, Python packages and Gurobi license.
"""
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Timer
import webbrowser

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "Stage_3_dashboard.html"
HOST, PORT = "127.0.0.1", 8765
REFERENCE_CAPITAL = 100_000.0
GAMMAS = (0.1, 1.0, 10.0, 100.0)
MODELS = {
    "S1": ("random_walk", 1, (("turnover_multiplier", 10),)),
    "S2": ("direct", 1, (("turnover_multiplier", 5), ("krd_tolerance", 0.5))),
    "S3": ("constant", 3, (("turnover_multiplier", 10), ("krd_tolerance", 0.5))),
    "S4": ("constant", 3, (("uncertainty_scale", 0.5), ("turnover_multiplier", 5))),
    "S5": ("random_walk", 3, (("turnover_multiplier", 10), ("krd_tolerance", 0.5))),
}
SELECTION = {
    0.1: {10: "S4", 25: "S4", 50: "S4"},
    1.0: {10: "S3", 25: "S4", 50: "S4"},
    10.0: {10: "S4", 25: "S4", 50: "S4"},
    100.0: {10: "S2", 25: "S3", 50: "S1"},
}
LOCK = Lock()
DATA = None
CACHE = {}


class OptimizationFailure(RuntimeError):
    """Only optimization/solution failures permit a whole-path S5 retry."""


def reference_bucket(k):
    # Never round an investor's hard cap UP in historical replay.
    # K<10 has no calibrated replay; use the closest reference model, then solve exact K.
    return 50 if k == 50 else 25 if k >= 25 else 10


def validate_profile(k, gamma):
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50:
        raise ValueError("Maximum bond quantity must be an integer from 1 to 50.")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma not in GAMMAS:
        raise ValueError("Risk aversion must correspond to 0.1, 1, 10 or 100.")


def load_data():
    """Reuse the verified Stage 2 data loading and point-in-time error history."""
    global DATA
    if DATA is not None:
        return DATA
    import Stage_2_backtest as bt
    if (bt.START_MONTH, bt.END_MONTH) != ("2023-12", "2025-09"):
        raise ValueError("The Stage 2 window has changed; refresh the Stage 3 calibration archive first.")
    if (bt.CANDIDATE_COUNT, bt.TRANSACTION_COST, bt.MAX_CASH_WEIGHT, bt.MAX_WEIGHT) != (150, 0.001, 0.01, 1.0):
        raise ValueError("Stage 2 fixed settings no longer match the calibrated model.")
    forecasts, errors = bt.load_forecast_history(3)
    data = bt.process_all_data(bond_columns=bt.BOND_COLUMNS)
    macro, curve = bt.macro_features(bt.history_for_features(data, "curve"),
                                     bt.history_for_features(data, "fred"))
    panel = bt.load_panel(data["bonds"][1], macro, curve)
    DATA = (panel, curve, forecasts, errors)
    return DATA


def as_number(value):
    """JSON does not have NaN or Infinity; use null for missing values."""
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def pack_plan(metrics):
    """Use the same compact schema as the embedded historical data."""
    return [[h, str(p["month"]), int(p["holdings"]), float(p["cash_weight"]),
             float(p["expected_tracking_error"]), float(p["expected_return"]),
             float(p["benchmark_expected_return"]), float(p["duration_gap"]),
             float(p["convexity_gap"]), *map(float, p["krd_gaps"]),
             float(p["cashflow_risk_coverage"])]
            for h, p in enumerate(metrics["monthly_plan"], 1)]


def run_path(sid, k, gamma):
    """Reoptimize the complete calibration window from cash at the EXACT cap.

    Solve at reference capital because all weights/fees are homogeneous in
    capital. The browser scales dollar amounts to the investor's input.
    No future returns enter the optimizer; evaluate and settle only afterwards.
    """
    import numpy as np
    import pandas as pd
    import Stage_2_backtest as bt

    panel, curve, forecasts, errors = load_data()
    model, h, changes = MODELS[sid]
    settings = bt.strategy_parameters(gamma, changes)
    expected = {"duration_tolerance": 0.5, "convexity_tolerance": 25.0,
                "krd_tolerance": 2.0, "turnover_multiplier": 1.0,
                "uncertainty_scale": 1.0, "terminal_te_cap": 0.025,
                **dict(changes), "gamma": gamma}
    if settings != expected:
        raise ValueError("Stage 2 settings differ from the calibrated strategy.")
    state, value = None, REFERENCE_CAPITAL
    monthly, holdings, plans = [], [], []
    conditional_moments = []
    by_origin = {month: rows for month, rows in forecasts.groupby("month", sort=False)}
    for asof in pd.period_range(bt.START_MONTH, bt.END_MONTH, freq="M"):
        label = str(asof)
        print(f"{sid} | K={k} | gamma={gamma:g} | {label}", flush=True)
        snapshot = panel.loc[panel.month.eq(asof)].copy()
        history = panel.loc[panel.month.between(asof - 35, asof),
                            ["cusip", "month", "RET_EOM", "DURATION", "time_to_maturity", "RATING_NUM"]].copy()
        bundle = {"asof": label, "snapshot": snapshot, "benchmark_snapshot": snapshot.copy(),
                  "history": history, "curve_params": curve}
        stage = bt.prepare_strategy_inputs(bundle, by_origin[label], errors, state, value, model, h)
        try:
            solution, metrics = bt.run_configured_solver(stage, k, settings)
        except (RuntimeError, ValueError, bt.optimizer.gp.GurobiError) as exc:
            raise OptimizationFailure(f"{sid} at {label}: {exc}") from exc
        if metrics["K"] != k or not np.isclose(metrics["gamma"], gamma):
            raise ValueError("Optimizer did not receive the requested investor settings.")
        held, bond_return, benchmark_return, coverage, _ = bt.evaluate_month(stage, solution)
        fee = float(metrics["first_trade_cost_dollars"])
        before = value
        state, _, net_return, _ = bt.settle_with_cash(held, solution, stage, bond_return, fee / before)
        value *= 1 + net_return
        w = held.weight.to_numpy(float)
        if len(w) == 0 or len(w) > k:
            raise OptimizationFailure(f"Invalid held bond count at {label}: {len(w)}.")
        first = metrics["monthly_plan"][0]
        turnover = (solution["buys"].sum() + solution["sells"].sum()
                    + stage["outside_pretrade_weight"]) / 2
        monthly.append([
            str(asof + 1), label, float(net_return), float(benchmark_return), fee,
            float(turnover), int(len(w)), float(w.sum()**2 / np.sum(w**2)),
            float(w.max()), float(first["expected_tracking_error"]), float(solution["cash_weight"]),
            float(state["cash_weight"]), float(coverage), float(metrics["terminal_tracking_error"]),
            bool(metrics["optimal"]), float(value),
        ])
        held = held.copy()
        held["capital_allocated"] = held.weight * before
        held["expected_return"] = held.cusip.map(
            pd.Series(stage["periods"][0]["mu"], index=stage["candidates"].cusip))
        holdings.append([[str(row.cusip), as_number(row.weight), as_number(row.capital_allocated),
                          as_number(row.expected_return), as_number(row.duration), as_number(row.next_return),
                          str(row.return_source)]
                         for row in held.sort_values("weight", ascending=False).itertuples(index=False)])
        plans.append(pack_plan(metrics))
        if label == bt.END_MONTH:
            # Project the bond-level factor covariance onto each planned portfolio
            # and the complete benchmark. This is the same risk model as Stage 2.
            Q = np.asarray(stage['candidate_covariance'], float)
            cross = -0.5 * np.asarray(stage['risk_linear'], float)
            benchmark_var = float(stage['risk_constant'])
            for weights in solution['planned_weights']:
                var_p = float(weights @ Q @ weights)
                cov_pb = float(cross @ weights)
                if not np.isfinite([var_p, benchmark_var, cov_pb]).all() or min(var_p, benchmark_var) < -1e-10:
                    raise ValueError('Invalid conditional bond-factor covariance.')
                var_p, var_b = max(0.0, var_p), max(0.0, benchmark_var)
                limit = math.sqrt(var_p * var_b)
                if abs(cov_pb) > limit + 1e-8:
                    raise ValueError('Conditional portfolio/benchmark covariance is not positive semidefinite.')
                conditional_moments.append([var_p, var_b, max(-limit, min(limit, cov_pb))])
    if len(monthly) != 22:
        raise ValueError("Incomplete rolling backtest; no partial performance will be displayed.")
    r = np.array([m[2] for m in monthly])
    b = np.array([m[3] for m in monthly])
    if not np.isfinite(np.r_[r, b]).all() or np.min(np.r_[r, b]) <= -1:
        raise ValueError("Invalid realized return path.")
    return {"sid": sid, "monthly": monthly, "holdings": holdings, "plans": plans,
            "net": float(np.prod(1 + r) - 1), "br": float(np.prod(1 + b) - 1),
            "te": float(np.std(r-b, ddof=1) * np.sqrt(12)),
            "conditional_moments": conditional_moments}


def archive_rankings(bucket, gamma):
    """Reference comparison only; never imply these are exact-K rankings."""
    html = HTML.read_text(encoding="utf-8")
    marker = '<script id="archive" type="application/json">'
    raw = html.split(marker, 1)[1].split("</script>", 1)[0]
    return json.loads(raw)["profiles"][f"{bucket}|{gamma:g}"]["ranking"]


def optimize(k, gamma):
    validate_profile(k, gamma)
    gamma = float(gamma)
    with LOCK:
        key = (k, gamma)
        if key in CACHE:
            return {**CACHE[key], "cached": True}
        bucket = reference_bucket(k)
        primary = SELECTION[gamma][bucket]
        fallback = None
        try:
            result = run_path(primary, k, gamma)
        except OptimizationFailure as exc:
            fallback = str(exc)
            print(f"Primary failed; restart S5 from cash: {fallback}", flush=True)
            try:
                result = run_path("S5", k, gamma)
            except OptimizationFailure as fallback_exc:
                raise OptimizationFailure(
                    f"No complete solution at K={k}. Primary: {exc}. "
                    f"Fallback: {fallback_exc}. Try a larger bond limit or different risk aversion; "
                    "a solver failure is not proof of mathematical infeasibility."
                ) from fallback_exc
        result.update({"primary": primary, "fallback": fallback,
                       "provisional": (bucket, gamma) in {(10, 1.0), (50, 1.0)},
                       "referenceK": bucket, "ranking": archive_rankings(bucket, gamma),
                       "cached": False})
        CACHE[key] = result
        return result


class Handler(BaseHTTPRequestHandler):
    """Loopback-only server. Serve one HTML file and one JSON endpoint."""

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path not in ("/", "/Stage_3_dashboard.html"):
            self.send_error(404)
            return
        data = HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/run":
            self.send_error(404)
            return
        # Do not accept cross-origin browser requests or arbitrary data/file paths.
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"):
            self.send_json(403, {"error": "Only the local dashboard may request a solve."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048 or self.headers.get_content_type() != "application/json":
                raise ValueError("Invalid request.")
            payload = json.loads(self.rfile.read(length))
            k, gamma = payload["k"], payload["gamma"]
            validate_profile(k, gamma)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        try:
            result = optimize(k, gamma)
        except Exception as exc:
            print(f"Stage 3 request failed: {exc}", flush=True)
            self.send_json(422, {"error": str(exc)})
            return
        self.send_json(200, result)


def main():
    if not HTML.is_file():
        raise FileNotFoundError(f"Missing {HTML}; pull the latest repository changes.")
    url = f"http://{HOST}:{PORT}/"
    with ThreadingHTTPServer((HOST, PORT), Handler) as server:
        print(f"GoalPath-Bond: {url}\nNo Streamlit required. Ctrl+C to stop.\n"
              "Fresh runs use your existing Stage 2 environment and Gurobi license.", flush=True)
        Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nGoalPath-Bond stopped.", flush=True)


if __name__ == "__main__":
    main()
