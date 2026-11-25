"""GoalPath-Bond | Stage 3 investor implementation.

Run: python Stage_3_dashboard.py
Historical-market demonstration: fresh monthly optimization, not archived
portfolio outcomes or future-market scenario generation.
"""
import math
from threading import RLock, Lock, Thread, Timer
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import webbrowser

import numpy as np
import pandas as pd

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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
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


def evaluate_goal(k, gamma, capital, months, goal, progress):
    target = capital * math.exp(math.log1p(goal) * months / 12)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Target wealth is outside the supported numerical range.")
    primary = SELECTION[gamma][reference_k(k)]
    progress.progress(0.0, text="Loading bond history and forecasts")
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


def result_payload(result):
    """Same metrics as the previous dashboard, serialized for the browser."""
    m = result["monthly"]
    capital, target = result["capital"], result["target"]
    ending, benchmark = float(m.portfolio_value.iloc[-1]), float(m.benchmark_value.iloc[-1])
    r, b = ending / capital - 1, benchmark / capital - 1
    annualized = math.expm1(math.log(ending / capital) * 12 / result["months"])
    active = m.net_return - m.benchmark_return
    te = float(active.std(ddof=1) * np.sqrt(12))
    utility = 12 * active.mean() - result["gamma"] * te**2
    sid = result["sid"]
    return {
        "model": f"{sid} — {MODEL_NAMES[sid]}", "sid": sid,
        "primary": result["primary"], "fallback": result["fallback"],
        "start": result["start"], "end": str(m.month.iloc[-1]),
        "months": result["months"], "k": result["k"], "gamma": result["gamma"],
        "reference_k": reference_k(result["k"]),
        "provisional": (reference_k(result["k"]), result["gamma"]) in {(10, 1), (50, 1)},
        "nonoptimal": int((~m.optimal).sum()),
        "net": r, "benchmark": b, "ending": ending, "benchmark_ending": benchmark,
        "te": te if math.isfinite(te) else None,
        "utility": float(utility) if math.isfinite(utility) else None,
        "annualized": annualized, "goal": result["goal"], "target": target,
        "fees": float(m.fees.sum()), "effective": float(m.effective_holdings.mean()),
        "capital": capital,
        "wealth": [[str(pd.Period(result["start"], freq="M") - 1), capital, capital]]
                  + [[str(row.month), float(row.portfolio_value), float(row.benchmark_value)]
                     for row in m.itertuples(index=False)],
        "holdings": json.loads(result["holdings"].to_json(orient="records")),
    }


# Local browser UI; no Streamlit and no separate HTML file.
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoalPath-Bond</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:#172126;font:16px system-ui,-apple-system,sans-serif}
main{max-width:980px;margin:24px auto;padding:24px;background:white;border:1px solid #e4e9eb;border-radius:16px}
header{background:linear-gradient(155deg,#173b50,#102f40);color:#fff;border-radius:12px;padding:24px}
h1{font-size:29px;margin:0 0 6px}header p{color:#c0dddd;margin:0}h2{font-size:21px;margin:24px 0 14px}
label{display:block;color:#526169;margin:16px 0 7px}input,select,button{font:inherit}
input[type=number],input[type=text],select{width:100%;padding:11px 13px;border:1px solid #d3dcdf;border-radius:9px;background:#fff;color:#172126}
input[type=range]{width:100%;accent-color:#008b83}.radio{display:flex;flex-wrap:wrap;gap:8px}
.radio label{margin:0;cursor:pointer}.radio input{position:absolute;opacity:0}.radio span{display:block;border:1px solid #dbe2e4;padding:9px 16px;border-radius:20px;color:#53616a}
.radio input:checked+span{background:#e0f2ef;border-color:#008b83;color:#00635d;font-weight:650}.radio input:focus-visible+span{outline:2px solid #008b83}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.metric{padding:16px;border:1px solid #e0e5e7;border-radius:10px;min-width:0}
.metric small,.note{color:#627078;font-size:14px;line-height:1.5}.metric strong{display:block;font-size:25px;margin:8px 0 4px;overflow-wrap:anywhere}
button{border:none;border-radius:9px;background:#008b83;color:white;padding:13px 16px;font-weight:650;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
#run{width:100%;margin-top:16px}hr{border:0;border-top:1px solid #e0e5e7;margin:25px 0}
progress{width:100%;accent-color:#008b83;height:16px}.notice{padding:12px 15px;border-radius:8px;background:#edf4f5;line-height:1.5}
.warn{background:#fff4e1;color:#814700}.good{background:#e5f5eb;color:#146038}.hidden{display:none}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px}
.bar{height:16px;border-radius:6px;background:#edf0f2;overflow:hidden;margin:7px 0}.fill{height:100%;border-radius:6px;background:#008b83}
#benchBar{background:#8595ad}#chart{width:100%;height:250px;display:block;margin:12px 0}
details{margin:20px 0}summary{cursor:pointer;font-weight:650}.tablewrap{overflow:auto;max-height:420px;margin-top:12px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #e1e5e7;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:#fff}
@media(max-width:600px){main{margin:0;padding:16px;border:0;border-radius:0}.metric strong{font-size:20px}header{padding:18px}}
</style></head><body><main>
<header><h1>GoalPath-Bond</h1><p>Investor Studio | Personalized fixed-income optimization</p></header>
<h2>Configure your portfolio</h2>
<form id="form">
<label for="capital">Initial capital (USD)</label><input id="capital" type="text" inputmode="decimal" value="100000" required>
<label for="k">Maximum bond quantity — <strong id="kValue">25</strong> bonds</label>
<input id="k" type="range" min="1" max="50" step="1" value="25">
<label>Investor risk aversion</label><div class="radio">
<label><input type="radio" name="risk" value="0.1"><span>Low</span></label>
<label><input type="radio" name="risk" value="1" checked><span>Medium</span></label>
<label><input type="radio" name="risk" value="10"><span>High</span></label>
<label><input type="radio" name="risk" value="100"><span>Very high</span></label></div>
<div class="grid"><div><label for="goal">Desired annualized return (%)</label><input id="goal" type="number" step="any" value="6" required></div>
<div><label for="months">Investment horizon (months)</label><input id="months" type="number" min="1" step="1" value="36" required></div></div>
<p class="note">Historical data from 2016; investment begins after model warm-up (January 2018 or later).</p>
<button id="run" type="submit">Evaluate my goal</button></form>
<div id="status" class="hidden" aria-live="polite"><p id="statusText" class="notice"></p><progress id="progress" max="1" value="0"></progress></div>
<section id="results" class="hidden"><hr><p class="note">Automatically selected configuration</p><h2 id="model"></h2><p id="period" class="note"></p><p id="warnings" class="notice warn hidden"></p>
<div class="grid" id="metrics"></div>
<h2>Portfolio vs. benchmark</h2>
<div class="row"><span>GoalPath-Bond</span><strong id="portReturn"></strong></div><div class="bar"><div class="fill" id="portBar"></div></div><p id="portEnd" class="note"></p>
<div class="row"><span>Benchmark proxy</span><strong id="benchReturn"></strong></div><div class="bar"><div class="fill" id="benchBar"></div></div><p id="benchEnd" class="note"></p>
<canvas id="chart" role="img" aria-label="Portfolio and benchmark wealth over the investment period"></canvas>
<p class="note"><span style="color:#008b83">● GoalPath-Bond</span> &nbsp; <span style="color:#8595ad">● Benchmark</span></p>
<div class="grid" id="costs"></div>
<h2>Investment goal</h2><div class="grid" id="goalMetrics"></div><p id="verdict" class="notice"></p>
<details><summary>Monthly portfolio holdings</summary><label for="holdMonth">Investment month</label><select id="holdMonth"></select><div class="tablewrap"><table><thead><tr><th>CUSIP</th><th>Weight (%)</th><th>Allocation (USD)</th><th>Realized return (%)</th></tr></thead><tbody id="holdings"></tbody></table></div></details>
<p class="note">Fresh monthly solves using point-in-time forecast inputs and historical returns. This is a retrospective demonstration, not future returns or actual trades. Strategy selection was calibrated on later historical data; this is not an independent out-of-sample test. Bond amounts are continuous-dollar allocations, not executable lots.</p>
</section></main>
<script>
const $=id=>document.getElementById(id), cash=x=>x.toLocaleString("en-US",{style:"currency",currency:"USD"}), pct=x=>x===null?"N/A":(100*x).toFixed(2)+"%", pp=x=>(x>=0?"+":"")+(100*x).toFixed(2)+" pp";
let result=null, polling=false;
$("k").addEventListener("input",()=>{$("kValue").textContent=$("k").value});
function metric(parent,title,value,note=""){
  const box=document.createElement("div");box.className="metric";
  for(const [tag,text] of [["small",title],["strong",value],["small",note]]){const el=document.createElement(tag);el.textContent=text;box.append(el)}
  $(parent).append(box);
}
function drawChart(){
  if(!result)return;const c=$("chart"),dpr=window.devicePixelRatio||1,W=c.clientWidth,H=c.clientHeight;
  if(!W||!H)return;c.width=W*dpr;c.height=H*dpr;const ctx=c.getContext("2d");ctx.scale(dpr,dpr);
  const rows=result.wealth,L=Math.min(115,W*.3),R=12,T=15,B=30,vals=rows.flatMap(r=>r.slice(1));
  let lo=Math.min(...vals),hi=Math.max(...vals),pad=(hi-lo||Math.max(Math.abs(hi),1)*.1)*.12;lo-=pad;hi+=pad;
  const x=i=>L+(W-L-R)*i/(rows.length-1),y=v=>T+(H-T-B)*(hi-v)/(hi-lo);
  ctx.font="11px system-ui";ctx.fillStyle="#627078";ctx.strokeStyle="#e0e5e7";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,yy=y(v);ctx.beginPath();ctx.moveTo(L,yy);ctx.lineTo(W-R,yy);ctx.stroke();ctx.textAlign="right";ctx.fillText(cash(v),L-7,yy+4,L-10)}
  ctx.textAlign="center";const ticks=[0,Math.floor((rows.length-1)/2),rows.length-1];for(const i of new Set(ticks)){ctx.fillText(rows[i][0],Math.max(24,Math.min(W-24,x(i))),H-8)}
  for(const [j,color] of [[1,"#008b83"],[2,"#8595ad"]]){ctx.beginPath();ctx.lineWidth=2.5;ctx.strokeStyle=color;rows.forEach((r,i)=>i?ctx.lineTo(x(i),y(r[j])):ctx.moveTo(x(i),y(r[j])));ctx.stroke()}
}
function showHoldings(){
  const tbody=$("holdings");tbody.replaceChildren();if(!result)return;
  for(const h of result.holdings.filter(h=>h.month===$("holdMonth").value).sort((a,b)=>b.weight-a.weight)){
    const tr=document.createElement("tr");for(const v of [h.cusip,(h.weight*100).toFixed(3),cash(h.allocation),(h.next_return*100).toFixed(3)]){const td=document.createElement("td");td.textContent=v;tr.append(td)}tbody.append(tr);
  }
}
function display(r){
  result=r;$("results").classList.remove("hidden");
  $("model").textContent=r.model;$("period").textContent=r.start+" – "+r.end+" | "+r.months+" months | K ≤ "+r.k+" | γ = "+r.gamma+" | Calibration reference K = "+r.reference_k;
  const warnings=[];if(r.fallback)warnings.push("Primary "+r.primary+" failed; S5 completed the entire path. "+r.fallback);
  if(r.provisional)warnings.push("Provisional calibration: one competing reference strategy did not complete.");
  if(r.nonoptimal)warnings.push(r.nonoptimal+" monthly solutions were feasible but not certified optimal.");
  $("warnings").textContent=warnings.join(" ");$("warnings").classList.toggle("hidden",!warnings.length);
  for(const id of ["metrics","costs","goalMetrics"])$(id).replaceChildren();
  metric("metrics","Net cumulative return",pct(r.net),pp(r.net-r.benchmark)+" vs benchmark");
  metric("metrics","Annualized net tracking error",pct(r.te),"Benchmark-relative risk");
  metric("metrics","Ending portfolio value",cash(r.ending));metric("metrics","Annualized net utility",pct(r.utility));
  $("portReturn").textContent=pct(r.net);$("benchReturn").textContent=pct(r.benchmark);
  $("portEnd").textContent="Ending value: "+cash(r.ending);$("benchEnd").textContent="Ending value: "+cash(r.benchmark_ending);
  const max=Math.max(Math.abs(r.net),Math.abs(r.benchmark),1e-12);
  $("portBar").style.width=(100*Math.abs(r.net)/max)+"%";$("benchBar").style.width=(100*Math.abs(r.benchmark)/max)+"%";
  $("portBar").style.background=r.net<0?"#b66c57":"#008b83";$("benchBar").style.background=r.benchmark<0?"#b66c57":"#8595ad";
  metric("costs","Total transaction costs",cash(r.fees));metric("costs","Average effective holdings",r.effective.toFixed(2));
  metric("goalMetrics","Target terminal wealth",cash(r.target));metric("goalMetrics","Realized annualized return",pct(r.annualized),pp(r.annualized-r.goal)+" vs target");
  const met=r.ending>=r.target;$("verdict").className="notice "+(met?"good":"warn");
  $("verdict").textContent=met?"Goal met in this historical investment: "+cash(r.ending-r.target)+" above target.":"Goal not met: "+cash(r.target-r.ending)+" below target. Try a different risk aversion or bond limit, or revise the return target/horizon. A change does not guarantee a better result.";
  $("holdMonth").replaceChildren();for(const row of r.wealth.slice(1)){const o=document.createElement("option");o.value=o.textContent=row[0];$("holdMonth").append(o)}
  $("holdMonth").selectedIndex=$("holdMonth").options.length-1;showHoldings();drawChart();
}
$("holdMonth").addEventListener("change",showHoldings);window.addEventListener("resize",drawChart);
function status(text,progress,error=false){$("status").classList.remove("hidden");$("statusText").textContent=text;$("statusText").className="notice"+(error?" warn":"");$("progress").classList.toggle("hidden",error);$("progress").value=progress}
function busy(on){$("run").disabled=on;for(const el of $("form").querySelectorAll("input"))el.disabled=on}
async function poll(){
  if(polling)return;polling=true;busy(true);
  try{while(true){const res=await fetch("/api/status",{cache:"no-store"});if(!res.ok)throw Error("Cannot read optimization status.");const j=await res.json();
    if(j.state==="done"){display(j.result);$("status").classList.add("hidden");break}
    if(j.state==="error"){status("Unable to complete this investment: "+j.error+". No partial or archived result is substituted.",0,true);break}
    if(j.state!=="running"){ $("status").classList.add("hidden");break}
    status(j.text,j.progress);await new Promise(resolve=>setTimeout(resolve,700));
  }}catch(e){status(e.message,0,true)}finally{polling=false;busy(false)}
}
$("form").addEventListener("submit",async e=>{
  e.preventDefault();if(polling||$("run").disabled)return;const raw=$("capital").value.replace(/,/g,"").trim();
  const capital=Number(raw),k=Number($("k").value),gamma=Number(document.querySelector('input[name="risk"]:checked').value),goal=Number($("goal").value)/100,months=Number($("months").value);
  if(!raw||!Number.isFinite(capital)||capital<=0||!Number.isFinite(goal)||goal<=-1||!Number.isSafeInteger(months)||months<1){status("Enter positive finite capital, an annual return above -100%, and a positive integer horizon.",0,true);return}
  busy(true);result=null;$("results").classList.add("hidden");status("Loading bond history and forecasts",0);
  try{const res=await fetch("/api/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({capital,k,gamma,goal,months})});const j=await res.json();if(!res.ok&&res.status!==409)throw Error(j.error||"Could not start optimization.");await poll()}
  catch(e){status(e.message,0,true);busy(false)}
});
</script></body></html>"""


HOST, PORT = "127.0.0.1", 8765
JOB_LOCK = Lock()
JOB = {"state": "idle"}


class Progress:
    """Translate engine progress calls to a thread-safe status endpoint."""
    def progress(self, value, text=""):
        with JOB_LOCK:
            JOB.update(progress=float(value), text=text)

    def empty(self):
        pass


def run_job(k, gamma, capital, months, goal):
    try:
        result = result_payload(evaluate_goal(k, gamma, capital, months, goal, Progress()))
        # Validate the entire response before marking the job successful.
        json.dumps(result, allow_nan=False)
        with JOB_LOCK:
            JOB.update(state="done", progress=1.0, result=result)
    except Exception as exc:
        with JOB_LOCK:
            JOB.update(state="error", error=str(exc))


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, data):
        self.send_bytes(status, json.dumps(data, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8")

    def allowed_host(self):
        return self.headers.get("Host") in {f"{HOST}:{PORT}", f"localhost:{PORT}"}

    def do_GET(self):
        if not self.allowed_host():
            self.send_error(403)
        elif self.path == "/":
            self.send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            with JOB_LOCK:
                status = dict(JOB)
            self.send_json(200, status)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.allowed_host() or self.headers.get("Origin") not in {
                f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"}:
            self.send_json(403, {"error": "Only the local dashboard can request a solve."})
            return
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2048 or self.headers.get_content_type() != "application/json":
                raise ValueError("Invalid request.")
            p = json.loads(self.rfile.read(size))
            k, gamma, capital, months, goal = (p[x] for x in ("k", "gamma", "capital", "months", "goal"))
            if (type(k) is not int or not 1 <= k <= 50 or
                    type(months) is not int or months < 1 or
                    type(gamma) not in (int, float) or gamma not in GAMMA_VALUES or
                    type(capital) not in (int, float) or not math.isfinite(capital) or capital <= 0 or
                    type(goal) not in (int, float) or not math.isfinite(goal) or goal <= -1):
                raise ValueError("Invalid investor inputs.")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        with JOB_LOCK:
            already_running = JOB["state"] == "running"
            if not already_running:
                JOB.clear()
                JOB.update(state="running", progress=0.0, text="Loading bond history and forecasts")
        if already_running:
            self.send_json(409, {"error": "An investment is already running."})
            return
        Thread(target=run_job, args=(k, gamma, capital, months, goal), daemon=True).start()
        self.send_json(202, {"state": "running"})


def dashboard():
    url = f"http://{HOST}:{PORT}/"
    with ThreadingHTTPServer((HOST, PORT), Handler) as server:
        print(f"GoalPath-Bond: {url}\nPress Ctrl+C to stop.", flush=True)
        Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nGoalPath-Bond stopped.", flush=True)


if __name__ == "__main__":
    dashboard()
