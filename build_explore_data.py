"""
build_explore_data.py
---------------------
Precomputes everything the interactive page (explore.html) needs from the
two daily CSVs, and writes a single JavaScript data file (explore_data.js).

No raw portfolio returns or attention data leave this script: the output is a
table of summary statistics (state-conditional mean returns, Fama-MacBeth
slopes, t-statistics) at each attention-state cutoff, plus the smoothed,
standardized time series already published in the papers.

Usage:
    python build_explore_data.py deciles_daily.csv premium_news_daily.csv explore_data.js
"""
import sys, json
import numpy as np, pandas as pd

dec_path, prem_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(dec_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
p = pd.read_csv(prem_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

VW = [f"vw_ret_{i}" for i in range(1, 11)]
EW = [f"ew_ret_{i}" for i in range(1, 11)]
BE = [f"beta_{i}" for i in range(1, 11)]
R = {"vw": d[VW].values, "ew": d[EW].values}
B = d[BE].values
T = len(d)

# ---- state construction: residual below its trailing one-year quantile ----
# (expanding window for the first year, then a 252-day trailing window)
u = d["uhat_med5"]
def state_at(cut):
    thr = u.rolling(252, min_periods=60).quantile(cut / 100.0)
    s = (u < thr).values.copy()
    s[thr.isna().values] = False
    return s

# ---- Fama-MacBeth within a state: r_{t+1} on beta_t, state measured at t+1 ----
def fm(Rm, mask):
    sl, ic, r2 = [], [], []
    for t in range(T - 1):
        if not mask[t + 1]:
            continue
        y, x = Rm[t + 1], B[t]
        X = np.column_stack([np.ones(10), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ b
        sl.append(b[1]); ic.append(b[0]); r2.append(1 - res.var() / y.var())
    sl, ic = np.array(sl), np.array(ic)
    tstat = lambda a: float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 else float("nan")
    return dict(n=int(len(sl)), slope=float(100 * sl.mean()), t_slope=tstat(sl),
                intercept=float(100 * ic.mean()), t_int=tstat(ic), r2=float(np.mean(r2)))

def summarize(mask):
    out = {"share": float(mask.mean())}
    # average excess return per decile (bps) and full-sample mean betas (as in the paper's figures)
    for w in ("vw", "ew"):
        sub = R[w][mask]
        out[w] = dict(mean_ret_bps=[float(x) for x in 100 * sub.mean(axis=0)], **fm(R[w], mask))
    out["beta"] = [float(x) for x in B.mean(axis=0)]
    return out

cutoffs = list(range(5, 101, 5))
states = {str(c): summarize(state_at(c)) for c in cutoffs}
states["100"] = summarize(np.ones(T, bool))               # unconditional
paper = {"low50": summarize(d["state_low50"].values == 1),  # the paper's own definitions
         "low25": summarize(d["state_low25"].values == 1)}

# ---- time series for the second panel (daily, standardized in the page) ----
def z(x):
    x = pd.Series(x); return (x - x.mean()) / x.std()
ts = dict(
    date=[s.strftime("%Y-%m-%d") for s in p["date"]],
    fm_vw=[round(float(x), 4) for x in p["fm_premium_vw"]],
    fm_ew=[round(float(x), 4) for x in p["fm_premium_ew"]],
    uhat=[round(float(x), 5) for x in d["uhat_med5"]],
    aia=[round(float(x), 4) for x in p["aia"]],
    iai=[None if pd.isna(x) else round(float(x), 4) for x in p["iai"]],
)

payload = dict(
    sample=dict(start=ts["date"][0], end=ts["date"][-1], days=T),
    cutoffs=cutoffs, states=states, paper=paper, ts=ts,
)
with open(out_path, "w") as f:
    f.write("window.EXPLORE = " + json.dumps(payload) + ";\n")
print(f"wrote {out_path}: {T} days, {len(cutoffs)} cutoffs")
print("check vs paper (VW, below-median):", paper["low50"]["vw"]["slope"], paper["low50"]["vw"]["t_slope"])
