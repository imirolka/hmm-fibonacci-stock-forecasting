import os
import numpy as np
import pandas as pd

OUTPUT_FOLDER = "post_results"
DECILES = 10
TAIL_PCTS = [0.10, 0.20]
EPS_UNCHANGED = 0.001


# locating prediction files
def find_prediction_files(root="."):
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.endswith("_predictions.csv"):
                out.append(os.path.join(r, f))
    out.sort()
    return out


def ticker_from_pred_filename(path):
    name = os.path.basename(path)
    if name.endswith("_predictions.csv"):
        return name.replace("_predictions.csv", "")
    if name.endswith("predictions.csv"):
        # e.g. SPG_predictions.csv
        base = name.replace("predictions.csv", "")
        return base.strip("_")
    return os.path.splitext(name)[0]


def infer_sector_from_path(pred_csv_path, ticker):

    p = pred_csv_path.replace("\\", "/").split("/")
    if len(p) < 2:
        return ""

    parent = p[-2]
    if parent.upper() == str(ticker).upper() and len(p) >= 3:
        sector = p[-3]
    else:
        sector = parent

    return sector.replace("_", " ")


def locate_results_root():
    here = find_prediction_files(".")
    if len(here) > 0:
        return ".", here

    if os.path.isdir("results"):
        there = find_prediction_files("results")
        if len(there) > 0:
            return "results", there

    return ".", []


# helpers
def to_float_array(x, fill_value=0.0):
    a = np.asarray(x, dtype=float)
    a = np.where(np.isfinite(a), a, fill_value)
    return a


def signal_to_position(sig_series):
    s = sig_series.astype(str).values
    pos = np.zeros(len(s), dtype=float)
    pos[s == "Buy"] = 1.0
    pos[s == "Sell"] = -1.0
    return pos


# metrics
def cum_return(strategy_returns):
    r = to_float_array(strategy_returns, fill_value=0.0)
    if len(r) == 0:
        return np.nan
    eq = np.cumprod(1.0 + r)
    return float(eq[-1] - 1.0)


def max_drawdown(strategy_returns):
    r = to_float_array(strategy_returns, fill_value=0.0)
    if len(r) == 0:
        return np.nan
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(np.min(dd))


def exposure_from_position(pos):
    p = to_float_array(pos, fill_value=0.0)
    if len(p) == 0:
        return np.nan
    return float(np.mean(np.abs(p) > 0))


def entries_from_position(pos):
    p = to_float_array(pos, fill_value=0.0)
    if len(p) == 0:
        return np.nan
    prev = np.r_[0.0, p[:-1]]
    curr = p
    return float(np.sum((prev == 0) & (curr != 0)))


def fibonacci_confirm_cancel_rate(df, pos_hmm, pos_fr):
    if "Signal_HMM" in df.columns and "Signal_HMM_FR" in df.columns:
        hmm_dir = df["Signal_HMM"].isin(["Buy", "Sell"])
        total = int(hmm_dir.sum())
        if total == 0:
            return np.nan, np.nan
        confirmed = int((hmm_dir & (df["Signal_HMM_FR"] == df["Signal_HMM"])).sum())
        cancelled = int((hmm_dir & (df["Signal_HMM_FR"] == "Hold")).sum())
        return confirmed / total, cancelled / total

    total = int(np.sum(np.abs(pos_hmm) > 0))
    if total == 0:
        return np.nan, np.nan
    confirmed = int(np.sum((pos_hmm != 0) & (pos_fr == pos_hmm)))
    cancelled = int(np.sum((pos_hmm != 0) & (pos_fr == 0)))
    return confirmed / total, cancelled / total


# per-ticker summary
def summarize_ticker(pred_csv_path):
    df = pd.read_csv(pred_csv_path)

    ticker = ticker_from_pred_filename(pred_csv_path)
    sector = infer_sector_from_path(pred_csv_path, ticker)

    if "Position_HMM" in df.columns:
        pos_hmm = to_float_array(df["Position_HMM"].values, fill_value=0.0)
    elif "Signal_HMM" in df.columns:
        pos_hmm = signal_to_position(df["Signal_HMM"])
    else:
        raise ValueError("Missing Position_HMM / Signal_HMM")

    if "Position_HMM_FR" in df.columns:
        pos_fr = to_float_array(df["Position_HMM_FR"].values, fill_value=0.0)
    elif "Signal_HMM_FR" in df.columns:
        pos_fr = signal_to_position(df["Signal_HMM_FR"])
    else:
        raise ValueError("Missing Position_HMM_FR / Signal_HMM_FR")

    if "ActualReturn" in df.columns:
        actual_ret = to_float_array(df["ActualReturn"].values, fill_value=0.0)
    elif "ActualClose" in df.columns and "LastClose" in df.columns:
        ac = to_float_array(df["ActualClose"].values, fill_value=np.nan)
        lc = to_float_array(df["LastClose"].values, fill_value=np.nan)
        actual_ret = np.where(np.isfinite(ac) & np.isfinite(lc) & (lc != 0), (ac - lc) / lc, 0.0)
    else:
        raise ValueError("Missing ActualReturn (or ActualClose/LastClose)")

    if "StrategyReturn_HMM" in df.columns:
        r_hmm = to_float_array(df["StrategyReturn_HMM"].values, fill_value=0.0)
    else:
        r_hmm = pos_hmm * actual_ret

    if "StrategyReturn_HMM_FR" in df.columns:
        r_fr = to_float_array(df["StrategyReturn_HMM_FR"].values, fill_value=0.0)
    else:
        r_fr = pos_fr * actual_ret

    cum_hmm = cum_return(r_hmm)
    cum_fr = cum_return(r_fr)

    dd_hmm = max_drawdown(r_hmm)
    dd_fr = max_drawdown(r_fr)

    exp_hmm = exposure_from_position(pos_hmm)
    exp_fr = exposure_from_position(pos_fr)

    ent_hmm = entries_from_position(pos_hmm)
    ent_fr = entries_from_position(pos_fr)

    conf_rate, cancel_rate = fibonacci_confirm_cancel_rate(df, pos_hmm, pos_fr)

    return {
        "Sector": sector,
        "Ticker": ticker,
        "n_days": int(df.shape[0]),

        "cum_return_hmm": cum_hmm,
        "cum_return_hmm_fr": cum_fr,
        "delta_cum_return": cum_fr - cum_hmm,

        "max_dd_hmm": dd_hmm,
        "max_dd_hmm_fr": dd_fr,
        "delta_max_dd": dd_fr - dd_hmm,

        "exposure_hmm": exp_hmm,
        "exposure_hmm_fr": exp_fr,
        "delta_exposure": exp_fr - exp_hmm,

        "entries_hmm": ent_hmm,
        "entries_hmm_fr": ent_fr,

        "fib_confirm_rate": conf_rate,
        "fib_cancel_rate": cancel_rate,
    }


# quantiles and tails
def make_quantile_table(ticker_df, q=DECILES):
    x = ticker_df.dropna(subset=["cum_return_hmm"]).copy()
    if len(x) == 0:
        return pd.DataFrame()

    ranks = x["cum_return_hmm"].rank(method="first")
    bins = min(q, len(x))
    x["quantile"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1

    agg = x.groupby("quantile", as_index=False).agg({
        "Ticker": "count",
        "cum_return_hmm": "mean",
        "cum_return_hmm_fr": "mean",
        "delta_cum_return": "mean",
        "max_dd_hmm": "mean",
        "max_dd_hmm_fr": "mean",
        "delta_max_dd": "mean",
        "exposure_hmm": "mean",
        "exposure_hmm_fr": "mean",
        "fib_confirm_rate": "mean",
        "fib_cancel_rate": "mean",
    })
    agg = agg.rename(columns={"Ticker": "n"})
    return agg


def tail_summary(ticker_df):
    x = ticker_df.dropna(subset=["cum_return_hmm"]).sort_values("cum_return_hmm").reset_index(drop=True)
    n = len(x)
    if n == 0:
        return pd.DataFrame()

    rows = []
    for p in TAIL_PCTS:
        k = max(1, int(np.ceil(p * n)))

        bottom = x.head(k)
        top = x.tail(k)

        for name, sub in [("bottom_%d%%" % int(p * 100), bottom), ("top_%d%%" % int(p * 100), top)]:
            rows.append({
                "bucket": name,
                "n": int(len(sub)),
                "mean_cum_return_hmm": float(sub["cum_return_hmm"].mean()),
                "mean_cum_return_hmm_fr": float(sub["cum_return_hmm_fr"].mean()),
                "mean_delta_cum_return": float(sub["delta_cum_return"].mean()),
                "mean_max_dd_hmm": float(sub["max_dd_hmm"].mean()),
                "mean_max_dd_hmm_fr": float(sub["max_dd_hmm_fr"].mean()),
                "mean_delta_max_dd": float(sub["delta_max_dd"].mean()),
                "mean_exposure_hmm": float(sub["exposure_hmm"].mean()),
                "mean_exposure_hmm_fr": float(sub["exposure_hmm_fr"].mean()),
                "mean_fib_confirm_rate": float(sub["fib_confirm_rate"].mean()),
                "mean_fib_cancel_rate": float(sub["fib_cancel_rate"].mean()),
            })
    return pd.DataFrame(rows)


def impact_counts(ticker_df):
    x = ticker_df.dropna(subset=["cum_return_hmm", "cum_return_hmm_fr"]).copy()
    if len(x) == 0:
        return pd.DataFrame()

    delta = x["cum_return_hmm_fr"] - x["cum_return_hmm"]
    higher = int((delta > EPS_UNCHANGED).sum())
    lower = int((delta < -EPS_UNCHANGED).sum())
    unchanged = int((np.abs(delta) <= EPS_UNCHANGED).sum())

    return pd.DataFrame([{
        "n_tickers": int(len(x)),
        "higher_cum_return_fr": higher,
        "lower_cum_return_fr": lower,
        "unchanged_cum_return": unchanged,
        "eps_unchanged": EPS_UNCHANGED
    }])


# main
def main():

    base, pred_files = locate_results_root()
    if len(pred_files) == 0:
        print("No predictions found. I searched for files ending with:")
        print(" - *_predictions.csv")
        print(" - *predictions.csv")
        print("in: .  and (if present) ./results")
        return

    out_dir = os.path.join(base, OUTPUT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    skipped = 0

    for pf in pred_files:
        try:
            rows.append(summarize_ticker(pf))
        except Exception as e:
            skipped += 1
            print("[SKIP] %s  -> %s" % (pf, str(e)))

    if len(rows) == 0:
        print("No valid tickers processed (all files skipped).")
        return

    ticker_df = pd.DataFrame(rows)

    # per-ticker table
    out_ticker = os.path.join(out_dir, "post_summary_by_ticker.csv")
    ticker_df.to_csv(out_ticker, index=False)

    # sector means
    num_cols = [c for c in ticker_df.columns if c not in ["Sector", "Ticker"]]
    sector_df = ticker_df.groupby("Sector", as_index=False)[num_cols].mean(numeric_only=True)
    out_sector = os.path.join(out_dir, "post_summary_by_sector.csv")
    sector_df.to_csv(out_sector, index=False)

    # quantiles / tails
    q_df = make_quantile_table(ticker_df, q=DECILES)
    out_q = os.path.join(out_dir, "post_quantiles_by_hmm_cumret.csv")
    q_df.to_csv(out_q, index=False)

    tail_df = tail_summary(ticker_df)
    out_tail = os.path.join(out_dir, "post_tail_summary.csv")
    tail_df.to_csv(out_tail, index=False)

    # impact
    cnt_df = impact_counts(ticker_df)
    out_cnt = os.path.join(out_dir, "post_impact_counts.csv")
    cnt_df.to_csv(out_cnt, index=False)

    print("\nSaved:")
    print(" - %s" % out_ticker)
    print(" - %s" % out_sector)
    print(" - %s" % out_q)
    print(" - %s" % out_tail)
    print(" - %s" % out_cnt)
    print("\nProcessed %d tickers, skipped %d files." % (len(rows), skipped))


if __name__ == "__main__":
    main()
