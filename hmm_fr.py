import os
import json
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import erf

# 1. loading data
def load_companies_from_json(json_path="companies.json"):
    with open(json_path, "r") as f:
        companies_by_sector = json.load(f)
    return companies_by_sector


def load_preprocessed_csv_for_ticker(data_folder, ticker):
    csv_path = os.path.join(data_folder, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    return df


# 2. HMM feature matrix X 
def augment_features(df):
    required = [
        "Open", "Close", "Adj Close",
        "Daily_Return", "MA50", "MA100", "MA200",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in preprocessed data: {missing}")

    data = df[required].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # intraday move:  (Close - Open) / Open
    open_prices = data["Open"].values
    close_prices = data["Close"].values
    frac_clop = np.where(open_prices != 0, (close_prices - open_prices) / open_prices, 0.0)

    # gaps to moving averages 
    adj = data["Adj Close"].values
    eps = 1e-8  # pamietaj cholero nie dziel przez zero :)
    ma50_gap = (adj - data["MA50"].values) / (adj + eps)
    ma100_gap = (adj - data["MA100"].values) / (adj + eps)
    ma200_gap = (adj - data["MA200"].values) / (adj + eps)

    feat_df = pd.DataFrame(
        {
            "Daily_Return": data["Daily_Return"].values, 
            "CloseOpen": frac_clop,
            "MA50_Gap": ma50_gap,
            "MA100_Gap": ma100_gap,
            "MA200_Gap": ma200_gap,
        },
        index=data.index,
    )

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feat_df


def split_train_test(data, train_ratio=0.8):
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    return train_data, test_data



# 3. HMM training and one-step forecast
def train_hmm(feature_df, n_components=3):
    # X shape: (n_days, n_features)
    X = feature_df.values
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) == 0:
        raise ValueError("Training data contains only NaN or infinite values after cleaning.")

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=500,
        tol=1e-3,
    )

    model.fit(X)
    return model


def gaussian_cdf(x, mean, std):
    x = float(np.asarray(x).reshape(-1)[0])
    mean = float(np.asarray(mean).reshape(-1)[0])
    std = float(np.asarray(std).reshape(-1)[0])

    if std <= 0.0:
        return 1.0 if x >= mean else 0.0

    z = (x - mean) / (std * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def hmm_one_step_forecast(model, features, prices, t, lookback=50,
                          prob_buy=0.55, prob_sell=0.45):
    # history window
    start_idx = max(0, t - lookback + 1)
    hist_X = features.iloc[start_idx:t+1].values

    # distribution over states at time
    gamma_t = model.predict_proba(hist_X)[-1]

    # state distribution at t+1
    next_state_probs = np.dot(gamma_t, model.transmat_)

    next_state_probs = np.clip(next_state_probs, 0.0, 1.0)
    total = next_state_probs.sum()
    if total <= 0:
        n_comp = model.n_components
        next_state_probs = np.full(n_comp, 1.0 / n_comp)
    else:
        next_state_probs /= total

    state_means = model.means_[:, 0]
    state_stds = np.sqrt(model.covars_[:, 0])

    # expected next-day return
    expected_return = float(np.dot(next_state_probs, state_means))

    # probability next return positive
    p_up_components = []
    for mu, sigma in zip(state_means, state_stds):
        p_up_comp = 1.0 - gaussian_cdf(0.0, mu, sigma)
        p_up_components.append(p_up_comp)
    p_up_components = np.array(p_up_components)
    p_up = float(np.dot(next_state_probs, p_up_components))

    # signals
    if p_up >= prob_buy:
        signal = "Buy"
    elif p_up <= prob_sell:
        signal = "Sell"
    else:
        signal = "Hold"

    last_close = prices["Close"].iloc[t]
    pred_close = last_close * (1.0 + expected_return)

    return {
        "expected_return": expected_return,
        "p_up": p_up,
        "signal_hmm": signal,
        "predicted_close_hmm": pred_close,
    }


# 4. fibonacci
def calculate_fibonacci_levels(price_window):
    high_price = float(price_window["High"].max())
    low_price = float(price_window["Low"].min())
    if not np.isfinite(high_price) or not np.isfinite(low_price) or high_price == low_price:
        return None

    diff = high_price - low_price
    levels = {
        "0.0%": high_price,
        "23.6%": high_price - 0.236 * diff,
        "38.2%": high_price - 0.382 * diff,
        "50.0%": high_price - 0.500 * diff,
        "61.8%": high_price - 0.618 * diff,
        "78.6%": high_price - 0.786 * diff,
        "100.0%": low_price,
    }
    return levels


def apply_fibonacci_confirmation(signal, predicted_price, fib_levels,
                                 tolerance=0.02):
    if fib_levels is None or signal == "Hold":
        return signal, predicted_price, None

    if signal == "Buy":
        candidate_levels = ["61.8%", "78.6%", "100.0%"]
    elif signal == "Sell":
        candidate_levels = ["0.0%", "23.6%", "38.2%"]
    else:
        return signal, predicted_price, None

    chosen_level = None
    adjusted_price = predicted_price

    for level_name in candidate_levels:
        level_price = fib_levels[level_name]
        if not np.isfinite(level_price):
            continue
        if abs(predicted_price - level_price) / level_price <= tolerance:
            adjusted_price = level_price
            chosen_level = level_name
            break

    if chosen_level is None:
        # reject
        return "Hold", predicted_price, None
    else:
        return signal, adjusted_price, chosen_level


# 5. metrics, backtesting
def calculate_regression_metrics(actual, predicted):
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)
    return rmse, mape


def directional_accuracy(actual_returns, predicted_returns):
    actual_sign = np.sign(actual_returns)
    pred_sign = np.sign(predicted_returns)
    mask = actual_sign != 0
    if mask.sum() == 0:
        return np.nan
    return float((actual_sign[mask] == pred_sign[mask]).mean())


def backtest_ticker(data, ticker, train_ratio=0.8,
                    n_components=3, lookback_hmm=50, lookback_fib=30,
                    prob_buy=0.55, prob_sell=0.45,
                    fib_tolerance=0.02):
    required_cols = [
        "Open", "High", "Low", "Close", "Adj Close",
        "Daily_Return", "MA50", "MA100", "MA200",
    ]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"{ticker}: missing required preprocessed columns: {missing}")

    # keep only rows where all key cols are available
    data = data[required_cols].dropna()

    if len(data) < lookback_hmm + 2:
        raise ValueError(f"Not enough preprocessed data for {ticker} after cleaning.")

    # build feature matrix
    features = augment_features(data)

    # price series
    prices = data[["Open", "High", "Low", "Close"]].loc[features.index]

    train_prices, test_prices = split_train_test(prices, train_ratio=train_ratio)
    train_features, test_features = split_train_test(features, train_ratio=train_ratio)

    model = train_hmm(train_features, n_components=n_components)

    full_features = features
    full_prices = prices
    split_idx = len(train_features)

    rows = []

    # predict day t+1, so t runs until len-2
    for t in tqdm(range(split_idx - 1, len(full_features) - 1),
                  desc=f"Backtesting {ticker}"):
        forecast = hmm_one_step_forecast(
            model, full_features, full_prices, t,
            lookback=lookback_hmm,
            prob_buy=prob_buy,
            prob_sell=prob_sell,
        )

        # actual next-day values
        date_next = full_prices.index[t + 1]
        last_close = full_prices["Close"].iloc[t]
        actual_close = full_prices["Close"].iloc[t + 1]
        actual_return = (actual_close - last_close) / last_close

        # apply fibonacci on recent price window ending at t
        start_fib = max(0, t + 1 - lookback_fib)
        fib_window = full_prices.iloc[start_fib:t+1]
        fib_levels = calculate_fibonacci_levels(fib_window)

        signal_fr, pred_close_fr, fib_level_used = apply_fibonacci_confirmation(
            forecast["signal_hmm"],
            forecast["predicted_close_hmm"],
            fib_levels,
            tolerance=fib_tolerance,
        )

        pred_return_hmm = (forecast["predicted_close_hmm"] - last_close) / last_close
        pred_return_fr = (pred_close_fr - last_close) / last_close

        pos_hmm = 1 if forecast["signal_hmm"] == "Buy" else -1 if forecast["signal_hmm"] == "Sell" else 0
        pos_fr = 1 if signal_fr == "Buy" else -1 if signal_fr == "Sell" else 0

        strat_ret_hmm = pos_hmm * actual_return
        strat_ret_fr = pos_fr * actual_return

        rows.append({
            "Date": date_next,
            "LastClose": last_close,
            "ActualClose": actual_close,
            "ActualReturn": actual_return,
            "PredictedClose_HMM": forecast["predicted_close_hmm"],
            "PredictedClose_HMM_FR": pred_close_fr,
            "PredictedReturn_HMM": pred_return_hmm,
            "PredictedReturn_HMM_FR": pred_return_fr,
            "P_up": forecast["p_up"],
            "Signal_HMM": forecast["signal_hmm"],
            "Signal_HMM_FR": signal_fr,
            "FibLevelUsed": fib_level_used,
            "Position_HMM": pos_hmm,
            "Position_HMM_FR": pos_fr,
            "StrategyReturn_HMM": strat_ret_hmm,
            "StrategyReturn_HMM_FR": strat_ret_fr,
        })

    results = pd.DataFrame(rows).set_index("Date")

    # restrict metrics to pure test period
    test_results = results.loc[test_prices.index.intersection(results.index)]

    # price-level metrics
    rmse_hmm, mape_hmm = calculate_regression_metrics(
        test_results["ActualClose"], test_results["PredictedClose_HMM"]
    )
    rmse_fr, mape_fr = calculate_regression_metrics(
        test_results["ActualClose"], test_results["PredictedClose_HMM_FR"]
    )

    # directional prediction accuracy
    dpa_hmm = directional_accuracy(
        test_results["ActualReturn"].values,
        test_results["PredictedReturn_HMM"].values,
    )
    dpa_fr = directional_accuracy(
        test_results["ActualReturn"].values,
        test_results["PredictedReturn_HMM_FR"].values,
    )

    # simple trading performance
    cum_return_hmm = float((1.0 + test_results["StrategyReturn_HMM"]).prod() - 1.0)
    cum_return_fr = float((1.0 + test_results["StrategyReturn_HMM_FR"]).prod() - 1.0)

    metrics = {
        "rmse_hmm": rmse_hmm,
        "mape_hmm": mape_hmm,
        "rmse_hmm_fr": rmse_fr,
        "mape_hmm_fr": mape_fr,
        "dpa_hmm": dpa_hmm,
        "dpa_hmm_fr": dpa_fr,
        "cum_return_hmm": cum_return_hmm,
        "cum_return_hmm_fr": cum_return_fr,
    }

    return results, metrics


# 6. helpers
def plot_all_predictions(results_df, output_folder, ticker):
    plt.figure(figsize=(15, 6))
    plt.plot(results_df.index, results_df["ActualClose"], label="Actual Close", linewidth=1.5)
    plt.plot(results_df.index, results_df["PredictedClose_HMM"], label="Predicted (HMM)", linestyle="--")
    plt.plot(results_df.index, results_df["PredictedClose_HMM_FR"], label="Predicted (HMM + Fibonacci)", linestyle=":")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Stock Price Prediction for {ticker}: HMM vs HMM + Fibonacci")
    os.makedirs(output_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{ticker}_prediction_plot.png"))
    plt.close()


def save_metrics(metrics, output_folder, ticker):
    os.makedirs(output_folder, exist_ok=True)
    metrics_path = os.path.join(output_folder, f"{ticker}_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def save_summary_tables(summary_df, results_folder):
    os.makedirs(results_folder, exist_ok=True)

    # ticker-level metrics
    summary_csv_path = os.path.join(results_folder, "summary_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # sector-level averages
    numeric_cols = [c for c in summary_df.columns
                    if c not in ["Sector", "Company", "Ticker"]]
    agg_dict = {c: "mean" for c in numeric_cols}

    sector_summary = summary_df.groupby("Sector", as_index=False).agg(agg_dict)

    sector_csv_path = os.path.join(results_folder, "summary_by_sector.csv")
    sector_summary.to_csv(sector_csv_path, index=False)

    print("\nSaved summary metrics to:")
    print(f"  - {summary_csv_path}")
    print(f"  - {sector_csv_path}")

    return sector_summary


def plot_sector_bars(summary_df, results_folder):
 
    if summary_df.empty:
        return

    sectors = summary_df["Sector"].unique()

    for sector in sectors:
        sub = summary_df[summary_df["Sector"] == sector].copy()
        if sub.empty:
            continue

        sector_folder = os.path.join(results_folder, sector.replace(" ", "_"))
        os.makedirs(sector_folder, exist_ok=True)

        tickers = sub["Ticker"].tolist()
        x = np.arange(len(tickers))
        width = 0.35

        # directional accuracy
        plt.figure(figsize=(10, 6))
        dpa_hmm = sub["dpa_hmm"].values
        dpa_fr = sub["dpa_hmm_fr"].values

        plt.bar(x - width / 2, dpa_hmm, width, label="HMM")
        plt.bar(x + width / 2, dpa_fr, width, label="HMM + Fibonacci")

        plt.xticks(x, tickers, rotation=45, ha="right")
        plt.ylabel("Directional accuracy")
        plt.ylim(0.0, 1.0)
        plt.title(f"Directional accuracy by ticker - {sector}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sector_folder, "sector_dpa_comparison.png"))
        plt.close()

        # cumulative return
        plt.figure(figsize=(10, 6))
        cr_hmm = sub["cum_return_hmm"].values
        cr_fr = sub["cum_return_hmm_fr"].values

        plt.bar(x - width / 2, cr_hmm, width, label="HMM")
        plt.bar(x + width / 2, cr_fr, width, label="HMM + Fibonacci")

        plt.xticks(x, tickers, rotation=45, ha="right")
        plt.ylabel("Cumulative return")
        plt.title(f"Strategy cumulative return by ticker - {sector}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sector_folder, "sector_cumreturn_comparison.png"))
        plt.close()


# main
if __name__ == "__main__":
    data_folder = "data"
    results_folder = "results"
    companies_json_path = "companies.json"

    os.makedirs(results_folder, exist_ok=True)

    companies_by_sector = load_companies_from_json(companies_json_path)

    # summary across all tickers
    summary_rows = []

    for sector, companies in companies_by_sector.items():
        print(f"\n=== Sector: {sector} ===")

        for company in companies:
            name = company["name"]
            ticker = company["ticker"]

            print(f"Processing {ticker} ({name})...")

            df = load_preprocessed_csv_for_ticker(data_folder, ticker)
            if df is None or df.empty:
                print(f"  -> Skipping {ticker}: no preprocessed CSV found or empty.")
                continue

            sector_folder = os.path.join(results_folder, sector.replace(" ", "_"))
            ticker_folder = os.path.join(sector_folder, ticker)

            try:
                results_df, metrics = backtest_ticker(df, ticker)
            except Exception as e:
                print(f"  -> Error processing {ticker}: {e}")
                continue

            # save results
            os.makedirs(ticker_folder, exist_ok=True)
            results_df.to_csv(os.path.join(ticker_folder, f"{ticker}_predictions.csv"))

            # save metrics and plot
            save_metrics(metrics, ticker_folder, ticker)
            plot_all_predictions(results_df, ticker_folder, ticker)

            row = {
                "Sector": sector,
                "Company": name,
                "Ticker": ticker,
            }
            row.update(metrics)
            summary_rows.append(row)

            # console
            print(f"  -> Metrics for {ticker}:")
            for k, v in metrics.items():
                print(f"     {k}: {v}")

    # save summary metrics, tables, plots
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        sector_summary = save_summary_tables(summary_df, results_folder)
        plot_sector_bars(summary_df, results_folder)
    else:
        print("\nNo successful results to summarize.")
