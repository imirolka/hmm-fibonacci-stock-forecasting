import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# globals
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# utils
def load_companies_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def read_ticker_csv(data_folder, ticker):
    csv_path = os.path.join(data_folder, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found for {ticker} -> {csv_path}")
        return None
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    if 'Daily_Return' not in df.columns and 'Adj Close' in df.columns:
        df['Daily_Return'] = df['Adj Close'].pct_change()
    if 'MA50' not in df.columns and 'Adj Close' in df.columns:
        df['MA50'] = df['Adj Close'].rolling(window=50, min_periods=50).mean()
    if 'MA100' not in df.columns and 'Adj Close' in df.columns:
        df['MA100'] = df['Adj Close'].rolling(window=100, min_periods=100).mean()
    if 'MA200' not in df.columns and 'Adj Close' in df.columns:
        df['MA200'] = df['Adj Close'].rolling(window=200, min_periods=200).mean()
    return df

def all_tickers_from_companies(companies_by_sector):
    return [c['ticker'] for sector in companies_by_sector.values() for c in sector]


# price trend for each ticker with MAs
def plot_price_trends(data_folder, companies_by_sector, out_folder):
    create_directory(out_folder)
    tickers = all_tickers_from_companies(companies_by_sector)
    for t in tickers:
        df = read_ticker_csv(data_folder, t)
        if df is None or df.empty or 'Adj Close' not in df.columns:
            print(f"Warning: {t} missing data for price trend. Skipping.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        df['Adj Close'].plot(ax=ax, label='Adj Close')
        if 'MA50' in df.columns:
            df['MA50'].plot(ax=ax, label='MA50', linestyle='-', alpha=0.9)
        if 'MA100' in df.columns:
            df['MA100'].plot(ax=ax, label='MA100', linestyle='--', alpha=0.9)
        if 'MA200' in df.columns:
            df['MA200'].plot(ax=ax, label='MA200', linestyle=':', alpha=0.9)

        ax.set_title(f"{t} | Price & MAs", loc='left')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

        fig.tight_layout()
        fp = os.path.join(out_folder, f"{t}_price_ma.png")
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"Saved: {fp}")


#sector performance comparison (rebased=100, top-N)
def _rebase_to_100(series):
    series = series.dropna()
    if series.empty:
        return series
    return (series / series.iloc[0]) * 100.0

def _sector_index_equal_weight(prices_df):
    if prices_df.empty:
        return prices_df.sum(axis=1)
    # rebase each ticker then equal-weight average
    rebased = prices_df.apply(_rebase_to_100)
    rebased = rebased.dropna(how='any')
    if rebased.empty:
        return rebased.sum(axis=1)
    sector_idx = rebased.mean(axis=1)
    # rebase again to 100 to be clean
    return _rebase_to_100(sector_idx)

def _right_label_positions(y_values, min_gap):
    adjusted = []
    for name, y in y_values:
        if not adjusted:
            adjusted.append((name, y))
        else:
            prev_y = adjusted[-1][1]
            y = max(y, prev_y + min_gap)
            adjusted.append((name, y))
    return adjusted

def plot_sector_performance_comparison(data_folder, companies_by_sector, out_folder,
                                       resample_monthly=True, top_n=6):
    create_directory(out_folder)

    sector_series = {}
    for sector, comps in companies_by_sector.items():
        tickers = [c['ticker'] for c in comps]
        price_cols = []
        for t in tickers:
            df = read_ticker_csv(data_folder, t)
            if df is None or df.empty or 'Adj Close' not in df.columns:
                continue
            price_cols.append(df['Adj Close'].rename(t))
        if not price_cols:
            continue
        prices = pd.concat(price_cols, axis=1)
        prices = prices.dropna(how='any')  # common dates
        if prices.empty:
            continue
        sector_idx = _sector_index_equal_weight(prices)
        sector_series[sector] = sector_idx

    if not sector_series:
        print("No sector data for performance comparison.")
        return

    df = pd.DataFrame(sector_series).dropna(how='any')
    if resample_monthly:
        df = df.resample('ME').last()

    # pick top N by final value
    last_vals = df.tail(1).T.sort_values(df.index[-1], ascending=False)
    keep = list(last_vals.index[:top_n])
    df = df[keep]

    # manual plotting cause labels were not labelling 
    fig, ax = plt.subplots(figsize=(12, 5))

    # extend for label space
    x0, x1 = df.index[0], df.index[-1]
    x_range = (x1 - x0)
    xpad = x_range * 0.10
    ax.set_xlim(x0, x1 + xpad)

    order = df.columns[::-1]
    for col in order:
        ax.plot(df.index, df[col].values, label=col, linewidth=1.6, alpha=0.95)

    ax.set_title("Sector Performance (Rebased = 100)", loc='left')
    ax.set_ylabel("Index (start=100)")
    ax.grid(True, alpha=0.25)

    y_last = df.tail(1).iloc[0]
    ymin, ymax = float(df.min().min()), float(df.max().max())
    min_gap = max(6.0, (ymax - ymin) * 0.05) if len(df.columns) > 1 else 0.0
    labels = sorted([(col, float(y_last[col])) for col in df.columns], key=lambda x: x[1])
    adjusted = _right_label_positions(labels, min_gap)

    top_overflow = adjusted[-1][1] - ymax
    if top_overflow > 0:
        adjusted = [(name, y - top_overflow) for name, y in adjusted]

    x_label = x1 + xpad * 0.92
    x_link  = x1 + xpad * 0.85
    for name, y_pos in adjusted:
        ax.plot([x1, x_link], [y_last[name], y_pos], linewidth=0.8, alpha=0.6)
        ax.text(x_label, y_pos, f"{name} ({y_last[name]:.1f})",
                va='center', ha='left', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.2),
                clip_on=False)

    ax.margins(x=0.02, y=0.05)
    fig.tight_layout()
    fp = os.path.join(out_folder, "sector_performance_rebased.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"Saved: {fp}")


# rolling volatility per ticker (annualized)
def plot_volatility(data_folder, companies_by_sector, out_folder, window=20):
    create_directory(out_folder)
    tickers = all_tickers_from_companies(companies_by_sector)
    for t in tickers:
        df = read_ticker_csv(data_folder, t)
        if df is None or df.empty or 'Daily_Return' not in df.columns:
            print(f"Warning: {t} missing returns for volatility.")
            continue
        vol = df['Daily_Return'].rolling(window, min_periods=window).std() * np.sqrt(252)
        if vol.dropna().empty:
            print(f"Warning: {t} has no rolling vol (window={window}).")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        vol.plot(ax=ax, label=f'Rolling {window}D Ann. Vol')

        ax.set_title(f"{t} | Rolling {window}D Annualized Volatility", loc='left')
        ax.grid(True, alpha=0.25)
        if vol.notna().any():
            med = vol.median()
            ax.axhline(med, linestyle='--', linewidth=1, alpha=0.6, label=f"Median: {med:.2%}")
            ax.legend(frameon=False)

        fig.tight_layout()
        fp = os.path.join(out_folder, f"{t}_rolling_vol_{window}d.png")
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f"Saved: {fp}")


#correlation matrices
def _collect_prices_for_tickers(data_folder, tickers):
    cols = []
    for t in tickers:
        df = read_ticker_csv(data_folder, t)
        if df is None or df.empty or 'Adj Close' not in df.columns:
            continue
        cols.append(df['Adj Close'].rename(t))
    if not cols:
        return pd.DataFrame()
    prices = pd.concat(cols, axis=1)
    prices = prices.dropna(how='any')
    return prices

def _corr_from_prices(prices):
    if prices.empty:
        return pd.DataFrame()
    logret = np.log(prices).diff().dropna(how='any')
    return logret.corr()

def _n_obs_from_prices(prices):
    if prices.empty:
        return 0
    return np.log(prices).diff().dropna(how='any').shape[0]

def _order_tickers_by_sector(companies_by_sector, tickers):
    # preserve sector grouping order in companies_by_sector
    order = []
    seen = set()
    for sector, comps in companies_by_sector.items():
        for c in comps:
            t = c.get('ticker')
            if t in tickers and t not in seen:
                order.append(t)
                seen.add(t)
    for t in tickers:
        if t not in seen:
            order.append(t)
    return order

def _save_corr_heatmap(corr, out_path, title):
    if corr.empty:
        print(f"Warning: empty correlation for {title}.")
        return

    n = corr.shape[0]
    show_full = n <= 8  # annotate small matrices

    fig_size = max(8, min(0.45 * n + 4, 18))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if show_full:
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='coolwarm', origin='upper', interpolation='none', aspect='equal')
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color='black')
    else:
        # mask upper triangle for large matrices
        mask = np.triu(np.ones_like(corr, dtype=bool))
        data = np.ma.array(corr.values, mask=mask)
        cmap = plt.get_cmap('coolwarm').copy()
        cmap.set_bad(color='white')
        im = ax.imshow(data, vmin=-1, vmax=1, cmap=cmap, origin='upper', interpolation='none', aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    ax.set_title(title, loc='left')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

def generate_correlation_matrix(data_folder, companies_by_sector, out_folder):
    create_directory(out_folder)
    tickers = all_tickers_from_companies(companies_by_sector)
    prices = _collect_prices_for_tickers(data_folder, tickers)
    if prices.empty:
        print("No prices available for correlation matrix.")
        return
    tick_order = _order_tickers_by_sector(companies_by_sector, list(prices.columns))
    prices = prices[tick_order]
    corr = _corr_from_prices(prices)
    n_obs = _n_obs_from_prices(prices)
    out_path = os.path.join(out_folder, "correlation_all_tickers.png")
    _save_corr_heatmap(corr, out_path, f"Ticker Correlation (log returns, common dates, N={n_obs})")

def generate_sector_correlation(data_folder, companies_by_sector, out_folder):
    create_directory(out_folder)
    for sector, comps in companies_by_sector.items():
        tickers = [c['ticker'] for c in comps]
        if len(tickers) < 2:
            continue
        prices = _collect_prices_for_tickers(data_folder, tickers)
        if prices.empty or prices.shape[1] < 2:
            continue
        corr = _corr_from_prices(prices)
        n_obs = _n_obs_from_prices(prices)
        out_path = os.path.join(out_folder, f"corr_{sector.replace(' ', '_')}.png")
        _save_corr_heatmap(corr, out_path, f"{sector} | Correlation (log returns, N={n_obs})")

def summarize_sector_correlation(data_folder, companies_by_sector, out_csv):
    import itertools
    rows = []
    for sector, comps in companies_by_sector.items():
        tickers = [c['ticker'] for c in comps]
        if len(tickers) < 2:
            continue
        prices = _collect_prices_for_tickers(data_folder, tickers)
        if prices.empty or prices.shape[1] < 2:
            continue
        corr = _corr_from_prices(prices)
        # flatten pairwise (i<j)
        vals = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                vals.append((cols[i], cols[j], float(corr.iloc[i, j])))
        if not vals:
            continue
        mean_corr = sum(v for _, _, v in vals) / len(vals)
        min_pair = min(vals, key=lambda x: x[2])
        max_pair = max(vals, key=lambda x: x[2])
        rows.append({
            "sector": sector,
            "n_tickers": len(tickers),
            "n_pairs": len(vals),
            "mean_corr": mean_corr,
            "min_pair": f"{min_pair[0]}-{min_pair[1]}",
            "min_corr": min_pair[2],
            "max_pair": f"{max_pair[0]}-{max_pair[1]}",
            "max_corr": max_pair[2],
        })
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")


def create_plots(json_file, data_folder, price_trend_folder, sector_performance_folder,
                 volatility_folder, correlation_matrix_folder):
    companies_by_sector = load_companies_from_json(json_file)

    # 1. per-ticker price trend
    plot_price_trends(data_folder, companies_by_sector, price_trend_folder)

    # 2. sector performance (rebased=100, top N)
    plot_sector_performance_comparison(data_folder, companies_by_sector, sector_performance_folder)

    # 3. rolling volatility per ticker
    plot_volatility(data_folder, companies_by_sector, volatility_folder)

    # 4. correlation matrices
    generate_correlation_matrix(data_folder, companies_by_sector, correlation_matrix_folder)
    generate_sector_correlation(data_folder, companies_by_sector, correlation_matrix_folder)
    summarize_sector_correlation(data_folder, companies_by_sector,
                                 os.path.join(correlation_matrix_folder, "sector_corr_summary.csv"))


json_file = "companies.json"
data_folder = "data"
price_trend_folder = "plots/price_trends"
sector_performance_folder = "plots/sector_performance_comparison"
volatility_folder = "plots/volatility"
correlation_matrix_folder = "plots/correlation_matrix"

if __name__ == "__main__":
    create_plots(json_file, data_folder, price_trend_folder, sector_performance_folder,
                 volatility_folder, correlation_matrix_folder)
