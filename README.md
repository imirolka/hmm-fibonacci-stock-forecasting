# HMM + Fibonacci Retracements (thesis code)

Python code for **one-day-ahead stock forecasting** and **Buy/Sell/Hold signal generation** using a **Gaussian Hidden Markov Model (HMM)**, with and without **Fibonacci retracement (FR)** confirmation layer. Tickers are defined in `companies.json`.

## Files
- `companies.json` – sectors + company names + tickers
- `data_preprocessing.py` – downloads data (Yahoo Finance) + computes `Daily_Return` and MA(50/100/200), saves to `data/`
- `hmm_fr.py` – trains HMM, generates signals, runs backtests (HMM vs HMM+FR), saves to `results/`
- `data_plots.py` – EDA plots (price+MAs, sector performance, volatility, correlations) → `plots/`
- `post_processing.py` – aggregates results from `*_predictions.csv` → `post_results/`


pip install yfinance numpy pandas matplotlib hmmlearn scikit-learn tqdm
