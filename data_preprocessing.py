import yfinance as yf
import json
import os

# load companies from JSON file
def load_companies_from_json(file_path):
    with open(file_path, 'r') as file:
        companies_by_sector = json.load(file)
    return companies_by_sector

# download stock data
def download_stock_data(tickers, start, end):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data is None or data.empty:
            print(f"Warning: {ticker} returned no data.")
            continue
        stock_data[ticker] = data.sort_index()
    return stock_data

# calculate missing data for all columns in stock data
def calculate_missing_values(stock_data):
    missing_values = {}
    for ticker, data in stock_data.items():
        na_count = data.isna().sum().sum()
        missing_values[ticker] = na_count
    return missing_values

# preprocess data - clean, add daily returns and moving averages
def preprocess_data(stock_data):
    preprocessed_data = {}
    for ticker, data in stock_data.items():
        data_cleaned = data.dropna().copy()
        
        data_cleaned['Daily_Return'] = data_cleaned['Adj Close'].pct_change()

        data_cleaned['MA50'] = data_cleaned['Adj Close'].rolling(window=50, min_periods=50).mean()
        data_cleaned['MA100'] = data_cleaned['Adj Close'].rolling(window=100, min_periods=100).mean()
        data_cleaned['MA200'] = data_cleaned['Adj Close'].rolling(window=200, min_periods=200).mean()

        data_cleaned = data_cleaned.dropna()

        preprocessed_data[ticker] = data_cleaned
    
    return preprocessed_data

# save data to CSV
def save_data(data, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for ticker, df in data.items():
        df.to_csv(os.path.join(folder, f"{ticker}.csv"))

#  save and create data
def data_preprocessing(json_file, start, end, folder):
    companies_by_sector = load_companies_from_json(json_file)

    # extract tickers from JSON
    tickers = [company['ticker'] for sector in companies_by_sector.values() for company in sector]

    # download and save raw data
    stock_data = download_stock_data(tickers, start, end)
    # print(stock_data.get('AAPL', '').head())

    # there is no missing data in any of the columns
    # missing_data = calculate_missing_values(stock_data)
    # print(missing_data)

    processed_stock_data = preprocess_data(stock_data)
    # print(processed_stock_data.get('AAPL', '').head())

    save_data(processed_stock_data, folder)


json_file = "companies.json"
start = "2019-08-01"
end = "2024-08-01"
folder = "data"

if __name__ == "__main__":
    data_preprocessing(json_file, start, end, folder)
