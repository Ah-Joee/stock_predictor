import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import logging
import asyncio
import aiohttp
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
DATE = pd.Timestamp('today').date()


async def fetch_ticker_data(session, ticker):
    # Request data for the last 200 trading days (approximately 1 year)
    end = int(time.time())
    start = end - (86400 * 365)  # 86400 seconds per day * 365 days
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d"
    async with session.get(url) as response:
        return await response.json()

async def fetch_all_tickers(tickers):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(session, ticker) for ticker in tickers]
        return await asyncio.gather(*tasks)

def process_single_ticker(ticker_data):

    try:
        if not ticker_data or 'chart' not in ticker_data or 'result' not in ticker_data['chart'] or not ticker_data['chart']['result']:
            logging.error(f"Invalid data structure received from API")
            return None

        ticker = ticker_data['chart']['result'][0]['meta']['symbol']
        hist = pd.DataFrame(ticker_data['chart']['result'][0]['indicators']['quote'][0])
        hist.index = pd.to_datetime(ticker_data['chart']['result'][0]['timestamp'], unit='s')
        
        info = yf.Ticker(ticker).info
        financial = yf.Ticker(ticker).financials.transpose()
        
        # Calculate daily financial metrics
        metrics = pd.DataFrame(index=[0])
        metrics['Date'] = str(DATE)
        metrics['Ticker'] = ticker
        metrics['Open'] = hist['open'].iloc[-1]
        metrics['Close'] = hist['close'].iloc[-1]
        metrics['High'] = hist['high'].iloc[-1]
        metrics['Low'] = hist['low'].iloc[-1]
        metrics['Volume'] = hist['volume'].iloc[-1]
        
        metrics['PE_Ratio'] = calculate_pe_ratio(info)
        metrics['ROE'] = calculate_roe(info)
        metrics['EPS_Growth'] = calculate_eps_growth(ticker)
        metrics['PB_Ratio'] = calculate_pb_ratio(info, metrics['Close'].iloc[0])
        metrics['Quick_Ratio'] = calculate_quick_ratio(financial)
        metrics['Debt_to_Equity'] = calculate_debt_to_equity(info)
        metrics['EBITDA'] = calculate_ebitda(financial) 
        metrics['Net_Profit_Margin'] = calculate_net_profit_margin(financial) 
        metrics['Dividend_Yield'] = calculate_dividend_yield(info)
        metrics['Beta'] = calculate_beta(info)
        metrics['RSI'] = calculate_rsi(hist)
        ma_50, ma_200 = calculate_moving_averages(hist)
        metrics['MA_50'] = ma_50
        metrics['MA_200'] = ma_200
        metrics['MACD'], metrics['Signal_Line'] = calculate_macd(hist)
        metrics['Upper_Band'], metrics['Lower_Band'] = calculate_bollinger_bands(hist)
        
        return metrics

    except Exception as e:
        logging.error(f"Error processing {ticker}: {str(e)}")
        return None

def calculate_pe_ratio(info):
    return info.get('forwardPE', None)

def calculate_roe(info):
    return info.get('returnOnEquity', None)

def calculate_eps_growth(ticker):
    df = yf.Ticker(ticker).financials.transpose() 
    return (df['Diluted EPS'].iloc[0] / df['Diluted EPS'].iloc[1] - 1) * 100 if len(df) >= 2 else None

def calculate_pb_ratio(info, price):
    book_value_per_share = info.get('bookValue', None)
    return price / book_value_per_share if book_value_per_share else None

def calculate_quick_ratio(financials):
    try:
        gross_profit = financials['Gross Profit'].iloc[0]
        total_operating_income = financials['Total Operating Income As Reported'].iloc[0]
        quick_assets = gross_profit
        operating_expenses = total_operating_income - gross_profit
        current_liabilities = operating_expenses
        quick_ratio = quick_assets / current_liabilities
        return quick_ratio
    except:
        return None

def calculate_debt_to_equity(info):
    return info.get('debtToEquity', None)

def calculate_ebitda(financial):
    return financial['EBITDA'].iloc[0] if 'EBITDA' in financial else None

def calculate_net_profit_margin(financial):
    if 'Net Income' in financial and 'Total Revenue' in financial:
        return financial['Net Income'].iloc[0] / financial['Total Revenue'].iloc[0]
    return None

def calculate_dividend_yield(info):
    return info.get('dividendYield', 0.0)

def calculate_beta(info):
    return info.get('beta', None)

def calculate_rsi(hist, period=14):
    delta = hist['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_moving_averages(hist):
    ma_50 = hist['close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
    ma_200 = hist['close'].rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else None
    return ma_50, ma_200

def calculate_macd(hist, short_period=12, long_period=26, signal_period=9):
    short_ema = hist['close'].ewm(span=short_period, adjust=False).mean()
    long_ema = hist['close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_bollinger_bands(hist, period=20, num_std_dev=2):
    middle_band = hist['close'].rolling(window=period).mean()
    std_dev = hist['close'].rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)
    return upper_band.iloc[-1], lower_band.iloc[-1]

async def fetch_and_preprocess_data(tickers):
    ticker_data = await fetch_all_tickers(tickers)
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_single_ticker, ticker_data))
    
    combined_metrics = pd.concat([result for result in results if result is not None], ignore_index=True)
    return combined_metrics

def train_model(ticker_data):
    features = ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = (ticker_data['Close'].shift(-1) > ticker_data['Close']).astype(int)
    train_size = len(features) - 21
    X_train, y_train = features[:train_size], target[:train_size]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(ticker, predictions, actual):
    return accuracy_score(actual, predictions)