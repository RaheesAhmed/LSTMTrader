import ccxt
import pandas as pd
import datetime

def fetch_binance_data(symbol="ASTR/USDT", timeframe="1h", since_days=30):
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=since_days)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Fetch and save data
symbol = "ASTR/USDT"
timeframe = "1h"
data = fetch_binance_data(symbol, timeframe)
data.to_csv(f"{symbol.replace('/', '_')}_data.csv", index=False)
print(f"Data for {symbol} saved to CSV!")
