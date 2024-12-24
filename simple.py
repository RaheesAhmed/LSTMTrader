import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import optuna
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    rolling_mean = pd.Series(prices).rolling(window=window).mean()
    rolling_std = pd.Series(prices).rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_volume_indicators(volume, close, period=20):
    """Calculate volume-based indicators."""
    volume = pd.Series(volume)
    close = pd.Series(close)
    
    # Volume EMA
    vol_ema = volume.ewm(span=period, adjust=False).mean()
    
    # On-Balance Volume (OBV)
    obv = (np.sign(close.diff()) * volume).cumsum()
    
    return vol_ema, obv

def calculate_rsi(prices, periods=14):
    deltas = np.diff(prices)
    seed = deltas[:periods+1]
    up = seed[seed >= 0].sum()/periods
    down = -seed[seed < 0].sum()/periods
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:periods] = 100. - 100./(1. + rs)
    
    for i in range(periods, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up*(periods-1) + upval)/periods
        down = (down*(periods-1) + downval)/periods
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def calculate_ema(prices, span=20):
    return pd.Series(prices).ewm(span=span, adjust=False).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    fast_ema = pd.Series(prices).ewm(span=fast, adjust=False).mean()
    slow_ema = pd.Series(prices).ewm(span=slow, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ====================================
# 1) Fetch Data + Indicators
# ====================================
def fetch_binance_data(symbol="ASTR/USDT", timeframe="1h", since_days=30, testnet=False):
    """Enhanced data fetching with testnet support."""
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'} if testnet else {}
    })
    
    if testnet:
        exchange.set_sandbox_mode(True)
    
    since = exchange.parse8601((datetime.now() - timedelta(days=since_days)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate all technical indicators
    df['rsi'] = calculate_rsi(df['close'].values)
    df['ema_20'] = calculate_ema(df['close'].values)
    macd, signal = calculate_macd(df['close'].values)
    df['macd'] = macd
    df['macd_signal'] = signal
    
    # New indicators
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'].values)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    vol_ema, obv = calculate_volume_indicators(df['volume'], df['close'])
    df['volume_ema'] = vol_ema
    df['obv'] = obv
    
    # Drop any NaN values
    df = df.dropna()
    
    return df

# ====================================
# 2) Dataset
# ====================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_out = torch.sum(attention_weights * hidden_states, dim=1)
        return attended_out, attention_weights

class EnhancedGRU(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        attended_out, _ = self.attention(out)
        out = self.fc1(attended_out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)

class PriceDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        self.feats = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_20', 
                     'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 
                     'atr', 'volume_ema']
        # Scale data
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(df[self.feats])
        self.data = pd.DataFrame(scaled, index=df.index, columns=self.feats)
        # Build sequences
        self.seqs, self.targets = [], []
        for i in range(len(self.data)-seq_len):
            self.seqs.append(self.data.iloc[i:i+seq_len].values)
            self.targets.append(self.data.iloc[i+seq_len]['close'])
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.seqs[idx])
        y = torch.FloatTensor([self.targets[idx]])
        return x, y

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    
    # Create model with suggested hyperparameters
    model = EnhancedGRU(
        input_size=14,  # Updated for new features
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train and validate
    model.train()
    for epoch in range(5):  # Quick training for optimization
        train_loss = 0
        for X, y in train_loader:  # Assuming global train_loader
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:  # Assuming global val_loader
            X, y = X.to(device), y.to(device)
            output = model(X)
            val_loss += criterion(output, y).item()
    
    return val_loss / len(val_loader)

def backtest(model, dataset, threshold_multiplier=1.5):
    """Enhanced backtesting with dynamic thresholds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    capital, position = 1000.0, 0.0
    trades = []
    df_scaled = dataset.data
    seq_len = dataset.seq_len
    feats = dataset.feats
    
    # Track metrics
    max_capital = capital
    max_drawdown = 0
    winning_trades = 0
    total_trades = 0
    
    with torch.no_grad():
        for i in range(seq_len, len(df_scaled)-1):
            # Get current market state
            cur_close_scaled = df_scaled.iloc[i]['close']
            cur_atr_scaled = df_scaled.iloc[i]['atr']
            
            # Dynamic threshold based on ATR
            threshold = cur_atr_scaled * threshold_multiplier
            
            seq = df_scaled.iloc[i-seq_len:i].values
            seq_torch = torch.FloatTensor(seq).unsqueeze(0).to(device)
            pred_scaled = model(seq_torch).item()
            
            # Inverse transform prices
            dummy = np.zeros((1, len(feats)))
            dummy[0, feats.index('close')] = pred_scaled
            pred_price = dataset.scaler.inverse_transform(dummy)[0, feats.index('close')]
            
            dummy[0, feats.index('close')] = cur_close_scaled
            cur_price = dataset.scaler.inverse_transform(dummy)[0, feats.index('close')]
            
            # Calculate price change percentage
            price_change_pct = (pred_price - cur_price) / cur_price
            
            # Enhanced trading logic with multiple indicators
            rsi = df_scaled.iloc[i]['rsi']
            macd = df_scaled.iloc[i]['macd']
            macd_signal = df_scaled.iloc[i]['macd_signal']
            bb_upper = df_scaled.iloc[i]['bb_upper']
            bb_lower = df_scaled.iloc[i]['bb_lower']
            
            # Buy conditions: predicted up + oversold + MACD crossover + near lower BB
            buy_signal = (
                price_change_pct > threshold and 
                rsi < 0.3 and  # RSI < 30 (scaled)
                macd > macd_signal and
                cur_close_scaled <= bb_lower
            )
            
            # Sell conditions: predicted down + overbought + MACD crossover + near upper BB
            sell_signal = (
                price_change_pct < -threshold and 
                rsi > 0.7 and  # RSI > 70 (scaled)
                macd < macd_signal and
                cur_close_scaled >= bb_upper
            )
            
            # Execute trades with position sizing based on ATR
            if buy_signal and position == 0:
                # Position size based on ATR
                risk_per_trade = capital * 0.02  # 2% risk per trade
                position_size = risk_per_trade / (cur_price * cur_atr_scaled)
                
                trade_cost = capital * 0.001  # 0.1% transaction cost
                capital -= trade_cost
                position = position_size
                capital = 0
                trades.append(("BUY", i, cur_price, trade_cost))
                total_trades += 1
                
            elif sell_signal and position > 0:
                capital = position * cur_price
                trade_cost = capital * 0.001
                capital -= trade_cost
                position = 0
                trades.append(("SELL", i, cur_price, trade_cost))
                total_trades += 1
                
                if capital > 1000:
                    winning_trades += 1
            
            # Track metrics
            current_value = capital + (position * cur_price if position > 0 else 0)
            max_capital = max(max_capital, current_value)
            current_drawdown = (max_capital - current_value) / max_capital
            max_drawdown = max(max_drawdown, current_drawdown)
    
    # Final exit
    if position > 0:
        final_scaled = df_scaled.iloc[-1]['close']
        dummy[0, feats.index('close')] = final_scaled
        final_price = dataset.scaler.inverse_transform(dummy)[0, feats.index('close')]
        capital = position * final_price
        trade_cost = capital * 0.001
        capital -= trade_cost
        trades.append(("FINAL_SELL", len(df_scaled)-1, final_price, trade_cost))
    
    # Calculate metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((capital - 1000) / 1000 * 100)
    sharpe_ratio = calculate_sharpe_ratio(trades, capital)
    
    print(f"\nBacktest Results:")
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return capital, trades, {
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def calculate_sharpe_ratio(trades, final_capital, risk_free_rate=0.02):
    """Calculate Sharpe Ratio for the trading strategy."""
    if not trades:
        return 0
    
    # Calculate daily returns
    returns = []
    initial_capital = 1000
    current_capital = initial_capital
    
    for trade in trades:
        if trade[0] == "SELL" or trade[0] == "FINAL_SELL":
            returns.append((trade[2] - current_capital) / current_capital)
        current_capital = trade[2]
    
    if not returns:
        return 0
    
    # Calculate annualized Sharpe Ratio
    returns = np.array(returns)
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if len(excess_returns) < 2:
        return 0
    
    sharpe_ratio = np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
    return sharpe_ratio

def main():
    # Test on multiple symbols
    symbols = ["ETH/USDT", "BTC/USDT", "BNB/USDT"]
    timeframes = ["1h", "4h"]
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nTesting {symbol} on {timeframe} timeframe")
            
            # Fetch data
            df = fetch_binance_data(symbol, timeframe, since_days=90)
            print(f"Data rows: {len(df)}")
            
            # Create dataset
            dataset = PriceDataset(df, seq_len=50)
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_data, val_data, test_data = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)
            test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
            
            # Hyperparameter optimization
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=20)
            
            # Create model with best parameters
            best_params = study.best_params
            model = EnhancedGRU(
                input_size=14,
                hidden_size=best_params['hidden_size'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            )
            
            # Train model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
            
            # Training loop
            for epoch in range(20):
                model.train()
                train_loss = 0
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(X)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        output = model(X)
                        val_loss += nn.MSELoss()(output, y).item()
                
                print(f"Epoch {epoch+1}/20, Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            # Backtest
            final_capital, trades, metrics = backtest(model, dataset)
            
            # Save results
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'final_capital': final_capital,
                'metrics': metrics,
                'best_params': best_params
            }
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'results': results
            }, f'model_{symbol.replace("/", "_")}_{timeframe}.pth')

if __name__ == "__main__":
    main()
