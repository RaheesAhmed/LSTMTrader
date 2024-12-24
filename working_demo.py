import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

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
def fetch_binance_data(symbol="ASTR/USDT", timeframe="1h", since_days=30):
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.now() - timedelta(days=since_days)).strftime('%Y-%m-%dT%H:%M:%SZ'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate technical indicators
    df['rsi'] = calculate_rsi(df['close'].values)
    df['ema_20'] = calculate_ema(df['close'].values)
    macd, signal = calculate_macd(df['close'].values)
    df['macd'] = macd
    df['macd_signal'] = signal
    
    # Drop any NaN values that might have been created
    df = df.dropna()
    
    return df

# ====================================
# 2) Dataset
# ====================================
class PriceDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.seq_len = seq_len
        self.feats = ['open','high','low','close','volume','rsi','ema_20','macd','macd_signal']
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

# ====================================
# 3) Model
# ====================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc2(out)

# ====================================
# 4) Train
# ====================================
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, Y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device)
                pv = model(Xv)
                val_loss += loss_fn(pv, Yv).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

# ====================================
# 5) Backtest
# ====================================
def backtest(model, dataset, threshold=0.002, transaction_cost=0.001):
    """
    Enhanced backtest function with transaction costs and improved trading logic.
    
    Args:
        model: Trained PyTorch model
        dataset: PriceDataset instance
        threshold: Minimum price change threshold for trading (0.2%)
        transaction_cost: Cost per trade as a fraction (0.1%)
    """
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
            cur_close_scaled = df_scaled.iloc[i]['close']
            seq = df_scaled.iloc[i-seq_len:i].values
            seq_torch = torch.FloatTensor(seq).unsqueeze(0).to(device)
            pred_scaled = model(seq_torch).item()
            
            # Inverse transform prices
            dummy_pred = np.zeros((1, len(feats)))
            dummy_pred[0, feats.index('close')] = pred_scaled
            pred_price = dataset.scaler.inverse_transform(dummy_pred)[0, feats.index('close')]
            
            dummy_cur = np.zeros((1, len(feats)))
            dummy_cur[0, feats.index('close')] = cur_close_scaled
            cur_price = dataset.scaler.inverse_transform(dummy_cur)[0, feats.index('close')]
            
            # Calculate price change percentage
            price_change_pct = (pred_price - cur_price) / cur_price
            
            # Enhanced trading logic with RSI consideration
            rsi = df_scaled.iloc[i]['rsi']
            macd = df_scaled.iloc[i]['macd']
            macd_signal = df_scaled.iloc[i]['macd_signal']
            
            # Buy conditions: predicted up + oversold + MACD crossover
            buy_signal = (price_change_pct > threshold and 
                         rsi < 0.3 and  # RSI < 30 (scaled)
                         macd > macd_signal)
            
            # Sell conditions: predicted down + overbought + MACD crossover
            sell_signal = (price_change_pct < -threshold and 
                          rsi > 0.7 and  # RSI > 70 (scaled)
                          macd < macd_signal)
            
            if buy_signal and position == 0:
                # Apply transaction cost
                trade_cost = capital * transaction_cost
                capital -= trade_cost
                position = (capital / cur_price)
                capital = 0
                trades.append(("BUY", i, cur_price, trade_cost))
                total_trades += 1
                
            elif sell_signal and position > 0:
                # Apply transaction cost
                capital = position * cur_price
                trade_cost = capital * transaction_cost
                capital -= trade_cost
                position = 0
                trades.append(("SELL", i, cur_price, trade_cost))
                total_trades += 1
                
                # Track winning trades
                if capital > 1000:  # If profitable after transaction costs
                    winning_trades += 1
            
            # Track maximum capital and drawdown
            current_value = capital + (position * cur_price if position > 0 else 0)
            max_capital = max(max_capital, current_value)
            current_drawdown = (max_capital - current_value) / max_capital
            max_drawdown = max(max_drawdown, current_drawdown)
    
    # Final exit
    if position > 0:
        final_scaled = df_scaled.iloc[-1]['close']
        dummy_fin = np.zeros((1, len(feats)))
        dummy_fin[0, feats.index('close')] = final_scaled
        final_price = dataset.scaler.inverse_transform(dummy_fin)[0, feats.index('close')]
        capital = position * final_price
        trade_cost = capital * transaction_cost
        capital -= trade_cost
        position = 0
        trades.append(("FINAL_SELL", len(df_scaled)-1, final_price, trade_cost))
        total_trades += 1
    
    # Calculate metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((capital - 1000) / 1000 * 100)
    total_transaction_costs = sum(trade[3] for trade in trades)
    
    print(f"\nBacktest Results:")
    print(f"Final Capital: ${capital:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Total Transaction Costs: ${total_transaction_costs:.2f}")
    
    return capital, trades, {
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'transaction_costs': total_transaction_costs
    }

# ====================================
# MAIN
# ====================================
def main():
    # Fetch more historical data
    df = fetch_binance_data("ETH/USDT", "1h", since_days=60)  # Increased to 60 days
    print(f"Data rows: {len(df)}")
    
    # Create dataset & splits
    dataset = PriceDataset(df, seq_len=50)  # Increased sequence length
    total_len = len(dataset)
    train_len = int(0.7 * total_len)  # 70% train
    val_len = int(0.15 * total_len)   # 15% validation
    test_len = total_len - train_len - val_len  # 15% test
    
    # Create splits
    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )
    
    # DataLoaders with more workers
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
    
    # Enhanced model & training
    model = SimpleLSTM(input_size=9, hidden_size=128, num_layers=2, dropout=0.2)
    train_model(model, train_loader, val_loader, epochs=10, lr=0.001)  # More epochs
    
    # Backtest with transaction costs
    final_capital, trade_history, metrics = backtest(
        model, dataset, threshold=0.002, transaction_cost=0.001
    )

if __name__ == "__main__":
    main()
