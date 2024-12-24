# @title LSTMTrader v2

import ccxt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas_ta as ta  # Added for technical indicators
import optuna
from sklearn.metrics import r2_score
import scipy.stats as st
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.multiprocessing as mp



# Set up logger
logger.add("trading_system.log", rotation="500 MB")

# Set device to CPU
device = torch.device("cpu")
logger.info(f"Using device: {device}")

def plot_training_curves(train_losses, val_losses):
    """Plot training curves using Plotly"""
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Scatter(y=train_losses, name="Training Loss", line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=val_losses, name="Validation Loss", line=dict(color='red')),
        row=1, col=1
    )
    
    fig.update_layout(
        title="Training and Validation Losses",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def plot_predictions(predictions, actual_prices=None):
    """Plot predictions using Plotly"""
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(
        go.Scatter(y=predictions, name="Predictions", line=dict(color='green')),
        row=1, col=1
    )
    
    if actual_prices is not None:
        fig.add_trace(
            go.Scatter(y=actual_prices, name="Actual Prices", line=dict(color='blue')),
            row=1, col=1
        )
    
    fig.update_layout(
        title="Price Predictions",
        xaxis_title="Time Steps",
        yaxis_title="Price",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    return fig

def plot_portfolio_performance(portfolio_values, trades=None):
    """Plot portfolio performance with trade markers"""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("Portfolio Value", "Drawdown"),
                       shared_xaxes=True,
                       vertical_spacing=0.1)
    
    # Portfolio value
    dates = [v[0] for v in portfolio_values]
    values = [v[1] for v in portfolio_values]
    
    fig.add_trace(
        go.Scatter(x=dates, y=values, name="Portfolio Value", line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add trade markers if provided
    if trades is not None:
        for trade in trades:
            color = 'green' if trade['profit'] > 0 else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade['timestamp']], 
                    y=[trade['value']],
                    mode='markers',
                    marker=dict(color=color, size=10),
                    name=f"{trade['type'].capitalize()} {trade['symbol']}",
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Drawdown
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / peaks * 100
    
    fig.add_trace(
        go.Scatter(x=dates, y=drawdowns, 
                  name="Drawdown", 
                  fill='tozeroy',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title="Portfolio Performance",
        hovermode='x unified',
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig

# 3. Modified Data Collection Function using Binance US
def fetch_construction_data(symbols=["BTC/USDT","ETH/USDT","BNB/USDT"], 
                          timeframe="1h", 
                          since_days=30):
    exchange = ccxt.binanceus()
    all_data = []
    
    for symbol in symbols:
        try:
            # Get timestamp for 30 days ago
            since = int((datetime.now() - timedelta(days=since_days)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['symbol'] = symbol
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                all_data.append(df)
                print(f"Data fetched for {symbol}: {len(df)} rows")
            else:
                print(f"No data returned for {symbol}, using sample data")
                sample_data = create_sample_data(since_days)
                sample_data['symbol'] = symbol
                all_data.append(sample_data)
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}, using sample data")
            sample_data = create_sample_data(since_days)
            sample_data['symbol'] = symbol
            all_data.append(sample_data)
    
    # Ensure we have data
    if not all_data:
        print("No data fetched, creating complete sample dataset")
        return create_sample_data(since_days)
    
    final_data = pd.concat(all_data, axis=0)
    print(f"Final dataset shape: {final_data.shape}")
    final_data = final_data.set_index('timestamp') # Set 'timestamp' as index
    return final_data


# 8. Save/Load Functions
def save_model(model, optimizer, epoch, val_loss, filepath='best_model.pth'):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, filepath)
    logger.info(f"Model saved at epoch {epoch} with validation loss: {val_loss:.4f}")


def load_model(filepath='construction_model.pth'):
    """Load model checkpoint"""
    model = ConstructionLSTM()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)

# Create sample data function
def create_sample_data(days=30):
    """Create more realistic sample data"""
    periods = days * 24  # hourly data
    timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='H')
    
    np.random.seed(42)  # for reproducibility
    
    # Generate more realistic price movements
    base_price = 100
    volatility = 0.02
    returns = np.random.normal(0, volatility, periods)
    price_path = base_price * np.exp(np.cumsum(returns))
    
    # Create DataFrame with OHLCV data
    data = pd.DataFrame({
        'timestamp': timestamps,
        'close': price_path,
        'open': price_path * (1 + np.random.normal(0, 0.001, periods)),
        'high': price_path * (1 + abs(np.random.normal(0, 0.002, periods))),
        'low': price_path * (1 - abs(np.random.normal(0, 0.002, periods))),
        'volume': np.random.lognormal(10, 1, periods)
    })
    
    # Ensure proper OHLC relationship
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

# Enhanced Data Preprocessing with Technical Indicators
def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize columns with NaN
    for col in ['rsi', 'sma_20', 'sma_50', 'ema_20', 'atr', 'stoch_k', 'stoch_d',
               'bb_upper', 'bb_middle', 'bb_lower', 'macd', 'macd_signal']:
        df[col] = np.nan
    
    # If we have multiple symbols, process each symbol separately
    if 'symbol' in df.columns:
        for symbol in df['symbol'].unique():
            try:
                symbol_data = df[df['symbol'] == symbol].copy()
                
                # Ensure we have numeric data
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    symbol_data[col] = pd.to_numeric(symbol_data[col], errors='coerce')
                
                # Calculate RSI
                symbol_data['rsi'] = ta.rsi(symbol_data['close'].fillna(method='ffill'), length=14)
                
                # Calculate Moving Averages
                symbol_data['sma_20'] = ta.sma(symbol_data['close'].fillna(method='ffill'), length=20)
                symbol_data['sma_50'] = ta.sma(symbol_data['close'].fillna(method='ffill'), length=50)
                symbol_data['ema_20'] = ta.ema(symbol_data['close'].fillna(method='ffill'), length=20)
                
                # Calculate ATR
                symbol_data['atr'] = ta.atr(symbol_data['high'].fillna(method='ffill'), 
                                          symbol_data['low'].fillna(method='ffill'), 
                                          symbol_data['close'].fillna(method='ffill'), 
                                          length=14)
                
                # Calculate Stochastic
                stoch = ta.stoch(symbol_data['high'].fillna(method='ffill'), 
                               symbol_data['low'].fillna(method='ffill'), 
                               symbol_data['close'].fillna(method='ffill'), 
                               length=14)
                if stoch is not None:
                    symbol_data['stoch_k'] = stoch['STOCHk_14_3_3']
                    symbol_data['stoch_d'] = stoch['STOCHd_14_3_3']
                
                # Calculate Bollinger Bands
                bb = ta.bbands(symbol_data['close'].fillna(method='ffill'), length=20)
                if bb is not None:
                    symbol_data['bb_upper'] = bb['BBU_20_2.0']
                    symbol_data['bb_middle'] = bb['BBM_20_2.0']
                    symbol_data['bb_lower'] = bb['BBL_20_2.0']
                
                # Calculate MACD
                macd = ta.macd(symbol_data['close'].fillna(method='ffill'), fast=12, slow=26)
                if macd is not None:
                    symbol_data['macd'] = macd['MACD_12_26_9']
                    symbol_data['macd_signal'] = macd['MACDs_12_26_9']
                
                # Update the main dataframe
                for col in symbol_data.columns:
                    if col in df.columns:
                        df.loc[symbol_data.index, col] = symbol_data[col]
                
            except Exception as e:
                logger.warning(f"Error calculating indicators for {symbol}: {str(e)}")
                continue
    else:
        try:
            # Single symbol case - process directly
            # Ensure we have numeric data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['rsi'] = ta.rsi(df['close'].fillna(method='ffill'), length=14)
            df['sma_20'] = ta.sma(df['close'].fillna(method='ffill'), length=20)
            df['sma_50'] = ta.sma(df['close'].fillna(method='ffill'), length=50)
            df['ema_20'] = ta.ema(df['close'].fillna(method='ffill'), length=20)
            
            df['atr'] = ta.atr(df['high'].fillna(method='ffill'), 
                             df['low'].fillna(method='ffill'), 
                             df['close'].fillna(method='ffill'), 
                             length=14)
            
            stoch = ta.stoch(df['high'].fillna(method='ffill'), 
                           df['low'].fillna(method='ffill'), 
                           df['close'].fillna(method='ffill'), 
                           length=14)
            if stoch is not None:
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']
            
            bb = ta.bbands(df['close'].fillna(method='ffill'), length=20)
            if bb is not None:
                df['bb_upper'] = bb['BBU_20_2.0']
                df['bb_middle'] = bb['BBM_20_2.0']
                df['bb_lower'] = bb['BBL_20_2.0']
            
            macd = ta.macd(df['close'].fillna(method='ffill'), fast=12, slow=26)
            if macd is not None:
                df['macd'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                
        except Exception as e:
            logger.warning(f"Error calculating indicators: {str(e)}")
    
    # Forward fill and then backfill any remaining NaN values
    df = df.ffill().bfill()
    
    # Replace any remaining NaN or infinite values with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# 4. Data Preprocessing
class ConstructionDataset(Dataset):
    def __init__(self, data, sequence_length=24):
        self.data = data
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
        # Add technical indicators
        data = add_technical_indicators(data.copy())
        
        # Enhanced feature set including new indicators
        features = ['open', 'high', 'low', 'close', 'volume',
                   'rsi', 'sma_20', 'sma_50', 'ema_20',
                   'bb_upper', 'bb_middle', 'bb_lower',
                   'macd', 'macd_signal', 'atr', 'stoch_k', 'stoch_d']
        
        # Ensure all features exist in the data
        for feature in features:
            if feature not in data.columns:
                data[feature] = 0
        
        # Scale the features
        self.scaled_data = self.scaler.fit_transform(data[features])
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        # Ensure we have enough data for at least one sequence
        if len(self.scaled_data) > sequence_length:
            for i in range(len(self.scaled_data) - sequence_length):
                sequence = self.scaled_data[i:i + sequence_length]
                target = self.scaled_data[i + sequence_length]
                
                # Only add if we have valid data
                if not np.isnan(sequence).any() and not np.isnan(target).any():
                    self.sequences.append(sequence)
                    self.targets.append(target)
        
        # Convert to numpy arrays for efficiency
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.sequences)} sequences")
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor(self.targets[idx]))

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 5. Model Architecture
class ConstructionLSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, num_layers=3, dropout=0.3):
        super(ConstructionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
        
        self.gru = nn.GRU(hidden_size, hidden_size // 2,
                         num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size // 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        x = self.fc1(gru_out[:, -1, :])
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 6. Training Function
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    writer = SummaryWriter()  # For TensorBoard logging
    
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    min_improvement = 0.01  # 1% improvement threshold
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Early stopping check
        early_stopping(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}]')
            logger.info(f'Training Loss: {avg_train_loss:.4f}')
            logger.info(f'Validation Loss: {avg_val_loss:.4f}')
            logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save model only if significant improvement
        if avg_val_loss < best_val_loss * (1 - min_improvement):
            best_val_loss = avg_val_loss
            save_model(model, optimizer, epoch, avg_val_loss)
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    return training_losses, validation_losses


class EnhancedPaperTradingSimulator:
    def __init__(self, initial_capital=100000, risk_manager=None, regime_detector=None,
                 regime_adapter=None, performance_analytics=None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        
        # Initialize components
        self.risk_manager = risk_manager or RiskManager(
            trailing_stop_pct=0.02,
            max_drawdown_pct=0.15,
            risk_per_trade=0.02,
            volatility_scaling=True
        )
        
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.regime_adapter = regime_adapter or RegimeStrategyAdapter()
        self.performance_analytics = performance_analytics or PerformanceAnalytics()
        
    def simulate_trading(self, data, predictions, confidence_intervals, timestamps, symbols):
        """Enhanced trading simulation with all components"""
        portfolio_values = [self.initial_capital]
        current_regime = None
        
        # Ensure data has technical indicators
        if 'rsi' not in data.columns:
            data = add_technical_indicators(data.copy())
        
        # Reset index to make sure we can properly iterate
        data = data.reset_index(drop=True)
        
        # Get the number of unique timestamps
        n_steps = min(len(predictions)-1, len(data) // len(symbols))
        
        for t in range(n_steps):
            try:
                # Update market regime using the correct slice of data
                start_idx = max(0, t*len(symbols))
                end_idx = (t+1)*len(symbols)
                regime_data = data.iloc[start_idx:end_idx]
                current_regime = self.regime_detector.detect_regime(regime_data)
                
                # Adjust strategy based on regime
                strategy_params = self.regime_adapter.adapt_parameters(current_regime)
                
                # Portfolio optimization
                optimal_allocations = self.optimize_portfolio(
                    predictions[t],
                    confidence_intervals,
                    data['atr'].iloc[t*len(symbols)] if 'atr' in data else 0.02
                )
                
                # Process each symbol
                for i, symbol in enumerate(symbols):
                    current_idx = t*len(symbols) + i
                    if current_idx >= len(data):
                        continue
                        
                    symbol_data = data.iloc[current_idx]
                    position_size = optimal_allocations.get(symbol, 0)
                    
                    # Check entry conditions
                    if symbol not in self.positions:
                        if self.check_entry_conditions(
                            symbol,
                            predictions[t+1],
                            confidence_intervals,
                            symbol_data['close'],
                            symbol_data
                        ):
                            self._open_position(
                                symbol,
                                symbol_data['close'],
                                position_size,
                                timestamps[t] if t < len(timestamps) else timestamps[-1]
                            )
                    
                    # Check exit conditions for existing positions
                    elif symbol in self.positions:
                        exit_signal = self.check_exit_conditions(
                            symbol,
                            timestamps[t] if t < len(timestamps) else timestamps[-1],
                            self.positions[symbol],
                            predictions[t+1],
                            confidence_intervals,
                            symbol_data['close']
                        )
                        
                        if exit_signal:
                            self._close_position(
                                symbol,
                                symbol_data['close'],
                                timestamps[t] if t < len(timestamps) else timestamps[-1],
                                "Strategy exit signal"
                            )
                
                # Calculate and record portfolio value
                portfolio_value = self.calculate_portfolio_value(data, t*len(symbols))
                portfolio_values.append(portfolio_value)
                self.performance_analytics.add_portfolio_value(
                    portfolio_value,
                    timestamps[t] if t < len(timestamps) else timestamps[-1]
                )
                
                # Check global risk limits
                if self.risk_manager.check_drawdown_limit(portfolio_value, self.initial_capital):
                    print(f"Maximum drawdown limit reached at {timestamps[t]}")
                    self.close_all_positions(data, t*len(symbols), timestamps[t])
                    break
                    
            except Exception as e:
                logger.warning(f"Error in simulation step {t}: {str(e)}")
                continue
        
        # Calculate final results
        final_capital = portfolio_values[-1]
        returns = (final_capital - self.initial_capital) / self.initial_capital
        
        # Calculate metrics
        metrics = self.performance_analytics.calculate_metrics()
        
        return {
            'final_capital': final_capital,
            'returns': returns,
            'portfolio_values': portfolio_values,
            'trade_history': self.trade_history,
            'metrics': metrics
        }
    
    def check_entry_conditions(self, symbol, prediction, confidence_intervals, current_price, technical_data):
        """Check if entry conditions are met using technical indicators"""
        # Basic trend check
        trend_up = (
            technical_data['sma_20'] > technical_data['sma_50']
            if 'sma_20' in technical_data and 'sma_50' in technical_data
            else True
        )
        
        # Momentum check
        good_momentum = (
            30 < technical_data['rsi'] < 70
            if 'rsi' in technical_data
            else True
        )
        
        # Volatility check
        reasonable_volatility = (
            technical_data['atr'] < current_price * 0.03
            if 'atr' in technical_data
            else True
        )
        
        # Price prediction check
        predicted_return = (prediction - current_price) / current_price
        good_prediction = abs(predicted_return) > 0.01
        
        return trend_up and good_momentum and reasonable_volatility and good_prediction
    
    def check_exit_conditions(self, symbol, timestamp, position, prediction, confidence_intervals, current_price):
        """Check if exit conditions are met"""
        # Calculate holding time
        holding_time = (timestamp - position['entry_time']).total_seconds() / 3600
        
        # Check maximum holding period (48 hours)
        if holding_time > 48:
            return True
        
        # Check trailing stop
        if self.risk_manager.update_trailing_stop(symbol, current_price, position):
            return True
        
        # Check prediction reversal
        if prediction < position['entry_price']:
            return True
        
        # Confidence loss
        price_range = confidence_intervals[1] - confidence_intervals[0]
        confidence_score = 1 - (price_range / prediction)
        if confidence_score < self.confidence_threshold:
            return True, "Low confidence"
        
        return False, None
    
    def optimize_portfolio(self, predictions, confidence_intervals, volatility):
        """Simple portfolio optimization based on predictions and risk"""
        n_assets = len(predictions) if isinstance(predictions, (list, np.ndarray)) else 1
        equal_weight = 1.0 / n_assets
        
        # For now, return equal weights (can be enhanced with proper optimization)
        return {i: equal_weight for i in range(n_assets)}
    
    def calculate_portfolio_value(self, data, t):
        """Calculate current portfolio value"""
        portfolio_value = self.capital
        
        for symbol, position in self.positions.items():
            try:
                symbol_idx = t + list(self.positions.keys()).index(symbol)
                if symbol_idx < len(data):
                    symbol_data = data.iloc[symbol_idx]
                    current_price = symbol_data['close']
                    portfolio_value += position['size'] * current_price
            except Exception as e:
                logger.warning(f"Error calculating portfolio value for {symbol}: {str(e)}")
                continue
        
        return portfolio_value
    
    def _open_position(self, symbol, price, size, timestamp):
        """Open a new position"""
        position_value = self.capital * size
        position_size = position_value / price
        
        if position_value <= self.capital:
            self.positions[symbol] = {
                'size': position_size,
                'entry_price': price,
                'entry_time': timestamp
            }
            self.capital -= position_value
            
            trade = {
                'type': 'buy',
                'symbol': symbol,
                'price': price,
                'size': position_size,
                'value': position_value,
                'timestamp': timestamp
            }
            self.trade_history.append(trade)
            self.performance_analytics.add_trade(trade)
    
    def _close_position(self, symbol, price, timestamp, reason=""):
        """Close an existing position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position_value = position['size'] * price
            self.capital += position_value
            
            profit = position_value - (position['size'] * position['entry_price'])
            
            trade = {
                'type': 'sell',
                'symbol': symbol,
                'price': price,
                'size': position['size'],
                'value': position_value,
                'profit': profit,
                'timestamp': timestamp,
                'reason': reason
            }
            self.trade_history.append(trade)
            self.performance_analytics.add_trade(trade)
            
            del self.positions[symbol]
    
    def close_all_positions(self, data, t, timestamp):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            try:
                symbol_idx = t + list(self.positions.keys()).index(symbol)
                if symbol_idx < len(data):
                    symbol_data = data.iloc[symbol_idx]
                    self._close_position(
                        symbol,
                        symbol_data['close'],
                        timestamp,
                        "Risk management - closing all positions"
                    )
            except Exception as e:
                logger.warning(f"Error closing position for {symbol}: {str(e)}")
                continue

class WalkForwardOptimizer:
    def __init__(self, train_window=30, test_window=7, step_size=7):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
    def optimize(self, data, model, parameter_ranges):
        """Perform walk-forward optimization"""
        results = []
        dates = data.index.unique()
        
        for start_idx in range(0, len(dates) - self.train_window - self.test_window, self.step_size):
            # Split data into train and test periods
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            train_data = data.loc[dates[start_idx:train_end]]
            test_data = data.loc[dates[train_end:test_end]]
            
            # Optimize parameters on training data
            study = optuna.create_study(direction='minimize')  # Minimize negative Sharpe ratio
            study.optimize(lambda trial: self._objective(
                trial, model, train_data, parameter_ranges
            ), n_trials=20)
            
            # Test optimized parameters
            best_params = study.best_params
            performance = self._evaluate_parameters(
                best_params, model, test_data
            )
            
            results.append({
                'period_start': dates[start_idx],
                'period_end': dates[test_end],
                'parameters': best_params,
                'performance': performance
            })
        
        # Find best performing parameters across all periods
        best_period = max(results, key=lambda x: x['performance']['metrics']['sharpe_ratio'])
        return best_period['parameters']
    
    def _objective(self, trial, model, train_data, parameter_ranges):
        """Optimization objective function"""
        # Create dataset from train_data
        dataset = ConstructionDataset(train_data)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Suggest parameters for trading system
        trading_params = {
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            'trailing_stop_pct': trial.suggest_float('trailing_stop_pct', 0.01, 0.05),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.6, 0.9),
            'position_sizing_multiplier': trial.suggest_float('position_sizing_multiplier', 0.5, 2.0)
        }
        
        # Create trading simulator with suggested parameters
        simulator = EnhancedPaperTradingSimulator(
            risk_manager=RiskManager(
                trailing_stop_pct=trading_params['trailing_stop_pct'],
                risk_per_trade=trading_params['risk_per_trade']
            )
        )
        
        # Get initial sequence for predictions
        initial_sequence = dataset[0][0].unsqueeze(0)
        
        # Generate predictions
        predictions = multi_step_forecast(model, dataset, initial_sequence, steps=48)
        mean, ci = calculate_confidence_interval(predictions)
        
        # Run simulation
        results = simulator.simulate_trading(
            data=train_data,
            predictions=predictions,
            confidence_intervals=ci,
            timestamps=train_data.index,
            symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        )
        
        # Return negative Sharpe ratio (since Optuna minimizes)
        return -results['metrics']['sharpe_ratio']
    
    def _evaluate_parameters(self, params, model, test_data):
        """Evaluate parameter set on test data"""
        # Create dataset from test data
        dataset = ConstructionDataset(test_data)
        initial_sequence = dataset[0][0].unsqueeze(0)
        
        # Generate predictions
        predictions = multi_step_forecast(model, dataset, initial_sequence, steps=48)
        mean, ci = calculate_confidence_interval(predictions)
        
        # Create simulator with optimized parameters
        simulator = EnhancedPaperTradingSimulator(
            risk_manager=RiskManager(
                trailing_stop_pct=params['trailing_stop_pct'],
                risk_per_trade=params['risk_per_trade']
            )
        )
        
        # Run simulation
        results = simulator.simulate_trading(
            data=test_data,
            predictions=predictions,
            confidence_intervals=ci,
            timestamps=test_data.index,
            symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        )
        
        return results


class MarketRegimeDetector:
    def __init__(self, window_size=20):
        self.window_size = window_size
        
    def detect_regime(self, data):
        """Detect market regime using multiple indicators"""
        # First ensure data has all required indicators
        data = add_technical_indicators(data.copy())
        
        # Calculate regime indicators
        volatility = self._calculate_volatility(data)
        trend = self._calculate_trend(data)
        momentum = self._calculate_momentum(data)
        volume_profile = self._calculate_volume_profile(data)
        
        # Combine indicators using fuzzy logic
        regime_scores = self._fuzzy_regime_classification(
            volatility, trend, momentum, volume_profile
        )
        
        # Return dominant regime
        return self._get_dominant_regime(regime_scores)
    
    def _calculate_volatility(self, data):
        """Calculate volatility regime"""
        returns = data['close'].pct_change()
        vol = returns.rolling(self.window_size).std() * np.sqrt(252)
        vol_zscore = (vol - vol.mean()) / vol.std()
        
        return {
            'high_volatility': vol_zscore > 1,
            'normal_volatility': (-1 <= vol_zscore) & (vol_zscore <= 1),
            'low_volatility': vol_zscore < -1
        }
    
    def _calculate_trend(self, data):
        """Calculate trend regime"""
        sma_short = data['sma_20']
        sma_long = data['sma_50']
        macd = data['macd']
        
        return {
            'strong_uptrend': (sma_short > sma_long) & (macd > 0),
            'weak_uptrend': (sma_short > sma_long) & (macd <= 0),
            'sideways': abs(sma_short - sma_long) < data['close'].std(),
            'weak_downtrend': (sma_short < sma_long) & (macd >= 0),
            'strong_downtrend': (sma_short < sma_long) & (macd < 0)
        }
    
    def _calculate_momentum(self, data):
        """Calculate momentum regime"""
        rsi = data['rsi']
        stoch_k = data['stoch_k']
        
        return {
            'overbought': (rsi > 70) & (stoch_k > 80),
            'bullish_momentum': (rsi > 50) & (stoch_k > 50),
            'neutral_momentum': (45 <= rsi) & (rsi <= 55),
            'bearish_momentum': (rsi < 50) & (stoch_k < 50),
            'oversold': (rsi < 30) & (stoch_k < 20)
        }
    
    def _calculate_volume_profile(self, data):
        """Calculate volume profile regime"""
        volume_sma = data['volume'].rolling(self.window_size).mean()
        
        return {
            'high_volume_bullish': (data['volume'] > volume_sma) & (data['close'] > data['open']),
            'high_volume_bearish': (data['volume'] > volume_sma) & (data['close'] < data['open']),
            'low_volume_bullish': (data['volume'] < volume_sma) & (data['close'] > data['open']),
            'low_volume_bearish': (data['volume'] < volume_sma) & (data['close'] < data['open'])
        }
    
    def _fuzzy_regime_classification(self, volatility, trend, momentum, volume_profile):
        """Combine different indicators using fuzzy logic"""
        regime_scores = {
            'strong_bull_market': self._calculate_bull_score(
                volatility, trend, momentum, volume_profile
            ),
            'weak_bull_market': self._calculate_weak_bull_score(
                volatility, trend, momentum, volume_profile
            ),
            'neutral_market': self._calculate_neutral_score(
                volatility, trend, momentum, volume_profile
            ),
            'weak_bear_market': self._calculate_weak_bear_score(
                volatility, trend, momentum, volume_profile
            ),
            'strong_bear_market': self._calculate_bear_score(
                volatility, trend, momentum, volume_profile
            )
        }
        
        return regime_scores
    
    def _calculate_bull_score(self, volatility, trend, momentum, volume_profile):
        """Calculate strong bull market score"""
        score = 0.0
        score += 0.3 * trend['strong_uptrend'].mean()
        score += 0.2 * momentum['bullish_momentum'].mean()
        score += 0.2 * volume_profile['high_volume_bullish'].mean()
        score += 0.15 * (1 - volatility['high_volatility'].mean())
        score += 0.15 * (1 - momentum['overbought'].mean())
        return score
    
    def _calculate_weak_bull_score(self, volatility, trend, momentum, volume_profile):
        """Calculate weak bull market score"""
        score = 0.0
        score += 0.3 * trend['weak_uptrend'].mean()
        score += 0.2 * momentum['bullish_momentum'].mean()
        score += 0.2 * volume_profile['low_volume_bullish'].mean()
        score += 0.15 * volatility['normal_volatility'].mean()
        score += 0.15 * (1 - momentum['overbought'].mean())
        return score
    
    def _calculate_neutral_score(self, volatility, trend, momentum, volume_profile):
        """Calculate neutral market score"""
        score = 0.0
        score += 0.4 * trend['sideways'].mean()
        score += 0.3 * momentum['neutral_momentum'].mean()
        score += 0.3 * volatility['normal_volatility'].mean()
        return score
    
    def _calculate_weak_bear_score(self, volatility, trend, momentum, volume_profile):
        """Calculate weak bear market score"""
        score = 0.0
        score += 0.3 * trend['weak_downtrend'].mean()
        score += 0.2 * momentum['bearish_momentum'].mean()
        score += 0.2 * volume_profile['low_volume_bearish'].mean()
        score += 0.15 * volatility['normal_volatility'].mean()
        score += 0.15 * (1 - momentum['oversold'].mean())
        return score
    
    def _calculate_bear_score(self, volatility, trend, momentum, volume_profile):
        """Calculate strong bear market score"""
        score = 0.0
        score += 0.3 * trend['strong_downtrend'].mean()
        score += 0.2 * momentum['bearish_momentum'].mean()
        score += 0.2 * volume_profile['high_volume_bearish'].mean()
        score += 0.15 * volatility['high_volatility'].mean()
        score += 0.15 * (1 - momentum['oversold'].mean())
        return score
    
    def _get_dominant_regime(self, regime_scores):
        """Get the dominant market regime based on scores"""
        return max(regime_scores.items(), key=lambda x: x[1])[0]

class RegimeStrategyAdapter:
    def __init__(self):
        self.base_parameters = {
            'risk_per_trade': 0.02,
            'trailing_stop_pct': 0.02,
            'confidence_threshold': 0.7,
            'position_sizing_multiplier': 1.0,
            'max_holding_period': 48
        }
        
    def adapt_parameters(self, regime):
        """Adapt strategy parameters based on market regime"""
        if regime == 'strong_bull_market':
            return {
                'risk_per_trade': self.base_parameters['risk_per_trade'] * 1.2,
                'trailing_stop_pct': self.base_parameters['trailing_stop_pct'] * 1.5,
                'confidence_threshold': self.base_parameters['confidence_threshold'] * 0.9,
                'position_sizing_multiplier': 1.2,
                'max_holding_period': self.base_parameters['max_holding_period'] * 1.5
            }
        
        elif regime == 'weak_bull_market':
            return {
                'risk_per_trade': self.base_parameters['risk_per_trade'] * 1.1,
                'trailing_stop_pct': self.base_parameters['trailing_stop_pct'] * 1.2,
                'confidence_threshold': self.base_parameters['confidence_threshold'] * 0.95,
                'position_sizing_multiplier': 1.1,
                'max_holding_period': self.base_parameters['max_holding_period'] * 1.2
            }
        
        elif regime == 'neutral_market':
            return self.base_parameters.copy()
        
        elif regime == 'weak_bear_market':
            return {
                'risk_per_trade': self.base_parameters['risk_per_trade'] * 0.8,
                'trailing_stop_pct': self.base_parameters['trailing_stop_pct'] * 0.8,
                'confidence_threshold': self.base_parameters['confidence_threshold'] * 1.1,
                'position_sizing_multiplier': 0.8,
                'max_holding_period': self.base_parameters['max_holding_period'] * 0.8
            }
        
        elif regime == 'strong_bear_market':
            return {
                'risk_per_trade': self.base_parameters['risk_per_trade'] * 0.6,
                'trailing_stop_pct': self.base_parameters['trailing_stop_pct'] * 0.6,
                'confidence_threshold': self.base_parameters['confidence_threshold'] * 1.2,
                'position_sizing_multiplier': 0.6,
                'max_holding_period': self.base_parameters['max_holding_period'] * 0.6
            }
        
        return self.base_parameters.copy()  # Default case
    
    def get_trading_rules(self, regime):
        """Get specific trading rules based on market regime"""
        base_rules = {
            'min_profit_target': 0.02,
            'max_loss': 0.02,
            'trend_confirmation_required': True,
            'volume_confirmation_required': True
        }
        
        regime_rules = {
            'strong_bull_market': {
                'min_profit_target': 0.03,
                'max_loss': 0.025,
                'trend_confirmation_required': False,
                'volume_confirmation_required': False
            },
            'weak_bull_market': {
                'min_profit_target': 0.025,
                'max_loss': 0.02,
                'trend_confirmation_required': True,
                'volume_confirmation_required': False
            },
            'neutral_market': base_rules,
            'weak_bear_market': {
                'min_profit_target': 0.015,
                'max_loss': 0.015,
                'trend_confirmation_required': True,
                'volume_confirmation_required': True
            },
            'strong_bear_market': {
                'min_profit_target': 0.01,
                'max_loss': 0.01,
                'trend_confirmation_required': True,
                'volume_confirmation_required': True
            }
        }
        
        return regime_rules.get(regime, base_rules)

# Enhanced Multi-step Forecast Function
def multi_step_forecast(model, dataset, initial_sequence, steps=48):
    """Multi-step forecast with CPU"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        current_sequence = initial_sequence.clone()
        
        for _ in range(steps):
            pred = model(current_sequence)
            predictions.append(pred.numpy()[0])
            
            # Update sequence for next prediction
            current_sequence = torch.cat((
                current_sequence[:, 1:, :],
                pred.unsqueeze(1)
            ), dim=1)
    
    return np.array(predictions)

# Adding evaluation metrics
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            outputs = model(sequences)
            predictions.extend(outputs.numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def objective(trial, train_loader, val_loader):
    """Optuna objective function for hyperparameter tuning"""
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 64, 256)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Create model with suggested hyperparameters
    model = ConstructionLSTM(input_size=17, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train for a few epochs
    n_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

def calculate_confidence_interval(predictions, alpha=0.05):
    """Calculate confidence intervals for predictions"""
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    ci = st.norm.interval(confidence=1-alpha, loc=mean, scale=std)
    return mean, ci

def calculate_risk_metrics(predictions, actual_prices, confidence_intervals=None):
    """Calculate trading risk metrics"""
    returns = np.diff(predictions) / predictions[:-1]
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    excess_returns = returns - 0.02/252
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    # Maximum Drawdown
    cumulative = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (rolling_max - cumulative) / rolling_max
    max_drawdown = np.max(drawdowns)
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)
    
    # Prediction Uncertainty
    uncertainty = None
    if confidence_intervals is not None:
        uncertainty = np.mean(confidence_intervals[1] - confidence_intervals[0])
    
    metrics = {
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_drawdown,
        'VaR_95': var_95,
        'Mean_Return': np.mean(returns),
        'Prediction_Uncertainty': uncertainty,
        'Win_Rate': np.mean(np.sign(returns) == np.sign(actual_returns))
    }
    
    return metrics

def calculate_position_size(prediction, confidence_interval, max_position=1.0, base_volatility=0.02):
    """Calculate suggested position size based on prediction confidence"""
    # Calculate normalized uncertainty
    uncertainty = (confidence_interval[1] - confidence_interval[0]) / 2
    pred_volatility = uncertainty / prediction
    
    # Adjust position size based on volatility ratio
    volatility_ratio = base_volatility / pred_volatility
    position_size = max_position * min(1.0, volatility_ratio)
    
    # Risk-based position sizing
    max_loss = abs(confidence_interval[0] - prediction) / prediction
    risk_adjusted_size = min(position_size, 0.02 / max_loss)  # 2% max risk per trade
    
    # Market conditions adjustment
    market_trend = prediction / confidence_interval[0] - 1
    trend_multiplier = np.clip(abs(market_trend) * 5, 0.5, 1.5)
    
    final_size = risk_adjusted_size * trend_multiplier
    
    sizing_info = {
        'position_size': final_size,
        'confidence_score': 1 - pred_volatility/base_volatility,
        'risk_score': max_loss,
        'trend_strength': market_trend
    }
    
    return sizing_info

class RiskManager:
    def __init__(self, trailing_stop_pct=0.02, max_drawdown_pct=0.15, 
                 risk_per_trade=0.02, volatility_scaling=True):
        self.trailing_stop_pct = trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_per_trade = risk_per_trade
        self.volatility_scaling = volatility_scaling
        self.trailing_stops = {}
        
    def update_trailing_stop(self, symbol, current_price, position):
        """Update trailing stop for a position"""
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = current_price * (1 - self.trailing_stop_pct)
        else:
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > self.trailing_stops[symbol]:
                self.trailing_stops[symbol] = new_stop
        
        return current_price <= self.trailing_stops[symbol]
    
    def calculate_position_size(self, predicted_price, confidence_interval, 
                              current_price, portfolio_value, volatility):
        """Calculate optimal position size based on risk metrics"""
        # Risk-adjusted position sizing
        price_range = confidence_interval[1] - confidence_interval[0]
        uncertainty = price_range / predicted_price
        
        # Volatility scaling
        if self.volatility_scaling:
            base_volatility = 0.02  # 2% baseline volatility
            vol_adjustment = base_volatility / volatility
            risk_adjusted = self.risk_per_trade * vol_adjustment
        else:
            risk_adjusted = self.risk_per_trade
        
        # Kelly Criterion adjustment
        win_prob = 0.5 + (predicted_price - current_price) / (2 * price_range)
        edge = (predicted_price / current_price - 1)
        kelly_size = win_prob - (1 - win_prob) / edge if edge != 0 else 0
        kelly_size = max(0, min(kelly_size, 0.5))  # Conservative Kelly
        
        # Final position size
        position_size = portfolio_value * risk_adjusted * kelly_size
        
        return position_size
    
    def check_drawdown_limit(self, portfolio_value, initial_capital):
        """Check if maximum drawdown limit is breached"""
        drawdown = (initial_capital - portfolio_value) / initial_capital
        return drawdown > self.max_drawdown_pct


class TradeRules:
    def __init__(self, confidence_threshold=0.7, min_edge=0.01, 
                 max_holding_period=48, trend_confirmation=True):
        self.confidence_threshold = confidence_threshold
        self.min_edge = min_edge
        self.max_holding_period = max_holding_period
        self.trend_confirmation = trend_confirmation
        self.positions_timing = {}
    
    def check_entry_conditions(self, symbol, prediction, confidence_interval, 
                             current_price, technical_indicators):
        """Check if entry conditions are met"""
        # Calculate prediction confidence
        price_range = confidence_interval[1] - confidence_interval[0]
        confidence_score = 1 - (price_range / prediction)
        
        # Calculate expected edge
        expected_return = (prediction - current_price) / current_price
        
        # Trend confirmation using technical indicators
        trend_confirmed = True
        if self.trend_confirmation:
            sma_20 = technical_indicators['sma_20']
            sma_50 = technical_indicators['sma_50']
            rsi = technical_indicators['rsi']
            trend_confirmed = (
                current_price > sma_20 > sma_50 and  # Uptrend
                rsi > 40 and rsi < 70  # Not overbought/oversold
            )
        
        return (confidence_score > self.confidence_threshold and
                abs(expected_return) > self.min_edge and
                trend_confirmed)
    
    def check_exit_conditions(self, symbol, current_time, position_info,
                            prediction, confidence_interval, risk_manager):
        """Check if exit conditions are met"""
        # Time-based exit
        holding_time = (current_time - position_info['entry_time']).total_seconds() / 3600
        if holding_time > self.max_holding_period:
            return True, "Max holding period exceeded"
        
        # Trailing stop hit
        if risk_manager.update_trailing_stop(symbol, prediction, position_info):
            return True, "Trailing stop triggered"
        
        # Prediction reversal
        entry_price = position_info['entry_price']
        if prediction < entry_price * (1 - self.min_edge):
            return True, "Prediction reversal"
        
        # Confidence loss
        price_range = confidence_interval[1] - confidence_interval[0]
        confidence_score = 1 - (price_range / prediction)
        if confidence_score < self.confidence_threshold:
            return True, "Low confidence"
        
        return False, None


class PerformanceAnalytics:
    def __init__(self):
        self.daily_returns = []
        self.trades = []
        self.portfolio_values = []
        
    def add_trade(self, trade):
        self.trades.append(trade)
        
    def add_portfolio_value(self, value, timestamp):
        self.portfolio_values.append((timestamp, value))
        
    def calculate_metrics(self, risk_free_rate=0.02):
        """Calculate comprehensive performance metrics"""
        returns = np.diff([v[1] for v in self.portfolio_values]) / \
                 np.array([v[1] for v in self.portfolio_values[:-1]])
        
        # Basic metrics
        total_return = (self.portfolio_values[-1][1] / self.portfolio_values[0][1]) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (np.mean(returns) - risk_free_rate/252) / np.std(returns) * np.sqrt(252)
        
        # Drawdown analysis
        portfolio_values = np.array([v[1] for v in self.portfolio_values])
        peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (peaks - portfolio_values) / peaks
        max_drawdown = np.max(drawdowns)
        
        # Trade analysis
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        losing_trades = [t for t in self.trades if t['profit'] <= 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        return {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(returns)),
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t['profit'] for t in winning_trades) / 
                               sum(t['profit'] for t in losing_trades)) if losing_trades else float('inf'),
            'VaR_95': var_95,
            'CVaR_95': cvar_95
        }
    
    def plot_performance(self):
        """Generate comprehensive performance plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        dates = [v[0] for v in self.portfolio_values]
        values = [v[1] for v in self.portfolio_values]
        ax1.plot(dates, values)
        ax1.set_title('Portfolio Value Over Time')
        
        # Drawdown visualization
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks * 100
        ax2.fill_between(dates, 0, drawdowns, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)')
        
        # Return distribution
        returns = np.diff(values) / values[:-1]
        ax3.hist(returns, bins=50, density=True, alpha=0.7)
        ax3.set_title('Return Distribution')
        
        # Trade profit/loss
        profits = [t['profit'] for t in self.trades]
        ax4.bar(range(len(profits)), profits)
        ax4.set_title('Trade Profit/Loss')
        
        plt.tight_layout()
        plt.show()

class UnifiedTradingSystem:
    def __init__(self, initial_capital=100000):
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.regime_adapter = RegimeStrategyAdapter()
        self.risk_manager = RiskManager()
        self.performance_analytics = PerformanceAnalytics()
        self.current_regime = None
        self.initial_capital = initial_capital
        
    def run_simulation(self, data, model, prediction_steps=72):
        """Run complete trading simulation with all components"""
        # Prepare dataset with technical indicators
        data = add_technical_indicators(data.copy())
        dataset = ConstructionDataset(data)
        
        # Generate predictions with confidence intervals
        initial_sequence = dataset[0][0].unsqueeze(0)
        predictions = multi_step_forecast(model, dataset, initial_sequence, steps=prediction_steps)
        mean, ci = calculate_confidence_interval(predictions)
        
        # Prepare trading data
        actual_prices, pred_transformed, ci_transformed = prepare_trading_data(
            data, predictions, ci, dataset.scaler
        )
        
        # Initialize trading simulator with advanced components
        simulator = EnhancedPaperTradingSimulator(
            initial_capital=self.initial_capital,
            risk_manager=self.risk_manager,
            regime_detector=self.regime_detector,
            regime_adapter=self.regime_adapter,
            performance_analytics=self.performance_analytics
        )
        
        # Run simulation
        results = simulator.simulate_trading(
            data=data,
            predictions=pred_transformed,
            confidence_intervals=ci_transformed,
            timestamps=data['timestamp'].values,
            symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        )
        
        # Generate comprehensive analysis
        self._generate_analysis_report(results, data)
        
        return results
    
    def _generate_analysis_report(self, results, data):
        """Generate comprehensive trading analysis report"""
        print("\n=== Trading Performance Report ===")
        
        # Overall performance metrics
        print("\nOverall Performance:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['returns']*100:.2f}%")
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
        
        # Regime-specific performance
        print("\nPerformance by Market Regime:")
        regime_performance = self._analyze_regime_performance(results['trade_history'])
        for regime, metrics in regime_performance.items():
            print(f"\n{regime.replace('_', ' ').title()}:")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Average Trade: ${metrics['avg_trade']:,.2f}")
        
        # Risk metrics
        print("\nRisk Metrics:")
        print(f"Value at Risk (95%): {results['metrics']['VaR_95']*100:.2f}%")
        print(f"Conditional VaR (95%): {results['metrics']['CVaR_95']*100:.2f}%")
        print(f"Annualized Volatility: {results['metrics']['volatility']*100:.2f}%")
        
        # Generate visualization plots
        self._plot_comprehensive_analysis(results, data)
    
    def _analyze_regime_performance(self, trade_history):
        """Analyze performance metrics by market regime"""
        from collections import defaultdict
        
        regime_trades = defaultdict(list)
        
        for trade in trade_history:
            regime = trade.get('regime', 'unknown')
            regime_trades[regime].append(trade)
        
        regime_metrics = {}
        for regime, trades in regime_trades.items():
            profits = [t['profit'] for t in trades]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p <= 0]
            
            regime_metrics[regime] = {
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf'),
                'avg_trade': np.mean(profits) if profits else 0,
                'total_trades': len(trades)
            }
        
        return regime_metrics
    
    def _plot_comprehensive_analysis(self, results, data):
        """Generate comprehensive analysis plots"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3)
        
        # Portfolio value and drawdown
        ax1 = plt.subplot(gs[0, :])
        self._plot_portfolio_performance(ax1, results['portfolio_values'])
        
        # Regime transitions
        ax2 = plt.subplot(gs[1, :])
        self._plot_regime_transitions(ax2, data)
        
        # Trade distribution
        ax3 = plt.subplot(gs[2, 0])
        self._plot_trade_distribution(ax3, results['trade_history'])
        
        # Position sizing
        ax4 = plt.subplot(gs[2, 1])
        self._plot_position_sizing(ax4, results['trade_history'])
        
        # Risk metrics evolution
        ax5 = plt.subplot(gs[2, 2])
        self._plot_risk_metrics_evolution(ax5, results)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_portfolio_performance(self, ax, portfolio_values):
        """Plot portfolio value and drawdown"""
        dates = [v[0] for v in portfolio_values]
        values = [v[1] for v in portfolio_values]
        
        # Plot portfolio value
        ax.plot(dates, values, label='Portfolio Value', color='blue')
        ax.set_title('Portfolio Performance and Drawdown')
        ax.set_ylabel('Portfolio Value ($)')
        
        # Add drawdown
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks * 100
        ax2 = ax.twinx()
        ax2.fill_between(dates, 0, drawdowns, alpha=0.3, color='red', label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_regime_transitions(self, ax, data):
        """Plot market regime transitions and key events"""
        regimes = self.regime_detector.detect_regime(data)
        dates = data.index
        
        # Plot regime changes
        ax.plot(dates, regimes, label='Market Regime', marker='o')
        ax.set_title('Market Regime Transitions')
        ax.set_ylabel('Regime')
        
        # Add key events
        for trade in self.performance_analytics.trades:
            if trade['profit'] > 0:
                ax.axvline(trade['timestamp'], color='green', alpha=0.2)
            else:
                ax.axvline(trade['timestamp'], color='red', alpha=0.2)
    
    def _plot_trade_distribution(self, ax, trade_history):
        """Plot trade profit/loss distribution"""
        profits = [t['profit'] for t in trade_history]
        ax.hist(profits, bins=30, density=True, alpha=0.7)
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title('Trade Profit/Loss Distribution')
        ax.set_xlabel('Profit/Loss ($)')
    
    def _plot_position_sizing(self, ax, trade_history):
        """Plot position sizing evolution"""
        dates = [t['timestamp'] for t in trade_history]
        sizes = [t['value'] for t in trade_history]
        ax.scatter(dates, sizes, alpha=0.5)
        ax.set_title('Position Sizing Evolution')
        ax.set_ylabel('Position Size ($)')
    
    def _plot_risk_metrics_evolution(self, ax, results):
        """Plot evolution of risk metrics"""
        window = 30  # 30-day rolling window
        returns = np.diff(results['portfolio_values']) / results['portfolio_values'][:-1]
        vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        sharpe = pd.Series(returns).rolling(window).mean() / pd.Series(returns).rolling(window).std() * np.sqrt(252)
        
        ax.plot(vol.index, vol.values, label='Volatility')
        ax.plot(sharpe.index, sharpe.values, label='Sharpe Ratio')
        ax.set_title('Risk Metrics Evolution')
        ax.legend()


def prepare_trading_data(data, predictions, ci, scaler):
    """Prepare data for paper trading simulation"""
    # Get close prices from original data
    close_prices = data[data['symbol'] == 'BTC/USDT']['close'].values
    eth_prices = data[data['symbol'] == 'ETH/USDT']['close'].values
    bnb_prices = data[data['symbol'] == 'BNB/USDT']['close'].values
    
    # Stack actual prices
    actual_prices = np.column_stack((close_prices, eth_prices, bnb_prices))
    
    # Inverse transform predictions
    pred_transformed = scaler.inverse_transform(predictions)
    ci_lower_transformed = scaler.inverse_transform(ci[0])
    ci_upper_transformed = scaler.inverse_transform(ci[1])
    
    return actual_prices, pred_transformed, (ci_lower_transformed, ci_upper_transformed)

def main():
    # Set up logging
    logger.info("Starting trading system...")
    logger.info(f"Using device: {device}")
    
    # Fetch and prepare data
    try:
        data = fetch_construction_data(since_days=60)
        if len(data) == 0:
            logger.warning("No data received, using complete sample dataset")
            data = create_sample_data(60)
    except Exception as e:
        logger.error(f"Error in data fetching: {str(e)}, using sample data")
        data = create_sample_data(60)
    
    # Create dataset
    dataset = ConstructionDataset(data)
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Split data
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders with multiple workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0  # Changed from 1 to 0 for CPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0  # Changed from 1 to 0 for CPU
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0  # Changed from 1 to 0 for CPU
    )
    
    logger.info("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader),
                  n_trials=20)
    
    # Get best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters found: {best_params}")
    
    # Train model with best parameters
    model = ConstructionLSTM(
        input_size=17,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    logger.info("Starting model training...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader,
        learning_rate=best_params['learning_rate']
    )
    
    # Plot training curves
    fig_losses = plot_training_curves(train_losses, val_losses)
    fig_losses.show()
    
    # Run simulation with walk-forward optimization
    logger.info("Starting walk-forward optimization...")
    walk_forward = WalkForwardOptimizer(train_window=30, test_window=7)
    parameter_ranges = {
        'risk_per_trade': (0.01, 0.05),
        'trailing_stop_pct': (0.01, 0.05),
        'confidence_threshold': (0.6, 0.9),
        'position_sizing_multiplier': (0.5, 2.0)
    }
    results = walk_forward.optimize(data, model, parameter_ranges)
    
    # Log optimization results
    logger.info("Walk-forward optimization completed")
    logger.info(f"Best parameters found: {results}")
    
    # Run final simulation with optimized parameters
    logger.info("Running final simulation...")
    trading_system = UnifiedTradingSystem(initial_capital=100000)
    final_results = trading_system.run_simulation(data, model)
    
    # Generate predictions for visualization
    initial_sequence = dataset[0][0].unsqueeze(0)
    predictions = multi_step_forecast(model, dataset, initial_sequence, steps=48)
    
    # Plot predictions
    fig_predictions = plot_predictions(predictions)
    fig_predictions.show()
    
    # Plot portfolio performance
    fig_portfolio = plot_portfolio_performance(
        final_results['portfolio_values'],
        final_results['trade_history']
    )
    fig_portfolio.show()
    
    logger.info("Trading simulation completed successfully")
    logger.info(f"Final portfolio value: ${final_results['final_capital']:,.2f}")
    logger.info(f"Total return: {final_results['returns']*100:.2f}%")
    logger.info(f"Sharpe ratio: {final_results['metrics']['sharpe_ratio']:.2f}")
    
    return final_results

if __name__ == "__main__":
    
    mp.set_start_method('spawn', force=True)
    main()

