# 2. Import Libraries
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
    return final_data


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
    df = df.reset_index(drop=True)

    df['rsi'] = df.ta.rsi(length=14)
    df['sma_20'] = df.ta.sma(length=20)
    df['sma_50'] = df.ta.sma(length=50)
    df['ema_20'] = df.ta.ema(length=20)
    
    # New indicators
    df['atr'] = df.ta.atr(length=14)
    stoch = df.ta.stoch(length=14)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Bollinger Bands
    bb = df.ta.bbands(length=20)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_middle'] = bb['BBM_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    
    # MACD
    macd = df.ta.macd(fast=12, slow=26)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    
    df = df.ffill()
    df = df.fillna(0)
    
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
        
        self.scaled_data = self.scaler.fit_transform(data[features])
        
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.scaled_data) - sequence_length):
            sequence = self.scaled_data[i:i + sequence_length]
            target = self.scaled_data[i + sequence_length]
            self.sequences.append(sequence)
            self.targets.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    min_improvement = 0.01  # 1% improvement threshold
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for sequences, targets in train_loader:
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
                outputs = model(sequences)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save model only if significant improvement
        if avg_val_loss < best_val_loss * (1 - min_improvement):
            best_val_loss = avg_val_loss
            save_model(model, optimizer, epoch, avg_val_loss)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return training_losses, validation_losses


# Enhanced Multi-step Forecast Function
def multi_step_forecast(model, dataset, initial_sequence, steps=48):
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
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train for a few epochs
    n_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        for sequences, targets in train_loader:
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
    
    return {
        'Volatility': volatility,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_drawdown,
        'VaR_95': var_95,
        'Mean_Return': np.mean(returns),
        'Prediction_Uncertainty': uncertainty,
        'Win_Rate': np.mean(np.sign(returns) == np.sign(actual_returns))
    }

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
    
    return {
        'position_size': final_size,
        'confidence_score': 1 - pred_volatility/base_volatility,
        'risk_score': max_loss,
        'trend_strength': market_trend
    }

# 7. Main Execution
def main():
    # Fetch and prepare data
    try:
        data = fetch_construction_data(since_days=60)
        if len(data) == 0:
            print("No data received, using complete sample dataset")
            data = create_sample_data(60)
    except Exception as e:
        print(f"Error in data fetching: {str(e)}, using sample data")
        data = create_sample_data(60)
    
    # Create dataset
    dataset = ConstructionDataset(data)
    
    # Split data
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader),
                  n_trials=20)
    
    # Get best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    
    # Initialize model with best parameters
    model = ConstructionLSTM(input_size=17,
                           hidden_size=best_params['hidden_size'],
                           num_layers=best_params['num_layers'],
                           dropout=best_params['dropout'])
    
    # Train model with best parameters
    train_losses, val_losses = train_model(model, train_loader, val_loader,
                                         learning_rate=best_params['learning_rate'])
    
    # Evaluate and plot results
    metrics = evaluate_model(model, test_loader)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate predictions with confidence intervals
    initial_sequence = dataset[0][0].unsqueeze(0)
    predictions = multi_step_forecast(model, dataset, initial_sequence, steps=72)
    mean, ci = calculate_confidence_interval(predictions)
    
    # Plot predictions with confidence intervals
    plt.figure(figsize=(12, 6))
    hours = np.arange(len(mean))
    plt.plot(hours, mean[:, 3], label='Predicted Close Price')
    plt.fill_between(hours, ci[0][:, 3], ci[1][:, 3], alpha=0.2, label='95% CI')
    plt.title('72-Hour Price Prediction with Confidence Intervals')
    plt.xlabel('Hours')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# 8. Save/Load Functions
def save_model(model, optimizer, epoch, val_loss, filepath='best_model.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, filepath)
    print(f"Model saved at epoch {epoch} with validation loss: {val_loss:.4f}")


def load_model(filepath='construction_model.pth'):
    model = ConstructionLSTM()
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# 9. Run Everything
if __name__ == "__main__":
    main()