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
    
    # Bollinger Bands
    bb = df.ta.bbands(length=20)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_middle'] = bb['BBM_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    
    # MACD
    macd = df.ta.macd(fast=12, slow=26)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    
    # Replace deprecated fillna method
    df = df.ffill()  # Forward fill
    df = df.fillna(0)  # Fill any remaining NaNs with 0
    
    return df


# 4. Data Preprocessing
class ConstructionDataset(Dataset):
    def __init__(self, data, sequence_length=24):
        self.data = data
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
        # Add technical indicators
        data = add_technical_indicators(data.copy())
        
        # Enhanced feature set
        features = ['open', 'high', 'low', 'close', 'volume',
                   'rsi', 'sma_20', 'sma_50', 'ema_20',
                   'bb_upper', 'bb_middle', 'bb_lower',
                   'macd', 'macd_signal']
        
        self.scaled_data = self.scaler.fit_transform(data[features])
        
        # Prepare sequences
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

# Add enhanced technical indicators and features
def add_enhanced_features(df):
    """Add enhanced technical indicators and features"""
    df = df.copy()
    
    # Volatility Indicators
    df['atr'] = df.ta.atr(length=14)
    df['natr'] = df.ta.natr(length=14)
    
    # Trend Indicators
    df['adx'] = df.ta.adx(length=14)
    df['cci'] = df.ta.cci(length=20)
    
    # Volume Indicators
    df['obv'] = df.ta.obv()
    df['cmf'] = df.ta.cmf(length=20)
    
    # Price Momentum
    df['mom'] = df.ta.mom(length=10)
    df['roc'] = df.ta.roc(length=10)
    
    # Time-based Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Volatility Features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=24).std()
    
    # Handle NaN values
    df = df.ffill()
    df = df.fillna(0)
    
    return df

# 5. Model Architecture
class ConstructionLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, num_layers=3):
        super(ConstructionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
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

# Attentation layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output, attention_weights

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size=25, hidden_size=128, num_layers=3):
        super(EnhancedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=0.3)
        
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended_out, attention_weights = self.attention(lstm_out)
        x = self.fc1(attended_out)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, attention_weights

def calculate_trading_metrics(predictions, actuals, risk_free_rate=0.02):
    """Calculate trading-specific metrics"""
    
    # Calculate returns
    pred_returns = np.diff(predictions) / predictions[:-1]
    actual_returns = np.diff(actuals) / actuals[:-1]
    
    # Sharpe Ratio
    excess_returns = pred_returns - (risk_free_rate / 252)  # Daily risk-free rate
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    # Maximum Drawdown
    cumulative_returns = np.cumprod(1 + pred_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns)
    
    # Directional Accuracy
    correct_directions = np.sum(np.sign(pred_returns) == np.sign(actual_returns))
    directional_accuracy = correct_directions / len(pred_returns)
    
    return {
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Directional_Accuracy': directional_accuracy
    }

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
def evaluate_model_extended(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            outputs = model(sequences)
            if isinstance(outputs, tuple):  # For attention model
                outputs = outputs[0]
            predictions.extend(outputs.numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Standard metrics
    standard_metrics = {
        'MSE': np.mean((predictions - actuals) ** 2),
        'MAE': np.mean(np.abs(predictions - actuals)),
        'RMSE': np.sqrt(np.mean((predictions - actuals) ** 2))
    }
    
    # Trading metrics
    close_price_idx = 3  # Assuming close price is at index 3
    trading_metrics = calculate_trading_metrics(
        predictions[:, close_price_idx],
        actuals[:, close_price_idx]
    )
    
    return {**standard_metrics, **trading_metrics}
# 7. Main Execution
def main():
    # Fetch and prepare data
    try:
        data = fetch_construction_data(since_days=60)  # Increased historical data
        if len(data) == 0:
            print("No data received, using complete sample dataset")
            data = create_sample_data(60)
    except Exception as e:
        print(f"Error in data fetching: {str(e)}, using sample data")
        data = create_sample_data(60)
    
    # Create dataset
    dataset = ConstructionDataset(data)
    
    # Split data into train, validation, and test sets
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
    
    # Initialize and train model
    model = ConstructionLSTM()
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # Plot training and validation losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Evaluate on test set
    metrics = evaluate_model(model, test_loader)
    print("\nTest Set Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate future predictions
    initial_sequence = dataset[0][0].unsqueeze(0)
    predictions = multi_step_forecast(model, dataset, initial_sequence, steps=48)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(predictions[:, 3], label='Predicted Close Price')  # Index 3 for close price
    plt.title('48-Hour Price Prediction')
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