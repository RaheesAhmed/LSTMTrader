# Cryptocurrency Trading System with Deep Learning

A sophisticated cryptocurrency trading system that uses LSTM and advanced technical indicators for price prediction and trading strategy development.

## Features

- Real-time cryptocurrency data fetching from Binance US
- Advanced technical indicators and feature engineering
- LSTM-based deep learning model with attention mechanism
- Hyperparameter optimization (in custom_model.py)
- Risk management and position sizing
- Comprehensive trading metrics and evaluation
- Interactive visualizations

## Clone the repository

```bash
git clone https://github.com/RaheesAhmed/LSTMTrader.git
```

## cd into the repository

```bash
cd LSTMTrader
```

## Install the dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

- `custom_prediction_model.py`: Advanced model with hyperparameter optimization and risk metrics
- ` .py`: Streamlined version for quick testing and development
- `local_usage.py`: Example implementation and usage patterns
- `requirements.txt`: Project dependencies

## Usage

### Basic Usage (simple_lstm_model.py)

```python
from simple_lstm_model import ConstructionLSTM, fetch_construction_data, ConstructionDataset

# Fetch data
data = fetch_construction_data(symbols=["BTC/USDT"], timeframe="1h", since_days=30)

# Create dataset
dataset = ConstructionDataset(data)

# Initialize and train model
model = ConstructionLSTM()
# ... continue with training and prediction
```

### Advanced Usage (custom_prediction_model.py)

```python
from custom_prediction_model import ConstructionLSTM, fetch_construction_data, calculate_risk_metrics

# Fetch data with multiple symbols
data = fetch_construction_data(
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    timeframe="1h",
    since_days=60
)

# Initialize model with optimized parameters
model = ConstructionLSTM(
    input_size=17,
    hidden_size=128,
    num_layers=3,
    dropout=0.3
)

# Get predictions and risk metrics
predictions = model.predict(data)
risk_metrics = calculate_risk_metrics(predictions, actual_prices)
```

## Model Comparison

### custom_prediction_model.py (Advanced Version)

- Hyperparameter optimization with Optuna
- Extended technical indicators
- Risk management features
- Position sizing calculations
- Confidence intervals
- Advanced analytics

### simple_lstm_model.py (Streamlined Version)

- Simplified architecture
- Basic technical indicators
- Faster training and inference
- Lower computational requirements
- Suitable for quick testing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Use it at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this system.
