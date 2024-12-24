# Advanced Cryptocurrency Trading System with Deep Learning

A sophisticated cryptocurrency trading system that combines GRU (Gated Recurrent Unit) neural networks with attention mechanisms and advanced technical indicators for cryptocurrency price prediction and automated trading strategy execution.

## Key Features

### Deep Learning Architecture

- **Enhanced GRU Model**: Utilizes GRU neural networks with attention mechanism for better sequence learning
- **Attention Layer**: Custom attention mechanism for focusing on relevant price patterns
- **Dynamic Feature Engineering**: 14 technical indicators including:
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - ATR (Average True Range)
  - Volume EMAs
  - On-Balance Volume (OBV)

### Trading Strategy

- **Dynamic Position Sizing**: Based on ATR for risk management
- **Multi-Signal Entry/Exit**: Combines multiple indicators for trade decisions:
  - Price prediction threshold
  - RSI oversold/overbought conditions
  - MACD crossovers
  - Bollinger Band touches
- **Risk Management**:
  - 2% risk per trade
  - Dynamic stop-loss based on ATR
  - Transaction cost consideration (0.1% per trade)

### Optimization & Evaluation

- **Hyperparameter Optimization**: Using Optuna for:
  - Hidden layer size
  - Number of GRU layers
  - Dropout rate
  - Learning rate
- **Comprehensive Metrics**:
  - Win rate
  - Maximum drawdown
  - Sharpe ratio
  - Total return
  - Transaction costs

### Data Management

- **Real-time Data**: Integration with Binance API
- **Multiple Timeframes**: Support for different trading intervals (1h, 4h)
- **Multiple Symbols**: Can trade various cryptocurrency pairs

## Installation

```bash
git clone https://github.com/yourusername/crypto-trading-system.git
cd crypto-trading-system
pip install -r requirements.txt
```

## Dependencies

- PyTorch
- ccxt
- pandas
- numpy
- scikit-learn
- optuna
- python-dotenv

## Usage

1. Set up your environment variables:

```bash
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

2. Run the trading system:

```bash
python simple.py
```

## Model Architecture

### EnhancedGRU

- Input Layer: 14 features (technical indicators)
- GRU Layers: Configurable (default: 2 layers)
- Attention Layer: Custom implementation for pattern focus
- Output Layer: Single value prediction (next price movement)

### Dataset Structure

- Sequence Length: 50 candles
- Features: OHLCV + 9 technical indicators
- Train/Val/Test Split: 70/15/15

## Performance Metrics

Recent backtest results with optimized parameters:

```
Final Capital: $1067.29
Total Return: 6.73%
Number of Trades: 4
Win Rate: 25.00%
Maximum Drawdown: 6.50%
Transaction Costs: $4.25
```

## Trading Logic

The system employs a sophisticated multi-factor approach:

1. **Entry Conditions**:

   - Predicted price movement > ATR-based threshold
   - RSI < 30 (oversold)
   - MACD crosses above signal line
   - Price near lower Bollinger Band

2. **Exit Conditions**:

   - Predicted price movement < -ATR-based threshold
   - RSI > 70 (overbought)
   - MACD crosses below signal line
   - Price near upper Bollinger Band

3. **Position Sizing**:
   - Risk per trade: 2% of capital
   - Position size adjusted based on ATR

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries significant risks, and past performance does not guarantee future results. Use this system at your own risk. The authors and contributors are not responsible for any financial losses incurred through the use of this system.
