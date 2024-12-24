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

old folder:

- `custom_prediction_model.py`: Advanced model with hyperparameter optimization and risk metrics
- `local_usage.py`: Example implementation and usage patterns
- `requirements.txt`: Project dependencies -`simple.py`: New version of the model with improved trading logic and transaction costs for binance

`I am testing the new version of the model with improved trading logic and transaction costs for binance so update are in the simple.py file. I have no GPU usage thats why i switch to the simpler version of it and stop working on the custom_prediction_model.py file.`

Here are the results of the backtest:

```
Epoch 10/10, Train: 0.0060, Val: 0.0037

Backtest Results:
Final Capital: $1063.55
Total Return: 6.36%
Number of Trades: 2
Win Rate: 50.00%
Maximum Drawdown: 4.53%
Total Transaction Costs: $2.06

```

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
