# Backtest Algorithmic Trading Using Deep Learning

<a href="">
  <img src="https://github.com/Saeidhoseinipour/Algorithmic-Trading/blob/main/Images/AT.png" alt="Algorithmic trading, Trding, Text mining, Matrix factorization, Co-clustering, Saeid Hoseinipour" style="width: 70%;">
</a>

## Overview

This repository provides a comprehensive guide and implementation of backtesting algorithmic trading strategies using deep learning in Python. It covers the entire process from data collection and preprocessing to strategy implementation and performance analysis.

## Table of Contents

- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Building and Training](#model-building-and-training)
- [Making Predictions](#making-predictions)
- [Implementing Arbitrage Strategy](#implementing-arbitrage-strategy)
- [Backtesting and Performance Analysis](#backtesting-and-performance-analysis)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the scripts and notebooks in this repository, you need to have Python installed on your system. You can download Python from [Python.org](https://www.python.org/).

Additionally, you need to install the following Python packages:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow keras matplotlib


## Data Collection
We use the yfinance package to download historical stock price data. Here is an example script to download data for Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), and Amazon (AMZN):

import yfinance as yf

```python
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN']
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Close']
```

## Data Preprocessing
Data preprocessing involves scaling the data using MinMaxScaler and creating datasets with a look-back period for LSTM input.

Example script:
```python
from sklearn.preprocessing import MinMaxScaler

scalers = {}
scaled_data = {}
for ticker in tickers:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data[ticker] = scaler.fit_transform(data[ticker].values.reshape(-1, 1))
    scalers[ticker] = scaler

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
train_size = int(len(data) * 0.8)
train_data = {ticker: scaled_data[ticker][:train_size] for ticker in tickers}
test_data = {ticker: scaled_data[ticker][train_size - look_back:] for ticker in tickers}

train_X = {}
train_Y = {}
test_X = {}
test_Y = {}

for ticker in tickers:
    train_X[ticker], train_Y[ticker] = create_dataset(train_data[ticker], look_back)
    test_X[ticker], test_Y[ticker] = create_dataset(test_data[ticker], look_back)

# Reshape input to be [samples, time steps, features]
for ticker in tickers:
    train_X[ticker] = np.reshape(train_X[ticker], (train_X[ticker].shape[0], train_X[ticker].shape[1], 1))
    test_X[ticker] = np.reshape(test_X[ticker], (test_X[ticker].shape[0], test_X[ticker].shape[1], 1))
```

## Model Building and Training
Build and train the LSTM model for each stock.

Example script:


```python
from keras.models import Sequential
from keras.layers import Dense, LSTM

def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

models = {}
for ticker in tickers:
    model = build_model()
    model.fit(train_X[ticker], train_Y[ticker], batch_size=1, epochs=1)
    models[ticker] = model
```




## Making Predictions
Use the trained LSTM models to make predictions.

Example script:
```python
predictions = {}
for ticker in tickers:
    pred = models[ticker].predict(test_X[ticker])
    predictions[ticker] = scalers[ticker].inverse_transform(pred)
```



## Implementing Arbitrage Strategy
Generate buy and sell signals based on the predicted and actual price differences.

Example script:

```python
signals = pd.DataFrame(index=data.index[train_size + look_back:])
for ticker in tickers:
    signals[ticker] = data[ticker][train_size + look_back:]
    signals[f'{ticker}_Pred'] = predictions[ticker]

# Generate buy and sell signals based on the predicted and actual price differences
for ticker in tickers:
    signals[f'{ticker}_Signal'] = 0
    signals.loc[signals[f'{ticker}_Pred'] > signals[ticker], f'{ticker}_Signal'] = 1
    signals.loc[signals[f'{ticker}_Pred'] < signals[ticker], f'{ticker}_Signal'] = -1
```


## Backtesting and Performance Analysis
Simulate the portfolio's performance based on the generated signals and calculate performance metrics.

Example script:
```python
portfolio = signals[[f'{ticker}_Signal' for ticker in tickers]].sum(axis=1).shift(1) * signals[[ticker for ticker in tickers]].pct_change().sum(axis=1)
cumulative_returns = (1 + portfolio).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(cumulative_returns, label='Cumulative Returns')
plt.title('Arbitrage Strategy Using LSTM Predictions')
plt.legend()
plt.show()

# Print performance metrics
total_return = cumulative_returns[-1] - 1
annualized_return = (1 + total_return) ** (252 / len(cumulative_returns)) - 1
sharpe_ratio = (portfolio.mean() / portfolio.std()) * np.sqrt(252)
max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

print(f"Total Return: {total_return:.2f}")
print(f"Annualized Return: {annualized_return:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}")
```



## Examples
The examples directory contains various example scripts demonstrating different trading strategies and their backtesting results.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new strategies to add.


