# jpm-strategy
A financial strategy for JPM using OHLC data from other top-performing financials stocks (BAC, WFC, C) and basic technical indictors (RSI, VWAP, MACD).

The model uses yfinance to retrieve historical stock data for training and walk-foward testing. For feature engineering, technical indicators are imported from ta. The model aims to predict daily change in JPM stock price, that is, whether the price will close on a higher or lower price on the next day, given historical data. The strategy compares two potential models: XGBoost (hyperparameters were tuned using random search with time-series cross validation) and GRU (used pytorch's Adam algorithm for stochastic optimization). Both models were trained on 2023-2024 historical data. The walk-forward test was performed on 2024-2025 historical data; models were benchmarked by calculating cumulative daily log returns using the predicted signals. 

It was found that GRU performed significantly better than XGBoost: 

Walk-forward Profit Factor (XGBoost): 1.1513593544067728
Walk-forward Profit Factor (GRU): 1.3974468768919466
