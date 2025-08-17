# Import
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

"""# Load Stock Data"""

symbols = ['JPM', 'BAC', 'WFC', 'C']

print("Downloading historical stock data...")

ohlc = yf.download(symbols, start='2023-01-01', end='2025-01-01', group_by='tickers', auto_adjust=True)
ohlc

"""Format Data"""

all_data = pd.concat(
    [ohlc[ticker].loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']] for ticker in symbols],
    axis=1
)

all_data = all_data.set_axis([f'{ticker}_{val}' for ticker in symbols for val in ['Open', 'High', 'Low', 'Close', 'Volume']], axis=1)
next_day = all_data.shift(-1)
"""# Feature Engineering"""

from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import MACD

print("Creating features..")
all_data['rsi_14'] = RSIIndicator(all_data['JPM_Close'], window=14).rsi()

all_data['vwap'] = VolumeWeightedAveragePrice(
    all_data['JPM_High'],
    all_data['JPM_Low'],
    all_data['JPM_Close'],
    all_data['JPM_Volume'],
    window=14
).volume_weighted_average_price()

all_data['macd'] = MACD(all_data['JPM_Close'], window_slow=26, window_fast=12).macd_diff()

# signal represent if stock will be up (1) or down (0) the next day
all_data['JPM_signal'] = pd.Series(np.full(len(all_data), 0), index=all_data.index)
signal = next_day['JPM_Close'] > all_data['JPM_Close']
all_data.loc[signal, 'JPM_signal'] = 1

all_data.dropna(inplace=True)

jpm_signals = all_data['JPM_signal']
all_data.drop('JPM_signal', axis=1, inplace=True)

print("Done")

"""# Split Training and Walk Forward Data"""

train_data = all_data[all_data.index < '2024-01-01'].copy()
wf_jpm_signals = jpm_signals[jpm_signals.index >= '2024-01-01']
train_jpm_signals = jpm_signals.drop(jpm_signals[jpm_signals.index >= '2024-01-01'].index)
wf_data = all_data[all_data.index >= '2024-01-01'].copy()

"""# Model

## XGBoost
"""

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

template_model = XGBClassifier()
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 2, 3, 4]
}

random_search = RandomizedSearchCV(
    estimator=template_model,
    param_distributions=param_grid,
    n_iter=5,
    scoring='accuracy',
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# tune hyperparameters
print("Tuning XGBoost...")
random_search.fit(train_data, train_jpm_signals)

print(f"Best Model: {random_search.best_estimator_}")
print(f"Best Accuracy: {random_search.best_score_}")

"""## GRU"""

import torch.nn as nn
import torch.optim as optim

class GRUBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUBinaryClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out # logits

train_data_tensor = torch.tensor(train_data.to_numpy(), dtype=torch.float32).unsqueeze(0)
wf_data_tensor = torch.tensor(wf_data.to_numpy(), dtype=torch.float32).unsqueeze(0)
train_jpm_signals_tensor = torch.tensor(train_jpm_signals.to_numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(2)
wf_jpm_signals_tensor = torch.tensor(wf_jpm_signals.to_numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(2)

hidden_size = 32
input_size = len(train_data.columns)

model_gru = GRUBinaryClassifier(input_size, hidden_size, 5)
pos_weight = torch.tensor([train_jpm_signals_tensor.numel() / train_jpm_signals_tensor.sum() - 1])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model_gru.parameters(), lr=0.002)

print("Training GRU...")
epochs = 5
for epoch in range(epochs):
    model_gru.train()

    logits = model_gru(train_data_tensor)
    loss = criterion(logits, train_jpm_signals_tensor)

    loss.backward()
    optimizer.step()
    model_gru.zero_grad()

    model_gru.eval()
    with torch.no_grad():
        test_logits = model_gru(wf_data_tensor)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > 0.5).float()
        acc = (test_preds == wf_jpm_signals_tensor).float().mean().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")
print("Done")

"""# In-sample Trading Performance Test"""

print("In-sample Testing...")
model_xgb = random_search.best_estimator_

def backtest_xgb(model, data):
    log_rets = np.log(data['JPM_Close']).diff().shift(-1)
    log_rets.fillna(0, inplace=True)
    signal = model.predict(data)
    signal[signal == 0] = -1
    sig_rets = log_rets * signal
    sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()
    return sig_pf

def backtest_gru(model, data):
    model.eval()
    log_rets = np.log(data['JPM_Close']).diff().shift(-1)
    log_rets.fillna(0, inplace=True)
    data_tensor = torch.tensor(data.to_numpy(), dtype=torch.float32).unsqueeze(0)
    probs = torch.sigmoid(model(data_tensor)).squeeze().detach().numpy()
    decisions = probs > 0.5
    signal = [1 if decision else -1 for decision in decisions]
    sig_rets = log_rets * signal
    sig_pf = sig_rets[sig_rets > 0].sum() / sig_rets[sig_rets < 0].abs().sum()
    return sig_pf

print(f"In-sample Profit Factor (XGBoost): {backtest_xgb(model=model_xgb, data=train_data)}")
print(f"In-sample Profit Factor (GRU): {backtest_gru(model=model_gru, data=train_data)}")

"""# Walk Forward Performance Test

Test on 2024-2025 data
"""
print("Walk-forward Testing...")

print(f"Walk-forward Profit Factor (XGBoost): {backtest_xgb(model=model_xgb, data=wf_data)}")
print(f"Walk-forward Profit Factor (GRU): {backtest_gru(model=model_gru, data=wf_data)}")

