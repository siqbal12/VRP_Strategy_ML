import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tickers = ["SPY", "QQQ", "^VIX"]

# data = yf.download(
#     tickers,
#     start="2010-01-01",
#     end="2026-01-01",
#     auto_adjust=True,
#     progress=False
# )["Close"]
# data = data.rename(columns={'^VIX', 'VIX'})

data = pd.read_csv('dataset.csv')

data['QQQ_Returns'] = data['QQQ'].pct_change()
data['SPY_Returns'] = data['SPY'].pct_change()

data['QQQ_Log_Returns'] = np.log(data['QQQ'] / data['QQQ'].shift(1))
data['SPY_Log_Returns'] = np.log(data['SPY'] / data['SPY'].shift(1))

data['Backward_RV_21'] = data['SPY_Log_Returns'].rolling(21).std() * np.sqrt(252)
data['Forward_RV_21'] = (
    data['SPY_Log_Returns'].shift(-1)
    .rolling(21)
    .std()
    .shift(-(21-1))   # align back to time t
    * np.sqrt(252)
)

data['IV'] = data['VIX'] / 100

data['VRP'] = data['IV'] - data['Forward_RV_21']
data['Is_VRP_Positive'] = (data['VRP'] > 0).astype(int)

data['SPY_Log_Returns_Lag1'] = data['SPY_Log_Returns'].shift(1)
data['SPY_Log_Returns_Lag5'] = data['SPY_Log_Returns'].shift(5)
data['SPY_Log_Returns_Lag21'] = data['SPY_Log_Returns'].shift(21)
data['SPY_Volatility_21'] = data['SPY_Log_Returns'].rolling(21).std() * np.sqrt(252)
data['VIX_Change_5'] = data['VIX'] - data['VIX'].shift(5)
data['Is_High_Vol_Regime'] = (data['VIX'] > 25).astype(int)
spy_running_max = data['SPY'].cummax()
data['SPY_Drawdown'] = (data['SPY'] / spy_running_max) - 1
data['Is_Bear_Regime'] = (data['SPY_Drawdown'] < -0.10).astype(int)

data = data.dropna()

#VRP Mean
print('VRP Mean: ', data['VRP'].mean())

#VRP Histogram
# plt.figure()
# plt.hist(data['VRP'])
# plt.show()

#VRP Over Time
# plt.figure(figsize=(8, 8))
# plt.scatter(data['Date'], data['VRP'], label='VRP (Scatter)', color='blue')
# plt.plot(data['Date'], data['VRP'], label='VRP (Line)', color='red')
# plt.plot(data['Date'], data['IV'], label='IV (Line)', color='green')
# plt.xlabel('Date')
# plt.ylabel('VRP')
# plt.xticks(data['Date'][::500], rotation=60)
# plt.suptitle('VRP Over Time')
# plt.legend()
# plt.show()

#VRP (Based on Regime)
print('VRP Mean (High Vol): ', data.loc[data['Is_High_Vol_Regime']==1, 'VRP'].mean())
print('VRP Mean (Low Vol): ', data.loc[data['Is_High_Vol_Regime']==0, 'VRP'].mean())

data.to_csv('dataset_cleaned.csv')


x = True