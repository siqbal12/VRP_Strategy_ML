import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRANSACTION_COST = 0.0005

strategy = 'Machine Learning'

#Given our y_pred from data_modeling.py, sees how well our strategy performs in test dataset
df_test = pd.read_csv('df_test.csv')


#Get signal
signal_threshold = np.percentile(df_test['VRP_pred'], 90)
df_test['Signal'] = (df_test['VRP_pred'] > signal_threshold).astype(int)


#Based on signal, trade!
# Trade package A: If IV > RV, vol is overpriced -> bullish equity market -> buy spy
df_test['Position'] = df_test['Signal'] if strategy == 'Machine Learning' else 1
# df_test['Position'] = 1 #Buy and Hold
# df_test['Position'] = (np.random.rand(df_test.shape[0]) < df_test['Signal'].mean()).astype(int) # Random position

df_test['Strategy Return'] = df_test['Position'] * df_test['SPY_Returns'].shift(-1)
df_test['Cumulative Return'] = (1 + df_test['Strategy Return']).cumprod()

# Trade package B: If IV > RV, vol is overpriced -> short volatility
# df_test['Position'] = df_test['Signal']
# df_test['Strategy Return'] = df_test['Position'] * df_test['VRP']
# df_test['Cumulative Return'] = (1 + df_test['Strategy Return']).cumprod()

df_test['Trades'] = df_test['Signal'].diff().abs()
df_test['Trading Costs'] = df_test['Trades'] * TRANSACTION_COST
df_test['Strategy Return'] = df_test['Strategy Return'] - df_test['Trading Costs']

df_test = df_test.dropna()

annualized_sharpe = df_test['Strategy Return'].mean() * np.sqrt(252) / df_test['Strategy Return'].std()

#Max drawdown
running_max = df_test['Cumulative Return'].cummax()
drawdown = (df_test['Cumulative Return'] / running_max) - 1
max_drawdown = drawdown.min()

#Cumulative Return plot
plt.figure(figsize=(5,5))
plt.plot(df_test['Date'], df_test['Cumulative Return'], label='Cumulative Return', color='blue')
plt.plot(df_test['Date'], [1] * df_test.shape[0], label='Baseline', color='red', ls='--')
plt.title(f"Cumulative Return")
plt.xlabel('Time')
plt.ylabel('Cumulative Return')
plt.xticks(df_test['Date'][::100],rotation=45)
# plt.ylim(overall_min * 0.9, overall_max * 1.1)
plt.legend()
plt.tight_layout()
plt.show()


#Diagnostics

#When are we in the market?
# We do not buy good days
# But we enter before bad/volatile/declining periods
spy_behavior = df_test.loc[df_test['Position']==1, 'SPY_Returns'].mean()


#Regime behavior
# We do not invest in bull markets
# But we will invest in bear markets (counter-cyclical exposure timing)
regime_behavior = df_test.groupby('Is_Bear_Regime')['Position'].mean()

#Returns in quantiles
# Low VRP_pred (slighlty positive returns)
# Med VRP_pred (flat/worse)
# High VRP_pred (best returns)
df_test['VRP_rank'] = df_test['VRP_pred'].rank(pct=True)
df_test.groupby(pd.cut(df_test['VRP_rank'], [0,0.5,0.9,1.0]))['SPY_Returns'].mean()

x = True


