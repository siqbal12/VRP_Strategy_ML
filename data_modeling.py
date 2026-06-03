import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset_cleaned.csv')
df = df.dropna()
df_train = df.loc[df['Date'] < '2019-01-01', :]
df_validation = df.loc[(df['Date'] >= '2019-01-01') & (df['Date'] < '2022-01-01'), :]
df_test = df.loc[df['Date'] >= '2022-01-01', :]


X_columns = ['VIX', 'VIX_Change_5',
             'SPY_Log_Returns_Lag1', 'SPY_Log_Returns_Lag5','SPY_Log_Returns_Lag21',
             'Backward_RV_21', 'SPY_Drawdown']
y_column = 'VRP'

X_train = df_train.loc[:, X_columns]
X_validation = df_validation.loc[:, X_columns]
X_test = df_test.loc[:, X_columns]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

y_train = df_train[y_column]
y_validation = df_validation[y_column]
y_test = df_test[y_column]

lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)
lin_reg.fit(X_train_scaled, y_train)

# y_validation_pred = lin_reg.predict(X_validation)
y_validation_pred = lin_reg.predict(X_validation_scaled)
mse = mean_squared_error(y_validation, y_validation_pred)
r2 = r2_score(y_validation, y_validation_pred)

# y_test_pred = lin_reg.predict(X_test)
y_test_pred = lin_reg.predict(X_test_scaled)
df_test['VRP_pred'] = y_test_pred

df_test.to_csv('df_test.csv')



x = True