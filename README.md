Volatility Risk Premium Strategy via Machine Learning	6/26
- Machine Learning: Developed a regression framework to forecast the volatility risk premium (implied vol – realized vol) using historical SPY/VIX data, enabling market regime classification
- Systematic Trading: Generated signals from regression predictions to trade in high VRP regimes; dynamically timed SPY exposure to increase Sharpe (0.64 to 1.19) from classic buy-and-hold strategy (out-of-sample)
- Risk Management: Systematically de-risked during volatility stress to reduce maximum drawdown (23% to 6%)

- To collect the data:
    - Run data_playground.py
- To build the ML models:
    - Run data_modeling.py
- To re-run the trading strategy:
    - Run trading_strategy.py
