# Install dependencies (run these once in your environment)
# pip uninstall ipython-sql
# pip install wrds PyPortfolioOpt pandas_datareader

import wrds
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import scipy.optimize as sco
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# ------------------------------------------------------
# 1. Connect to WRDS and fetch stock price data
# ------------------------------------------------------

conn = wrds.Connection()

#Ticker symbols for the stocks

ticker_symbols = [
    'ZIN',   # BMO Equal Weight Industrials Index ETF
    'KKR',      # KKR & Co. Inc.
    'VGT',      # Vanguard World Fund - Vanguard Information Technology ETF
    'MSFT',     # Microsoft Corporation
    'BLK' ,     # BlackRock, Inc.
    'XFN',      #iShares S&P/TSX Capped Financials Index ETF
    'MCK',      #McKesson Corporation
    'PLC',      #Park Lawn Corporation
    'V',        #Visa Inc.
    'FENY',     #Fidelity Covington Trust - Fidelity MSCI Energy Index ETF
    'MNST',     #Monster Beverage Corporation
    'CIGI',     #Colliers International Group Inc.
    'TOI',      #Topicus.com Inc.
    'TTWO',     #Take-Two Interactive Software, Inc.
    'JNJ',      #Johnson & Johnson
    'MFC',      #Manulife Financial Corporation
    'XMA',      #iShares S&P/TSX Capped Materials Index ETF
    'VCR',      #Vanguard World Fund - Vanguard Consumer Discretionary ETF
    'BEP.UN',   #Brookfield Renewable Partners L.P.
    'DD',       #DuPont de Nemours, Inc.
    'TPZ',      #Topaz Energy Corp.
    'BNS',      #The Bank of Nova Scotia
    'SU',       #Suncor Energy Inc.
    'DOW',      #Dow Inc.
    'IYZ',      #iShares Trust - iShares U.S. Telecommunications ETF
    'PINS',     #Pinterest, Inc.
    'CTVA',     #Corteva, Inc.
    'FTS',      #Fortis Inc.
    'BEPC',     #Brookfield Renewable Corporation
]

ticker_string = "','".join(ticker_symbols)

# Step 1: Map tickers to CUSIPs
mapping_data = conn.raw_sql(f"""
    SELECT ticker, cusip
    FROM crsp.stocknames
    WHERE ticker IN ('{ticker_string}')
""")

cusip_string = "','".join(mapping_data['cusip'].tolist())

# Step 2: Pull pricing data from CRSP MSF
data = conn.raw_sql(f"""
    SELECT cusip, date, prc, ret, shrout
    FROM crsp.msf
    WHERE cusip IN ('{cusip_string}')
    AND date >= '2021-12-31'
""")

conn.close()

# ------------------------------------------------------
# 2. Clean and prepare data
# ------------------------------------------------------

# Calculate daily returns
data['ret'] = data.groupby('cusip')['prc'].pct_change()
data.dropna(inplace=True)

# Pivot to get return series
returns_data = data.pivot(index='date', columns='cusip', values='ret')
number_assets = len(returns_data.columns)

# ------------------------------------------------------
# 3. Mean-variance optimization using PyPortfolioOpt
# ------------------------------------------------------

mu = returns_data.mean()           # Daily mean returns
S = returns_data.cov()             # Covariance matrix

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
optimal_weights = ef.clean_weights()
print("Optimal Weights:", optimal_weights)

# ------------------------------------------------------
# 4. Custom efficient frontier (manual method)
# ------------------------------------------------------

bounds = [(0, 1) for _ in range(number_assets)]

def calc_portfolio_perf(weights, mu, S):
    r = np.dot(mu, weights)
    sigma = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    return r, sigma

def efficient_frontier(mu, S, returns_range):
    efficients = []
    for ret in returns_range:
        constraints = (
            {'type': 'eq', 'fun': lambda w: calc_portfolio_perf(w, mu, S)[0] - ret},
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        result = sco.minimize(lambda w: calc_portfolio_perf(w, mu, S)[1],
                              [1./number_assets]*number_assets,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints)
        efficients.append(result)
    return efficients

returns_range = np.linspace(mu.min(), mu.max(), 50)
efficients = efficient_frontier(mu, S, returns_range)


#Plot of the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter([p['fun'] for p in efficients], returns_range, marker='x', color='red', s=100)
plt.title('Efficient Frontier of Telfer Capital Fund’s Stocks')
plt.xlabel('Volatility')
plt.ylabel('Expected returns')
plt.show()

# ------------------------------------------------------
# 5. Check if equal-weighted portfolio is on the frontier
# ------------------------------------------------------

def check_on_frontier(weights, mu, S, returns_range, efficients):
    port_return, port_std = calc_portfolio_perf(weights, mu, S)
    for efficient_return, efficient_risk in zip(returns_range, [p['fun'] for p in efficients]):
        if np.isclose(efficient_return, port_return) and np.isclose(efficient_risk, port_std):
            return True, port_return, port_std
    return False, port_return, port_std

equal_weights = [1./number_assets]*number_assets
is_on_frontier, examined_return, examined_std = check_on_frontier(equal_weights, mu, S, returns_range, efficients)

print(f"The portfolio is {'on' if is_on_frontier else 'not on'} the efficient frontier.")
print(f"Portfolio Return: {examined_return}")
print(f"Portfolio Risk: {examined_std}")

# ------------------------------------------------------
# 6. Plot: Risk-Return Diagram
# ------------------------------------------------------

# Annualize return and risk
annualized_returns = mu * 252
individual_risks = np.sqrt(np.diag(S)) * np.sqrt(252)

optimal_weights_list = list(optimal_weights.values())
port_return = np.dot(annualized_returns, optimal_weights_list)
port_vol = np.sqrt(np.dot(optimal_weights_list, np.dot(S * 252, optimal_weights_list)))

plt.figure(figsize=(10, 6))
plt.scatter(individual_risks, annualized_returns, color='blue', label='Individual Stocks')
plt.scatter(port_vol, port_return, color='green', marker='*', s=200, label='Optimized Portfolio')
plt.scatter([p['fun']*np.sqrt(252) for p in efficients], returns_range * 252,
            color='red', marker='x', s=100, label='Efficient Frontier')
plt.title('Risk-Return Diagram (Annualized)')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
