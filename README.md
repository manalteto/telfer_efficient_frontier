# Efficient Frontier Analysis with WRDS and PyPortfolioOpt

This project uses **WRDS** data and the **PyPortfolioOpt** library to evaluate whether a given portfolio is positioned on the **efficient frontier**. It retrieves daily prices from CRSP, calculates returns and covariances, builds optimal portfolios, and visually compares:

- Individual stock positions
- A hypothetical equally weighted portfolio
- The optimized portfolio using maximum Sharpe ratio
- The entire efficient frontier

### Tools Used:
- Python
- WRDS (via `wrds` package)
- `PyPortfolioOpt` for portfolio optimization
- `scipy.optimize` for custom efficient frontier calculation
- `pandas_datareader`, `matplotlib`, and `statsmodels`

### Output:
- Risk-return plots with:
  - Individual assets
  - Optimal and examined portfolios
  - Efficient frontier

