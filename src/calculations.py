import numpy as np
import pandas as pd
from typing import Tuple

def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns from price data.
    
    Args:
        prices: DataFrame with stock prices (columns = tickers, index = dates)
    
    Returns:
        DataFrame of daily returns
    
    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if prices.empty or prices.isnull().all().all():
        raise ValueError("Input price data is empty or contains only NaN values.")
    # Handle missing values explicitly
    prices = prices.ffill().bfill()
    return prices.pct_change(fill_method=None).dropna()

def annualize_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Annualize daily returns assuming 252 trading days/year.
    
    Args:
        daily_returns: DataFrame of daily returns
    
    Returns:
        Series of annualized returns per stock
    
    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if daily_returns.empty:
        raise ValueError("Input daily returns data is empty.")
    return daily_returns.mean() * 252

def calculate_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate annualized covariance matrix of returns.
    
    Args:
        daily_returns: DataFrame of daily returns
    
    Returns:
        Annualized covariance matrix (252 trading days)
    
    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if daily_returns.empty:
        raise ValueError("Input daily returns data is empty.")
    return daily_returns.cov() * 252

def portfolio_stats(
    weights: np.ndarray, 
    annual_returns: pd.Series, 
    cov_matrix: pd.DataFrame, 
    risk_free_rate: float = 0.02
) -> Tuple[float, float, float]:
    """
    Calculate portfolio statistics: return, volatility, Sharpe ratio.
    
    Args:
        weights: Asset allocation weights (sum to 1.0)
        annual_returns: Annualized returns per stock
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Risk-free rate for Sharpe ratio
    
    Returns:
        (portfolio_return, portfolio_volatility, sharpe_ratio)
    
    Raises:
        ValueError: If weights do not sum to 1.0 or dimensions mismatch.
    """
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.0.")
    if len(weights) != len(annual_returns):
        raise ValueError("Weights and annual returns must have the same length.")
    if cov_matrix.shape[0] != cov_matrix.shape[1] or cov_matrix.shape[0] != len(weights):
        raise ValueError("Covariance matrix dimensions must match the number of assets.")
    
    port_return = np.dot(weights, annual_returns)
    port_vol = np.sqrt(np.matmul(np.matmul(weights.T, cov_matrix.values), weights))
    sharpe = (port_return - risk_free_rate) / port_vol

    # Debugging statements
    print(f"Weights: {weights}")
    print(f"Annual Returns: {annual_returns}")
    print(f"Covariance Matrix:\n{cov_matrix}")
    print(f"Calculated Volatility: {port_vol}")
    return port_return, port_vol, sharpe
