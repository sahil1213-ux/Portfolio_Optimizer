import numpy as np
import pandas as pd
from typing import Tuple, Dict
from src.calculations import portfolio_stats

def generate_random_weights(
    n_assets: int,
    n_portfolios: int = 10000,
    max_weight: float = 1.0,
    min_stocks: int = None
) -> np.ndarray:
    """
    Generate random portfolio weights using Dirichlet distribution.

    Args:
        n_assets: Number of assets in the portfolio.
        n_portfolios: Number of random portfolios to generate.
        max_weight: Maximum weight allowed for any asset.
        min_stocks: Minimum number of stocks with non-zero allocation.

    Returns:
        np.ndarray: Array of shape (n_portfolios, n_assets) with weights summing to 1.

    Raises:
        ValueError: If n_assets or n_portfolios is less than 1.
    """
    # Validate inputs
    if n_assets < 1 or n_portfolios < 1:
        raise ValueError("Number of assets and portfolios must be greater than 0.")
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be between 0 and 1.")
    if min_stocks is not None and (min_stocks < 1 or min_stocks > n_assets):
        raise ValueError("min_stocks must be between 1 and n_assets.")

    # Generate weights
    if min_stocks:
        weights = np.zeros((n_portfolios, n_assets))
        for i in range(n_portfolios):
            weights[i] = _generate_weights_with_constraints(n_assets, max_weight, min_stocks)
    else:
        weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
        weights = np.clip(weights, 0, max_weight)
        weights = weights / weights.sum(axis=1, keepdims=True)  # Renormalize

    return weights


def _generate_weights_with_constraints(n_assets: int, max_weight: float, min_stocks: int) -> np.ndarray:
    """
    Generate a single portfolio weight vector honoring constraints.

    Args:
        n_assets: Number of assets in the portfolio.
        max_weight: Maximum weight allowed for any asset.
        min_stocks: Minimum number of stocks with non-zero allocation.

    Returns:
        np.ndarray: Weight vector for a single portfolio.
    """
    while True:
        weights = np.random.dirichlet(np.ones(n_assets))
        weights[weights < 0.01] = 0  # Treat <1% as unallocated
        weights = weights / weights.sum()  # Renormalize
        if (weights > 0).sum() >= min_stocks and np.all(weights <= max_weight):
            return weights

def simulate_portfolios(
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_portfolios: int = 10000
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run Monte Carlo simulation to generate random portfolios.
    
    Args:
        annual_returns: Annualized returns for each asset
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Risk-free rate for Sharpe ratio
        n_portfolios: Number of portfolios to simulate
    
    Returns:
        Tuple of (weights, results) where results is a dict containing:
        - 'returns': Portfolio returns
        - 'volatility': Portfolio volatilities
        - 'sharpe': Sharpe ratios
    
    Raises:
        ValueError: If annual_returns or cov_matrix is empty
    """
    if annual_returns.empty:
        raise ValueError("Annual returns data is empty.")
    if cov_matrix.empty:
        raise ValueError("Covariance matrix is empty.")
    if n_portfolios < 1:
        raise ValueError("Number of portfolios must be greater than 0.")
    
    n_assets = len(annual_returns)
    weights = generate_random_weights(n_assets, n_portfolios)
    
    results = {
        'returns': np.zeros(n_portfolios),
        'volatility': np.zeros(n_portfolios),
        'sharpe': np.zeros(n_portfolios)
    }
    
    for i in range(n_portfolios):
        try:
            ret, vol, sharpe = portfolio_stats(
                weights[i], annual_returns, cov_matrix, risk_free_rate
            )
            results['returns'][i] = ret
            results['volatility'][i] = vol
            results['sharpe'][i] = sharpe
        except Exception as e:
            raise ValueError(f"Error calculating portfolio stats for portfolio {i}: {str(e)}")
    
    return weights, results

def find_optimal_portfolio(
    weights: np.ndarray, 
    results: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Identify optimal portfolios (max Sharpe, min volatility).
    
    Args:
        weights: Array of portfolio weights from simulation
        results: Simulation results dict
    
    Returns:
        Dict of optimal portfolios:
        - 'max_sharpe': Weights/return/volatility for max Sharpe portfolio
        - 'min_vol': Weights/return/volatility for min volatility portfolio
    
    Raises:
        ValueError: If results dict is empty or invalid
    """
    if not results or not all(key in results for key in ['returns', 'volatility', 'sharpe']):
        raise ValueError("Results dictionary is invalid or incomplete.")
    
    max_sharpe_idx = np.argmax(results['sharpe'])
    min_vol_idx = np.argmin(results['volatility'])
    
    return {
        'max_sharpe': {
            'weights': weights[max_sharpe_idx],
            'return': results['returns'][max_sharpe_idx],
            'volatility': results['volatility'][max_sharpe_idx],
            'sharpe': results['sharpe'][max_sharpe_idx]
        },
        'min_vol': {
            'weights': weights[min_vol_idx],
            'return': results['returns'][min_vol_idx],
            'volatility': results['volatility'][min_vol_idx],
            'sharpe': results['sharpe'][min_vol_idx]
        }
    }