import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from typing import Dict
import logging
logging.basicConfig(level=logging.INFO)

def calculate_efficient_frontier(
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    target_returns: np.ndarray = None,
    max_weight_per_stock: float = 1.0,
    min_stocks: int = None
) -> Dict[str, np.ndarray]:
    """
    Calculate the efficient frontier using quadratic optimization.
    """
    # Input validation
    if annual_returns is None or cov_matrix is None:
        raise ValueError("annual_returns and cov_matrix cannot be None")
    
    if len(annual_returns) == 0 or len(cov_matrix) == 0:
        raise ValueError("Inputs cannot be empty")
    
    n_assets = len(annual_returns)
    
    # Validate covariance matrix shape
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"cov_matrix must be {n_assets}x{n_assets}")
    
    # Set default target returns if not provided
    if target_returns is None:
        logging.debug("target_returns is None, initializing default values.")
        min_return = max(annual_returns.min(), 0)  # Avoid negative returns
        target_returns = np.linspace(
            min_return,
            annual_returns.max(), 
            20
        )
    logging.info(f"target_returns initialized with {len(target_returns)} points.")
    
    frontier = {
        'target_returns': target_returns,
        'volatility': np.zeros_like(target_returns),
        'weights': np.zeros((len(target_returns), n_assets)),
        'sharpe': np.zeros_like(target_returns)  # Initialize Sharpe ratios as zeros
        
    }
    
    for i, target in enumerate(target_returns):
        logging.debug(f"Optimizing for target return: {target:.2%}")
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, annual_returns) - target}
        ]
        bounds = [(0, max_weight_per_stock) for _ in range(n_assets)]
        
        # Minimum stocks constraint (if specified)
        if min_stocks and min_stocks > 0:
            constraints.append(
                NonlinearConstraint(
                    fun=lambda w: np.sum(w > 0.01),  # Count stocks with >1% allocation
                    lb=min_stocks,
                    ub=n_assets
                )
            )
        # Enhanced optimization settings
        result = minimize(
            fun=lambda w: np.sqrt(w.T @ cov_matrix.values @ w),
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': 1000,          # Increased from default 100
                'ftol': 1e-10,            # Tighter tolerance
                'disp': False
            }
        )
        
        if not result.success:
            logging.warning(f"Optimization failed for target return {target:.2%} - using fallback")
            result.x = np.ones(n_assets) / n_assets  # Fallback to equal weights
        
        frontier['volatility'][i] = result.fun
        frontier['weights'][i] = result.x
        frontier['sharpe'][i] = (target - risk_free_rate) / result.fun  # Calculate Sharpe ratio
    
    return frontier

def find_tangency_portfolio(
    annual_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    max_weight_per_stock: float = 1.0
) -> Dict[str, float]:
    """
    Find the tangency portfolio (max Sharpe ratio) using optimization.
    
    Args:
        annual_returns: Expected returns for each asset
        cov_matrix: Covariance matrix of returns
        risk_free_rate: Risk-free rate
        max_weight_per_stock: Maximum allowed weight per asset
    
    Returns:
        Dictionary containing weights, return, volatility, and Sharpe ratio
    
    Raises:
        ValueError: If annual_returns or cov_matrix is empty
    """
    if annual_returns.empty:
        raise ValueError("Annual returns data is empty.")
    if cov_matrix.empty:
        raise ValueError("Covariance matrix is empty.")
    if max_weight_per_stock <= 0 or max_weight_per_stock > 1:
        raise ValueError("max_weight_per_stock must be between 0 and 1.")
    
    n_assets = len(annual_returns)
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, max_weight_per_stock) for _ in range(n_assets)]

    result = minimize(
    fun=lambda w: np.sqrt(w.T @ cov_matrix.values @ w),  # Portfolio volatility
    x0=np.ones(n_assets) / n_assets,  # Initial guess (equal weights)
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}  # Increase the iteration limit
)
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    weights = result.x
    port_return = np.dot(weights, annual_returns)
    port_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
    sharpe = (port_return - risk_free_rate) / port_vol
    
    return {
        'weights': weights,
        'return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe
    }