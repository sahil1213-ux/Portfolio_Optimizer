import pytest
import numpy as np
import pandas as pd
from src.simulation import *

@pytest.fixture
def sample_inputs():
    annual_returns = pd.Series({'A': 0.10, 'B': 0.15})
    cov_matrix = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], 
                            index=['A', 'B'], columns=['A', 'B'])
    return annual_returns, cov_matrix

def test_generate_random_weights():
    weights = generate_random_weights(n_assets=3, n_portfolios=100)
    assert weights.shape == (100, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)

def test_simulate_portfolios(sample_inputs):
    annual_returns, cov_matrix = sample_inputs
    weights, results = simulate_portfolios(annual_returns, cov_matrix, n_portfolios=1000)
    
    assert len(results['returns']) == 1000
    assert len(results['volatility']) == 1000
    assert not np.isnan(results['sharpe']).any()

def test_optimal_portfolios(sample_inputs):
    annual_returns, cov_matrix = sample_inputs
    weights, results = simulate_portfolios(annual_returns, cov_matrix)
    optimal = find_optimal_portfolio(weights, results)
    
    assert optimal['max_sharpe']['sharpe'] >= optimal['min_vol']['sharpe']
    assert optimal['min_vol']['volatility'] <= results['volatility'].mean()

    """
    Calculate the efficient frontier using quadratic optimization.
    
    Args:
        annual_returns: Expected returns for each asset
        cov_matrix: Covariance matrix of returns
        risk_free_rate: Risk-free rate for tangency portfolio
        target_returns: Array of target returns for frontier points
        max_weight: Maximum allowed weight per asset (default: no limit)
    
    Returns:
        Dictionary containing:
        - 'target_returns': Target return for each frontier point
        - 'volatility': Portfolio volatility at each target return
        - 'weights': Optimal weights for each target return
    
    Raises:
        ValueError: If annual_returns or cov_matrix is empty
    """    