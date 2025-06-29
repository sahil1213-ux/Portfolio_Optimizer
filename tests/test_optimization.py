import pytest
import numpy as np
import pandas as pd
from src.optimization import *

@pytest.fixture
def sample_inputs():
    annual_returns = pd.Series({'A': 0.10, 'B': 0.15})
    cov_matrix = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], 
                            index=['A', 'B'], columns=['A', 'B'])
    return annual_returns, cov_matrix

def test_efficient_frontier(sample_inputs):
    annual_returns, cov_matrix = sample_inputs
    frontier = calculate_efficient_frontier(annual_returns, cov_matrix)
    
    assert len(frontier['target_returns']) == 20
    assert frontier['volatility'].shape == (20,)
    assert np.all(frontier['weights'].sum(axis=1) == pytest.approx(1.0))

def test_tangency_portfolio(sample_inputs):
    annual_returns, cov_matrix = sample_inputs
    tangency = find_tangency_portfolio(annual_returns, cov_matrix)
    
    assert tangency['sharpe'] > 0
    assert tangency['volatility'] > 0
    assert np.isclose(tangency['weights'].sum(), 1.0)

def test_weight_constraints(sample_inputs):
    annual_returns, cov_matrix = sample_inputs
    frontier = calculate_efficient_frontier(annual_returns, cov_matrix, max_weight_per_stock=0.5)
    assert np.all(frontier['weights'] <= 0.5 + 1e-8)  # Allow for floating-point error
    