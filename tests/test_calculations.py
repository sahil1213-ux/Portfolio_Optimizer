import pytest
import numpy as np
import pandas as pd
from src.calculations import *

@pytest.fixture
def sample_data():
    prices = pd.DataFrame({
        'AAPL': [100, 101, 103, 102],
        'MSFT': [200, 202, 201, 205]
    })
    return prices

def test_daily_returns(sample_data):
    returns = calculate_daily_returns(sample_data)
    assert np.isclose(returns.iloc[0, 0], 0.01)  # AAPL day 1 return
    assert np.isclose(returns.iloc[1, 1], -0.00495, rtol=1e-3)  # MSFT day 2 return

def test_annualized_returns(sample_data):
    daily_returns = calculate_daily_returns(sample_data)
    annual = annualize_returns(daily_returns)
    assert annual.index.tolist() == ['AAPL', 'MSFT']

def test_cov_matrix(sample_data):
    cov = calculate_covariance_matrix(calculate_daily_returns(sample_data))
    assert cov.shape == (2, 2)
    assert cov.loc['AAPL', 'MSFT'] == cov.loc['MSFT', 'AAPL']  # Symmetry check

def test_portfolio_stats():
    weights = np.array([0.5, 0.5])
    annual_returns = pd.Series([0.10, 0.15], index=['A', 'B'])
    cov_matrix = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=['A', 'B'], columns=['A', 'B'])
    
    ret, vol, sharpe = portfolio_stats(weights, annual_returns, cov_matrix)
    assert np.isclose(ret, 0.125)
    assert np.isclose(vol, 0.1936, rtol=1e-2)
    assert sharpe > 0

    # pytest tests/test_cli.py