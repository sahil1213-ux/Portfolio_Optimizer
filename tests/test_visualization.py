import pytest
import numpy as np
import matplotlib
from src.visualization import *

@pytest.fixture
def sample_data():
    # Mock Monte Carlo results
    np.random.seed(42)
    mc_returns = np.random.normal(0.1, 0.05, 1000)
    mc_volatility = np.random.normal(0.15, 0.03, 1000)
    mc_sharpe = (mc_returns - 0.02) / mc_volatility
    
    # Mock efficient frontier
    frontier_returns = np.linspace(0.05, 0.15, 10)
    frontier_volatility = np.linspace(0.1, 0.2, 10)
    
    # Mock portfolios
    tangency = {
        'return': 0.12,
        'volatility': 0.14,
        'weights': {'A': 0.6, 'B': 0.4}
    }
    
    min_vol = {
        'return': 0.08,
        'volatility': 0.09,
        'weights': {'A': 0.3, 'B': 0.7}
    }
    
    return mc_returns, mc_volatility, mc_sharpe, frontier_returns, frontier_volatility, tangency, min_vol

def test_plot_efficient_frontier(sample_data):
    mc_returns, mc_volatility, mc_sharpe, frontier_returns, frontier_volatility, tangency, min_vol = sample_data
    fig = plot_efficient_frontier(
        mc_returns, mc_volatility, mc_sharpe,
        frontier_returns, frontier_volatility,
        tangency, min_vol
    )
    assert isinstance(fig, matplotlib.figure.Figure)

def test_plot_weight_allocation():
    weights = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOG': 0.2}
    fig = plot_weight_allocation(weights)
    assert isinstance(fig, matplotlib.figure.Figure)