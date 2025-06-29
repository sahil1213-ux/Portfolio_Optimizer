import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from matplotlib.ticker import PercentFormatter, FuncFormatter
import pandas as pd

def plot_efficient_frontier(
    mc_returns: np.ndarray,
    mc_volatility: np.ndarray,
    mc_sharpe: np.ndarray,
    frontier_returns: np.ndarray,
    frontier_volatility: np.ndarray,
    tangency_portfolio: Dict[str, float],
    min_vol_portfolio: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Monte Carlo simulations, efficient frontier, and optimal portfolios.
    
    Args:
        mc_returns: Array of Monte Carlo portfolio returns
        mc_volatility: Array of Monte Carlo portfolio volatilities
        mc_sharpe: Array of Monte Carlo Sharpe ratios
        frontier_returns: Efficient frontier target returns
        frontier_volatility: Efficient frontier volatilities
        tangency_portfolio: Max-Sharpe portfolio stats
        min_vol_portfolio: Minimum volatility portfolio stats (optional)
        save_path: If provided, save the plot to this path
    
    Returns:
        matplotlib Figure object
    
    Raises:
        ValueError: If any input array is empty or invalid
    """
    # Input validation
    if mc_returns.size == 0 or mc_volatility.size == 0 or mc_sharpe.size == 0:
        raise ValueError("Monte Carlo simulation data is empty.")
    if frontier_returns.size == 0 or frontier_volatility.size == 0:
        raise ValueError("Efficient frontier data is empty.")
    if not tangency_portfolio or 'volatility' not in tangency_portfolio or 'return' not in tangency_portfolio:
        raise ValueError("Tangency portfolio data is invalid.")
    if min_vol_portfolio and ('volatility' not in min_vol_portfolio or 'return' not in min_vol_portfolio):
        raise ValueError("Minimum volatility portfolio data is invalid.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Monte Carlo simulations
    sc = ax.scatter(
        mc_volatility, 
        mc_returns, 
        c=mc_sharpe, 
        cmap='viridis', 
        alpha=0.5,
        label='Random Portfolios'
    )
    plt.colorbar(sc, label='Sharpe Ratio')
    
    # Plot efficient frontier
    ax.plot(
        frontier_volatility, 
        frontier_returns, 
        'r--', 
        linewidth=2,
        label='Efficient Frontier'
    )
    
    # Highlight optimal portfolios
    ax.scatter(
        tangency_portfolio['volatility'], 
        tangency_portfolio['return'], 
        c='black', 
        s=200, 
        marker='*',
        label='Tangency Portfolio (Max Sharpe)'
    )
    
    if min_vol_portfolio:
        ax.scatter(
            min_vol_portfolio['volatility'], 
            min_vol_portfolio['return'], 
            c='blue', 
            s=200, 
            marker='o',
            label='Min Volatility Portfolio'
        )
    
    # Formatting
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Portfolio Optimization')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_weight_allocation(
    weights: Dict[str, float],
    title: str = "Portfolio Allocation",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a pie chart of portfolio weights.
    
    Args:
        weights: Dictionary of {ticker: weight}
        title: Chart title
        save_path: If provided, save the plot to this path
    
    Returns:
        matplotlib Figure object
    
    Raises:
        ValueError: If weights dictionary is empty or invalid
    """
    # Input validation
    if not weights or not all(isinstance(v, (int, float)) for v in weights.values()):
        raise ValueError("Weights dictionary is empty or contains invalid values.")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Sort weights for consistent colors
    sorted_weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
    
    ax.pie(
        sorted_weights.values(),
        labels=sorted_weights.keys(),
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    ax.set_title(title)
    ax.axis('equal')  # Equal aspect ratio ensures pie is circular
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def plot_backtest_results(
    portfolio_values: pd.Series,
    benchmarks: dict,
    initial_value: float = 10000,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot portfolio vs benchmarks over time.
    
    Args:
        portfolio_values: Portfolio values over time as a pandas.Series.
        benchmarks: Dictionary of benchmark values over time as pandas.Series.
        initial_value: Initial portfolio value for normalization.
        save_path: If provided, save the plot to this path.
    
    Returns:
        matplotlib Figure object.
    
    Raises:
        ValueError: If portfolio_values or benchmarks are invalid.
    """
    # Input validation
    if not isinstance(portfolio_values, pd.Series):
        raise ValueError("Portfolio values must be a pandas.Series.")
    if not benchmarks or not all(isinstance(v, pd.Series) for v in benchmarks.values()):
        raise ValueError("Benchmarks must be a dictionary of pandas.Series.")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize portfolio values to initial value
    portfolio_norm = portfolio_values / initial_value
    ax.plot(portfolio_norm, label='Optimized Portfolio', lw=2)
    
    # Normalize benchmark values and plot
    for name, values in benchmarks.items():
        bench_norm = values / values.iloc[0]
        ax.plot(bench_norm, label=name, alpha=0.7)
    
    ax.set_title('Backtest Performance')
    ax.set_ylabel('Growth of $1')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig