import pandas as pd
from src.data_loader import fetch_stock_data
from src.calculations import calculate_daily_returns, annualize_returns, calculate_covariance_matrix
from src.simulation import simulate_portfolios, find_optimal_portfolio
from src.optimization import calculate_efficient_frontier, find_tangency_portfolio
from src.visualization import plot_efficient_frontier, plot_weight_allocation
from typing import List, Optional
from src.utils.generate_optimization_report import generate_optimization_report
from pathlib import Path
import numpy as np

def run_optimization(args, output_dir):
        # Run pipeline
        try:
            results = optimization(
                tickers=args.tickers,
                period=args.period,
                risk_free_rate=args.risk_free_rate,
                max_weight=args.max_weight,
                output_dir=output_dir,
                generate_pdf=args.pdf,
            )
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return
    
        
        # Print summary
        print("\n=== Optimal Portfolio ===")
        print(f"Expected Return: {results['tangency']['return']:.2%}")
        print(f"Volatility: {results['tangency']['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['tangency']['sharpe']:.2f}")
        print("\nAllocation:")
        for ticker, weight in results['tangency']['weights'].items():
            print(f"  {ticker}: {weight:.1%} (${args.amount * weight:.2f})")
        
        return results


def optimization(
    tickers: List[str],
    period: str = "5y",
    risk_free_rate: float = 0.02,
    max_weight: float = 1.0,
    min_stocks: Optional[int] = None,
    amount: float = 10000.0,
    output_dir: Optional[Path] = None,
    generate_pdf: bool = False
) -> dict:
    """
    Core optimization pipeline
    
    Args:
        tickers: List of stock tickers
        period: Data period for historical stock data
        risk_free_rate: Risk-free rate for Sharpe ratio
        max_weight: Maximum weight per asset
        output_dir: Directory to save results
    
    Returns:
        Dictionary containing optimization results
    
    Raises:
        ValueError: If any step fails
    """
    # Step 1: Data loading
    try:
        prices = fetch_stock_data(tickers, period=period)
        # print("Prices DataFrame:")
        # print(prices.head())
        if prices.empty or prices.isnull().all().all():
            raise ValueError("Fetched price data is empty or contains only NaN values.")
        
        daily_returns = calculate_daily_returns(prices)
        annual_returns = annualize_returns(daily_returns)
        if annual_returns.empty:
            raise ValueError("Annual returns data is empty. Check the input price data.")
        
        cov_matrix = calculate_covariance_matrix(daily_returns)
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        tickers = [ticker for ticker in tickers if ticker not in str(e)]
        if not tickers:
            raise ValueError("No valid tickers available after failed downloads.")
        prices = fetch_stock_data(tickers, period=period)
    
    # Step 2: Simulation
    try:
        weights, mc_results = simulate_portfolios(
            annual_returns, 
            cov_matrix, 
            risk_free_rate=risk_free_rate
        )

        optimal_mc = find_optimal_portfolio(weights, mc_results)

    except Exception as e:
        raise ValueError(f"Error during simulation: {str(e)}")
    
    # Validate calculations
    if annual_returns.empty or cov_matrix.empty:
        raise ValueError("Invalid returns/covariance data")
    
    # Step 3: Optimization
    try:
        frontier = calculate_efficient_frontier(
            annual_returns, 
            cov_matrix, 
            risk_free_rate=risk_free_rate,
            max_weight_per_stock=max_weight
        )
        
        # Filter out failed points
        valid_idx = ~np.isnan(frontier['volatility'])
        frontier = {k: v[valid_idx] for k,v in frontier.items()}

        tangency = find_tangency_portfolio(
            annual_returns, 
            cov_matrix, 
            risk_free_rate=risk_free_rate,
            max_weight_per_stock=max_weight
        )
    except Exception as e:
        raise ValueError(f"Error during optimization: {str(e)}")
    
    # Convert weights to dict
    tangency['weights'] = dict(zip(tickers, tangency['weights']))
    optimal_mc['max_sharpe']['weights'] = dict(zip(tickers, optimal_mc['max_sharpe']['weights']))
    optimal_mc['min_vol']['weights'] = dict(zip(tickers, optimal_mc['min_vol']['weights']))
    
    # Step 4: Visualization
    if output_dir:
        try:
            plot_efficient_frontier(
                mc_results['returns'],
                mc_results['volatility'],
                mc_results['sharpe'],
                frontier['target_returns'],
                frontier['volatility'],
                tangency,
                optimal_mc['min_vol'],
                save_path=str(output_dir / "efficient_frontier.png")
            )
            
            plot_weight_allocation(
                tangency['weights'],
                title="Optimal Portfolio Allocation",
                save_path=str(output_dir / "allocation.png")
            )
            
            # Save raw data
            pd.DataFrame({
                'ticker': tickers,
                'weights': tangency['weights'].values(),
                'annual_return': annual_returns.values
            }).to_csv(output_dir / "weights.csv", index=False)
        except Exception as e:
            raise ValueError(f"Error during visualization: {str(e)}")
    
        print(f"Results saved to: {output_dir.resolve()}")
        
    # Step 5: Generate PDF report
    pdf_data = {
        'tickers': tickers,
        'period': period,
        'risk_free_rate': risk_free_rate,
        'amount': amount,
        'tangency': tangency,
        'annual_returns': annual_returns.to_dict(),
        'max_weight': max_weight if max_weight < 1.0 else None,
        'min_stocks': min_stocks
    }
    
    # Generate PDF if requested
    pdf_path = None
    if generate_pdf and output_dir:
        efficient_frontier_path = output_dir / "efficient_frontier.png"
        allocation_path = output_dir / "allocation.png"
        
        pdf_path = generate_optimization_report(
            output_dir,
            pdf_data,
            efficient_frontier_path,
            allocation_path,
        )
        print(f"PDF report generated: {pdf_path}")
    
    return {
        'tangency': tangency,
        'monte_carlo': optimal_mc,
        'frontier': frontier
    }
