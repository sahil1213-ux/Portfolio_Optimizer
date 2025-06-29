from src.visualization import plot_backtest_results
from src.utils.generate_backtest_report import generate_backtest_report
import pandas as pd
from pathlib import Path

def backtest_command(args, output_dir):
    """Handle backtest subcommand"""
    from ..backtest import PortfolioBacktester
    
    # Convert output_dir to Path object
    output_dir = Path(output_dir)

    bt = PortfolioBacktester(
        tickers=args.tickers,
        initial_amount=args.amount
    )
    
    bt.load_data(args.start_date, args.end_date)
    results = bt.run_backtest(
        train_period=args.train_period,
        test_period=args.test_period,
        rebalance_freq=args.rebalance,
        transaction_cost=args.transaction_cost
    )
    
    # Print results
    print(f"\n=== Backtest Results ({args.start_date} to {args.end_date}) ===")
    print(f"CAGR: {results['cagr']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    
    print("\nBenchmarks:")
    for name, ret in results['benchmarks'].items():
        print(f"  {name}: {ret:.2%}")
    
    # Convert benchmarks to pandas.Series
    benchmark_series = {
        name: pd.Series(data, index=results['values'].index)
        for name, data in results['benchmarks'].items()
    }
    
    # Save plot
    plot_backtest_results(
        portfolio_values=results['values'],
        benchmarks=benchmark_series,
        save_path=output_dir / "backtest_performance.png"
    )
    print(f"Backtest performance plot saved to: {output_dir / 'backtest_performance.png'}")

    # PDF
    pdf_data = {
        'CAGR': results['cagr'],
        'Sharpe Ratio': results['sharpe'],
        'Max Drawdown': results['max_drawdown'],
        'benchmarks': results['benchmarks']
    }

    # Generate PDF if requested
    pdf_path = None
    if args.pdf:
        backtest_path = output_dir / "backtest_performance.png"
        
        pdf_path = generate_backtest_report(
            output_dir,
            pdf_data,
            backtest_path,
        )
        print(f"PDF report generated: {pdf_path}")