import argparse
from pathlib import Path
from src.utils.validation import validate_arguments
import sys
from src.commands.optimization import run_optimization
from src.commands.backtest import backtest_command

# Forcefully set the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

def main():
    parser = argparse.ArgumentParser(description="Stock Portfolio Optimizer")
    
    # Subcommands
    parser.add_argument(
        "command", 
        choices=["optimize", "backtest"], 
        help="Choose whether to optimize a portfolio or backtest a strategy"
    )

    # Shared arguments
    parser.add_argument("--tickers", nargs="+", required=True, help="List of stock tickers (e.g., AAPL MSFT GOOG)")
    parser.add_argument("--amount", type=float, default=10000, help="Total investment amount in USD (default: 10000)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results (default: ./results)")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report")
    
    # Optimization-specific arguments
    parser.add_argument("--period", default="5y", help="Data period (1y, 3y, 5y, etc., default: 5y)")
    parser.add_argument("--risk-free-rate", type=float, default=0.02, help="Risk-free rate for Sharpe ratio (default: 0.02)")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Maximum weight per asset (0.0-1.0, default: no limit)")
    parser.add_argument("--min-stocks", type=int, help="Minimum number of stocks to include (e.g., 3)")
    
    # Backtesting-specific arguments
    parser.add_argument("--start-date", default="2015-01-01", help="Start date for backtesting (default: 2015-01-01)")
    parser.add_argument("--end-date", default="2023-12-31", help="End date for backtesting (default: 2023-12-31)")
    parser.add_argument("--train-period", default="3Y", help="Training period for backtesting (default: 3Y)")
    parser.add_argument("--test-period", default="1Y", help="Testing period for backtesting (default: 1Y)")
    parser.add_argument("--rebalance", choices=["M", "Q", "Y"], default="Q", help="Rebalancing frequency (default: Q)")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Transaction cost (default: 0.001)")
    
    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Handle subcommands
    if args.command == "optimize":
        run_optimization(args, output_dir)
    elif args.command == "backtest":
        backtest_command(args, output_dir)

    
if __name__ == "__main__":
    main()

# Portfolio Optimization with Default Parameters
# python src/cli.py optimize --tickers AAPL MSFT --amount 5000 --max-weight 0.8 --output-dir optimised_results

# Backtest with Custom Parameters
#  python src/cli.py backtest --tickers AAPL MSFT --amount 5000 --start-date 2015-01-01 --end-date 2023-12-31 --train-period 3Y --test-period 1Y --rebalance Q --transaction-cost 0.001 --output-dir backtest_results

# Portfolio Optimization with Minimum Stocks Constraint
# python src/cli.py optimize --tickers AAPL MSFT GOOG TSLA --amount 10000 --max-weight 0.5 --min-stocks 3 --output-dir constrained_results --pdf

# Backtest with PDF Report
# python src/cli.py backtest --tickers AAPL MSFT --amount 5000 --start-date 2015-01-01 --end-date 2023-12-31 --train-period 3Y --test-period 1Y --rebalance M --transaction-cost 0.002 --pdf --output-dir backtest_results_pdf

# NOTE: Sometimes it may not run as excepted to try like this:-
# set PYTHONPATH=C:\Users\sahil\Downloads\Data_Analytics\Internships\ML-DevifyX\Portfolio_Optimizer python src/cli.py optimize --tickers AAPL MSFT --amount 5000 --max-weight 0.8 --output-dir optimised_results





