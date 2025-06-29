# Portfolio Optimizer

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Modern Portfolio Theory (MPT) implementation with Monte Carlo simulation, efficient frontier optimization, and backtesting capabilities.

## Features

### Core Features

- 📊 **Data Ingestion**: Fetch historical prices from Yahoo Finance
- 📈 **Return Analysis**: Daily/annualized returns calculation
- 🔄 **Covariance Matrix**: Asset risk relationships
- 🎲 **Monte Carlo Simulation**: 20,000+ random portfolios
- 📉 **Efficient Frontier**: Optimal risk-return portfolios
- ⭐ **Optimal Portfolios**: Max Sharpe & Min Volatility
- 💻 **CLI Interface**: Simple command-line control

### Advanced Features

- 🔒 **Constraints**:
  - Max allocation per stock (e.g., `--max-weight 0.3`)
  - Minimum number of stocks (e.g., `--min-stocks 2`)
- 📤 **Exports**:
  - PDF reports with visualizations
  - CSV files with weights/returns
- 🔄 **Rebalancing**: Quarterly/Monthly/Annual
- ⏳ **Backtesting**: Out-of-sample performance testing

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/portfolio-optimizer.git
   cd portfolio-optimizer

   ```

2. Create virtual environment:
```

   conda activate -p venv python==3.12
   conda activate venv/
```
3. Install dependencies:

```
pip install -r requirements.txt

```

## USAGE

### Portfolio Optimization with Default Parameters

```
python src/cli.py optimize --tickers AAPL MSFT --amount 5000 --max-weight 0.8 --output-dir optimised_results
```

### Backtest with Custom Parameters

```
python src/cli.py backtest --tickers AAPL MSFT --amount 5000 --start-date 2015-01-01 --end-date 2023-12-31 --train-period 3Y --test-period 1Y --rebalance Q --transaction-cost 0.001 --output-dir backtest_results
```

### Portfolio Optimization with Minimum Stocks Constraint

```
python src/cli.py optimize --tickers AAPL MSFT GOOG TSLA --amount 10000 --max-weight 0.5 --min-stocks 3 --output-dir constrained_results --pdf
```

### Backtest with PDF Report

```
python src/cli.py backtest --tickers AAPL MSFT --amount 5000 --start-date 2015-01-01 --end-date 2023-12-31 --train-period 3Y --test-period 1Y --rebalance M --transaction-cost 0.002 --pdf --output-dir backtest_results_pdf
```

## Command Reference

### Optimization Mode

![Image](https://github.com/user-attachments/assets/a6eb9cb2-6473-45b0-a240-e378b6704800)

### Backtest Mode

![Image](https://github.com/user-attachments/assets/4d934aff-d731-4580-bb6f-82682357fb0a)

## Sample Outputs

![Image](https://github.com/user-attachments/assets/9a521bee-296e-4748-9841-d43f5ae20b35)

![Image](https://github.com/user-attachments/assets/431e9536-b713-4cdd-b98a-a92d7b964215)

![Image](https://github.com/user-attachments/assets/9a6d669a-5034-4687-8531-98f5d5fd60fd)

![Image](https://github.com/user-attachments/assets/759157e4-ff0c-4741-b780-7ce098d0b413)

![Image](https://github.com/user-attachments/assets/d56be76c-7043-410f-a6f9-dbf48b8a738c)

## Project Structure
```
portfolio-optimizer/
├── backtest_results/ # Backtesting outputs (CSV/plots)
├── backtest_results.pdf # Sample backtest report
├── constrained_results/ # Optimization runs with constraints
├── optimised_results/ # Unconstrained optimization outputs
├── src/ # Core source code
│ ├── commands/ # CLI command modules
│ ├── backtest.py # Backtesting engine
│ ├── optimization.py # core optimization module
│ └── utils/ # Utility modules
│ ├── generate_backtest_report.py # Backtest PDF generation
│ ├── generate_optimization_report.py # Optimization PDF generation
│ ├── validation.py # Data validation
│ ├── backtest.py # Backtest utilities
│ ├── calculations.py # Return/risk calculations
│ ├── cli.py # Command-line interface
│ ├── data_loader.py # Market data fetcher
│ ├── optimization.py # optimization utilities
│ ├── simulation.py # Monte Carlo simulation
│ └── visualization.py # Plot generation
├── tests/ # Unit tests
├── .gitignore # Git exclusion rules
├── README.md # This document
└── requirements.txt # Python dependencies
```
## Development

### Run tests:

```
pytest tests/
```

