import pandas as pd
import numpy as np
from .data_loader import fetch_stock_data
from .calculations import calculate_daily_returns, calculate_covariance_matrix, annualize_returns
from .optimization import find_tangency_portfolio
import logging

class PortfolioBacktester:
    def __init__(self, tickers: list, initial_amount: float = 10000):
        if not tickers or not isinstance(tickers, list):
            raise ValueError("Tickers must be a non-empty list.")
        if initial_amount <= 0:
            raise ValueError("Initial amount must be greater than 0.")
        self.tickers = tickers
        self.initial_amount = initial_amount
        self.full_data = None
        self.benchmarks = {
            'SPY': 'S&P 500',
            'BND': 'Aggregate Bonds'
        }
    
    def load_data(self, start_date: str, end_date: str):
        """Load full historical dataset"""
        if not start_date or not end_date:
            raise ValueError("Start and end dates must be provided.")
        
        self.full_data = fetch_stock_data(
            self.tickers + list(self.benchmarks.keys()),
            start=start_date,
            end=end_date
        )
        
        if self.full_data.empty or self.full_data.isnull().all().all():
            raise ValueError("Fetched data is empty or contains only NaN values.")
        
        return self

    logging.basicConfig(level=logging.INFO)

    def run_backtest(
        self,
        train_period: str = '3Y',
        test_period: str = '1Y',
        rebalance_freq: str = 'ME',
        transaction_cost: float = 0.001
    ) -> dict:
        logging.info("Starting backtest...")
        logging.info(f"Train period: {train_period}, Test period: {test_period}, Rebalance frequency: {rebalance_freq}")
        
        # Convert train_period and test_period to days
        def convert_to_days(period: str) -> int:
            if period.endswith('Y'):
                return int(period[:-1]) * 365  # Convert years to days
            elif period.endswith('M'):
                return int(period[:-1]) * 30   # Convert months to days
            else:
                raise ValueError(f"Unsupported period format: {period}")

        test_days = convert_to_days(test_period)

        # Split data
        train_end = self.full_data.index[-1] - pd.Timedelta(days=test_days)
        train_data = self.full_data.loc[:train_end]
        test_data = self.full_data.loc[train_end:]
        
        if train_data.empty or test_data.empty:
            raise ValueError("Train or test data is empty.")
        
        logging.info("Data split into training and testing periods.")
        
        # Optimize portfolio on training period
        train_returns = calculate_daily_returns(train_data)
        annual_returns = annualize_returns(train_returns[self.tickers])
        cov_matrix = calculate_covariance_matrix(train_returns[self.tickers])
        
        optimal = find_tangency_portfolio(annual_returns, cov_matrix)
        weights = optimal['weights']
        logging.info(f"Optimal weights: {weights}")
        
        # Initialize portfolio
        portfolio = {
            'value': [self.initial_amount],
            'cash': self.initial_amount,
            'weights': weights,
            'shares': {t: 0 for t in self.tickers}
        }
        
        # Rebalancing dates
        if rebalance_freq == 'M':
            logging.warning("'M' is deprecated. Using 'ME' (Month-End) instead.")
            rebalance_freq = 'ME'
        
        reb_dates = pd.date_range(
            start=test_data.index[0],
            end=test_data.index[-1],
            freq=rebalance_freq
        )
        logging.info(f"Rebalancing dates: {reb_dates}")
        
        # Backtesting loop
        for date, prices in test_data.iterrows():
            if date in reb_dates:
                portfolio = self._rebalance(portfolio, prices, transaction_cost)
                logging.info(f"Rebalanced portfolio on {date}.")
            
            portfolio['value'].append(self._calculate_value(portfolio, prices))

        # Convert portfolio['value'] to a pandas.Series
        portfolio_series = pd.Series(portfolio['value'], index=[test_data.index[0]] + test_data.index.tolist())
        # Calculate performance metrics
        results = self._calculate_metrics(
            portfolio_series, 
            test_data,
            test_period
        )
        logging.info("Backtest completed.")
        
        return results
    def _rebalance(self, portfolio, prices, transaction_cost):
        """
        Rebalance to target weights.
        
        Args:
            portfolio (dict): Portfolio dictionary containing cash, weights, and shares.
            prices (pd.Series): Current prices of the stocks.
            transaction_cost (float): Transaction cost as a fraction of trade value.
        
        Returns:
            dict: Updated portfolio after rebalancing.
        """
        # Calculate total portfolio value
        total_value = self._calculate_value(portfolio, prices)
        
        # Convert weights to a dictionary with tickers as keys
        weights_dict = {ticker: weight for ticker, weight in zip(self.tickers, portfolio['weights'])}
        
        # Calculate target values for each ticker
        target_values = {ticker: total_value * weight for ticker, weight in weights_dict.items()}
        
        # Calculate trade orders
        orders = {}
        for ticker in self.tickers:
            current_value = portfolio['shares'].get(ticker, 0) * prices[ticker]
            target_delta = target_values[ticker] - current_value
            
            # Calculate shares to buy/sell
            shares = target_delta / prices[ticker]
            if abs(shares) > 1e-6:  # Avoid tiny fractional shares
                orders[ticker] = shares
        
        # Execute trades with transaction costs
        trade_cost = sum(abs(shares) * prices[ticker] * transaction_cost for ticker, shares in orders.items())
        portfolio['cash'] -= trade_cost
        
        for ticker, shares in orders.items():
            portfolio['shares'][ticker] = portfolio['shares'].get(ticker, 0) + shares
        
        return portfolio
    
    def _calculate_metrics(self, values, test_data, test_period):
        """Calculate performance metrics"""
        if values.empty or len(values) < 2:
            raise ValueError("Insufficient data to calculate metrics.")
        
        portfolio_series = pd.Series(values, index=[test_data.index[0]] + test_data.index.tolist())
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        if portfolio_returns.empty:
            raise ValueError("Portfolio returns are empty.")
        
        # Benchmark returns
        bench_returns = {}
        for ticker, name in self.benchmarks.items():
            bench_returns[name] = calculate_daily_returns(test_data[[ticker]]).iloc[:, 0]
        
        # Convert test_period to days
        def convert_to_days(period: str) -> int:
            if period.endswith('Y'):
                return int(period[:-1]) * 365  # Convert years to days
            elif period.endswith('M'):
                return int(period[:-1]) * 30   # Convert months to days
            else:
                raise ValueError(f"Unsupported period format: {period}")
        
        test_days = convert_to_days(test_period)
        years = test_days / 365.25  # Convert days to years
        
        # Calculate metrics
        return {
            'cagr': (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) ** (1/years) - 1,
            'sharpe': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': (portfolio_series / portfolio_series.cummax() - 1).min(),
            'benchmarks': {
                name: (returns.add(1).prod() - 1)  # Total return
                for name, returns in bench_returns.items()
            },
            'values': portfolio_series
        }

    def _calculate_value(self, portfolio, prices):
        """
        Calculate the total value of the portfolio.
        
        Args:
            portfolio (dict): Portfolio dictionary containing cash, weights, and shares.
            prices (pd.Series): Current prices of the stocks.
        
        Returns:
            float: Total portfolio value.
        """
        # Calculate the value of all shares held
        shares_value = sum(portfolio['shares'][ticker] * prices[ticker] for ticker in self.tickers)
        
        # Add cash to the total value
        total_value = portfolio['cash'] + shares_value
        
        return total_value