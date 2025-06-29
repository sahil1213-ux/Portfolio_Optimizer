from unittest.mock import patch, MagicMock
from src.commands.optimization import run_optimization
import pandas as pd
import numpy as np

def test_run_optimization(tmp_path):
    # Create mock result that optimization would return
    mock_result = {
        'tangency': {
            'return': 0.15,
            'volatility': 0.2,
            'sharpe': 0.65,
            'weights': {'AAPL': 0.6, 'MSFT': 0.4}
        },
        'monte_carlo': {
            'max_sharpe': {
                'return': 0.14,
                'volatility': 0.21,
                'sharpe': 0.57,
                'weights': {'AAPL': 0.7, 'MSFT': 0.3}
            },
            'min_vol': {
                'return': 0.12,
                'volatility': 0.16,
                'sharpe': 0.63,
                'weights': {'AAPL': 0.3, 'MSFT': 0.7}
            }
        },
        'frontier': {
            'target_returns': np.array([0.05, 0.1, 0.15]),
            'volatility': np.array([0.1, 0.15, 0.2]),
            'weights': np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
        }
    }
    
    # Create mock args object
    mock_args = MagicMock()
    mock_args.tickers = ['AAPL', 'MSFT']
    mock_args.period = '5y'
    mock_args.risk_free_rate = 0.02
    mock_args.max_weight = 1.0
    mock_args.pdf = False
    mock_args.amount = 10000.0
    
    # Mock the optimization function in the same module
    with patch('src.commands.optimization.optimization', return_value=mock_result):
        # Run the function
        results = run_optimization(mock_args, tmp_path)
        
        # Verify outputs
        assert results is not None
        assert 'tangency' in results
        assert 'monte_carlo' in results
        assert 'frontier' in results
        assert isinstance(results['tangency'], dict)
        assert 'sharpe' in results['tangency']