import pytest
from src.data_loader import fetch_stock_data, validate_tickers

def test_valid_ticker_fetch():
    data = fetch_stock_data(['AAPL'], period='1mo')
    assert not data.empty
    assert 'AAPL' in data.columns
    assert len(data) > 15  # Should have >15 trading days

def test_invalid_ticker():
    with pytest.raises(ValueError):
        fetch_stock_data(['INVALID_TICKER_XYZ'])

def test_multi_ticker_fetch():
    data = fetch_stock_data(['MSFT', 'GOOG'], period='1y')
    assert {'MSFT', 'GOOG'}.issubset(set(data.columns))
    assert data.isnull().sum().sum() == 0

def test_ticker_validation():
    validate_tickers(['AAPL', 'MSFT'])  # Should pass
    
    with pytest.raises(ValueError):
        validate_tickers(['AAPL', 'INVALID_TICKER'])