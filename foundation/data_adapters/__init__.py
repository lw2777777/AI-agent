
"""
数据适配器 - 统一数据访问层
支持多数据源: Yahoo Finance, AKShare, FRED, Polygon
"""

from .yahoo_adapter import YahooFinanceAdapter
from .akshare_adapter import AKShareAdapter
from .fred_adapter import FREDAdapter
from .polygon_adapter import PolygonAdapter
from .yfinance_adapter import get_stock_data

__all__ = [
    'YahooFinanceAdapter',
    'AKShareAdapter', 
    'FREDAdapter',
    'PolygonAdapter'
]