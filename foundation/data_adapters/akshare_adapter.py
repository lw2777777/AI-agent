# foundation/data_adapters/akshare_adapter.py - 精简稳定版

"""
AKShare 数据适配器 - 稳定版
带多重重试 + 备用接口
"""

import pandas as pd
import time
import random
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AKShareAdapter:
    """AKShare 数据适配器 - 稳定版"""
    
    def __init__(self):
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        try:
            import akshare as ak
            return True
        except ImportError:
            logger.warning("AKShare not installed, run: pip install akshare")
            return False
    
    def get_stock_zh_a_hist(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        max_retries: int = 5
    ) -> Dict[str, Any]:
        """获取A股历史数据 - 多接口重试"""
        if not self._available:
            return {"error": "AKShare not available"}
        
        # 标准化日期
        start = start_date.replace('-', '') if '-' in start_date else start_date
        end = end_date.replace('-', '') if '-' in end_date else end_date
        
        # 多个接口轮流尝试
        for attempt in range(max_retries):
            try:
                import akshare as ak
                
                logger.info(f"🔄 尝试获取 {symbol} (第 {attempt+1}/{max_retries} 次)")
                
                # 方案1: 东方财富 (最稳定)
                if attempt % 3 == 0:
                    df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start,
                        end_date=end,
                        adjust=adjust
                    )
                # 方案2: 新浪
                elif attempt % 3 == 1:
                    df = ak.stock_zh_a_daily(
                        symbol=f"sz{symbol}" if symbol.startswith(('0', '3')) else f"sh{symbol}",
                        start_date=start,
                        end_date=end,
                        adjust=adjust
                    )
                # 方案3: 腾讯
                else:
                    df = ak.stock_zh_a_hist_tx(
                        symbol=f"sz{symbol}" if symbol.startswith(('0', '3')) else f"sh{symbol}",
                        start_date=start,
                        end_date=end,
                        adjust=adjust
                    )
                
                if df is not None and not df.empty:
                    df = self._standardize_columns(df)
                    logger.info(f"✅ 成功获取 {symbol}: {len(df)} 条")
                    return {
                        "success": True,
                        "data": df.to_dict('records'),
                        "count": len(df),
                        "error": None
                    }
                
            except Exception as e:
                error_msg = str(e)[:100]
                logger.warning(f"⚠️ 第 {attempt+1} 次失败: {error_msg}")
                
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt + random.uniform(0, 1), 10)
                    logger.info(f"⏳ 等待 {wait_time:.1f} 秒...")
                    time.sleep(wait_time)
        
        return {
            "success": False,
            "data": [],
            "count": 0,
            "error": f"All {max_retries} attempts failed"
        }
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        column_map = {
            '日期': 'date', 'date': 'date',
            '开盘': 'open', 'open': 'open',
            '收盘': 'close', 'close': 'close',
            '最高': 'high', 'high': 'high',
            '最低': 'low', 'low': 'low',
            '成交量': 'volume', 'volume': 'volume',
            '成交额': 'amount', 'amount': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        df = df.rename(columns=column_map)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        # 只保留需要的列
        keep_cols = ['date', 'open', 'close', 'high', 'low', 'volume']
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]
        
        return df
    
    def get_stock_zh_a_spot(self) -> Dict[str, Any]:
        """获取A股实时行情"""
        if not self._available:
            return {"error": "AKShare not available"}
        
        for attempt in range(3):
            try:
                import akshare as ak
                df = ak.stock_zh_a_spot_em()
                if not df.empty:
                    return {
                        "success": True,
                        "data": df.to_dict('records'),
                        "count": len(df)
                    }
            except Exception as e:
                logger.warning(f"实时行情失败 (尝试 {attempt+1}/3): {str(e)[:80]}")
                time.sleep(2)
        
        return {"success": False, "error": "实时行情不可用"}