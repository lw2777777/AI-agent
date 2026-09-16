"""
Polygon.io 美股数据适配器
需要注册获取API Key: https://polygon.io/
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PolygonAdapter:
    """Polygon.io 数据适配器"""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: str):
        """
        初始化适配器
        :param api_key: 从 Polygon.io Dashboard 获取的个人密钥
        """
        self.api_key = api_key
        self._last_request_time = 0
        
    def get_historical_data(
        self, 
        symbol: str, 
        days_back: int = 30,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        获取美股历史日线数据
        :param symbol: 股票代码 (如 AAPL, GOOGL)
        :param days_back: 回溯天数
        :param interval: 时间粒度 (day, minute, hour)
        :return: DataFrame 包含 open, high, low, close, volume
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # 映射时间粒度
        interval_map = {
            "day": "1/day",
            "minute": "1/minute",
            "hour": "1/hour"
        }
        range_str = interval_map.get(interval, "1/day")
        
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{range_str}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if response.status_code != 200:
                logger.error(f"API请求失败 (状态码 {response.status_code}): {data.get('error', 'Unknown error')}")
                return pd.DataFrame()
            
            if data.get('resultsCount', 0) > 0:
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df = df.rename(columns={
                    'o': 'open', 'h': 'high', 
                    'l': 'low', 'c': 'close', 'v': 'volume'
                })
                df = df.set_index('date')
                logger.info(f"✅ 成功获取 {symbol} 数据，共 {len(df)} 行")
                return df
            else:
                logger.warning(f"⚠️ {symbol} 无数据: {data.get('status', 'No results')}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 网络请求异常: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ 解析异常: {e}")
            return pd.DataFrame()
    
    def get_realtime_quote(self, symbol: str) -> dict:
        """获取实时报价 (仅限付费套餐)"""
        # 免费套餐可能不支持，可暂不实现
        pass


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    API_KEY = "63982487-e103-4988-8841-5a3af1dc826a"  # 换成你自己的
    adapter = PolygonAdapter(API_KEY)
    
    # 获取苹果近30天数据
    df = adapter.get_historical_data("AAPL", days_back=30)
    print(df.tail())