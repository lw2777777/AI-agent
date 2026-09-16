#!/usr/bin/env python3
"""
Quick Start - 快速启动AI交易Agent系统

功能:
1. 使用AKShare拉取A股数据
2. 初始化CoreAgent进行量化分析
3. 展示ReAct推理过程
4. 展示记忆系统
5. 展示多Agent协作
6. 生成交易建议

运行: python quick_start.py
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import akshare as ak

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# 第一部分: 数据获取 (使用AKShare)
# ============================================================

class DataFetcher:
    """数据获取器 - 使用AKShare拉取数据"""
    
    def __init__(self):
        self._check_availability()
    
    def _check_availability(self):
        try:
            import akshare as ak
            self.ak = ak
            self.available = True
            logger.info("✅ AKShare 已加载")
        except ImportError:
            logger.error("❌ AKShare 未安装，请运行: pip install akshare")
            self.available = False
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        if not self.available:
            return pd.DataFrame()
        try:
            df = self.ak.stock_zh_a_spot_em()
            logger.info(f"✅ 获取到 {len(df)} 只股票")
            return df
        except Exception as e:
            logger.error(f"❌ 获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """获取历史数据"""
        if not self.available:
            return pd.DataFrame()
        
        try:
            start = start_date.replace('-', '') if '-' in start_date else start_date
            end = end_date.replace('-', '') if '-' in end_date else end_date
            
            df = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust
            )
            
            if df.empty:
                logger.warning(f"⚠️ {symbol} 无数据")
                return df
            
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude',
                '涨跌幅': 'pct_change', '涨跌额': 'change',
                '换手率': 'turnover'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            logger.info(f"✅ {symbol}: {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"❌ 获取 {symbol} 失败: {e}")
            return pd.DataFrame()


# ============================================================
# 第二部分: 量化分析工具
# ============================================================

class QuantAnalyzer:
    """量化分析器"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period:
            return 50
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9) -> Dict[str, float]:
        """计算MACD"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
            'signal': signal_line.iloc[-1] if not signal_line.empty else 0,
            'histogram': histogram.iloc[-1] if not histogram.empty else 0
        }
    
    @staticmethod
    def calculate_bollinger(prices: pd.Series, period=20, std=2) -> Dict[str, float]:
        """计算布林带"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0}
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        return {
            'upper': (rolling_mean + rolling_std * std).iloc[-1],
            'middle': rolling_mean.iloc[-1],
            'lower': (rolling_mean - rolling_std * std).iloc[-1]
        }
    
    @staticmethod
    def analyze(data: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """综合分析"""
        if data.empty or len(data) < 20:
            return {'error': '数据不足'}
        
        close = data['close']
        current = close.iloc[-1]
        
        # 计算所有指标
        rsi = QuantAnalyzer.calculate_rsi(close)
        macd = QuantAnalyzer.calculate_macd(close)
        bb = QuantAnalyzer.calculate_bollinger(close)
        
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else 0
        
        # 趋势判断
        if current > ma5 > ma20 > ma60:
            trend = 'bullish'
        elif current < ma5 < ma20 < ma60:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # 信号生成
        signals = []
        if rsi < 30:
            signals.append('oversold')
        elif rsi > 70:
            signals.append('overbought')
        if macd['macd'] > macd['signal']:
            signals.append('macd_bullish')
        else:
            signals.append('macd_bearish')
        if current < bb['lower']:
            signals.append('below_bb_lower')
        elif current > bb['upper']:
            signals.append('above_bb_upper')
        
        # 建议
        buy_score = sum(1 for s in signals if s in ['oversold', 'macd_bullish', 'below_bb_lower'])
        sell_score = sum(1 for s in signals if s in ['overbought', 'macd_bearish', 'above_bb_upper'])
        
        if buy_score > sell_score and trend in ['bullish', 'neutral']:
            recommendation = 'buy'
            confidence = min(0.5 + buy_score * 0.15, 0.95)
        elif sell_score > buy_score and trend in ['bearish', 'neutral']:
            recommendation = 'sell'
            confidence = min(0.5 + sell_score * 0.15, 0.95)
        else:
            recommendation = 'hold'
            confidence = 0.5
        
        return {
            'symbol': symbol,
            'current_price': current,
            'trend': trend,
            'rsi': rsi,
            'macd': macd,
            'bollinger_bands': bb,
            'ma5': ma5,
            'ma20': ma20,
            'ma60': ma60,
            'signals': signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'data_points': len(data)
        }


# ============================================================
# 第三部分: 使用agent_framework的CoreAgent
# ============================================================

def run_with_core_agent(data: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用CoreAgent进行分析
    展示agent_framework的核心能力
    """
    print("\n" + "=" * 70)
    print("🤖 使用 CoreAgent 进行深度分析")
    print("=" * 70)
    
    try:
        # 导入CoreAgent (如果可用)
        from agent_framework.core.core_agent import CoreAgent
        
        # 初始化API Keys (可以配置DeepSeek等)
        api_keys = {
            'deepseek': os.getenv('DEEPSEEK_API_KEY', ''),
            'openai': os.getenv('OPENAI_API_KEY', ''),
        }
        
        agent = CoreAgent(api_keys=api_keys)
        
        # 构建查询
        query = f"""
        请对以下股票进行深度分析:
        股票代码: {analysis.get('symbol', 'UNKNOWN')}
        当前价格: {analysis.get('current_price', 0):.2f}
        趋势: {analysis.get('trend', 'neutral')}
        RSI: {analysis.get('rsi', 50):.2f}
        MACD: {analysis.get('macd', {})}
        布林带: {analysis.get('bollinger_bands', {})}
        检测信号: {analysis.get('signals', [])}
        
        请提供:
        1. 技术分析解读
        2. 风险提示
        3. 操作建议
        """
        
        # 配置Agent
        config = {
            "name": "QuantAnalyst",
            "role": "量化分析师",
            "instructions": "你是一位专业的量化分析师，擅长技术分析和投资决策。",
            "tools": ["calculate", "format_json"],
            "markdown": True,
            "reasoning": {"enabled": True, "min_steps": 2, "max_steps": 5},
        }
        
        # 如果有API Key，使用LLM
        if api_keys.get('deepseek') or api_keys.get('openai'):
            print("\n🧠 使用LLM进行分析...")
            if api_keys.get('deepseek'):
                config["model"] = {"provider": "deepseek", "model_id": "deepseek-chat"}
            elif api_keys.get('openai'):
                config["model"] = {"provider": "openai", "model_id": "gpt-4o-mini"}
            
            response = agent.run(query, config)
            content = agent.get_response_content(response)
            
            return {
                'success': True,
                'response': content,
                'agent_name': 'CoreAgent',
                'reasoning_enabled': True
            }
        else:
            print("\n⚠️ 未配置API Key，使用规则分析")
            return {
                'success': True,
                'response': generate_rule_based_analysis(analysis),
                'agent_name': 'RuleBasedAnalyzer',
                'reasoning_enabled': False
            }
            
    except ImportError as e:
        logger.warning(f"agent_framework模块未找到: {e}")
        print("\n⚠️ 使用内置规则分析")
        return {
            'success': True,
            'response': generate_rule_based_analysis(analysis),
            'agent_name': 'FallbackAnalyzer',
            'reasoning_enabled': False
        }
    except Exception as e:
        logger.error(f"CoreAgent执行失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'agent_name': 'ErrorHandler'
        }


def generate_rule_based_analysis(analysis: Dict[str, Any]) -> str:
    """基于规则生成分析报告"""
    symbol = analysis.get('symbol', 'UNKNOWN')
    price = analysis.get('current_price', 0)
    trend = analysis.get('trend', 'neutral')
    rsi = analysis.get('rsi', 50)
    signals = analysis.get('signals', [])
    rec = analysis.get('recommendation', 'hold')
    conf = analysis.get('confidence', 0.5)
    
    lines = []
    lines.append(f"📊 {symbol} 技术分析报告")
    lines.append("=" * 50)
    lines.append(f"当前价格: {price:.2f}")
    lines.append(f"趋势判断: {trend.upper()}")
    lines.append(f"RSI指标: {rsi:.2f} ({'超买' if rsi > 70 else '超卖' if rsi < 30 else '中性'})")
    
    # MACD
    macd = analysis.get('macd', {})
    if macd:
        lines.append(f"MACD: {macd.get('macd', 0):.4f}")
        lines.append(f"MACD信号: {macd.get('signal', 0):.4f}")
        lines.append(f"MACD柱: {macd.get('histogram', 0):.4f}")
    
    # 布林带
    bb = analysis.get('bollinger_bands', {})
    if bb:
        lines.append(f"布林带上轨: {bb.get('upper', 0):.2f}")
        lines.append(f"布林带中轨: {bb.get('middle', 0):.2f}")
        lines.append(f"布林带下轨: {bb.get('lower', 0):.2f}")
    
    lines.append(f"移动平均: MA5={analysis.get('ma5', 0):.2f}, MA20={analysis.get('ma20', 0):.2f}")
    
    lines.append("\n📈 信号检测:")
    for signal in signals:
        emoji = "🟢" if signal in ['oversold', 'macd_bullish', 'below_bb_lower'] else "🔴"
        lines.append(f"  {emoji} {signal}")
    
    lines.append(f"\n💡 操作建议: {rec.upper()} (置信度: {conf:.1%})")
    
    # 详细分析
    lines.append("\n📝 分析要点:")
    if rec == 'buy':
        lines.append("  - 技术指标显示买入信号")
        if rsi < 30:
            lines.append("  - RSI超卖，有反弹可能")
        if 'macd_bullish' in signals:
            lines.append("  - MACD金叉，上涨动能增强")
        if 'below_bb_lower' in signals:
            lines.append("  - 价格低于布林带下轨，可能反弹")
    elif rec == 'sell':
        lines.append("  - 技术指标显示卖出信号")
        if rsi > 70:
            lines.append("  - RSI超买，有回调风险")
        if 'macd_bearish' in signals:
            lines.append("  - MACD死叉，下跌动能增强")
        if 'above_bb_upper' in signals:
            lines.append("  - 价格高于布林带上轨，可能回调")
    else:
        lines.append("  - 技术指标信号混合，建议观望")
        lines.append("  - 等待更明确的信号")
    
    return "\n".join(lines)


# ============================================================
# 第四部分: 记忆系统演示
# ============================================================

def demo_memory_system():
    """演示记忆系统"""
    print("\n" + "=" * 70)
    print("🧠 记忆系统演示")
    print("=" * 70)
    
    try:
        from agent_framework.core.archival_memory import ArchivalMemoryStore
        
        store = ArchivalMemoryStore()
        
        # 存储记忆
        print("\n📝 存储记忆...")
        store.save(
            content="AAPL 在2026年9月表现强劲，技术面看涨",
            memory_type="analysis",
            metadata={"symbol": "AAPL", "date": "2026-09-09"}
        )
        store.save(
            content="市场波动率上升，建议降低仓位",
            memory_type="risk",
            metadata={"date": "2026-09-08"}
        )
        store.save(
            content="投资者偏好科技股，资金持续流入AI板块",
            memory_type="sentiment",
            metadata={"sector": "technology", "date": "2026-09-07"}
        )
        print("✅ 已存储3条记忆")
        
        # 搜索记忆
        print("\n🔍 搜索记忆 (查询: '科技股 风险')...")
        results = store.search("科技股 风险", k=3)
        for r in results:
            print(f"  - {r.get('content', '')[:80]}...")
        
        # 统计
        stats = store.list()
        print(f"\n📊 记忆统计: {len(stats)} 条记忆")
        
        return {'success': True, 'memory_count': len(stats)}
        
    except ImportError:
        print("⚠️ 记忆系统模块未安装")
        return {'success': False, 'error': 'Memory module not found'}


# ============================================================
# 第五部分: MCP协议演示
# ============================================================

def demo_mcp_protocol():
    """演示MCP协议"""
    print("\n" + "=" * 70)
    print("🔌 MCP协议演示")
    print("=" * 70)
    
    try:
        from agent_framework.mcp_protocol.server import MCPServer, create_mcp_server
        from agent_framework.mcp_protocol import MCPTool, MCPResource
        
        # 创建MCP服务器
        server = create_mcp_server(
            name="Quant MCP Server",
            version="1.0.0",
            instructions="量化交易工具服务器"
        )
        
        # 注册工具
        @server.tool(
            name="calculate_rsi",
            description="计算RSI指标",
            parameters={
                "prices": {"type": "array", "description": "价格数组"},
                "period": {"type": "integer", "description": "周期", "default": 14}
            }
        )
        def calculate_rsi(prices: list, period: int = 14) -> float:
            if len(prices) < period:
                return 50.0
            import pandas as pd
            s = pd.Series(prices)
            delta = s.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1] if not rsi.empty else 50.0)
        
        @server.tool(
            name="calculate_macd",
            description="计算MACD指标",
            parameters={
                "prices": {"type": "array", "description": "价格数组"},
                "fast": {"type": "integer", "description": "快线周期", "default": 12},
                "slow": {"type": "integer", "description": "慢线周期", "default": 26},
                "signal": {"type": "integer", "description": "信号线周期", "default": 9}
            }
        )
        def calculate_macd(prices: list, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
            if len(prices) < slow:
                return {"macd": 0, "signal": 0, "histogram": 0}
            import pandas as pd
            s = pd.Series(prices)
            exp1 = s.ewm(span=fast, adjust=False).mean()
            exp2 = s.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return {
                "macd": float(macd_line.iloc[-1]),
                "signal": float(signal_line.iloc[-1]),
                "histogram": float(histogram.iloc[-1])
            }
        
        print(f"✅ MCP服务器创建成功: {server.name}")
        print(f"📊 注册工具: {list(server.tools.keys())}")
        
        # 测试调用
        print("\n🧪 测试工具调用...")
        test_prices = [100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 
                      104, 105, 106, 105, 104, 103, 104, 105, 106, 107]
        result = server._handle_tools_call({
            'id': 1,
            'method': 'tools/call',
            'params': {
                'name': 'calculate_rsi',
                'arguments': {'prices': test_prices, 'period': 14}
            }
        })
        print(f"  RSI计算: {result.result}")
        
        return {'success': True, 'tools': list(server.tools.keys())}
        
    except ImportError as e:
        print(f"⚠️ MCP模块未找到: {e}")
        return {'success': False, 'error': str(e)}


# ============================================================
# 第六部分: 主程序
# ============================================================

def main():
    """主程序"""
    print("=" * 70)
    print("🚀 AI交易Agent系统 - Quick Start")
    print("📅", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # ============================================================
    # Step 1: 获取数据
    # ============================================================
    print("\n📊 Step 1: 获取市场数据...")
    fetcher = DataFetcher()
    
    # 选择股票
    symbols = ['000001', '600519', '000858']  # 平安银行、贵州茅台、五粮液
    symbol = symbols[0]
    
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    
    print(f"  标的: {symbol}")
    print(f"  日期: {start_date} - {end_date}")
    
    df = fetcher.get_historical_data(symbol, start_date, end_date)
    
    if df.empty:
        print("⚠️ 无法获取真实数据，使用模拟数据")
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.randn(100) * 0.3,
            'high': prices + np.abs(np.random.randn(100) * 0.5),
            'low': prices - np.abs(np.random.randn(100) * 0.5),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })
        df['symbol'] = symbol
    
    print(f"✅ 数据获取成功: {len(df)} 条记录")
    
    # 数据摘要
    print(f"\n📋 数据摘要:")
    print(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"  最新价格: {df['close'].iloc[-1]:.2f}")
    print(f"  最高价: {df['high'].max():.2f}")
    print(f"  最低价: {df['low'].min():.2f}")
    
    # ============================================================
    # Step 2: 量化分析
    # ============================================================
    print("\n📊 Step 2: 量化分析...")
    analysis = QuantAnalyzer.analyze(df, symbol)
    
    print(f"\n📈 分析结果:")
    print(f"  当前价格: {analysis.get('current_price', 0):.2f}")
    print(f"  趋势: {analysis.get('trend', 'neutral').upper()}")
    print(f"  RSI: {analysis.get('rsi', 50):.2f}")
    print(f"  信号: {analysis.get('signals', [])}")
    print(f"  建议: {analysis.get('recommendation', 'hold').upper()}")
    print(f"  置信度: {analysis.get('confidence', 0.5):.1%}")
    
    # ============================================================
    # Step 3: 使用CoreAgent深度分析
    # ============================================================
    result = run_with_core_agent(df, analysis)
    
    if result.get('success'):
        print("\n📄 分析报告:")
        print("-" * 50)
        print(result.get('response', ''))
        print("-" * 50)
        print(f"Agent: {result.get('agent_name', 'Unknown')}")
        if result.get('reasoning_enabled'):
            print("🧠 推理模式: 已启用")
    else:
        print(f"❌ 分析失败: {result.get('error', 'Unknown error')}")
    
    # ============================================================
    # Step 4: 记忆系统演示
    # ============================================================
    memory_result = demo_memory_system()
    
    # ============================================================
    # Step 5: MCP协议演示
    # ============================================================
    mcp_result = demo_mcp_protocol()
    
    # ============================================================
    # Step 6: 保存结果
    # ============================================================
    print("\n📊 Step 6: 保存结果...")
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    # 保存数据
    data_file = output_dir / f"{symbol}_data.csv"
    df.to_csv(data_file, index=False, encoding='utf-8-sig')
    print(f"  ✅ 数据保存: {data_file}")
    
    # 保存分析
    analysis_file = output_dir / f"{symbol}_analysis.json"
    analysis_copy = analysis.copy()
    for key in ['macd', 'bollinger_bands']:
        if key in analysis_copy and isinstance(analysis_copy[key], dict):
            pass
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_copy, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✅ 分析保存: {analysis_file}")
    
    # ============================================================
    # Step 7: 系统信息
    # ============================================================
    print("\n📊 Step 7: 系统信息...")
    
    # 尝试获取更多系统信息
    try:
        from agent_framework.core.core_agent import CoreAgent
        agent = CoreAgent()
        info = agent.get_system_info()
        print(f"  Agent框架: {info.get('version', 'unknown')}")
        print(f"  功能: {', '.join(info.get('features', [])[:5])}...")
    except:
        pass
    
    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ Quick Start 执行完成!")
    print("=" * 70)
    
    print("\n📖 总结:")
    print(f"  ✅ 数据拉取: {'成功' if not df.empty else '失败'}")
    print(f"  ✅ 量化分析: {analysis.get('recommendation', 'N/A').upper()}")
    print(f"  ✅ CoreAgent: {result.get('success', False)}")
    print(f"  ✅ 记忆系统: {memory_result.get('success', False)}")
    print(f"  ✅ MCP协议: {mcp_result.get('success', False)}")
    
    print("\n💡 下一步:")
    print("  1. 配置API Key: export DEEPSEEK_API_KEY='your_key'")
    print("  2. 运行完整系统: python -m agent_framework.core.runner")
    print("  3. 查看数据: ls data/")
    
    return {
        'symbol': symbol,
        'data_count': len(df),
        'analysis': analysis,
        'agent_result': result,
        'memory_result': memory_result,
        'mcp_result': mcp_result
    }


if __name__ == "__main__":
    result = main()