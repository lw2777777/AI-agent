"""
news_provider.py — 非结构化信息源

MarketAgent 用 LLM 处理这部分。
真实场景接：akshare 新闻 / 东财公告 / 舆情 API

提供：
  - NewsItem 数据结构
  - NewsProvider 抽象接口
  - SyntheticNewsProvider（合成，确定性可复现）
  - AkshareNewsProvider（真实，需 akshare）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    timestamp: str
    title: str
    content: str = ""
    source: str = "unknown"
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsProvider:
    """抽象接口"""
    def fetch(
        self,
        symbol: str,
        as_of: datetime,
        lookback_hours: int = 24,
        max_items: int = 10,
    ) -> List[NewsItem]:
        raise NotImplementedError


class SyntheticNewsProvider(NewsProvider):
    """
    合成新闻：按 symbol + 日期确定性生成，保证可复现。
    无真实新闻源时用它跑通闭环。
    """
    TEMPLATES = [
        "公司发布季度财报，营收同比增长 {x}%",
        "公司宣布重大资产重组，市场反应积极",
        "分析师上调评级至买入，目标价上调",
        "公司高管减持公告",
        "行业监管趋严，短期承压",
        "公司被曝财务造假传闻",
        "公司发布新产品，市场关注度提升",
        "股东大会通过分红方案",
    ]

    def fetch(self, symbol, as_of, lookback_hours=24, max_items=10):
        seed = hash((symbol, as_of.strftime("%Y%m%d"))) % (2**31)
        rng = np.random.default_rng(seed)
        #随机种子 只要symbol和日期相同 种子就相同

        n = int(rng.integers(0, min(4, max_items) + 1))
        items: List[NewsItem] = []
        for _ in range(n):
            title = self.TEMPLATES[int(rng.integers(0, len(self.TEMPLATES)))]
            title = title.format(x=int(rng.integers(-20, 40)))
            ts = as_of - timedelta(hours=float(rng.uniform(0, lookback_hours)))
            items.append(NewsItem(
                timestamp=ts.isoformat(),
                title=title,
                content="",
                source="synthetic",
            ))
        return items


class AkshareNewsProvider(NewsProvider):
    """真实新闻源（akshare）。失败时静默返回空列表。"""

    def fetch(self, symbol, as_of, lookback_hours=24, max_items=10):
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，NewsProvider 返回空")
            return []

        try:
            df = ak.stock_news_em(symbol=symbol)
            if df is None or df.empty:
                return []

            items: List[NewsItem] = []
            cutoff = as_of - timedelta(hours=lookback_hours)
            for _, row in df.head(max_items * 3).iterrows():
                title = str(row.get("新闻标题", row.get("title", "")))
                content = str(row.get("新闻内容", row.get("content", "")))
                ts_raw = row.get("发布时间", row.get("publish_time", ""))
                try:
                    ts = datetime.fromisoformat(str(ts_raw))
                except Exception:
                    ts = as_of
                if ts < cutoff:
                    continue
                items.append(NewsItem(
                    timestamp=ts.isoformat(),
                    title=title,
                    content=content[:500],
                    source="akshare_em",
                ))
                if len(items) >= max_items:
                    break
            return items
        except Exception as e:
            logger.warning("akshare 新闻获取失败: %s", e)
            return []


def get_news_provider(use_real: bool = False) -> NewsProvider:
    return AkshareNewsProvider() if use_real else SyntheticNewsProvider()