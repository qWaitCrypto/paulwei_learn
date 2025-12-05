"""
TradesData: 负责从 bitmex_trades.csv 读取并查询成交记录。

特点：
- 仅加载 execType == 'Trade' 的记录（排除 Funding/Settlement）。
- 内存缓存 DataFrame，便于快速过滤。
- 提供 list_symbols / get_trades 两个主要接口。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

DEFAULT_TRADES_CSV = Path("data/bitmex_trades.csv")


def _parse_utc(ts: Union[str, dt.datetime]) -> dt.datetime:
    """将输入解析为带 UTC 时区的 datetime。"""
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=dt.timezone.utc)
        return ts.astimezone(dt.timezone.utc)
    if isinstance(ts, str):
        s = ts.strip()
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(s)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


class TradesData:
    """简单的成交数据访问层，基于 bitmex_trades.csv。"""

    def __init__(self, trades_csv: Union[str, Path] = DEFAULT_TRADES_CSV):
        self.trades_csv = Path(trades_csv)
        if not self.trades_csv.exists():
            raise FileNotFoundError(f"Trades CSV not found: {self.trades_csv}")

        df = pd.read_csv(self.trades_csv)
        if "timestamp" not in df.columns or "symbol" not in df.columns:
            raise ValueError("Trades CSV must contain 'timestamp' and 'symbol' columns.")

        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        # 仅保留真实成交
        df = df[df["execType"] == "Trade"].copy()
        df.sort_values(["symbol", "timestamp"], inplace=True)
        self.df = df

    def list_symbols(self) -> List[str]:
        return sorted(self.df["symbol"].unique().tolist())

    def range(self, symbol: str) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        sub = self.df[self.df["symbol"] == symbol]
        if sub.empty:
            return None
        return (
            sub["timestamp"].min().to_pydatetime(),
            sub["timestamp"].max().to_pydatetime(),
        )

    def get_trades(
        self,
        symbol: str,
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """返回指定时间窗内的成交记录，按时间升序。"""
        start_ts = _parse_utc(start)
        end_ts = _parse_utc(end)
        if start_ts >= end_ts:
            raise ValueError("start must be before end")

        sub = self.df[self.df["symbol"] == symbol]
        if sub.empty:
            return sub
        mask = (sub["timestamp"] >= start_ts) & (sub["timestamp"] <= end_ts)
        sub = sub[mask].sort_values("timestamp")
        if limit is not None:
            sub = sub.head(limit)
        return sub.reset_index(drop=True)


__all__ = ["TradesData"]

