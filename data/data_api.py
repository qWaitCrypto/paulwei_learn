"""
BitMEX 1m K 线的 DuckDB 数据访问层。

核心思路：
- 初始化一次，保持活跃的 DuckDB 连接。
- 更高周期全部在 DuckDB 内，从 1m 即时聚合。
- 时间统一按 UTC 处理，输入先对齐到周期边界。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import duckdb
import pandas as pd


DEFAULT_DB_PATH = "bitmex_1m_kline.duckdb"
DEFAULT_TABLE = "kline_1m"


def _parse_utc(ts: Union[str, dt.datetime]) -> dt.datetime:
    """把输入解析成带时区的 UTC datetime。"""
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


def _to_iso_z(ts: dt.datetime) -> str:
    """把 UTC datetime 输出为以 Z 结尾的 ISO 字符串。"""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        ts = ts.astimezone(dt.timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _interval_to_duckdb(interval: str) -> str:
    """把周期字符串映射到 DuckDB INTERVAL 字面量。"""
    m = interval.strip().lower()
    mapping = {
        "1m": "INTERVAL '1 minute'",
        "5m": "INTERVAL '5 minute'",
        "15m": "INTERVAL '15 minute'",
        "30m": "INTERVAL '30 minute'",
        "1h": "INTERVAL '1 hour'",
        "4h": "INTERVAL '4 hour'",
        "1d": "INTERVAL '1 day'",
        "1w": "INTERVAL '7 day'",
    }
    if m not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[m]


def _interval_to_timedelta(interval: str) -> dt.timedelta:
    """把周期字符串映射到对应的 timedelta。"""
    m = interval.strip().lower()
    mapping = {
        "1m": dt.timedelta(minutes=1),
        "5m": dt.timedelta(minutes=5),
        "15m": dt.timedelta(minutes=15),
        "30m": dt.timedelta(minutes=30),
        "1h": dt.timedelta(hours=1),
        "4h": dt.timedelta(hours=4),
        "1d": dt.timedelta(days=1),
        "1w": dt.timedelta(days=7),
    }
    if m not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[m]


def _align_time(ts: dt.datetime, interval: str, mode: str) -> dt.datetime:
    """把时间戳对齐到指定周期的边界。"""
    ts = _parse_utc(ts)
    m = interval.strip().lower()

    # 分钟/小时周期用 epoch 分钟对齐；天/周单独处理
    if m.endswith("m") or m.endswith("h"):
        minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
        }.get(m)
        if minutes is None:
            raise ValueError(f"Unsupported interval for alignment: {interval}")
        epoch = int(ts.timestamp() // 60)
        if mode == "floor":
            aligned_epoch = epoch - (epoch % minutes)
        elif mode == "ceil":
            aligned_epoch = epoch + (-epoch % minutes)
        else:
            raise ValueError("mode must be 'floor' or 'ceil'")
        return dt.datetime.fromtimestamp(aligned_epoch * 60, tz=dt.timezone.utc)

    if m in ("1d", "1w"):
        if mode not in ("floor", "ceil"):
            raise ValueError("mode must be 'floor' or 'ceil'")
        # 先去掉小时/分钟，再按日/周处理
        day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        if m == "1d":
            if mode == "ceil" and ts != day_start:
                return day_start + dt.timedelta(days=1)
            return day_start
        # 周对齐：以周一为起点
        weekday = day_start.weekday()  # Monday=0
        week_start = day_start - dt.timedelta(days=weekday)
        if mode == "ceil" and ts != week_start:
            week_start = week_start + dt.timedelta(days=7)
        return week_start

    raise ValueError(f"Unsupported interval for alignment: {interval}")


class KlineData:
    """DuckDB 中 K 线的标准化访问接口。"""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH, table: str = DEFAULT_TABLE):
        self.db_path = Path(db_path)
        self.table = table
        self.conn = duckdb.connect(database=str(self.db_path))
        self._ensure_table()

    def close(self) -> None:
        self.conn.close()

    # -------------------- 元信息辅助 --------------------
    def _ensure_table(self) -> None:
        """检查表是否存在，不存在则抛出清晰提示。"""
        tables = {row[0] for row in self.conn.execute("SHOW TABLES").fetchall()}
        if self.table not in tables:
            raise RuntimeError(
                f"Table '{self.table}' not found in {self.db_path}. "
                "Ensure download_bitmex_1m_kline.py has populated it."
            )

    def list_symbols(self) -> List[str]:
        rows = self.conn.execute(
            f"SELECT DISTINCT symbol FROM {self.table} ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]

    def range(self, symbol: str) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        row = self.conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM {self.table} WHERE symbol = ?",
            (symbol,),
        ).fetchone()
        if not row or row[0] is None or row[1] is None:
            return None
        return _parse_utc(row[0]), _parse_utc(row[1])

    def ranges(self, symbols: Optional[Sequence[str]] = None) -> dict:
        """返回 {symbol: (start, end)}；symbols 为空则遍历全部。"""
        if symbols is None:
            symbols = self.list_symbols()
        result = {}
        for sym in symbols:
            rng = self.range(sym)
            if rng is not None:
                result[sym] = rng
        return result

    # -------------------- 主查询 --------------------
    def get_klines(
        self,
        symbols: Union[str, Iterable[str]],
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        interval: str = "1m",
        columns: str = "ohlcv",
        fill: str = "none",
    ) -> pd.DataFrame:
        """
        按指定 symbol + 时间窗聚合输出 K 线。

        symbols: 字符串或可迭代的字符串。
        start/end: UTC；会先按周期对齐（start 向下取整，end 向上取整）。
        interval: 1m,5m,15m,30m,1h,4h,1d,1w 之一。
        columns: 'ohlcv' 或 'full'（包含 trades/turnover/home/foreignNotional/vwap）。
        fill: 'none' 仅返回有数据的桶；'grid' 生成完整时间网格并允许缺失值。
        """
        sym_list = [symbols] if isinstance(symbols, str) else list(symbols)
        if not sym_list:
            raise ValueError("symbols cannot be empty")

        start_ts = _align_time(_parse_utc(start), interval, "floor")
        end_ts = _align_time(_parse_utc(end), interval, "ceil")
        if start_ts >= end_ts:
            raise ValueError("start must be before end after alignment")

        now_utc = dt.datetime.now(dt.timezone.utc)
        # 避免返回未收盘的当前桶：若用户请求超出当前时间，则剪裁到“当前时间向下对齐后的完整周期结束”
        safe_now_end = _align_time(now_utc, interval, "floor")
        end_effective = min(end_ts, safe_now_end)
        if end_effective <= start_ts:
            raise ValueError(
                "Effective end time is not after start; window too close to 'now'."
            )

        interval_literal = _interval_to_duckdb(interval)
        interval_delta = _interval_to_timedelta(interval)
        bucket_expr = f"time_bucket({interval_literal}, ts) AS bucket"

        select_cols = [
            "symbol",
            bucket_expr,
            "arg_min(open, ts) AS open",
            "max(high) AS high",
            "min(low) AS low",
            "arg_max(close, ts) AS close",
            "sum(volume) AS volume",
        ]
        if columns in ("ohlcv", "full"):
            select_cols.append("sum(trades) AS trades")
        if columns == "full":
            select_cols.extend(
                [
                    "sum(turnover) AS turnover",
                    "sum(homeNotional) AS homeNotional",
                    "sum(foreignNotional) AS foreignNotional",
                    "sum(turnover) / NULLIF(sum(homeNotional), 0) AS vwap",
                    "arg_max(lastSize, ts) AS lastSize",
                ]
            )

        placeholders = ", ".join(["?"] * len(sym_list))
        base_sql = f"""
            WITH base AS (
                SELECT
                    symbol,
                    CAST(timestamp AS TIMESTAMP) AS ts,
                    open, high, low, close, volume, trades,
                    vwap, turnover, homeNotional, foreignNotional, lastSize
                FROM {self.table}
                WHERE symbol IN ({placeholders})
                  AND ts >= ?
                  AND ts < ?
            ),
            agg AS (
                SELECT
                    {", ".join(select_cols)}
                FROM base
                GROUP BY symbol, bucket
            )
        """

        if fill == "grid":
            # 生成 [start_ts, end_effective) 的完整时间网格，最后一个桶起点为 end_effective - interval
            grid_end = end_effective - interval_delta
            grid_sql = f"""
                , symbol_list AS (
                    SELECT UNNEST(?) AS symbol
                ),
                grid AS (
                    SELECT
                        s.symbol,
                        g.bucket
                    FROM symbol_list s
                    CROSS JOIN generate_series(CAST(? AS TIMESTAMP), CAST(? AS TIMESTAMP), {interval_literal}) AS g(bucket)
                ),
                final AS (
                    SELECT
                        g.symbol,
                        g.bucket,
                        a.open,
                        a.high,
                        a.low,
                        a.close,
                        a.volume
                        {', a.trades' if columns in ('ohlcv', 'full') else ''}
                        {', a.turnover, a.homeNotional, a.foreignNotional, a.vwap, a.lastSize' if columns == 'full' else ''}
                    FROM grid g
                    LEFT JOIN agg a
                      ON a.symbol = g.symbol AND a.bucket = g.bucket
                )
                SELECT * FROM final ORDER BY symbol, bucket
            """
            sql = base_sql + grid_sql
            params = (*sym_list, _to_iso_z(start_ts), _to_iso_z(end_effective))
            params = (*params, sym_list, _to_iso_z(start_ts), _to_iso_z(grid_end))
        else:
            sql = base_sql + " SELECT * FROM agg ORDER BY symbol, bucket"
            params = (*sym_list, _to_iso_z(start_ts), _to_iso_z(end_effective))

        df = self.conn.execute(sql, params).fetchdf()
        return df


__all__ = ["KlineData"]
