"""
功能/性能测试：
1) 列出 symbol，并做 1h 对齐验证（原始 08:03→12:10，应返回 9/10/11/12 四根）。
2) 15m 示例输出。
3) 多组长窗口性能测试（从 1m 到 1d），覆盖更长时间。
4) 多 symbol + grid 补全示例，验证对齐/缺口填充。
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

# 确保可以导入上级 data 模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_api import KlineData, _align_time


def pick_base_day(rng: Tuple[dt.datetime, dt.datetime]) -> dt.datetime:
    """选择示例日期：优先 2021-01-02，否则用覆盖范围内的一天。"""
    base = dt.datetime(2021, 1, 2, tzinfo=dt.timezone.utc)
    if rng and not (rng[0] <= base <= rng[1]):
        base = rng[0].replace(hour=0, minute=0, second=0, microsecond=0)
    return base


def run_perf_case(
    kd: KlineData,
    symbol: str,
    rng: Tuple[dt.datetime, dt.datetime],
    interval: str,
    start: dt.datetime,
    end: dt.datetime,
) -> None:
    """执行单个性能测试用例并打印耗时/行数。"""
    start_aligned = _align_time(start, interval, "floor")
    end_aligned = _align_time(end, interval, "ceil")
    # 确保不超范围
    if start_aligned < rng[0]:
        start_aligned = _align_time(rng[0], interval, "floor")
    if end_aligned > rng[1]:
        end_aligned = _align_time(rng[1], interval, "ceil")

    t0 = time.perf_counter()
    df = kd.get_klines(
        symbols=symbol,
        start=start_aligned,
        end=end_aligned,
        interval=interval,
        columns="full",
    )
    cost = time.perf_counter() - t0
    print(f"{interval} | 窗口 {start_aligned} -> {end_aligned} | 行数 {len(df)} | 耗时 {cost:.3f}s")


def main() -> None:
    kd = KlineData(db_path="data/bitmex_1m_kline.duckdb")

    # 1) 列出 symbol
    symbols = kd.list_symbols()
    print(f"可用 symbol 数量：{len(symbols)}")
    print(f"前 10 个 symbol：{symbols[:10]}")

    # 选择一个常用 symbol
    symbol = "XBTUSD" if "XBTUSD" in symbols else symbols[0]
    rng = kd.range(symbol)
    print(f"测试 symbol：{symbol}")
    if rng:
        print(f"{symbol} 覆盖时间：{rng[0]} -> {rng[1]}")
    else:
        print(f"{symbol} 没有数据")
        return

    # 2) 1h 对齐示例：原始窗口 08:03 -> 12:10，向上对齐后应取 9/10/11/12 四根
    base_day = pick_base_day(rng)
    raw_start = base_day.replace(hour=8, minute=3, second=0, microsecond=0)
    raw_end = base_day.replace(hour=12, minute=10, second=0, microsecond=0)
    if raw_end > rng[1]:
        # 如超出范围，退回到覆盖范围末尾前的同一天
        base_day = (rng[1] - dt.timedelta(hours=13)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        raw_start = base_day.replace(hour=8, minute=3, second=0, microsecond=0)
        raw_end = base_day.replace(hour=12, minute=10, second=0, microsecond=0)

    aligned_start = _align_time(raw_start, "1h", "ceil")
    aligned_end = _align_time(raw_end, "1h", "ceil")
    print(f"\n原始请求窗口：{raw_start} -> {raw_end}")
    print(f"向上对齐到整点：{aligned_start} -> {aligned_end}（end 开区间，不含 {aligned_end} 桶）")

    df_1h = kd.get_klines(
        symbols=symbol,
        start=aligned_start,
        end=aligned_end,
        interval="1h",
        columns="ohlcv",
    )
    print("1h K 线（预期 9/10/11/12 四根）：")
    print(df_1h)

    # 3) 15m 级别，使用原始窗口（内部会 start 向下、end 向上对齐）
    df_15m = kd.get_klines(
        symbols=symbol,
        start=raw_start,
        end=raw_end,
        interval="15m",
        columns="ohlcv",
    )
    print("\n15m K 线（展示前 12 行）：")
    print(df_15m.head(12))

    # 4) 性能测试：不同周期，长窗口（更大体量）
    print("\n=== 性能测试（耗时、行数） ===")
    end_ts = _align_time(rng[1], "1m", "floor")
    perf_windows = [
        ("1m", end_ts - pd.Timedelta(days=180), end_ts),          # 近 6 个月
        ("1m", end_ts - pd.Timedelta(days=30), end_ts),           # 近 30 天
        ("5m", end_ts - pd.Timedelta(days=365), end_ts),          # 近 1 年
        ("15m", end_ts - pd.Timedelta(days=365 * 3), end_ts),     # 近 3 年
        ("1h", end_ts - pd.Timedelta(days=365 * 5), end_ts),      # 近 5 年
        ("1d", rng[0], rng[1]),                                   # 全量日线
    ]
    for interval, start_ts, end_ts_case in perf_windows:
        run_perf_case(kd, symbol, rng, interval, start_ts, end_ts_case)

    # 5) 多 symbol + grid 填充示例（短窗口，避免过大）
    print("\n=== 多 symbol + grid 填充示例（15m，3 天窗口） ===")
    sample_symbols: Iterable[str] = symbols[:3]
    end_short = _align_time(rng[1], "15m", "ceil")
    start_short = end_short - pd.Timedelta(days=3)
    grid_df = kd.get_klines(
        symbols=sample_symbols,
        start=start_short,
        end=end_short,
        interval="15m",
        columns="ohlcv",
        fill="grid",
    )
    print(f"symbols: {list(sample_symbols)} | 窗口 {start_short} -> {end_short} | 行数 {len(grid_df)}")
    print(grid_df.head(6))


if __name__ == "__main__":
    main()
