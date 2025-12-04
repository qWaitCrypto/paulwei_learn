"""
BitMEX 1m K 线历史数据下载脚本（带断点续传与幂等性）。

核心设计：
- 数据源：BitMEX 公共 REST 接口 `/trade/bucketed`，`binSize=1m`；
- 字段：保存 12 个字段：
  `timestamp, symbol, open, high, low, close, trades, volume, vwap, lastSize, turnover, homeNotional, foreignNotional`；
- 存储：使用本地 DuckDB 数据库文件，单表 `kline_1m`，并通过 `(symbol, timestamp)` 唯一约束
  和去重插入逻辑，确保多次运行 / 重复下载是幂等的（不会产生重复行）。

断点续传策略（resume）：
- 每次运行前，从 DuckDB 查询该 symbol 已有的最早和最晚 timestamp；
- 目标下载区间可以来自：
  1) 手动指定 `--start` / `--end`；
  2) 从 `bitmex_trades.csv` 中推导出该 symbol 的首笔/尾笔成交时间，并前后扩展若干天；
- 对于 [target_start, target_end] 区间：
  - 若数据库中该 symbol 没有数据，则直接从 target_start 拉到 target_end；
  - 若已有部分数据，则只补：
      [target_start, existing_min) 及 (existing_max, target_end] 两端缺失区间；
  - 插入时通过唯一索引 + 去重插入逻辑，保证重复运行不会产生重复记录；
- 可选地通过 `--check-gaps` 对已存在数据做“内部缺口”检查，并尝试补齐。

使用示例：
1) 基于交易记录的时间范围，为单个品种拉取 1m K 线：
   python download_bitmex_1m_kline.py --symbol XBTUSD --use-trade-range

2) 为全部在 bitmex_trades.csv 出现过的 symbol 依次拉取：
   python download_bitmex_1m_kline.py --all-symbols --use-trade-range

3) 手动指定时间范围（UTC），不依赖交易记录：
   python download_bitmex_1m_kline.py --symbol ETHUSD --start 2020-01-01 --end 2021-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd
import requests


BASE_URL = "https://www.bitmex.com/api/v1"
DEFAULT_DB_PATH = "bitmex_1m_kline.duckdb"
DEFAULT_TRADES_CSV = "bitmex_trades.csv"


@dataclass
class TimeRange:
    start: dt.datetime
    end: dt.datetime

    def clamp_to(self, other: "TimeRange") -> "TimeRange":
        """返回与 other 的交集（若无交集则抛错）。"""
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        if new_start >= new_end:
            raise ValueError("Time ranges do not overlap.")
        return TimeRange(start=new_start, end=new_end)


def parse_utc(s: str) -> dt.datetime:
    """解析类似 '2020-01-01' 或 ISO8601 字符串为 UTC 时间。"""
    s = s.strip()
    # 简单日期形式
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = dt.datetime.strptime(s, "%Y-%m-%d")
        return d.replace(tzinfo=dt.timezone.utc)
    # 其它情况直接交给 fromisoformat，支持带时区 / Z 结尾
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)


def to_iso_z(t: dt.datetime) -> str:
    """以 `YYYY-MM-DDTHH:MM:SSZ` 格式输出 UTC 时间。"""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    else:
        t = t.astimezone(dt.timezone.utc)
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """初始化 DuckDB 数据库和表结构（必要时自动升级列类型）。"""
    conn = duckdb.connect(database=str(db_path))
    # 目标 schema：数值列使用更宽的类型，避免溢出
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kline_1m (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            trades BIGINT,
            volume BIGINT,
            vwap DOUBLE,
            lastSize BIGINT,
            turnover BIGINT,
            homeNotional DOUBLE,
            foreignNotional DOUBLE
        )
        """
    )
    # 若表已存在且列类型较窄，则尝试升级为上述目标类型（向上转型安全）
    # 使用 information_schema.columns 检查并按需 ALTER。若需要 ALTER，则先暂时删除索引，
    # 完成所有列类型升级后再重建索引，避免 DependencyException。
    cols = conn.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'kline_1m'
        """
    ).fetchall()
    col_types = {name.lower(): dtype.upper() for name, dtype in cols}

    double_cols = ("open", "high", "low", "close", "vwap", "homeNotional", "foreignNotional")
    bigint_cols = ("trades", "volume", "lastSize", "turnover")

    need_alter = False
    for col in double_cols:
        if col_types.get(col.lower()) not in (None, "DOUBLE"):
            need_alter = True
            break
    if not need_alter:
        for col in bigint_cols:
            if col_types.get(col.lower()) not in (None, "BIGINT"):
                need_alter = True
                break

    if need_alter:
        # 删除可能依赖列类型的索引
        conn.execute("DROP INDEX IF EXISTS idx_kline_symbol_ts")
        # 执行列类型升级
        for col in double_cols:
            if col_types.get(col.lower()) not in (None, "DOUBLE"):
                conn.execute(f"ALTER TABLE kline_1m ALTER COLUMN {col} TYPE DOUBLE")
        for col in bigint_cols:
            if col_types.get(col.lower()) not in (None, "BIGINT"):
                conn.execute(f"ALTER TABLE kline_1m ALTER COLUMN {col} TYPE BIGINT")

    # 为 (symbol, timestamp) 建立唯一索引，用于保证幂等性
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_kline_symbol_ts
        ON kline_1m(symbol, timestamp)
        """
    )

    return conn


def load_symbol_ranges_from_trades(
    csv_path: Path,
) -> Dict[str, TimeRange]:
    """
    从 bitmex_trades.csv 中推导每个 symbol 的首笔/尾笔成交时间。

    返回：
        {symbol: TimeRange(start, end)}
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "symbol" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'symbol' columns.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

    ranges: Dict[str, TimeRange] = {}
    for symbol, group in df.groupby("symbol"):
        start_ts = group["timestamp"].min().to_pydatetime()
        end_ts = group["timestamp"].max().to_pydatetime()
        ranges[symbol] = TimeRange(start=start_ts, end=end_ts)
    return ranges


def get_existing_range(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
) -> Optional[TimeRange]:
    """查询数据库中该 symbol 已有的最早/最晚 timestamp。"""
    cur = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM kline_1m WHERE symbol = ?",
        (symbol,),
    )
    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return None

    min_ts_str, max_ts_str = row
    # 存储中是 ISO8601 字符串，带 Z
    min_ts = parse_utc(min_ts_str)
    max_ts = parse_utc(max_ts_str)
    return TimeRange(start=min_ts, end=max_ts)


def fetch_bitmex_bucketed_1m(
    symbol: str,
    start_time: dt.datetime,
    count: int = 750,
    timeout: int = 10,
) -> List[dict]:
    """
    调用 BitMEX trade/bucketed 接口，拉取给定 symbol 从 start_time 开始的 1m K 线。

    注意：
    - 以 startTime + count 的方式向前分页，reverse=false；
    - 返回的每条记录都应包含完整的 1 分钟 OHLCV 信息。
    """
    endpoint = f"{BASE_URL}/trade/bucketed"
    params = {
        "binSize": "1m",
        "symbol": symbol,
        "count": count,
        "reverse": "false",
        "partial": "false",
        "startTime": to_iso_z(start_time),
    }
    resp = requests.get(endpoint, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response type: {type(data)}")
    return data


def insert_candles(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    candles: Iterable[dict],
) -> int:
    """将一批 K 线插入 DuckDB，遇到重复 (symbol, timestamp) 时忽略。"""
    rows: List[Dict[str, object]] = []
    for c in candles:
        rows.append(
            {
                "symbol": symbol,
                "timestamp": c.get("timestamp"),
                "open": c.get("open"),
                "high": c.get("high"),
                "low": c.get("low"),
                "close": c.get("close"),
                "trades": c.get("trades"),
                "volume": c.get("volume"),
                "vwap": c.get("vwap"),
                "lastSize": c.get("lastSize"),
                "turnover": c.get("turnover"),
                "homeNotional": c.get("homeNotional"),
                "foreignNotional": c.get("foreignNotional"),
            }
        )

    if not rows:
        return 0

    df = pd.DataFrame(rows)
    # 在 DuckDB 中注册临时表
    conn.register("tmp_candles", df)
    # 通过 NOT EXISTS 避免插入重复 (symbol, timestamp) 记录
    conn.execute(
        """
        INSERT INTO kline_1m (
            symbol, timestamp, open, high, low, close,
            trades, volume, vwap, lastSize,
            turnover, homeNotional, foreignNotional
        )
        SELECT
            symbol, timestamp, open, high, low, close,
            trades, volume, vwap, lastSize,
            turnover, homeNotional, foreignNotional
        FROM tmp_candles t
        WHERE NOT EXISTS (
            SELECT 1 FROM kline_1m k
            WHERE k.symbol = t.symbol AND k.timestamp = t.timestamp
        )
        """
    )
    # 这里无法精确获取实际插入条数，返回尝试插入的条数作为近似
    inserted = len(df)
    conn.unregister("tmp_candles")
    return inserted


def download_range_forward(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    time_range: TimeRange,
    batch_size: int = 750,
    rate_limit_sleep: float = 1.0,
) -> None:
    """从 time_range.start 开始，向前下载到 time_range.end 或“当前时间”之前。

    为避免向“未来时间”请求：
    - 每次调用时，将 end 截断为 min(time_range.end, now_utc - 1min)；
    - 若已接近当前时间（current >= now_utc - 小缓冲）且返回空结果，则认为已经没有更多已完成的 K 线，停止下载。
    """
    current = time_range.start
    end = time_range.end

    # 将结束时间截断到“当前 UTC 时间之前”，避免向未来请求数据
    now_utc = dt.datetime.now(dt.timezone.utc)
    effective_end = min(end, now_utc - dt.timedelta(minutes=1))

    if current >= effective_end:
        print(
            f"[{symbol}] effective_end ({to_iso_z(effective_end)}) "
            f"is not after start ({to_iso_z(current)}), nothing to download.",
        )
        return

    print(
        f"[{symbol}] downloading forward from {to_iso_z(current)} "
        f"to {to_iso_z(effective_end)} (batch={batch_size})"
    )

    while current <= effective_end:
        try:
            candles = fetch_bitmex_bucketed_1m(symbol, current, count=batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[{symbol}] request failed at {to_iso_z(current)}: {exc!r}")
            print(f"[{symbol}] sleeping 10s then retrying...")
            time.sleep(10)
            continue

        if not candles:
            # 根据 current 与“现在”的距离，区分两种情况：
            # 1) 若已经非常接近当前时间：认为没有更多已完成的 1m K 线可取，直接停止；
            # 2) 若离现在还很远：说明可能处于合约未上市或已到期的区间，跳过一段时间继续尝试。
            now_utc = dt.datetime.now(dt.timezone.utc)
            near_now_threshold = now_utc - dt.timedelta(minutes=2)
            if current >= near_now_threshold:
                print(
                    f"[{symbol}] empty response near 'now' at {to_iso_z(current)}, "
                    "no more completed candles available, stopping.",
                )
                break

            print(
                f"[{symbol}] empty response at {to_iso_z(current)}, "
                "advancing by 1 day to skip inactive period.",
            )
            current = current + dt.timedelta(days=1)
            if current > effective_end:
                break
            continue

        # 过滤掉超出 effective_end 之后的蜡烛，避免把目标区间之外的数据写入数据库
        filtered: List[dict] = []
        for c in candles:
            ts_c = parse_utc(c["timestamp"])
            if ts_c <= effective_end:
                filtered.append(c)

        if not filtered:
            print(
                f"[{symbol}] all fetched candles beyond effective_end "
                f"({to_iso_z(effective_end)}), stopping.",
            )
            break

        inserted = insert_candles(conn, symbol, filtered)
        last_ts_str = filtered[-1]["timestamp"]
        last_ts = parse_utc(last_ts_str)
        current = last_ts + dt.timedelta(minutes=1)

        print(
            f"[{symbol}] fetched {len(candles)} candles, "
            f"inserted {inserted}, last={last_ts_str}"
        )

        # 简单限流，避免触发 BitMEX 频率限制
        time.sleep(rate_limit_sleep)


def find_internal_gaps(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    time_range: TimeRange,
) -> List[TimeRange]:
    """
    在数据库中检查指定 symbol 在给定时间区间内的内部缺口。

    返回：
        缺口列表，每个 TimeRange 表示 (prev+1min, next-1min) 的缺失区域。
    """
    cur = conn.execute(
        """
        SELECT timestamp
        FROM kline_1m
        WHERE symbol = ?
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
        """,
        (symbol, to_iso_z(time_range.start), to_iso_z(time_range.end)),
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        return []

    missing: List[TimeRange] = []
    prev_ts = parse_utc(rows[0][0])

    for (ts_str,) in rows[1:]:
        ts = parse_utc(ts_str)
        delta_minutes = (ts - prev_ts).total_seconds() / 60.0
        if delta_minutes > 1.5:
            gap_start = prev_ts + dt.timedelta(minutes=1)
            gap_end = ts - dt.timedelta(minutes=1)
            missing.append(TimeRange(start=gap_start, end=gap_end))
        prev_ts = ts

    return missing


def plan_download_segments(
    existing: Optional[TimeRange],
    target: TimeRange,
) -> List[TimeRange]:
    """
    根据已存在区间 existing 与目标区间 target，规划需要追加下载的区间。

    逻辑：
    - 若 existing 为空：直接返回 [target]；
    - 若有 existing：只补 target.start 到 existing.start 之前、
      以及 existing.end 到 target.end 之后的部分。
    """
    segments: List[TimeRange] = []
    if existing is None:
        segments.append(target)
        return segments

    if target.start < existing.start:
        segments.append(TimeRange(start=target.start, end=existing.start))
    if target.end > existing.end:
        segments.append(TimeRange(start=existing.end + dt.timedelta(minutes=1), end=target.end))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BitMEX 1m kline into DuckDB with resume.")
    parser.add_argument("--symbol", help="Single symbol to download, e.g. XBTUSD.")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Download for all symbols present in bitmex_trades.csv.",
    )
    parser.add_argument(
        "--use-trade-range",
        action="store_true",
        help="Derive time range from bitmex_trades.csv (per symbol).",
    )
    parser.add_argument(
        "--start",
        help="Manual start time (UTC), e.g. 2020-01-01 or 2020-01-01T00:00:00.",
    )
    parser.add_argument(
        "--end",
        help="Manual end time (UTC), e.g. 2025-01-01.",
    )
    parser.add_argument(
        "--buffer-days",
        type=int,
        default=15,
        help="Days to extend before first trade and after last trade when --use-trade-range.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"DuckDB database path (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--trades-csv",
        default=DEFAULT_TRADES_CSV,
        help=f"Trades CSV path (default: {DEFAULT_TRADES_CSV}).",
    )
    parser.add_argument(
        "--check-gaps",
        action="store_true",
        help="After downloading outer ranges, scan and attempt to refill internal gaps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=750,
        help="Batch size per HTTP request (<=1000).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep seconds between requests to avoid rate limits.",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    trades_csv_path = Path(args.trades_csv)

    if not args.symbol and not args.all_symbols:
        raise SystemExit("You must specify --symbol or --all-symbols.")

    if args.use_trade_range and not trades_csv_path.exists():
        raise SystemExit(f"Trades CSV not found: {trades_csv_path}")

    # 加载 symbol → trade time range 映射（若需要）
    symbol_ranges: Dict[str, TimeRange] = {}
    if args.use_trade_range:
        symbol_ranges = load_symbol_ranges_from_trades(trades_csv_path)

    # 确定要下载的 symbol 列表
    symbols: List[str]
    if args.all_symbols:
        if not symbol_ranges:
            raise SystemExit("Using --all-symbols requires --use-trade-range.")
        symbols = sorted(symbol_ranges.keys())
    else:
        if not args.symbol:
            raise SystemExit("When not using --all-symbols, you must provide --symbol.")
        symbols = [args.symbol]

    conn = init_db(db_path)

    for symbol in symbols:
        print(f"\n===== Processing symbol: {symbol} =====")
        # 1) 确定 target range
        if args.use_trade_range:
            if symbol not in symbol_ranges:
                print(f"[{symbol}] not found in trades CSV, skipping.")
                continue
            base_range = symbol_ranges[symbol]
            target_start = base_range.start - dt.timedelta(days=args.buffer_days)
            target_end = base_range.end + dt.timedelta(days=args.buffer_days)
        else:
            if not args.start or not args.end:
                print(
                    f"[{symbol}] must specify --start and --end when not using --use-trade-range.",
                )
                continue
            target_start = parse_utc(args.start)
            target_end = parse_utc(args.end)

        if target_start >= target_end:
            print(f"[{symbol}] invalid target range: start >= end, skipping.")
            continue

        target = TimeRange(start=target_start, end=target_end)
        print(
            f"[{symbol}] target range: {to_iso_z(target.start)} -> {to_iso_z(target.end)}",
        )

        # 2) 查询已有数据范围
        existing = get_existing_range(conn, symbol)
        if existing is None:
            print(f"[{symbol}] no existing data in DB.")
        else:
            print(
                f"[{symbol}] existing range: "
                f"{to_iso_z(existing.start)} -> {to_iso_z(existing.end)}",
            )

        # 3) 规划需要下载的区间（仅补两端）
        segments = plan_download_segments(existing, target)
        if not segments:
            print(f"[{symbol}] nothing to download for outer ranges.")
        else:
            for seg in segments:
                download_range_forward(
                    conn,
                    symbol,
                    seg,
                    batch_size=args.batch_size,
                    rate_limit_sleep=args.sleep,
                )

        # 4) 可选：检查并补内部缺口
        if args.check_gaps:
            print(f"[{symbol}] scanning for internal gaps within target range...")
            gaps = find_internal_gaps(conn, symbol, target)
            if not gaps:
                print(f"[{symbol}] no internal gaps detected.")
            else:
                print(f"[{symbol}] detected {len(gaps)} internal gaps, attempting to refill...")
                for gap in gaps:
                    print(
                        f"[{symbol}] refilling gap: "
                        f"{to_iso_z(gap.start)} -> {to_iso_z(gap.end)}",
                    )
                    download_range_forward(
                        conn,
                        symbol,
                        gap,
                        batch_size=args.batch_size,
                        rate_limit_sleep=args.sleep,
                    )

    conn.close()


if __name__ == "__main__":
    main()
