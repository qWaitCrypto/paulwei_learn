"""
本地后端服务：提供 K 线 / 成交数据 API，并挂载前端静态页面。

特性：
- 依赖已有的数据访问层：KlineData（DuckDB 1m K 线）和 TradesData（CSV 成交）。
- 提供规范化的 JSON 接口，便于前端 Lightweight Charts 适配。
- 完全本地运行：不依赖任何外部网络或云服务。
"""

from __future__ import annotations

import datetime as dt
import random
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data.data_api import KlineData, _align_time, _interval_to_timedelta
from data.trades_data import TradesData
import pandas as pd

from tv_backend.practice_engine import (
    Bar,
    Order as PracticeOrder,
    PracticeConfig,
    engine as practice_engine,
)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WEB_DIR = PROJECT_ROOT / "web"

# 初始化数据访问层
kline_data = KlineData(db_path=DATA_DIR / "bitmex_1m_kline.duckdb")
trades_data = TradesData(trades_csv=DATA_DIR / "bitmex_trades.csv")

# 支持的周期映射
SUPPORTED_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}

app = FastAPI(title="PaulWei Local Kline API", version="1.0.0")


# ----------------------------- 模型定义 -----------------------------
class KlineBar(BaseModel):
    t: int  # 秒级时间戳（UTC）
    o: float
    h: float
    l: float
    c: float
    v: float
    trades: Optional[float] = None
    turnover: Optional[float] = None
    homeNotional: Optional[float] = None
    foreignNotional: Optional[float] = None


class KlineResponse(BaseModel):
    s: str = Field("ok", description="ok / no_data / error")
    symbol: str
    interval: str
    from_ts: int
    to_ts: int
    bars: List[KlineBar]


class TradeItem(BaseModel):
    timestamp: str
    side: Optional[str] = None
    price: float
    qty: float
    ordType: Optional[str] = None
    orderID: Optional[str] = None
    execID: Optional[str] = None
    homeNotional: Optional[float] = None
    foreignNotional: Optional[float] = None
    commission: Optional[float] = None


class TradesResponse(BaseModel):
    s: str = Field("ok", description="ok / no_data / error")
    symbol: str
    from_ts: int
    to_ts: int
    trades: List[TradeItem]


# ----------------------------- 练习模式模型 -----------------------------
class PracticeSessionCreate(BaseModel):
    symbol: Optional[str] = Field(None, description="可选：指定品种；不填则随机")
    interval: str = Field("15m", description="周期，默认 15m")
    duration_minutes: int = Field(24 * 60, description="时长（分钟），默认 1 天")
    start: Optional[str] = Field(None, description="可选：指定开始时间（ISO 或 epoch 秒）")
    end: Optional[str] = Field(None, description="可选：指定结束时间（ISO 或 epoch 秒）")
    initial_cash: Optional[float] = Field(100000.0, description="初始资金")
    fee_rate: Optional[float] = Field(0.0006, description="手续费率，按成交额")
    slippage_pct: Optional[float] = Field(0.0005, description="滑点百分比")


class PracticeSessionInfo(BaseModel):
    session_id: str
    symbol: str
    interval: str
    start: str
    end: str


class PracticeBarsResponse(BaseModel):
    s: str = Field("ok")
    session_id: str
    symbol: str
    interval: str
    bars: List[KlineBar]


class PracticeTradesResponse(BaseModel):
    s: str = Field("ok")
    session_id: str
    symbol: str
    trades: List[TradeItem]


# ----------------------------- 工具函数 -----------------------------
def parse_utc(ts: str) -> dt.datetime:
    """解析 ISO 字符串或秒级时间戳为 UTC datetime。"""
    ts = ts.strip()
    # 如果是纯数字，按秒级 epoch 解析
    if ts.isdigit():
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc)
    # ISO 形式
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    dt_obj = dt.datetime.fromisoformat(ts)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return dt_obj.astimezone(dt.timezone.utc)


def iso_z(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        ts = ts.astimezone(dt.timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def choose_random_session(interval: str, duration_minutes: int) -> PracticeSessionInfo:
    """随机选择一个 symbol + 时间区间，保证区间长度足够。"""
    symbols = kline_data.list_symbols()
    if not symbols:
        raise RuntimeError("No symbols available in kline data.")

    # 尝试最多 100 次随机选取
    for _ in range(100):
        symbol = random.choice(symbols)
        rng = kline_data.range(symbol)
        if rng is None:
            continue
        start_dt, end_dt = rng
        total_minutes = (end_dt - start_dt).total_seconds() / 60.0
        if total_minutes < duration_minutes:
            continue
        # 随机选择窗口
        margin_minutes = total_minutes - duration_minutes
        offset_minutes = random.uniform(0, margin_minutes)
        candidate_start = start_dt + dt.timedelta(minutes=offset_minutes)
        candidate_end = candidate_start + dt.timedelta(minutes=duration_minutes)
        # 对齐到周期边界
        aligned_start = _align_time(candidate_start, interval, "floor")
        aligned_end = _align_time(candidate_end, interval, "ceil")
        session = PracticeSessionInfo(
            session_id=str(uuid.uuid4()),
            symbol=symbol,
            interval=interval,
            start=iso_z(aligned_start),
            end=iso_z(aligned_end),
        )
        return session
    raise RuntimeError("Failed to choose a random session; try different parameters.")


# ----------------------------- 路由 -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/symbols")
def list_symbols():
    kl_symbols = set(kline_data.list_symbols())
    tr_symbols = set(trades_data.list_symbols())
    symbols = sorted(kl_symbols.union(tr_symbols))
    return {"symbols": symbols}


@app.get("/api/kline", response_model=KlineResponse)
def get_kline(
    symbol: str = Query(..., description="合约代码，如 XBTUSD"),
    interval: str = Query(..., description="周期：1m/5m/15m/30m/1h/4h/1d/1w"),
    start: str = Query(..., description="开始时间，ISO 或 epoch 秒"),
    end: str = Query(..., description="结束时间，ISO 或 epoch 秒"),
    columns: str = Query("ohlcv", description="ohlcv 或 full（含交易量/委托等）"),
):
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval not supported: {interval}")

    start_ts = parse_utc(start)
    end_ts = parse_utc(end)
    if start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="start must be before end")

    try:
        df = kline_data.get_klines(
            symbols=symbol,
            start=start_ts,
            end=end_ts,
            interval=interval,
            columns="full" if columns == "full" else "ohlcv",
            fill="none",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if df.empty:
        return KlineResponse(
            s="no_data",
            symbol=symbol,
            interval=interval,
            from_ts=int(start_ts.timestamp()),
            to_ts=int(end_ts.timestamp()),
            bars=[],
        )

    bars: List[KlineBar] = []
    for _, row in df.iterrows():
        t = int(pd.Timestamp(row["bucket"]).timestamp())
        bar = {
            "t": t,
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]) if "volume" in row else None,
        }
        if columns == "full":
            bar.update(
                {
                    "trades": float(row["trades"]) if "trades" in row else None,
                    "turnover": float(row["turnover"]) if "turnover" in row else None,
                    "homeNotional": float(row["homeNotional"]) if "homeNotional" in row else None,
                    "foreignNotional": float(row["foreignNotional"])
                    if "foreignNotional" in row
                    else None,
                }
            )
        bars.append(KlineBar(**bar))

    return KlineResponse(
        s="ok",
        symbol=symbol,
        interval=interval,
        from_ts=int(start_ts.timestamp()),
        to_ts=int(end_ts.timestamp()),
        bars=bars,
    )


@app.get("/api/trades", response_model=TradesResponse)
def get_trades(
    symbol: str = Query(..., description="合约代码，如 XBTUSD"),
    start: str = Query(..., description="开始时间，ISO 或 epoch 秒"),
    end: str = Query(..., description="结束时间，ISO 或 epoch 秒"),
    limit: Optional[int] = Query(None, description="可选，返回前 N 条"),
):
    start_ts = parse_utc(start)
    end_ts = parse_utc(end)
    if start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="start must be before end")

    try:
        df = trades_data.get_trades(symbol=symbol, start=start_ts, end=end_ts, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if df.empty:
        return TradesResponse(
            s="no_data",
            symbol=symbol,
            from_ts=int(start_ts.timestamp()),
            to_ts=int(end_ts.timestamp()),
            trades=[],
        )

    trades_list: List[TradeItem] = []
    for _, row in df.iterrows():
        trades_list.append(
            TradeItem(
                timestamp=iso_z(row["timestamp"]),
                side=row.get("side"),
                price=float(row["price"]),
                qty=float(row["qty"]),
                ordType=row.get("ordType"),
                orderID=row.get("orderID"),
                execID=row.get("execID"),
                homeNotional=float(row["homeNotional"])
                if "homeNotional" in row and pd.notna(row["homeNotional"])
                else None,
                foreignNotional=float(row["foreignNotional"])
                if "foreignNotional" in row and pd.notna(row["foreignNotional"])
                else None,
                commission=float(row["commission"]) if "commission" in row else None,
            )
        )

    return TradesResponse(
        s="ok",
        symbol=symbol,
        from_ts=int(start_ts.timestamp()),
        to_ts=int(end_ts.timestamp()),
        trades=trades_list,
    )


# ----------------------------- 练习模式接口 -----------------------------
@app.post("/api/practice/session", response_model=PracticeSessionInfo)
def create_practice_session(payload: PracticeSessionCreate):
    # 参数校验
    interval = payload.interval
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"interval not supported: {interval}")
    duration_minutes = payload.duration_minutes
    if duration_minutes <= 0:
        raise HTTPException(status_code=400, detail="duration_minutes must be positive")

    symbol = payload.symbol
    # 确定时间区间
    if payload.start and payload.end:
        start_dt = parse_utc(payload.start)
        end_dt = parse_utc(payload.end)
        if start_dt >= end_dt:
            raise HTTPException(status_code=400, detail="start must be before end")
    else:
        # 随机选取时间段
        candidates = [payload.symbol] if payload.symbol else kline_data.list_symbols()
        selected = None
        for _ in range(200):
            sym = random.choice(candidates)
            rng = kline_data.range(sym)
            if rng is None:
                continue
            s_start, s_end = rng
            total_minutes = (s_end - s_start).total_seconds() / 60.0
            if total_minutes < duration_minutes:
                continue
            margin = total_minutes - duration_minutes
            offset = random.uniform(0, margin)
            start_dt = s_start + dt.timedelta(minutes=offset)
            end_dt = start_dt + dt.timedelta(minutes=duration_minutes)
            selected = (sym, start_dt, end_dt)
            break
        if selected is None:
            raise HTTPException(status_code=400, detail="failed to pick random session")
        symbol = selected[0]
        start_dt, end_dt = selected[1], selected[2]
    # 若用户指定了 symbol，覆盖随机结果；若仍为空则随机
    if symbol is None:
        if payload.symbol:
            symbol = payload.symbol
        else:
            symbol = random.choice(kline_data.list_symbols())

    aligned_start = _align_time(start_dt, interval, "floor")
    aligned_end = _align_time(end_dt, interval, "ceil")

    # 读取时间段内的 K 线，作为练习数据
    df = kline_data.get_klines(
        symbols=symbol,
        start=aligned_start,
        end=aligned_end,
        interval=interval,
        columns="ohlcv",
        fill="none",
    )
    if df.empty:
        raise HTTPException(status_code=400, detail="no kline data for selected range")
    bars = []
    for _, row in df.iterrows():
        bars.append(
            Bar(
                t=int(pd.Timestamp(row["bucket"]).timestamp()),
                o=float(row["open"]),
                h=float(row["high"]),
                l=float(row["low"]),
                c=float(row["close"]),
                v=float(row["volume"]),
            )
        )

    config = PracticeConfig(
        interval=interval,
        initial_cash=payload_dict_value(payload, "initial_cash", default=100000.0),
        fee_rate=payload_dict_value(payload, "fee_rate", default=0.0006),
        slippage_pct=payload_dict_value(payload, "slippage_pct", default=0.0005),
        duration_minutes=duration_minutes,
    )
    # 初始练习位置：默认从窗口中间开始，这样左侧有完整历史，右侧逐步推进
    start_index = max(0, min(len(bars) - 1, len(bars) // 2))
    session_state = practice_engine.create_session(
        symbol=symbol,
        bars=bars,
        config=config,
        start_index=start_index,
    )
    # 将对齐后的开始结束时间返回
    session_info = PracticeSessionInfo(
        session_id=session_state.session_id,
        symbol=symbol,
        interval=interval,
        start=iso_z(aligned_start),
        end=iso_z(aligned_end),
    )
    return session_info


def payload_dict_value(payload: PracticeSessionCreate, key: str, default: float) -> float:
    # 兼容未来扩展字段，不存在则返回默认
    d = payload.dict()
    return float(d.get(key, default)) if d.get(key) is not None else default


@app.get("/api/practice/session/{session_id}", response_model=PracticeSessionInfo)
def get_practice_session(session_id: str):
    try:
        state = practice_engine.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return PracticeSessionInfo(
        session_id=state.session_id,
        symbol=state.symbol,
        interval=state.config.interval,
        start=iso_z(dt.datetime.fromtimestamp(state.bars[0].t, tz=dt.timezone.utc))
        if state.bars
        else "",
        end=iso_z(dt.datetime.fromtimestamp(state.bars[-1].t, tz=dt.timezone.utc))
        if state.bars
        else "",
    )


def _serialize_state(state):
    current_bar = state.bars[state.current_index] if state.current_index < len(state.bars) else None
    pos = state.position
    unrealized = 0.0
    if pos.qty != 0 and current_bar:
        direction = 1 if pos.qty > 0 else -1
        unrealized = (current_bar.c - pos.avg_price) * abs(pos.qty) * direction
    equity = state.cash + state.realized_pnl + unrealized

    return {
        "session_id": state.session_id,
        "symbol": state.symbol,
        "interval": state.config.interval,
        "current_index": state.current_index,
        "total_bars": len(state.bars),
        "equity": equity,
        "cash": state.cash,
        "realized_pnl": state.realized_pnl,
        "fees": state.fees,
        "position": {
            "qty": pos.qty,
            "avg_price": pos.avg_price,
            "take_profit": pos.take_profit,
            "stop_loss": pos.stop_loss,
            "unrealized": unrealized,
        },
        "open_orders": [
            {
                "id": o.id,
                "side": o.side,
                "type": o.type,
                "qty": o.qty,
                "price": o.price,
                "stop_price": o.stop_price,
                "take_profit": o.take_profit,
                "stop_loss": o.stop_loss,
                "status": o.status,
            }
            for o in state.open_orders.values()
        ],
        "fills": [
            {
                "time": iso_z(f.time),
                "side": f.side,
                "qty": f.qty,
                "price": f.price,
                "fee": f.fee,
                "pnl": f.pnl,
            }
            for f in state.fills[-50:]
        ],
        "equity_curve": state.equity_curve[-500:],
        "bars": [{"t": b.t, "o": b.o, "h": b.h, "l": b.l, "c": b.c, "v": b.v} for b in state.bars[: state.current_index + 1]],
    }


@app.get("/api/practice/state")
def get_practice_state(session_id: str):
    try:
        state = practice_engine.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    # 更新权益（确保当前点的 equity 记录）
    if state.bars:
        bar = state.bars[state.current_index]
        # 追加当前 equity
        pos = state.position
        unrealized = 0.0
        if pos.qty != 0:
            direction = 1 if pos.qty > 0 else -1
            unrealized = (bar.c - pos.avg_price) * abs(pos.qty) * direction
        equity = state.cash + state.realized_pnl + unrealized
        if not state.equity_curve or state.equity_curve[-1][0] != bar.t:
            state.equity_curve.append((bar.t, equity))
    return _serialize_state(state)


@app.post("/api/practice/advance")
def practice_advance(session_id: str, steps: int = 1):
    try:
        state = practice_engine.advance(session_id, steps=steps)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _serialize_state(state)


class OrderPayload(BaseModel):
    side: str  # buy/sell
    type: str  # market/limit/stop
    qty: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@app.post("/api/practice/order")
def practice_order(session_id: str, payload: OrderPayload):
    # 校验
    if payload.side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="side must be buy/sell")
    if payload.type not in ("market", "limit", "stop"):
        raise HTTPException(status_code=400, detail="type must be market/limit/stop")
    if payload.qty <= 0:
        raise HTTPException(status_code=400, detail="qty must be positive")
    if payload.type == "limit" and (payload.price is None):
        raise HTTPException(status_code=400, detail="limit order requires price")
    if payload.type == "stop" and (payload.stop_price is None):
        raise HTTPException(status_code=400, detail="stop order requires stop_price")

    order = PracticeOrder(
        id=str(uuid.uuid4()),
        side=payload.side,
        type=payload.type,
        qty=payload.qty,
        price=payload.price,
        stop_price=payload.stop_price,
        take_profit=payload.take_profit,
        stop_loss=payload.stop_loss,
    )
    try:
        practice_engine.submit_order(session_id, order)
        state = practice_engine.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _serialize_state(state)


@app.post("/api/practice/cancel")
def practice_cancel(session_id: str, order_id: str):
    try:
        practice_engine.cancel_order(session_id, order_id)
        state = practice_engine.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session or order not found")
    return _serialize_state(state)

# ----------------------------- 静态文件挂载 -----------------------------
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
