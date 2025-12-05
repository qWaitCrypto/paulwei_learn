"""
练习模式引擎：管理会话、撮合订单、持仓与资金。

特性（当前版本）：
- 支持市价、限价、止损（stop-market）订单；
- 支持下单时附带止盈/止损价格，挂单触发后建仓或平仓；
- 手续费：统一费率（按成交额）；滑点：按百分比调整成交价；
- 持仓：净仓位模型（多/空各一条，允许加仓/反手，计算加权成本、已实现盈亏、浮盈亏）；
- 播放：基于预加载的 K 线序列，按索引推进（advance）时处理挂单与 TP/SL 触发；
- 状态查询：返回资金、持仓、挂单、历史成交、权益曲线、当前已播放到的 K 线。

限制（后续可扩展）：
- 挂单有效期、不区分 maker/taker 费率；
- 不处理逐笔仓位分层，使用净仓模式；
- 止盈/止损为固定价格触发。
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------- 数据结构 -----------------------------
@dataclass
class Bar:
    t: int  # 秒级时间戳
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class Order:
    id: str
    side: str  # "buy" / "sell"
    type: str  # "market" / "limit" / "stop"
    qty: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    status: str = "open"  # open / filled / cancelled
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    filled_at: Optional[dt.datetime] = None
    fill_price: Optional[float] = None


@dataclass
class Fill:
    time: dt.datetime
    side: str
    qty: float
    price: float
    fee: float
    pnl: float


@dataclass
class Position:
    qty: float = 0.0  # 正数多仓，负数空仓
    avg_price: float = 0.0
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class PracticeConfig:
    interval: str
    initial_cash: float = 100000.0
    fee_rate: float = 0.0006  # 按成交额
    slippage_pct: float = 0.0005  # 0.05%
    duration_minutes: int = 24 * 60


@dataclass
class PracticeSessionState:
    session_id: str
    symbol: str
    config: PracticeConfig
    bars: List[Bar]
    current_index: int = 0
    cash: float = 0.0
    realized_pnl: float = 0.0
    fees: float = 0.0
    position: Position = field(default_factory=Position)
    open_orders: Dict[str, Order] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)  # (timestamp, equity)

    def __post_init__(self):
        self.cash = self.config.initial_cash
        # 初始化第一条 equity
        if self.bars:
            first_bar = self.bars[0]
            self.equity_curve.append((first_bar.t, self.cash))


# ----------------------------- 工具函数 -----------------------------
def _slip_price(price: float, side: str, slippage_pct: float) -> float:
    if side == "buy":
        return price * (1 + slippage_pct)
    return price * (1 - slippage_pct)


def _order_can_fill_on_bar(order: Order, bar: Bar) -> bool:
    if order.type == "market":
        return True
    if order.type == "limit":
        if order.side == "buy":
            return bar.l <= order.price <= bar.h
        else:
            return bar.l <= order.price <= bar.h
    if order.type == "stop":
        if order.side == "buy":
            return bar.h >= order.stop_price
        else:
            return bar.l <= order.stop_price
    return False


def _fill_price(order: Order, bar: Bar, slippage_pct: float) -> float:
    base_price: float
    if order.type == "market":
        base_price = bar.c
    elif order.type == "limit":
        base_price = order.price
    elif order.type == "stop":
        base_price = order.stop_price
    else:
        base_price = bar.c
    return _slip_price(base_price, order.side, slippage_pct)


# ----------------------------- 核心引擎 -----------------------------
class PracticeEngine:
    def __init__(self):
        self.sessions: Dict[str, PracticeSessionState] = {}

    def create_session(
        self,
        symbol: str,
        bars: List[Bar],
        config: PracticeConfig,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        start_index: Optional[int] = None,
    ) -> PracticeSessionState:
        session_id = str(uuid.uuid4())
        state = PracticeSessionState(
            session_id=session_id,
            symbol=symbol,
            config=config,
            bars=bars,
            current_index=0,
        )
        # 如果指定了起始索引，则从该索引开始播放（用于随机时间点练习）
        if start_index is not None and bars:
          idx = max(0, min(start_index, len(bars) - 1))
          state.current_index = idx
        self.sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> PracticeSessionState:
        if session_id not in self.sessions:
            raise KeyError("session not found")
        return self.sessions[session_id]

    def submit_order(self, session_id: str, order: Order) -> Order:
        state = self.get_session(session_id)
        # 尝试立即撮合（使用当前 bar）
        if state.current_index < len(state.bars):
            current_bar = state.bars[state.current_index]
            self._try_fill_order(state, order, current_bar, immediate=True)
        if order.status == "open":
            state.open_orders[order.id] = order
        return order

    def cancel_order(self, session_id: str, order_id: str) -> None:
        state = self.get_session(session_id)
        if order_id in state.open_orders:
            state.open_orders[order_id].status = "cancelled"
            del state.open_orders[order_id]

    def advance(self, session_id: str, steps: int = 1) -> PracticeSessionState:
        state = self.get_session(session_id)
        steps = max(1, steps)
        for _ in range(steps):
            if state.current_index >= len(state.bars) - 1:
                break
            state.current_index += 1
            bar = state.bars[state.current_index]
            # 处理挂单触发
            self._process_open_orders(state, bar)
            # 处理持仓的止盈/止损
            self._process_position_tp_sl(state, bar)
            # 更新权益曲线
            self._update_equity(state, bar)
        return state

    def _process_open_orders(self, state: PracticeSessionState, bar: Bar):
        to_remove = []
        for oid, order in state.open_orders.items():
            if order.status != "open":
                to_remove.append(oid)
                continue
            if _order_can_fill_on_bar(order, bar):
                self._fill_order(state, order, bar)
                to_remove.append(oid)
        for oid in to_remove:
            if oid in state.open_orders:
                del state.open_orders[oid]

    def _try_fill_order(self, state: PracticeSessionState, order: Order, bar: Bar, immediate: bool = False):
        if order.type == "market":
            self._fill_order(state, order, bar)
        elif order.type in ("limit", "stop"):
            if immediate and _order_can_fill_on_bar(order, bar):
                self._fill_order(state, order, bar)
            # 否则挂单等待

    def _fill_order(self, state: PracticeSessionState, order: Order, bar: Bar):
        fill_price = _fill_price(order, bar, state.config.slippage_pct)
        fee = abs(order.qty * fill_price) * state.config.fee_rate
        pnl = 0.0

        # 更新持仓
        pos = state.position
        new_qty = pos.qty + (order.qty if order.side == "buy" else -order.qty)
        if pos.qty == 0:
            # 开仓
            pos.qty = new_qty
            pos.avg_price = fill_price
            pos.take_profit = order.take_profit
            pos.stop_loss = order.stop_loss
        else:
            # 同向加仓或反向减仓
            if pos.qty * new_qty > 0:
                # 同向，加权平均
                total_cost = pos.avg_price * abs(pos.qty) + fill_price * abs(order.qty if order.side == "buy" else -order.qty)
                pos.qty = new_qty
                pos.avg_price = total_cost / abs(pos.qty)
                # 若有新 TP/SL，覆盖
                if order.take_profit is not None:
                    pos.take_profit = order.take_profit
                if order.stop_loss is not None:
                    pos.stop_loss = order.stop_loss
            else:
                # 反向，部分或全部平仓
                close_qty = min(abs(pos.qty), abs(order.qty))
                direction = 1 if pos.qty > 0 else -1
                trade_pnl = (fill_price - pos.avg_price) * close_qty * direction
                pnl += trade_pnl
                pos.qty = new_qty
                if pos.qty == 0:
                    pos.avg_price = 0.0
                    pos.take_profit = None
                    pos.stop_loss = None
                else:
                    # 还有剩余，剩余仓位的均价就是原均价（假设反手超量，新开仓按 fill_price）
                    if pos.qty * direction < 0:
                        # 反手超量，新仓
                        pos.avg_price = fill_price
                        pos.take_profit = order.take_profit
                        pos.stop_loss = order.stop_loss

        state.cash -= fee
        state.fees += fee
        state.realized_pnl += pnl

        order.status = "filled"
        order.filled_at = dt.datetime.now(dt.timezone.utc)
        order.fill_price = fill_price
        state.fills.append(
            Fill(
                time=order.filled_at,
                side=order.side,
                qty=order.qty,
                price=fill_price,
                fee=fee,
                pnl=pnl,
            )
        )

    def _process_position_tp_sl(self, state: PracticeSessionState, bar: Bar):
        pos = state.position
        if pos.qty == 0:
            return
        tp = pos.take_profit
        sl = pos.stop_loss
        direction = 1 if pos.qty > 0 else -1
        trigger_price = None
        trigger_type = None
        if tp is not None:
            if direction > 0 and bar.h >= tp:
                trigger_price = tp
                trigger_type = "tp"
            elif direction < 0 and bar.l <= tp:
                trigger_price = tp
                trigger_type = "tp"
        if trigger_price is None and sl is not None:
            if direction > 0 and bar.l <= sl:
                trigger_price = sl
                trigger_type = "sl"
            elif direction < 0 and bar.h >= sl:
                trigger_price = sl
                trigger_type = "sl"
        if trigger_price is not None:
            qty_to_close = abs(pos.qty)
            side = "sell" if direction > 0 else "buy"
            order = Order(
                id=str(uuid.uuid4()),
                side=side,
                type="market",
                qty=qty_to_close,
            )
            # 用触发价作为基准成交价
            fill_price = _slip_price(trigger_price, side, state.config.slippage_pct)
            fee = abs(order.qty * fill_price) * state.config.fee_rate
            pnl = (fill_price - pos.avg_price) * qty_to_close * direction
            state.cash -= fee
            state.fees += fee
            state.realized_pnl += pnl
            pos.qty = 0.0
            pos.avg_price = 0.0
            pos.take_profit = None
            pos.stop_loss = None
            state.fills.append(
                Fill(
                    time=dt.datetime.now(dt.timezone.utc),
                    side=side,
                    qty=qty_to_close,
                    price=fill_price,
                    fee=fee,
                    pnl=pnl,
                )
            )

    def _update_equity(self, state: PracticeSessionState, bar: Bar):
        pos = state.position
        unrealized = 0.0
        if pos.qty != 0:
            direction = 1 if pos.qty > 0 else -1
            unrealized = (bar.c - pos.avg_price) * abs(pos.qty) * direction
        equity = state.cash + state.realized_pnl + unrealized
        state.equity_curve.append((bar.t, equity))


# 单例引擎
engine = PracticeEngine()
