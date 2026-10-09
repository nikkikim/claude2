"""리스크 관리. 백테스트와 실제 봇이 '같은 코드'를 사용합니다.

* 매수 금액 결정 (자산 비율 한도 + 손절 시 최대 손실 한도)
* 손절 / 익절 / 트레일링 스탑
* 일일 손실 한도, 하루 최대 매수 횟수, 손절 후 쿨다운
* 최대 낙폭(MDD) 도달 시 전량 매도 후 정지
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import RiskConfig


@dataclass
class Position:
    entry_price: float            # 실제 체결 평균가
    qty: float                    # 수량
    entry_ts: float               # 매수 시각 (epoch 초)
    entry_value: float            # 매수 대금 (수수료 제외)
    entry_fee: float              # 매수 수수료
    highest: float                # 매수 후 최고가 (트레일링 스탑용)
    exit_at_candle_end: bool = False
    entry_candle_ts: float = 0.0  # 매수한 캔들의 시작 시각


@dataclass
class RiskState:
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    trades_today: int = 0
    cooldown_until: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig, tz: str, timeframe_seconds: int, state: RiskState | None = None):
        self.cfg = cfg
        self.tz = ZoneInfo(tz)
        self.tf_seconds = timeframe_seconds
        self.state = state or RiskState()

    # ── 시간이 흐를 때마다 호출 ────────────────────────────
    def on_time(self, ts: float, equity: float) -> None:
        s = self.state
        day = datetime.fromtimestamp(ts, self.tz).strftime("%Y-%m-%d")
        if day != s.day:
            s.day, s.day_start_equity, s.day_pnl, s.trades_today = day, equity, 0.0, 0
        s.peak_equity = max(s.peak_equity, equity)
        limit = self.cfg.max_drawdown_pct
        if limit > 0 and s.peak_equity > 0 and not s.halted:
            drawdown = (s.peak_equity - equity) / s.peak_equity * 100
            if drawdown >= limit:
                s.halted = True
                s.halt_reason = f"최대 낙폭 한도 도달: 최고점 대비 -{drawdown:.1f}% (한도 {limit}%)"

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason

    # ── 매수 가능 여부 / 금액 ──────────────────────────────
    def can_enter(self, ts: float) -> tuple[bool, str]:
        s, c = self.state, self.cfg
        if s.halted:
            return False, s.halt_reason or "정지 상태"
        if c.max_daily_loss_pct > 0 and s.day_start_equity > 0:
            if s.day_pnl <= -s.day_start_equity * c.max_daily_loss_pct / 100:
                return False, f"일일 손실 한도 도달 (오늘 손익 {s.day_pnl:,.0f})"
        if c.max_trades_per_day > 0 and s.trades_today >= c.max_trades_per_day:
            return False, f"하루 최대 매수 횟수 {c.max_trades_per_day}회 도달"
        if ts < s.cooldown_until:
            return False, "손절 후 쿨다운 중"
        return True, ""

    def order_value(self, equity: float, cash: float, fee_rate: float) -> float:
        """이번에 매수할 금액(수수료 제외). 0 이면 매수하지 않음."""
        c = self.cfg
        value = equity * c.max_position_pct / 100
        if c.stop_loss_pct > 0 and c.risk_per_trade_pct > 0:
            # 손절가에서 팔았을 때 손실이 자산의 risk_per_trade_pct% 가 되도록 금액 제한
            value = min(value, equity * c.risk_per_trade_pct / c.stop_loss_pct)
        value = min(value, cash / (1 + fee_rate))  # 수수료 낼 돈은 남겨둠
        return value if value >= c.min_order_value else 0.0

    # ── 손절 / 익절 / 트레일링 ─────────────────────────────
    def check_exit(self, pos: Position, open_: float, high: float, low: float) -> tuple[str, float] | None:
        """이 캔들(또는 현재 가격)에서 청산해야 하는지. (사유, 기준가격) 을 돌려줍니다.
        손절과 익절이 한 캔들에 같이 가능하면 '손절이 먼저'라고 보수적으로 가정합니다."""
        c = self.cfg
        levels = []
        if c.stop_loss_pct > 0:
            levels.append((pos.entry_price * (1 - c.stop_loss_pct / 100), "stop_loss"))
        if c.trailing_stop_pct > 0:
            levels.append((pos.highest * (1 - c.trailing_stop_pct / 100), "trailing_stop"))
        if levels:
            level, reason = max(levels)
            if low <= level:
                return reason, min(open_, level)  # 시가부터 갭 하락하면 시가로 체결
        if c.take_profit_pct > 0:
            target = pos.entry_price * (1 + c.take_profit_pct / 100)
            if high >= target:
                return "take_profit", max(open_, target)
        return None

    # ── 거래 결과 기록 ─────────────────────────────────────
    def on_entry(self) -> None:
        self.state.trades_today += 1

    def on_exit(self, candle_ts: float, pnl: float, reason: str) -> None:
        """candle_ts: 청산이 일어난 캔들의 시작 시각. 손절한 캔들 이후 N개 캔들 동안 재매수를 막습니다."""
        self.state.day_pnl += pnl
        if reason == "stop_loss" and self.cfg.cooldown_candles > 0:
            self.state.cooldown_until = candle_ts + (self.cfg.cooldown_candles + 1) * self.tf_seconds
