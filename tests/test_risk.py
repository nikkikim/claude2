from datetime import datetime, timezone

import pytest

from trader.config import RiskConfig
from trader.risk import Position, RiskManager

DAY = 86400


def ts(day, hour=0):
    return datetime(2024, 1, day, hour, tzinfo=timezone.utc).timestamp()


def pos(entry=100.0, highest=None):
    return Position(entry_price=entry, qty=1, entry_ts=0, entry_value=entry, entry_fee=0, highest=highest or entry)


def rm(**kw):
    return RiskManager(RiskConfig(**kw), "UTC", DAY)


def test_order_value_uses_risk_budget():
    r = rm(max_position_pct=50, risk_per_trade_pct=1, stop_loss_pct=3, min_order_value=1)
    assert abs(r.order_value(1_000_000, 1_000_000, 0) - 1_000_000 / 3) < 1e-6   # 1% / 3% = 33%
    r = rm(max_position_pct=20, risk_per_trade_pct=1, stop_loss_pct=3, min_order_value=1)
    assert r.order_value(1_000_000, 1_000_000, 0) == 200_000                      # 비율 한도가 더 작음


def test_order_value_limits_cash_fee_and_min_order():
    r = rm(max_position_pct=100, risk_per_trade_pct=0, stop_loss_pct=0, min_order_value=5000)
    value = r.order_value(1_000_000, 1_000_000, 0.0005)
    assert abs(value * 1.0005 - 1_000_000) < 1e-6          # 수수료까지 합쳐 현금을 넘지 않음
    assert r.order_value(1_000_000, 4_000, 0) == 0         # 최소 주문 금액 미만이면 0


def test_stop_loss_take_profit_and_gap():
    r = rm(stop_loss_pct=5, take_profit_pct=10)
    assert r.check_exit(pos(), 100, 104, 96) is None
    assert r.check_exit(pos(), 100, 104, 94) == ("stop_loss", pytest.approx(95))
    assert r.check_exit(pos(), 90, 91, 89) == ("stop_loss", 90)          # 시가부터 갭 하락 → 시가 체결
    assert r.check_exit(pos(), 100, 112, 99) == ("take_profit", pytest.approx(110))
    assert r.check_exit(pos(), 115, 120, 114) == ("take_profit", 115)    # 갭 상승 → 시가 체결
    assert r.check_exit(pos(), 100, 112, 94)[0] == "stop_loss"           # 둘 다 가능하면 손절 우선


def test_trailing_stop():
    r = rm(trailing_stop_pct=10)
    p = pos(entry=100, highest=150)
    assert r.check_exit(p, 140, 141, 134) == ("trailing_stop", pytest.approx(135))   # 최고가 150 의 -10% = 135
    assert r.check_exit(p, 140, 141, 136) is None


def test_all_disabled_never_exits():
    assert rm(stop_loss_pct=0, take_profit_pct=0, trailing_stop_pct=0).check_exit(pos(), 1, 1000, 0.001) is None


def test_daily_loss_limit_and_reset():
    r = rm(max_daily_loss_pct=3)
    r.on_time(ts(1), 1_000_000)
    assert r.can_enter(ts(1))[0]
    r.on_exit(ts(1), -30_000, "signal")
    assert not r.can_enter(ts(1, 5))[0]
    r.on_time(ts(2), 970_000)               # 다음 날 → 초기화
    assert r.can_enter(ts(2))[0]


def test_max_trades_per_day():
    r = rm(max_trades_per_day=2)
    r.on_time(ts(1), 1000)
    r.on_entry(); r.on_entry()
    assert not r.can_enter(ts(1, 3))[0]
    r.on_time(ts(2), 1000)
    assert r.can_enter(ts(2))[0]


def test_cooldown_after_stop_loss_blocks_n_candles():
    r = rm(cooldown_candles=1)
    r.on_exit(ts(1), -10, "stop_loss")
    assert not r.can_enter(ts(2))[0]        # 바로 다음 캔들은 금지
    assert r.can_enter(ts(3))[0]            # 그다음부터 허용
    r2 = rm(cooldown_candles=1)
    r2.on_exit(ts(1), 10, "signal")
    assert r2.can_enter(ts(2))[0]           # 손절이 아니면 쿨다운 없음


def test_max_drawdown_halts():
    r = rm(max_drawdown_pct=20)
    r.on_time(ts(1), 1_000_000)
    r.on_time(ts(2), 1_200_000)
    r.on_time(ts(3), 1_000_000)             # 고점 대비 -16.7%
    assert not r.state.halted
    r.on_time(ts(4), 950_000)               # -20.8%
    assert r.state.halted and not r.can_enter(ts(4))[0]
