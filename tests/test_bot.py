import csv
from datetime import datetime, timezone

import pandas as pd
import pytest

from trader.bot import TradingBot
from trader.broker import OrderRejected, PaperBroker, PositionMissing
from trader.config import Config
from trader.strategies import Plan, build_strategy

DAY = 86400
T0 = datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp()   # 현재 캔들 시작 시각


def make_df(n_closed=9, cur_ts=T0, cur_open=100.0):
    """직전 캔들들은 변동폭 20 (고가 110, 저가 90) → 목표가 = 시가 + 0.5×20 = 110"""
    rows = []
    for k in range(n_closed):
        ts = cur_ts - (n_closed - k) * DAY
        rows.append((pd.Timestamp(ts, unit="s", tz="UTC"), 100, 110, 90, 100, 1.0))
    rows.append((pd.Timestamp(cur_ts, unit="s", tz="UTC"), cur_open, cur_open, cur_open, cur_open, 1.0))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


class FakeMarket:
    def __init__(self, df, price=100.0):
        self.df, self.price = df, price

    def get_price(self):
        return self.price

    def get_candles(self, n):
        return self.df.tail(n).reset_index(drop=True)

    def next_candle(self, open_price):
        ts = self.df["timestamp"].iat[-1] + pd.Timedelta(days=1)
        self.df = pd.concat([self.df, pd.DataFrame([(ts, open_price, open_price, open_price, open_price, 1.0)],
                                                   columns=self.df.columns)], ignore_index=True)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def cfg_for_test(**risk):
    cfg = Config(capital=1_000_000)
    cfg.costs.fee_pct, cfg.costs.slippage_pct = 0.1, 0.1
    r = cfg.risk
    r.max_position_pct, r.risk_per_trade_pct, r.stop_loss_pct = 100, 0, 3
    r.max_daily_loss_pct = r.max_drawdown_pct = r.max_trades_per_day = 0
    r.cooldown_candles = 0
    for k, v in risk.items():
        setattr(r, k, v)
    return cfg


def make_bot(tmp_path, cfg=None, market=None, broker=None, clock=None):
    cfg = cfg or cfg_for_test()
    market = market or FakeMarket(make_df())
    broker = broker or PaperBroker(market, cfg.costs.fee_pct, cfg.costs.slippage_pct)
    strat = build_strategy("volatility_breakout", {"k": 0.5})
    clock = clock or Clock(T0 + 3600)
    return TradingBot(cfg, broker, strat, "paper", root=tmp_path, clock=clock), market, clock


def test_no_buy_below_target_then_buy_on_cross(tmp_path):
    bot, market, _ = make_bot(tmp_path)
    market.price = 105
    bot.step()
    assert bot.state.position is None
    market.price = 111
    bot.step()
    pos = bot.state.position
    assert pos is not None and pos.entry_price == pytest.approx(111 * 1.001)
    assert bot.state.cash == pytest.approx(1_000_000 - pos.entry_value - pos.entry_fee, abs=1e-6)
    assert pos.entry_value + pos.entry_fee == pytest.approx(1_000_000)         # 현금 거의 전부(100%) 사용
    rows = list(csv.reader((tmp_path / "logs" / "trades_paper.csv").open(encoding="utf-8")))
    assert rows[0][:2] == ["time", "side"] and rows[1][1] == "BUY" and len(rows) == 2


def test_stop_loss_then_no_rebuy_in_same_candle(tmp_path):
    bot, market, _ = make_bot(tmp_path)
    market.price = 111
    bot.step()
    market.price = 100                       # 매수가 대비 -10% → 손절
    bot.step()
    assert bot.state.position is None
    rows = list(csv.reader((tmp_path / "logs" / "trades_paper.csv").open(encoding="utf-8")))
    assert rows[-1][1] == "SELL" and rows[-1][7] == "stop_loss" and float(rows[-1][5]) < 0
    market.price = 130                       # 다시 목표가 위로 올라가도 같은 캔들에서는 재매수 금지
    for _ in range(3):
        bot.step()
    assert bot.state.position is None
    assert len(list(csv.reader((tmp_path / "logs" / "trades_paper.csv").open(encoding="utf-8")))) == 3


def test_exit_at_candle_end_when_new_candle_starts(tmp_path):
    bot, market, clock = make_bot(tmp_path)
    market.price = 111
    bot.step()
    assert bot.state.position is not None
    bot.step()                                # 같은 캔들에서는 계속 보유
    assert bot.state.position is not None
    market.next_candle(112)                   # 새 캔들 시작
    market.price = 112
    clock.t = T0 + DAY + 60
    bot.step()
    assert bot.state.position is None
    rows = list(csv.reader((tmp_path / "logs" / "trades_paper.csv").open(encoding="utf-8")))
    assert rows[-1][7] == "candle_end"
    assert float(rows[-1][5]) > 0             # 111 → 112 : 수수료/슬리피지 후에도 이익 (+0.9% - 0.4%)


def test_state_survives_restart(tmp_path):
    bot, market, clock = make_bot(tmp_path)
    market.price = 111
    bot.step()
    saved = bot.state.position
    bot2, _, _ = make_bot(tmp_path, market=market, clock=clock)
    assert bot2.state.position == saved and bot2.state.cash == pytest.approx(bot.state.cash)
    bot2.step()                               # 재시작 후에도 중복 매수하지 않음
    assert bot2.state.position == saved


def test_drawdown_limit_sells_and_stops(tmp_path):
    cfg = cfg_for_test(stop_loss_pct=0, max_drawdown_pct=5)
    bot, market, _ = make_bot(tmp_path, cfg=cfg)
    market.price = 111
    bot.step()
    market.price = 100                        # 약 -10% → 낙폭 한도(5%) 초과
    bot.step()
    assert bot.state.position is None and bot.risk.state.halted and bot.finished
    rows = list(csv.reader((tmp_path / "logs" / "trades_paper.csv").open(encoding="utf-8")))
    assert rows[-1][7] == "max_drawdown_halt"


def test_stop_file_prevents_entry_and_exits(tmp_path):
    bot, market, _ = make_bot(tmp_path)
    (tmp_path / "STOP").write_text("")
    market.price = 111
    bot.step()
    assert bot.state.position is None and bot.finished


def test_stale_candles_skip(tmp_path):
    bot, market, clock = make_bot(tmp_path)
    market.price = 111
    clock.t = T0 + 5 * DAY                    # 최신 캔들이 5일 전 → 거래소 장애 가정
    bot.step()
    assert bot.state.position is None


def test_entry_at_open_signal_expires(tmp_path):
    bot, _, _ = make_bot(tmp_path)
    plan = Plan(enter=True, trigger=None)
    assert bot._may_enter(plan, T0, 100, T0 + 20)             # 캔들 시작 직후: OK
    assert not bot._may_enter(plan, T0, 100, T0 + 12 * 3600)  # 12시간 뒤 봇을 켰다면 늦게 사지 않음


class ScriptedBroker:
    def __init__(self, market, buy_exc=None, sell_exc=None):
        self.market, self.buy_exc, self.sell_exc = market, buy_exc, sell_exc
        self.paper = PaperBroker(market, 0, 0)

    def get_price(self): return self.market.get_price()
    def get_candles(self, n): return self.market.get_candles(n)

    def buy(self, value):
        if self.buy_exc:
            raise self.buy_exc
        return self.paper.buy(value)

    def sell(self, qty):
        if self.sell_exc:
            raise self.sell_exc
        return self.paper.sell(qty)


def test_rejected_buy_is_safe_and_not_retried_every_poll(tmp_path):
    market = FakeMarket(make_df(), 111)
    bot, _, _ = make_bot(tmp_path, market=market, broker=ScriptedBroker(market, buy_exc=OrderRejected("잔고 부족")))
    bot.step(); bot.step()
    assert bot.state.position is None and not bot.risk.state.halted and not bot.finished
    assert bot.state.cash == 1_000_000


def test_unknown_buy_failure_halts_to_avoid_double_buy(tmp_path):
    market = FakeMarket(make_df(), 111)
    bot, _, _ = make_bot(tmp_path, market=market, broker=ScriptedBroker(market, buy_exc=TimeoutError("timeout")))
    bot.step()
    assert bot.risk.state.halted and bot.finished and bot.state.position is None


def test_missing_coin_on_sell_halts(tmp_path):
    market = FakeMarket(make_df(), 111)
    broker = ScriptedBroker(market)
    bot, _, _ = make_bot(tmp_path, market=market, broker=broker)
    bot.step()
    assert bot.state.position is not None
    broker.sell_exc = PositionMissing("없음")
    market.price = 100
    bot.step()
    assert bot.risk.state.halted and bot.state.position is None


def test_rejected_sell_is_retried_next_poll(tmp_path):
    market = FakeMarket(make_df(), 111)
    broker = ScriptedBroker(market)
    bot, _, _ = make_bot(tmp_path, market=market, broker=broker)
    bot.step()
    broker.sell_exc = OrderRejected("일시 오류")
    market.price = 100
    bot.step()
    assert bot.state.position is not None                     # 아직 보유 (매도 실패)
    broker.sell_exc = None
    bot.step()
    assert bot.state.position is None                         # 재시도 성공
