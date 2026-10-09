"""백테스트와 실제 봇(step)이 같은 데이터에서 같은 매매를 하는지 비교한다.
봇에게는 캔들을 하나씩 '진행 중' 상태로 보여주고(시가만 공개), 가격 틱을 open → (목표가) → high → low → close 순서로 흘려보낸다."""
import csv

import pandas as pd
import pytest

from test_bot import Clock, FakeMarket
from trader.backtest import run_backtest
from trader.bot import TradingBot
from trader.broker import PaperBroker
from trader.config import Config
from trader.data import make_demo_data
from trader.strategies import STRATEGIES, build_strategy

COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def replay(df, strat, cfg, tmp_path):
    prepared = strat.prepare(df)
    market = FakeMarket(df.iloc[:1])
    clock = Clock(0.0)
    bot = TradingBot(cfg, PaperBroker(market, cfg.costs.fee_pct, cfg.costs.slippage_pct), strat, "paper",
                     root=tmp_path, clock=clock)
    n = len(df)
    for i in range(strat.warmup, n + 1):
        if i < n:
            row = df.iloc[i]
            partial = pd.DataFrame([(row.timestamp, row.open, row.open, row.open, row.open, row.volume)], columns=COLS)
            ticks = [row.open]
            plan = strat.plan(prepared, i, False)
            if plan.trigger is not None and plan.trigger <= row.high:
                ticks.append(max(plan.trigger, row.open))
            ticks += [row.high, "LOW", row.close]
            ts = row.timestamp
        else:                                   # 마지막 캔들 다음 캔들 시작 (대기 중이던 매도 실행용)
            ts = df.timestamp.iat[-1] + pd.Timedelta(days=1)
            partial = pd.DataFrame([(ts, df.close.iat[-1],) + (df.close.iat[-1],) * 3 + (1.0,)], columns=COLS)
            ticks = [df.close.iat[-1]]
        market.df = pd.concat([df.iloc[:i], partial], ignore_index=True)
        for k, price in enumerate(ticks):
            if price == "LOW":
                # 가격이 고가에서 저가로 '순간이동'하지 않고 손절선을 지나가도록 손절선 가격 틱을 먼저 흘려보냄
                pos = bot.state.position
                stop = pos.entry_price * (1 - cfg.risk.stop_loss_pct / 100) if pos else None
                if stop is not None and row.low <= stop:
                    market.price = float(stop)
                    clock.t = ts.timestamp() + 10 + 5 * k - 1
                    bot.step()
                price = row.low
            market.price = float(price)
            clock.t = ts.timestamp() + 10 + 5 * k
            bot.step()
            if bot.finished:
                break
    rows = list(csv.DictReader((tmp_path / "logs" / "trades_paper.csv").open(encoding="utf-8")))
    return [r for r in rows if r["side"] == "SELL"], bot


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_bot_matches_backtest(name, tmp_path):
    cfg = Config(capital=1_000_000)
    cfg.risk.max_drawdown_pct = 0       # 봇은 틱마다, 백테스트는 시가마다 낙폭을 보기 때문에 이 규칙만 제외
    cfg.risk.trailing_stop_pct = 0
    df = make_demo_data(420, "1d", seed=5)
    strat = build_strategy(name, None, 5)

    bt = [t for t in run_backtest(df, strat, cfg).trades if t.reason != "end_of_data"]
    sells, bot = replay(df, strat, cfg, tmp_path)

    assert len(bt) >= 3, "비교할 거래가 너무 적음 (테스트 데이터 조정 필요)"
    assert [s["reason"] for s in sells] == [t.reason for t in bt]
    for s, t in zip(sells, bt):
        assert float(s["price"]) == pytest.approx(t.exit_price, rel=1e-9)
        assert float(s["pnl"]) == pytest.approx(t.pnl, rel=1e-6, abs=0.02)
