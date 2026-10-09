import pytest

from conftest import candles
from trader.backtest import run_backtest
from trader.data import make_demo_data
from trader.strategies import STRATEGIES, build_strategy

# 변동성 돌파(k=0.5) 손계산용 데이터: 2번째 캔들 목표가 = 110 + 0.5 × (120-100) = 120
ROWS = [(100, 110, 90, 100), (100, 120, 100, 110), (110, 130, 105, 125), (125, 126, 120, 121)]


def vb():
    return build_strategy("volatility_breakout", {"k": 0.5})


def test_breakout_trade_matches_hand_calculation(plain_cfg):
    r = run_backtest(candles(ROWS), vb(), plain_cfg)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert (t.entry_price, t.exit_price, t.reason) == (120, 125, "candle_end")
    assert r.metrics["final_equity"] == pytest.approx(1_000_000 * 125 / 120)


def test_costs_are_applied(plain_cfg):
    plain_cfg.costs.fee_pct = 0.1
    plain_cfg.costs.slippage_pct = 0.1
    r = run_backtest(candles(ROWS), vb(), plain_cfg)
    t = r.trades[0]
    assert t.entry_price == pytest.approx(120 * 1.001) and t.exit_price == pytest.approx(125 * 0.999)
    value = 1_000_000 / 1.001                                 # 수수료까지 합쳐 현금 전부
    qty = value / t.entry_price
    expected = 1_000_000 - value * 1.001 + qty * t.exit_price * (1 - 0.001)
    assert r.metrics["final_equity"] == pytest.approx(expected)


def test_stop_loss_inside_entry_candle(plain_cfg):
    plain_cfg.risk.stop_loss_pct = 5
    rows = list(ROWS)
    rows[2] = (110, 130, 100, 125)            # 저가 100 < 손절가 114
    r = run_backtest(candles(rows), vb(), plain_cfg)
    t = r.trades[0]
    assert t.reason == "stop_loss" and t.exit_price == pytest.approx(114)
    assert r.metrics["final_equity"] == pytest.approx(1_000_000 * 0.95)


def test_no_entry_when_target_not_reached(plain_cfg):
    rows = list(ROWS)
    rows[2] = (110, 119, 105, 118)            # 목표가 120 에 못 미침
    assert run_backtest(candles(rows), vb(), plain_cfg).trades == []


def test_gap_above_target_fills_at_open(plain_cfg):
    rows = list(ROWS)
    rows[2] = (125, 130, 123, 128)            # 시가가 이미 목표가(145) 가 아니라 → 목표가 125+10=135 도달 못함
    assert run_backtest(candles(rows), vb(), plain_cfg).trades == []
    rows[2] = (118, 140, 117, 130)            # 목표가 118+10=128, 시가보다 위 → 128 체결
    t = run_backtest(candles(rows), vb(), plain_cfg).trades[0]
    assert t.entry_price == pytest.approx(128)


def test_risk_sizing_limits_loss_to_budget(plain_cfg):
    plain_cfg.risk.stop_loss_pct = 5
    plain_cfg.risk.risk_per_trade_pct = 1
    rows = list(ROWS)
    rows[2] = (110, 130, 100, 125)
    r = run_backtest(candles(rows), vb(), plain_cfg)
    assert r.metrics["final_equity"] == pytest.approx(1_000_000 - 10_000)  # 손절 시 정확히 자산의 1% 손실


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_accounting_identity_and_sanity(name, plain_cfg):
    cfg = plain_cfg
    cfg.costs.fee_pct, cfg.costs.slippage_pct = 0.05, 0.05
    cfg.risk.stop_loss_pct, cfg.risk.risk_per_trade_pct, cfg.risk.max_position_pct = 3, 1, 50
    cfg.risk.min_order_value = 5000
    df = make_demo_data(500, "1d", seed=11)
    strat = build_strategy(name, None, 5)
    r = run_backtest(df, strat, cfg)
    assert r.metrics["final_equity"] == pytest.approx(cfg.capital + sum(t.pnl for t in r.trades))
    assert (r.equity > 0).all()
    for t in r.trades:
        assert t.exit_time >= t.entry_time and t.qty > 0
    # 결정론적: 같은 입력이면 같은 결과
    assert run_backtest(df, strat, cfg).metrics == r.metrics


def test_max_drawdown_halt_closes_and_stops_trading(plain_cfg):
    plain_cfg.risk.max_drawdown_pct = 10
    plain_cfg.risk.stop_loss_pct = 0
    # 계속 하락 → MA 교차 매수 후 큰 손실
    closes = [100] * 10 + [110, 120, 130] + [130 - 8 * k for k in range(1, 12)] + [40] * 10
    rows = [(c, c, c, c) for c in closes]
    strat = build_strategy("ma_cross", {"fast": 2, "slow": 5})
    r = run_backtest(candles(rows), strat, plain_cfg)
    assert r.halted
    assert r.trades and r.trades[-1].reason in ("max_drawdown_halt", "signal")


def test_too_little_data_raises(plain_cfg):
    with pytest.raises(ValueError):
        run_backtest(candles(ROWS[:2]), vb(), plain_cfg)
