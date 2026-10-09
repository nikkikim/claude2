import numpy as np
import pytest

from conftest import candles
from trader.data import make_demo_data
from trader.indicators import ema, rsi, sma
from trader.strategies import STRATEGIES, build_strategy


def test_sma_ema_rsi_basic():
    s = candles([(1, 1, 1, v) for v in [1, 2, 3, 4, 5]])["close"]
    assert sma(s, 3).iloc[2] == 2 and np.isnan(sma(s, 3).iloc[1])
    assert np.isnan(ema(s, 3).iloc[1]) and ema(s, 3).iloc[4] > ema(s, 3).iloc[3]
    up = candles([(1, 1, 1, v) for v in range(1, 40)])["close"]
    assert rsi(up, 14).iloc[-1] == 100
    down = candles([(1, 1, 1, v) for v in range(40, 1, -1)])["close"]
    assert rsi(down, 14).iloc[-1] < 1


@pytest.mark.parametrize("name", list(STRATEGIES))
@pytest.mark.parametrize("trend", [0, 5])
def test_no_lookahead(name, trend):
    """i 번째 판단은 i-1 까지의 캔들과 i 의 시가만 써야 한다. 미래/현재 캔들의 나머지 값을 바꿔도 결과가 같아야 함."""
    df = make_demo_data(150, "1d", seed=3)
    strat = build_strategy(name, None, trend)
    base = strat.prepare(df)
    rng = np.random.default_rng(0)
    checked = 0
    for i in range(strat.warmup, 140, 7):
        for holding in (False, True):
            expected = strat.plan(base, i, holding)
            tampered = df.copy()
            tampered.loc[i, ["high", "low", "close", "volume"]] = rng.uniform(1, 1e9, 4)
            tampered.loc[i + 1:, ["open", "high", "low", "close"]] = rng.uniform(1, 1e9, (len(df) - i - 1, 4))
            assert strat.plan(strat.prepare(tampered), i, holding) == expected
            checked += 1
    assert checked > 10


def test_ma_cross_golden_and_dead():
    closes = [10] * 25 + [20, 30, 40, 50] + [5, 5, 5, 5, 5, 5]
    df = candles([(c, c, c, c) for c in closes])
    strat = build_strategy("ma_cross", {"fast": 2, "slow": 5})
    prepared = strat.prepare(df)
    enters = [i for i in range(strat.warmup, len(df)) if strat.plan(prepared, i, False).enter]
    assert enters == [25 + 1]  # 25번째 캔들(종가 20)에서 골든크로스 → 다음 캔들(26)에서 매수 신호
    assert strat.plan(prepared, 28, True).sell_now is False       # 상승 중에는 매도 신호 없음
    assert any(strat.plan(prepared, i, True).sell_now for i in range(29, 35))  # 급락 후 데드크로스


def test_volatility_breakout_target():
    df = candles([(100, 110, 90, 100), (100, 120, 100, 110), (110, 130, 105, 125), (125, 126, 120, 121)])
    strat = build_strategy("volatility_breakout", {"k": 0.5})
    plan = strat.plan(strat.prepare(df), 2, False)
    assert plan.enter and plan.trigger == 110 + 0.5 * 20 and plan.exit_at_candle_end


def test_trend_filter_blocks_entry_below_ma():
    closes = list(range(100, 60, -1)) + [61]  # 하락 추세
    df = candles([(c, c + 1, c - 1, c) for c in closes])
    plain = build_strategy("volatility_breakout", {"k": 0.5}, 0)
    filtered = build_strategy("volatility_breakout", {"k": 0.5}, 5)
    i = len(df) - 1
    assert plain.plan(plain.prepare(df), i, False).enter
    assert not filtered.plan(filtered.prepare(df), i, False).enter


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        build_strategy("ma_cross", {"fast": 30, "slow": 10})
    with pytest.raises(ValueError):
        build_strategy("ma_cross", {"fastt": 5})
    with pytest.raises(ValueError):
        build_strategy("volatility_breakout", {"k": 0})
    with pytest.raises(ValueError):
        build_strategy("nope")
