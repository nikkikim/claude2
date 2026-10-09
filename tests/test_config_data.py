from pathlib import Path

import pandas as pd
import pytest

from trader.config import ConfigError, load_config
from trader.data import fetch_ohlcv_range, load_csv, make_demo_data

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_shipped_config_is_valid():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.strategy == "volatility_breakout" and cfg.timeframe_seconds == 86400
    assert cfg.quote_currency == "KRW" and cfg.risk.stop_loss_pct == 3


def test_decimals_are_kept(tmp_path):
    cfg = load_config(write(tmp_path, "risk:\n  max_position_pct: 0.5\n  max_trades_per_day: 3\ncosts:\n  fee_pct: 0.04\n"))
    assert cfg.risk.max_position_pct == 0.5 and cfg.risk.max_trades_per_day == 3 and cfg.costs.fee_pct == 0.04


@pytest.mark.parametrize("text", [
    "risk:\n  stop_los_pct: 3\n",          # 오타
    "strategy: magic\n",
    "timeframe: 7m\n",
    "symbol: BTCKRW\n",
    "risk:\n  stop_loss_pct: 300\n",
    "risk:\n  stop_loss_pct: abc\n",
    "capital: 0\n",
    "colour: red\n",
    "timezone: Mars/Base\n",
    "risk:\n  max_position_pct: 0\n",
])
def test_bad_config_gives_friendly_error(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_load_csv_roundtrip(tmp_path):
    df = make_demo_data(30, "1d")
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    back = load_csv(p)
    assert len(back) == len(df) and back["close"].iloc[-1] == pytest.approx(df["close"].iloc[-1])
    ms = df.assign(timestamp=df["timestamp"].map(lambda t: int(t.timestamp() * 1000)))  # 밀리초 형식
    ms_path = tmp_path / "ms.csv"
    pd.DataFrame(ms).to_csv(ms_path, index=False)
    assert load_csv(ms_path)["timestamp"].iloc[0].year >= 2024


def test_load_csv_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("timestamp,open,close\n2024-01-01,1,2\n")
    with pytest.raises(ValueError):
        load_csv(p)


class FakeEx:
    """since 부터 limit 개를 돌려주는 가짜 거래소 (페이지 나눠 받기 검증용)"""
    def __init__(self, n, tf_ms=86_400_000, t0=1_700_000_000_000):
        self.t = [t0 + k * tf_ms for k in range(n)]
        self.calls = 0

    def fetch_ohlcv(self, symbol, tf, since=None, limit=200):
        self.calls += 1
        rows = [[t, 1, 2, 0.5, 1.5, 10] for t in self.t if t >= since][:limit]
        return rows


def test_fetch_ohlcv_range_paginates_without_duplicates():
    ex = FakeEx(450)
    df = fetch_ohlcv_range(ex, "BTC/KRW", "1d", ex.t[0], until_ms=ex.t[-1] + 86_400_000)
    assert len(df) == 450 and df["timestamp"].is_unique and ex.calls <= 5
