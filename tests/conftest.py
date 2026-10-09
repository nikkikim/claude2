import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trader.config import Config  # noqa: E402


def candles(rows, start="2024-01-01", freq="1D") -> pd.DataFrame:
    """rows: [(open, high, low, close), ...] → 캔들 DataFrame"""
    ts = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [float(r[0]) for r in rows], "high": [float(r[1]) for r in rows],
        "low": [float(r[2]) for r in rows], "close": [float(r[3]) for r in rows],
        "volume": 1.0,
    })


@pytest.fixture
def plain_cfg() -> Config:
    """수수료/슬리피지/리스크 규칙을 모두 끈 설정 (손으로 계산한 값과 비교하기 쉽게)."""
    cfg = Config(capital=1_000_000)
    cfg.costs.fee_pct = 0
    cfg.costs.slippage_pct = 0
    r = cfg.risk
    r.max_position_pct = 100
    r.risk_per_trade_pct = 0
    r.stop_loss_pct = 0
    r.take_profit_pct = 0
    r.trailing_stop_pct = 0
    r.max_daily_loss_pct = 0
    r.max_drawdown_pct = 0
    r.max_trades_per_day = 0
    r.cooldown_candles = 0
    r.min_order_value = 1
    return cfg
