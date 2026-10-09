"""기술적 지표. 모두 '과거 데이터만' 사용합니다 (미래 값을 엿보지 않음)."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """단순 이동평균 (MA)."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """지수 이동평균 (EMA). 첫 period 개는 값이 안정되지 않아 NaN 처리합니다."""
    out = series.ewm(span=period, adjust=False).mean()
    out.iloc[: period - 1] = float("nan")
    return out


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder 방식)."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0)  # 하락이 전혀 없으면 100
    out.iloc[:period] = float("nan")
    return out
