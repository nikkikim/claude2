"""캔들(시가/고가/저가/종가) 데이터: 거래소에서 받기, CSV 읽기, 연습용 가짜 데이터 만들기."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TIMEFRAME_SECONDS, ConfigError

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(exchange_id: str, api_key: str | None = None, secret: str | None = None):
    import ccxt

    if exchange_id not in ccxt.exchanges:
        raise ConfigError(f"지원하지 않는 거래소입니다: {exchange_id}")
    params = {"enableRateLimit": True}
    if api_key and secret:
        params.update({"apiKey": api_key, "secret": secret})
    return getattr(ccxt, exchange_id)(params)


def rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df.astype({c: "float64" for c in COLUMNS[1:]})


def fetch_ohlcv_range(ex, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None,
                      per_call: int = 200) -> pd.DataFrame:
    """since_ms 부터 지금(또는 until_ms)까지의 캔들을 여러 번 나눠서 받아옵니다."""
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    until_ms = until_ms or int(time.time() * 1000)
    rows: dict[int, list[float]] = {}
    cursor = since_ms
    for _ in range(10_000):  # 무한 루프 방지
        batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=per_call)
        new = [r for r in batch if r[0] not in rows]
        if not new:
            break
        for r in new:
            rows[r[0]] = r[:6]
        last = max(r[0] for r in new)
        if last + tf_ms > until_ms:
            break
        cursor = last + tf_ms
    return rows_to_frame([rows[k] for k in sorted(rows)]) if rows else pd.DataFrame(columns=COLUMNS)


def fetch_history(exchange_id: str, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """백테스트용 과거 데이터 (API 키 불필요)."""
    ex = make_exchange(exchange_id)
    since = int((time.time() - days * 86400) * 1000)
    df = fetch_ohlcv_range(ex, symbol, timeframe, since)
    if df.empty:
        raise RuntimeError("거래소에서 캔들 데이터를 받지 못했습니다. 심볼/타임프레임/인터넷 연결을 확인하세요.")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSV 파일 읽기. 열 이름: timestamp,open,high,low,close,volume
    timestamp 는 날짜문자열(2024-01-01 09:00:00) 또는 밀리초 숫자 모두 가능합니다."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = set(COLUMNS[:5]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV 에 필요한 열이 없습니다: {', '.join(sorted(missing))} (필요: {', '.join(COLUMNS)})")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    df = df[COLUMNS].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df.astype({c: "float64" for c in COLUMNS[1:]})


def make_demo_data(days: int, timeframe: str, seed: int = 7, start_price: float = 50_000_000) -> pd.DataFrame:
    """연습용 가짜 시세 (상승장/하락장/횡보장이 섞인 랜덤 데이터). 실제 시세가 아닙니다!"""
    rng = np.random.default_rng(seed)
    tf = TIMEFRAME_SECONDS[timeframe]
    n = max(int(days * 86400 / tf), 50)
    per_day = 86400 / tf
    vol = 0.035 / np.sqrt(per_day)              # 하루 변동성 약 3.5%
    regime_len = max(int(30 * per_day), 20)     # 약 30일마다 장세 변경
    drift = np.empty(n)
    for start in range(0, n, regime_len):
        drift[start:start + regime_len] = rng.choice([-1.0, 0.0, 1.0], p=[0.35, 0.35, 0.3]) * 0.25 * vol
    rets = drift + rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol * 0.6, n))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    end = pd.Timestamp("2025-01-01", tz="UTC")
    ts = pd.date_range(end=end, periods=n, freq=pd.Timedelta(seconds=tf))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(10, 100, n)})
