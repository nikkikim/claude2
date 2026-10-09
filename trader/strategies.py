"""매매 전략.

모든 전략은 같은 규칙을 따릅니다 (백테스트와 실제 봇이 똑같이 동작하도록):

  plan(df, i, holding)  →  "i 번째 캔들(= 지금 진행 중인 캔들)에서 무엇을 할지"

  * i-1 번째까지의 '완성된 캔들'과, i 번째 캔들의 '시가'만 사용합니다.
    (i 번째 캔들의 고가/저가/종가는 아직 모르는 값이므로 절대 쓰지 않습니다.)

새 전략을 추가하고 싶다면 맨 아래 STRATEGIES 에 클래스를 등록하면 됩니다.
(또는 Claude 에게 "이런 조건으로 매수/매도하는 전략을 추가해줘" 라고 요청하세요.)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import ema, rsi, sma


@dataclass
class Plan:
    enter: bool = False                 # 이번 캔들에서 매수할지
    trigger: float | None = None        # None: 시가에 바로 매수 / 숫자: 가격이 이 값 이상이 되면 매수
    sell_now: bool = False              # 보유 중이면 즉시 매도(전략의 매도 신호)
    exit_at_candle_end: bool = False    # 이 캔들이 끝날 때 무조건 매도 (변동성 돌파용)
    reason: str = ""


class Strategy:
    name = ""
    defaults: dict = {}

    def __init__(self, params: dict | None = None, trend_filter_ma: int = 0):
        params = dict(params or {})
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"전략 '{self.name}' 에 알 수 없는 설정이 있습니다: {', '.join(sorted(unknown))} "
                f"(사용 가능: {', '.join(self.defaults)})"
            )
        self.params = {**self.defaults, **params}
        self.trend_filter_ma = int(trend_filter_ma)
        self._validate()

    # ── 하위 클래스가 구현 ──────────────────────────────────
    def _validate(self) -> None:
        pass

    def _warmup(self) -> int:
        return 2

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _plan(self, df: pd.DataFrame, i: int, holding: bool) -> Plan:
        raise NotImplementedError

    # ── 공통 ────────────────────────────────────────────────
    @property
    def warmup(self) -> int:
        """지표가 안정되려면 필요한 최소 캔들 수."""
        return max(self._warmup(), self.trend_filter_ma + 2)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 계산 (전체 데이터에 한 번). 모든 지표는 과거 값만 사용합니다."""
        df = df.reset_index(drop=True).copy()
        if self.trend_filter_ma > 0:
            df["trend_ma"] = sma(df["close"], self.trend_filter_ma)
        return self._prepare(df)

    def plan(self, df: pd.DataFrame, i: int, holding: bool) -> Plan:
        if i < self.warmup:
            return Plan()
        p = self._plan(df, i, holding)
        if p.enter and self.trend_filter_ma > 0:
            trend, last_close = df["trend_ma"].iat[i - 1], df["close"].iat[i - 1]
            if not (last_close > trend):  # NaN 이어도 False
                return Plan(sell_now=p.sell_now, reason="추세 필터: 이동평균 아래라 매수 안 함")
        return p


class _CrossBase(Strategy):
    """이동평균 교차 (골든크로스 매수 / 데드크로스 매도)."""
    ma_func = staticmethod(sma)
    defaults = {"fast": 5, "slow": 20}

    def _validate(self) -> None:
        fast, slow = int(self.params["fast"]), int(self.params["slow"])
        if fast < 1 or fast >= slow:
            raise ValueError(f"'{self.name}' 은 fast < slow 여야 합니다. (현재 fast={fast}, slow={slow})")

    def _warmup(self) -> int:
        return int(self.params["slow"]) + 2

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["fast"] = self.ma_func(df["close"], int(self.params["fast"]))
        df["slow"] = self.ma_func(df["close"], int(self.params["slow"]))
        return df

    def _plan(self, df: pd.DataFrame, i: int, holding: bool) -> Plan:
        f1, s1 = df["fast"].iat[i - 1], df["slow"].iat[i - 1]
        f0, s0 = df["fast"].iat[i - 2], df["slow"].iat[i - 2]
        if pd.isna([f0, s0, f1, s1]).any():
            return Plan()
        golden = f0 <= s0 and f1 > s1
        return Plan(enter=bool(golden), sell_now=bool(f1 < s1),
                    reason="골든크로스" if golden else ("단기선<장기선" if f1 < s1 else ""))


class MaCross(_CrossBase):
    name = "ma_cross"
    ma_func = staticmethod(sma)


class EmaCross(_CrossBase):
    name = "ema_cross"
    ma_func = staticmethod(ema)
    defaults = {"fast": 12, "slow": 26}


class VolatilityBreakout(Strategy):
    """변동성 돌파: 목표가 = 이번 캔들 시가 + k × (직전 캔들 고가 - 저가).
    가격이 목표가를 넘으면 매수하고, 캔들이 끝날 때 매도합니다."""
    name = "volatility_breakout"
    defaults = {"k": 0.5}

    def _validate(self) -> None:
        if not 0 < float(self.params["k"]) <= 3:
            raise ValueError("volatility_breakout 의 k 는 0보다 크고 3 이하여야 합니다. (보통 0.3~0.7)")

    def _plan(self, df: pd.DataFrame, i: int, holding: bool) -> Plan:
        prev_range = df["high"].iat[i - 1] - df["low"].iat[i - 1]
        if pd.isna(prev_range) or prev_range <= 0:
            return Plan()
        target = df["open"].iat[i] + float(self.params["k"]) * prev_range
        return Plan(enter=True, trigger=float(target), exit_at_candle_end=True,
                    reason=f"목표가 {target:,.2f}")


class RsiReversion(Strategy):
    """RSI 과매도 구간에서 매수, RSI 가 회복되면 매도."""
    name = "rsi_reversion"
    defaults = {"period": 14, "oversold": 30, "exit_level": 55}

    def _validate(self) -> None:
        p = self.params
        if int(p["period"]) < 2:
            raise ValueError("rsi_reversion 의 period 는 2 이상이어야 합니다.")
        if not 0 < float(p["oversold"]) < float(p["exit_level"]) <= 100:
            raise ValueError("rsi_reversion 은 0 < oversold < exit_level <= 100 이어야 합니다.")

    def _warmup(self) -> int:
        return int(self.params["period"]) * 3

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = rsi(df["close"], int(self.params["period"]))
        return df

    def _plan(self, df: pd.DataFrame, i: int, holding: bool) -> Plan:
        value = df["rsi"].iat[i - 1]
        if pd.isna(value):
            return Plan()
        oversold = value < float(self.params["oversold"])
        recovered = value >= float(self.params["exit_level"])
        return Plan(enter=bool(oversold), sell_now=bool(recovered),
                    reason=f"RSI {value:.1f}")


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (VolatilityBreakout, MaCross, EmaCross, RsiReversion)
}


def build_strategy(name: str, params: dict | None = None, trend_filter_ma: int = 0) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략: {name} (사용 가능: {', '.join(STRATEGIES)})")
    return STRATEGIES[name](params, trend_filter_ma)
