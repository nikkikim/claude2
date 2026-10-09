"""주문 실행 담당.

  PaperBroker : 모의투자. 실제 시세를 보면서 가상으로 체결 (돈이 나가지 않음)
  LiveBroker  : 실전. ccxt 로 실제 거래소에 시장가 주문을 냅니다.

두 브로커는 같은 메서드를 가져서 봇(bot.py)은 어느 쪽인지 신경 쓰지 않습니다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from .config import TIMEFRAME_SECONDS
from .data import fetch_ohlcv_range, make_exchange

log = logging.getLogger("trader")


class OrderRejected(Exception):
    """거래소가 주문을 확실히 거절함 (돈이 나가지 않았음이 확실)."""


class PositionMissing(Exception):
    """팔려는 코인이 거래소 계좌에 없음."""


@dataclass
class Fill:
    price: float      # 평균 체결가
    qty: float        # 체결 수량
    fee: float        # 수수료 (결제 통화 기준)
    estimated: bool = False  # 거래소에서 체결 내역을 확인 못 해서 추정한 값이면 True


class MarketData:
    """거래소의 공개 시세 (API 키 불필요)."""

    def __init__(self, exchange_id: str, symbol: str, timeframe: str, api_key=None, secret=None):
        self.ex = make_exchange(exchange_id, api_key, secret)
        self.symbol = symbol
        self.timeframe = timeframe
        self.tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000

    def get_price(self) -> float:
        return float(self.ex.fetch_ticker(self.symbol)["last"])

    def get_candles(self, n: int) -> pd.DataFrame:
        """최근 n 개 캔들 (마지막 줄은 아직 진행 중인 캔들)."""
        since = int(time.time() * 1000) - (n + 1) * self.tf_ms
        df = fetch_ohlcv_range(self.ex, self.symbol, self.timeframe, since)
        return df.tail(n).reset_index(drop=True)


class PaperBroker:
    def __init__(self, market, fee_pct: float, slippage_pct: float):
        self.market = market
        self.fee = fee_pct / 100
        self.slip = slippage_pct / 100

    def get_price(self) -> float:
        return self.market.get_price()

    def get_candles(self, n: int) -> pd.DataFrame:
        return self.market.get_candles(n)

    def buy(self, value: float) -> Fill:
        price = self.market.get_price() * (1 + self.slip)
        return Fill(price=price, qty=value / price, fee=value * self.fee)

    def sell(self, qty: float) -> Fill:
        price = self.market.get_price() * (1 - self.slip)
        return Fill(price=price, qty=qty, fee=qty * price * self.fee)


class LiveBroker:
    """⚠ 실제 돈이 오가는 브로커. 시장가 주문만 사용합니다."""

    def __init__(self, market: MarketData, fee_pct: float):
        self.market = market
        self.ex = market.ex
        self.symbol = market.symbol
        self.fee_rate = fee_pct / 100
        self.base, self.quote = self.symbol.split("/")

    def get_price(self) -> float:
        return self.market.get_price()

    def get_candles(self, n: int) -> pd.DataFrame:
        return self.market.get_candles(n)

    def _free(self, currency: str) -> float:
        bal = self.ex.fetch_balance()
        return float((bal.get(currency) or {}).get("free") or 0.0)

    def check_connection(self) -> dict:
        """API 키가 올바른지 확인하고 잔고를 돌려줍니다."""
        return {self.quote: self._free(self.quote), self.base: self._free(self.base)}

    def buy(self, value: float) -> Fill:
        import ccxt

        free_cash = self._free(self.quote)
        if free_cash < value * (1 + self.fee_rate):
            raise OrderRejected(f"거래소 {self.quote} 잔고 부족: 필요 {value * (1 + self.fee_rate):,.2f}, 보유 {free_cash:,.2f}")
        price = self.get_price()
        amount = float(self.ex.amount_to_precision(self.symbol, value / price))
        try:
            # 업비트처럼 '금액 기준 시장가 매수'를 쓰는 거래소는 ccxt 가 amount*price 로 환산해서 보냅니다.
            order = self.ex.create_order(self.symbol, "market", "buy", amount, price)
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.AuthenticationError, ccxt.PermissionDenied) as exc:
            raise OrderRejected(str(exc)) from exc
        return self._to_fill(order, price, amount, side="buy")

    def sell(self, qty: float) -> Fill:
        import ccxt

        free_coin = self._free(self.base)
        qty = min(qty, free_coin)
        qty = float(self.ex.amount_to_precision(self.symbol, qty)) if qty > 0 else 0.0
        if qty <= 0:
            raise PositionMissing(f"거래소 계좌에 {self.base} 가 없습니다.")
        price = self.get_price()
        try:
            order = self.ex.create_order(self.symbol, "market", "sell", qty)
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.AuthenticationError, ccxt.PermissionDenied) as exc:
            raise OrderRejected(str(exc)) from exc
        return self._to_fill(order, price, qty, side="sell")

    def _to_fill(self, order: dict, ref_price: float, ref_qty: float, side: str) -> Fill:
        """주문이 체결될 때까지 잠깐 기다렸다가 실제 체결 내역을 읽습니다. 실패하면 추정값을 씁니다."""
        info = order
        for _ in range(10):
            if info.get("status") == "closed" and info.get("filled"):
                break
            time.sleep(1)
            try:
                info = self.ex.fetch_order(order["id"], self.symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("체결 내역 조회 실패(추정값 사용): %s", exc)
                break
        filled = float(info.get("filled") or 0)
        if info.get("status") == "closed" and filled > 0:
            cost = info.get("cost")
            price = float(info.get("average") or (cost / filled if cost else ref_price))
            fee_info = info.get("fee") or {}
            if fee_info.get("cost") is not None and fee_info.get("currency") in (None, self.quote):
                fee = float(fee_info["cost"])
            else:
                fee = price * filled * self.fee_rate
            return Fill(price=price, qty=filled, fee=fee)
        log.warning("주문 %s 의 체결을 확인하지 못해 추정값을 사용합니다. 거래소 앱에서 꼭 확인하세요!", order.get("id"))
        return Fill(price=ref_price, qty=ref_qty, fee=ref_price * ref_qty * self.fee_rate, estimated=True)
