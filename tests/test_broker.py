import ccxt
import pytest

from trader import broker as broker_module
from trader.broker import LiveBroker, OrderRejected, PositionMissing


class FakeEx:
    """ccxt 거래소 흉내 (실제 주문 없음)"""
    def __init__(self, quote_free=1_000_000, base_free=0.5, order=None, fetch_order_result=None, create_exc=None):
        self.balance = {"KRW": {"free": quote_free}, "BTC": {"free": base_free}}
        self.order, self.fetch_order_result, self.create_exc = order, fetch_order_result, create_exc
        self.created = []

    def fetch_balance(self):
        return self.balance

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.6f}"

    def create_order(self, *args):
        if self.create_exc:
            raise self.create_exc
        self.created.append(args)
        return self.order or {"id": "1", "status": "open"}

    def fetch_order(self, order_id, symbol):
        return self.fetch_order_result


class FakeMarket:
    symbol = "BTC/KRW"

    def __init__(self, ex):
        self.ex = ex

    def get_price(self):
        return 100_000_000.0


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(broker_module.time, "sleep", lambda s: None)


def live(ex):
    return LiveBroker(FakeMarket(ex), fee_pct=0.05)


def test_buy_reads_actual_fill():
    ex = FakeEx(order={"id": "1", "status": "open"},
                fetch_order_result={"id": "1", "status": "closed", "filled": 0.002, "average": 100_100_000.0,
                                   "cost": 200_200.0, "fee": {"cost": 100.0, "currency": "KRW"}})
    fill = live(ex).buy(200_000)
    assert ex.created == [("BTC/KRW", "market", "buy", 0.002, 100_000_000.0)]
    assert fill.price == 100_100_000.0 and fill.qty == 0.002 and fill.fee == 100.0 and not fill.estimated


def test_buy_rejects_when_exchange_cash_is_short():
    ex = FakeEx(quote_free=100_000)
    with pytest.raises(OrderRejected):
        live(ex).buy(200_000)
    assert ex.created == []                                   # 주문 자체를 내지 않음


def test_buy_maps_definite_rejections_but_not_unknown_errors():
    with pytest.raises(OrderRejected):
        live(FakeEx(create_exc=ccxt.InsufficientFunds("x"))).buy(200_000)
    with pytest.raises(OrderRejected):
        live(FakeEx(create_exc=ccxt.InvalidOrder("x"))).buy(200_000)
    with pytest.raises(ccxt.RequestTimeout):                  # 주문이 나갔는지 모르는 경우는 그대로 전달 → 봇이 정지
        live(FakeEx(create_exc=ccxt.RequestTimeout("x"))).buy(200_000)


def test_estimated_fill_when_status_unknown():
    class Broken(FakeEx):
        def fetch_order(self, order_id, symbol):
            raise ccxt.NetworkError("down")

    fill = live(Broken()).buy(200_000)
    assert fill.estimated and fill.qty == pytest.approx(0.002) and fill.price == 100_000_000.0


def test_sell_never_exceeds_exchange_balance_and_requires_coin():
    ex = FakeEx(base_free=0.001, order={"id": "2", "status": "closed", "filled": 0.001, "average": 99_000_000.0,
                                        "cost": 99_000.0, "fee": {"cost": 49.5, "currency": "KRW"}})
    fill = live(ex).sell(0.5)                                 # 기록상 0.5 개지만 거래소에는 0.001 개뿐
    assert ex.created == [("BTC/KRW", "market", "sell", 0.001)] and fill.qty == 0.001
    with pytest.raises(PositionMissing):
        live(FakeEx(base_free=0)).sell(0.5)
