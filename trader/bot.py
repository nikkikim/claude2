"""자동매매 봇 (모의투자 / 실전 공용).

poll_seconds 마다 step() 을 한 번씩 실행합니다:
  1) 최근 캔들과 현재가를 가져옴
  2) 보유 중이면: 전략 매도 신호 / 손절 / 익절 / 트레일링 / 캔들 종료 매도 확인
  3) 보유 중이 아니면: 전략 매수 신호 + 리스크 규칙 확인 후 매수
상태(보유 포지션, 현금, 일일 손익 등)는 state/*.json 에 저장되어 껐다 켜도 이어집니다.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .broker import OrderRejected, PositionMissing
from .config import Config
from .notify import Notifier
from .risk import Position, RiskManager, RiskState
from .strategies import Strategy

log = logging.getLogger("trader")

HISTORY_CANDLES = 300
TRADE_LOG_HEADER =["time", "side", "price", "qty", "fee", "pnl", "pnl_pct", "reason", "cash_after"]


@dataclass
class BotState:
    cash: float
    position: Position | None = None
    candle_ts: float = 0.0          # 마지막으로 본 캔들 시작 시각
    traded_candle_ts: float = 0.0   # 이 캔들에서는 이미 거래했음 (한 캔들에 한 번만 매수)
    risk: RiskState = field(default_factory=RiskState)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "BotState":
        pos = Position(**d["position"]) if d.get("position") else None
        return cls(cash=d["cash"], position=pos, candle_ts=d.get("candle_ts", 0.0),
                   traded_candle_ts=d.get("traded_candle_ts", 0.0), risk=RiskState(**d.get("risk", {})))


def load_state(path: Path) -> BotState | None:
    if not path.exists():
        return None
    return BotState.from_dict(json.loads(path.read_text(encoding="utf-8")))


class TradingBot:
    def __init__(self, cfg: Config, broker, strategy: Strategy, mode: str, root: Path = Path("."),
                 notifier: Notifier | None = None, clock: Callable[[], float] = time.time):
        self.cfg, self.broker, self.strategy, self.mode = cfg, broker, strategy, mode
        self.notifier = notifier or Notifier("", "")
        self.clock = clock
        self.fee_rate = cfg.costs.fee_pct / 100
        self.tf = cfg.timeframe_seconds
        # EMA/RSI 는 '처음부터의 이력'에 조금씩 영향을 받으므로, 짧게 받으면 백테스트와 값이 달라집니다.
        # 300개면 오차가 사실상 0 입니다. (tests/test_parity.py 가 백테스트와의 일치를 검증)
        self.n_candles = max(strategy.warmup + 5, HISTORY_CANDLES)
        self.state_path = root / "state" / f"{mode}_state.json"
        self.trade_log = root / "logs" / f"trades_{mode}.csv"
        self.stop_file = root / "STOP"
        self.state_path.parent.mkdir(exist_ok=True)
        self.trade_log.parent.mkdir(exist_ok=True)

        loaded = load_state(self.state_path)
        if loaded:
            log.info("이전 상태를 불러왔습니다 (현금 %s, 보유 %s)", f"{loaded.cash:,.0f}",
                     "있음" if loaded.position else "없음")
        self.state = loaded or BotState(cash=cfg.capital)
        self.risk = RiskManager(cfg.risk, cfg.timezone, self.tf, self.state.risk)
        self.finished = False

    # ── 저장 / 기록 ────────────────────────────────────────
    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(self.state.to_json(), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _log_trade(self, side: str, price: float, qty: float, fee: float, pnl: float | str,
                   pnl_pct: float | str, reason: str, now: float) -> None:
        new_file = not self.trade_log.exists()
        with self.trade_log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(TRADE_LOG_HEADER)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), side, price, qty, fee,
                        pnl, pnl_pct, reason, round(self.state.cash, 2)])

    def _alert(self, text: str) -> None:
        log.info(text)
        self.notifier.send(f"[{self.mode}] {text}")

    # ── 한 번의 점검 ───────────────────────────────────────
    def step(self) -> None:
        now = self.clock()
        df = self.strategy.prepare(self.broker.get_candles(self.n_candles))
        if len(df) <= self.strategy.warmup:
            log.warning("캔들이 부족합니다 (%d개 / 필요 %d개)", len(df), self.strategy.warmup + 1)
            return
        i = len(df) - 1
        cur_ts = df["timestamp"].iat[i].timestamp()
        if now - cur_ts > 2 * self.tf:
            log.warning("최신 캔들이 오래됐습니다 (거래소 점검 중일 수 있음). 이번 점검은 건너뜁니다.")
            return
        price = self.broker.get_price()
        pos = self.state.position

        equity = self.state.cash + (pos.qty * price if pos else 0.0)
        self.risk.on_time(now, equity)
        plan = self.strategy.plan(df, i, pos is not None)

        if cur_ts != self.state.candle_ts:
            self.state.candle_ts = cur_ts
            log.info("새 캔들 시작 | 현재가 %s | 자산 %s | 전략 판단: %s%s", f"{price:,.2f}", f"{equity:,.0f}",
                     "매수 대기" if plan.enter else "관망", f" ({plan.reason})" if plan.reason else "")

        if pos is not None:
            reason = None
            if self.risk.state.halted:
                reason = "max_drawdown_halt"
            elif pos.exit_at_candle_end and cur_ts > pos.entry_candle_ts:
                reason = "candle_end"
            elif plan.sell_now:
                reason = "signal"
            else:
                hit = self.risk.check_exit(pos, price, price, price)
                if hit:
                    reason = hit[0]
                else:
                    pos.highest = max(pos.highest, price)
            if reason:
                self._sell(reason, now, cur_ts)
        elif self._may_enter(plan, cur_ts, price, now):
            self._buy(plan, cur_ts, now, equity)

        self.state.risk = self.risk.state
        self._save()

        if self.state.position is None:
            if self.risk.state.halted:
                self._alert(f"🛑 봇을 정지합니다: {self.risk.state.halt_reason}")
                self.finished = True
            elif self.stop_file.exists():
                self._alert("STOP 파일이 있어 봇을 종료합니다. (보유 포지션 없음)")
                self.finished = True

    def _may_enter(self, plan, cur_ts: float, price: float, now: float) -> bool:
        if not plan.enter or self.state.traded_candle_ts == cur_ts:
            return False
        if self.risk.state.halted or self.stop_file.exists():
            return False
        if plan.trigger is not None:
            return price >= plan.trigger
        # '시가에 바로 매수' 신호는 캔들 초반에만 유효 (봇을 늦게 켰을 때 뒤늦게 사지 않도록)
        window = max(0.1 * self.tf, 3 * self.cfg.poll_seconds)
        return now - cur_ts <= window

    # ── 매수 / 매도 ────────────────────────────────────────
    def _buy(self, plan, cur_ts: float, now: float, equity: float) -> None:
        ok, why = self.risk.can_enter(now)
        if not ok:
            log.info("매수 신호가 있지만 리스크 규칙으로 건너뜀: %s", why)
            self.state.traded_candle_ts = cur_ts
            return
        value = self.risk.order_value(equity, self.state.cash, self.fee_rate)
        if value <= 0:
            log.info("매수 금액이 최소 주문 금액(%s) 미만이라 건너뜀", f"{self.cfg.risk.min_order_value:,.0f}")
            self.state.traded_candle_ts = cur_ts
            return
        try:
            fill = self.broker.buy(value)
        except OrderRejected as exc:
            self._alert(f"⚠ 매수 주문이 거절됐습니다: {exc}")
            self.state.traded_candle_ts = cur_ts
            return
        except Exception as exc:  # noqa: BLE001
            # 주문이 나갔는지 알 수 없는 상태 → 중복 매수를 막기 위해 정지하고 사람이 확인
            self.risk.halt(f"매수 주문 결과를 알 수 없음({exc}). 거래소 앱에서 체결 여부를 확인한 뒤 state 폴더를 정리하세요.")
            self._alert(f"🚨 {self.risk.state.halt_reason}")
            return
        entry_value = fill.price * fill.qty
        self.state.cash -= entry_value + fill.fee
        self.state.position = Position(entry_price=fill.price, qty=fill.qty, entry_ts=now, entry_value=entry_value,
                                       entry_fee=fill.fee, highest=fill.price,
                                       exit_at_candle_end=plan.exit_at_candle_end, entry_candle_ts=cur_ts)
        self.state.traded_candle_ts = cur_ts
        self.risk.on_entry()
        self._log_trade("BUY", fill.price, fill.qty, fill.fee, "", "", plan.reason, now)
        self._alert(f"🟢 매수 {self.cfg.symbol} {fill.qty:.8g}개 @ {fill.price:,.2f} ({entry_value:,.0f} {self.cfg.quote_currency})"
                    + (" [체결가 추정]" if fill.estimated else ""))

    def _sell(self, reason: str, now: float, cur_ts: float) -> None:
        pos = self.state.position
        try:
            fill = self.broker.sell(pos.qty)
        except PositionMissing as exc:
            self.risk.halt(f"기록상 보유 중인데 거래소에 코인이 없습니다({exc}). 직접 확인이 필요합니다.")
            self.state.position = None
            self._alert(f"🚨 {self.risk.state.halt_reason}")
            return
        except OrderRejected as exc:
            self._alert(f"⚠ 매도 주문이 거절됐습니다(다음 점검에서 다시 시도): {exc}")
            return
        proceeds = fill.price * fill.qty - fill.fee
        cost = pos.entry_value + pos.entry_fee
        pnl = proceeds - cost
        self.state.cash += proceeds
        self.state.position = None
        if reason != "candle_end":
            # 손절 등으로 판 캔들에서는 다시 사지 않음. 단, 캔들 종료 매도는 '지난 캔들의 거래'이므로
            # 새 캔들의 매수 기회를 막으면 안 됨 (백테스트도 같은 방식)
            self.state.traded_candle_ts = cur_ts
        self.risk.on_exit(cur_ts, pnl, reason)
        self._log_trade("SELL", fill.price, fill.qty, fill.fee, round(pnl, 2), round(pnl / cost * 100, 3), reason, now)
        self._alert(f"{'🔴' if pnl < 0 else '🔵'} 매도({reason}) {self.cfg.symbol} @ {fill.price:,.2f} | "
                    f"손익 {pnl:+,.0f} {self.cfg.quote_currency} ({pnl / cost * 100:+.2f}%)"
                    + (" [체결가 추정]" if fill.estimated else ""))

    # ── 계속 실행 ──────────────────────────────────────────
    def run_forever(self) -> None:
        errors = 0
        self._alert(f"봇 시작 | {self.cfg.exchange} {self.cfg.symbol} {self.cfg.timeframe} | 전략 {self.strategy.name}")
        try:
            while not self.finished:
                try:
                    self.step()
                    errors = 0
                except Exception as exc:  # noqa: BLE001  네트워크 오류 등은 잠시 후 재시도
                    errors += 1
                    if errors == 1:
                        log.exception("점검 중 오류 - 잠시 후 다시 시도합니다: %s", exc)  # 상세 내용은 처음 한 번만
                    else:
                        log.warning("점검 중 오류 (%d회 연속): %s: %s", errors, type(exc).__name__, str(exc)[:200])
                    if errors == 10:
                        self._alert("⚠ 오류가 10회 연속 발생했습니다. 인터넷/거래소 상태를 확인하세요.")
                time.sleep(self.cfg.poll_seconds if errors == 0 else min(60, self.cfg.poll_seconds * (1 + errors)))
        except KeyboardInterrupt:
            pos = self.state.position
            log.info("사용자가 봇을 종료했습니다. %s", "⚠ 보유 포지션이 남아 있습니다!" if pos else "보유 포지션 없음.")
        finally:
            self._save()
