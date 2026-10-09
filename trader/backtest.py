"""백테스트: 과거 데이터로 전략 + 리스크 설정을 시험해봅니다.

현실적으로 보이도록 다음을 가정합니다 (모두 보수적인 쪽):
  * 신호는 '완성된 캔들'로만 판단 → 이동평균/RSI 전략은 다음 캔들 시가에 체결
  * 변동성 돌파는 목표가에 체결 (시가가 이미 목표가 위면 시가), 캔들 종가에 매도
  * 수수료와 슬리피지(불리한 체결) 반영
  * 한 캔들 안에서 손절과 익절이 모두 가능하면 손절이 먼저라고 가정
  * 손절/매도한 캔들에서는 다시 매수하지 않음
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .risk import Position, RiskManager
from .strategies import Strategy


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    reason: str


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series          # 캔들 종가 기준 자산 추이 (index = 시간)
    metrics: dict
    halted: str = ""


def run_backtest(df: pd.DataFrame, strategy: Strategy, cfg: Config) -> BacktestResult:
    df = strategy.prepare(df)
    n = len(df)
    if n <= strategy.warmup + 1:
        raise ValueError(f"데이터가 너무 적습니다. (캔들 {n}개, 최소 {strategy.warmup + 2}개 필요)")

    times = list(df["timestamp"])
    tss = [t.timestamp() for t in times]
    opens, highs, lows, closes = (df[c].to_numpy() for c in ("open", "high", "low", "close"))

    fee = cfg.costs.fee_pct / 100
    slip = cfg.costs.slippage_pct / 100
    risk = RiskManager(cfg.risk, cfg.timezone, cfg.timeframe_seconds)

    cash = cfg.capital
    pos: Position | None = None
    pos_time: pd.Timestamp | None = None
    trades: list[Trade] = []
    curve: list[tuple[pd.Timestamp, float]] = []

    def sell(i: int, ref_price: float, reason: str) -> None:
        nonlocal cash, pos
        price = ref_price * (1 - slip)
        proceeds = pos.qty * price
        sell_fee = proceeds * fee
        cost = pos.entry_value + pos.entry_fee
        pnl = proceeds - sell_fee - cost
        cash += proceeds - sell_fee
        trades.append(Trade(pos_time, times[i], pos.entry_price, price, pos.qty, pnl, pnl / cost * 100, reason))
        risk.on_exit(tss[i], pnl, reason)
        pos = None

    start = strategy.warmup
    for i in range(start, n):
        o, h, l, c, ts = opens[i], highs[i], lows[i], closes[i], tss[i]
        risk.on_time(ts, cash + (pos.qty * o if pos else 0.0))
        exited = False

        if pos is not None:
            plan = strategy.plan(df, i, True)
            if risk.state.halted:
                sell(i, o, "max_drawdown_halt"); exited = True
            elif plan.sell_now:
                sell(i, o, "signal"); exited = True
            else:
                hit = risk.check_exit(pos, o, h, l)
                if hit:
                    sell(i, hit[1], hit[0]); exited = True
                else:
                    pos.highest = max(pos.highest, h)

        if pos is None and not exited and not risk.state.halted:
            plan = strategy.plan(df, i, False)
            if plan.enter and (plan.trigger is None or h >= plan.trigger):
                ok, _ = risk.can_enter(ts)
                ref = o if plan.trigger is None else max(plan.trigger, o)
                value = risk.order_value(cash, cash, fee) if ok else 0.0
                if value > 0:
                    price = ref * (1 + slip)
                    entry_fee = value * fee
                    cash -= value + entry_fee
                    pos = Position(entry_price=price, qty=value / price, entry_ts=ts, entry_value=value,
                                   entry_fee=entry_fee, highest=max(price, h),  # 같은 캔들에서 고점이 먼저였다고 보수적으로 가정
                                   exit_at_candle_end=plan.exit_at_candle_end, entry_candle_ts=ts)
                    pos_time = times[i]
                    risk.on_entry()
                    hit = risk.check_exit(pos, price, h, l)
                    if hit:
                        sell(i, hit[1], hit[0])

        if pos is not None and pos.exit_at_candle_end:
            sell(i, c, "candle_end")

        curve.append((times[i], cash + (pos.qty * c if pos else 0.0)))

    if pos is not None:
        sell(n - 1, closes[n - 1], "end_of_data")
        curve[-1] = (times[n - 1], cash)

    equity = pd.Series([v for _, v in curve], index=[t for t, _ in curve], name="equity")
    metrics = compute_metrics(equity, trades, cfg.capital, opens[start], closes[n - 1])
    return BacktestResult(trades, equity, metrics, risk.state.halt_reason)


def compute_metrics(equity: pd.Series, trades: list[Trade], capital: float, first_price: float,
                    last_price: float) -> dict:
    final = float(equity.iloc[-1])
    running_peak = pd.concat([pd.Series([capital]), equity.reset_index(drop=True)]).cummax().iloc[1:]
    mdd = float(((equity.reset_index(drop=True) - running_peak) / running_peak).min() * 100)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    gross_loss = -sum(losses)
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    total_return = (final / capital - 1) * 100
    return {
        "initial_capital": capital,
        "final_equity": final,
        "total_return_pct": total_return,
        "buy_and_hold_pct": (last_price / first_price - 1) * 100,
        "max_drawdown_pct": mdd,
        "num_trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
        "avg_trade_pct": (sum(t.pnl_pct for t in trades) / len(trades)) if trades else 0.0,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else (float("inf") if wins else 0.0),
        "period_days": days,
    }


def format_report(result: BacktestResult, cfg: Config, strategy_label: str, data_label: str) -> str:
    m = result.metrics
    pf = "∞" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
    reasons: dict[str, int] = {}
    for t in result.trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    reason_text = ", ".join(f"{k} {v}회" for k, v in sorted(reasons.items())) or "-"
    q = cfg.quote_currency
    lines = [
        "=" * 56,
        f" 백테스트 결과 | {strategy_label}",
        f" 데이터: {data_label} ({cfg.symbol}, {cfg.timeframe}, 약 {m['period_days']:.0f}일)",
        "=" * 56,
        f" 시작 자금        : {m['initial_capital']:>14,.0f} {q}",
        f" 최종 자산        : {m['final_equity']:>14,.0f} {q}",
        f" 총 수익률        : {m['total_return_pct']:>+13.2f} %",
        f" (참고) 그냥 보유 : {m['buy_and_hold_pct']:>+13.2f} %",
        f" 최대 낙폭(MDD)   : {m['max_drawdown_pct']:>13.2f} %",
        f" 거래 횟수        : {m['num_trades']:>14d} 회",
        f" 승률             : {m['win_rate_pct']:>13.1f} %",
        f" 거래당 평균 수익 : {m['avg_trade_pct']:>+13.2f} %",
        f" 손익비(PF)       : {pf:>14}",
        f" 청산 사유        : {reason_text}",
    ]
    if result.halted:
        lines.append(f" ⚠ 중간에 정지됨  : {result.halted}")
    lines += [
        "-" * 56,
        " ※ 과거 성과는 미래 수익을 보장하지 않습니다.",
        "   '그냥 보유'보다 못하거나 MDD 가 감당 안 되면 설정을 다시 고민하세요.",
    ]
    return "\n".join(lines)


def save_chart(result: BacktestResult, path: str, title: str) -> bool:
    """자산 추이 그래프 저장. matplotlib 가 없으면 건너뜁니다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result.equity.index, result.equity.values, lw=1.4)
    ax.axhline(result.metrics["initial_capital"], color="gray", lw=0.8, ls="--")
    ax.set_title(f"Equity curve - {title}")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
