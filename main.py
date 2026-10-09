#!/usr/bin/env python3
"""코인 자동매매 - 명령어 모음.

  python main.py backtest            과거 데이터로 전략 시험 (돈 안 나감)
  python main.py backtest --demo     인터넷/거래소 없이 가짜 데이터로 연습
  python main.py compare             모든 전략을 같은 조건으로 비교
  python main.py check               설정/거래소 연결 점검
  python main.py run                 모의투자 시작 (실제 시세, 가상 체결 - 돈 안 나감)
  python main.py run --live          실전 매매 시작 (⚠ 실제 돈으로 주문)
  python main.py status              봇의 현재 상태 보기
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from trader.backtest import format_report, run_backtest, save_chart  # noqa: E402
from trader.bot import TradingBot, load_state  # noqa: E402
from trader.broker import LiveBroker, MarketData, PaperBroker  # noqa: E402
from trader.config import STRATEGY_NAMES, ConfigError, load_config, load_env  # noqa: E402
from trader.data import fetch_history, load_csv, make_demo_data  # noqa: E402
from trader.notify import Notifier  # noqa: E402
from trader.strategies import build_strategy  # noqa: E402


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log = logging.getLogger("trader")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for handler in (logging.StreamHandler(), RotatingFileHandler(ROOT / "logs" / "bot.log", maxBytes=2_000_000,
                                                                   backupCount=3, encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def get_strategy(cfg, name: str | None = None):
    name = name or cfg.strategy
    params = cfg.strategy_params.get(name) or {}
    return build_strategy(name, params, cfg.trend_filter_ma)


def get_data(cfg, args):
    if args.demo:
        return make_demo_data(args.days or cfg.backtest_days, cfg.timeframe), "가짜 연습 데이터(실제 시세 아님)"
    if args.csv:
        return load_csv(args.csv), f"CSV {args.csv}"
    days = args.days or cfg.backtest_days
    print(f"{cfg.exchange} 에서 {cfg.symbol} {cfg.timeframe} 캔들 {days}일 치를 내려받는 중...")
    return fetch_history(cfg.exchange, cfg.symbol, cfg.timeframe, days), f"{cfg.exchange} 실제 시세"


def cmd_backtest(cfg, args) -> int:
    strategy = get_strategy(cfg, args.strategy)
    df, label = get_data(cfg, args)
    result = run_backtest(df, strategy, cfg)
    print(format_report(result, cfg, f"{strategy.name} {strategy.params}"
                        + (f" + 추세필터 MA{cfg.trend_filter_ma}" if cfg.trend_filter_ma else ""), label))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if result.trades:
        import pandas as pd
        pd.DataFrame([t.__dict__ for t in result.trades]).to_csv(out / f"trades_{strategy.name}.csv", index=False)
        print(f"\n거래 내역 저장: results/trades_{strategy.name}.csv")
    if args.chart and save_chart(result, str(out / f"equity_{strategy.name}.png"), strategy.name):
        print(f"자산 그래프 저장: results/equity_{strategy.name}.png")
    if args.demo:
        print("\n※ 가짜 데이터라 수익률 숫자 자체에는 의미가 없습니다. 사용법 연습용입니다.")
    return 0


def cmd_compare(cfg, args) -> int:
    df, label = get_data(cfg, args)
    rows = []
    for name in STRATEGY_NAMES:
        try:
            r = run_backtest(df, get_strategy(cfg, name), cfg)
        except ValueError as exc:
            print(f"- {name}: 건너뜀 ({exc})")
            continue
        m = r.metrics
        rows.append((name, m["total_return_pct"], m["max_drawdown_pct"], m["num_trades"], m["win_rate_pct"], m["buy_and_hold_pct"]))
    print(f"\n전략 비교 | {label} | {cfg.symbol} {cfg.timeframe} | 추세필터 MA{cfg.trend_filter_ma}\n")
    print(f"{'전략':<22}{'수익률%':>10}{'MDD%':>10}{'거래수':>8}{'승률%':>8}")
    for name, ret, mdd, n, win, _ in rows:
        print(f"{name:<22}{ret:>+10.2f}{mdd:>10.2f}{n:>8d}{win:>8.1f}")
    if rows:
        print(f"{'(그냥 보유)':<22}{rows[0][5]:>+10.2f}")
    print("\n※ 한 기간의 결과일 뿐입니다. 기간을 바꿔서도 확인하고, 수익률과 MDD 를 함께 보세요.")
    return 0


def make_live_broker(cfg) -> LiveBroker:
    key, secret = os.environ.get("EXCHANGE_API_KEY", ""), os.environ.get("EXCHANGE_API_SECRET", "")
    if not key or not secret:
        raise ConfigError(".env 파일에 EXCHANGE_API_KEY / EXCHANGE_API_SECRET 을 먼저 넣어주세요. (.env.example 참고)")
    return LiveBroker(MarketData(cfg.exchange, cfg.symbol, cfg.timeframe, key, secret), cfg.costs.fee_pct)


def cmd_check(cfg, args) -> int:
    strategy = get_strategy(cfg)
    print(f"설정 OK | {cfg.exchange} {cfg.symbol} {cfg.timeframe} | 전략 {strategy.name} {strategy.params}")
    print(f"리스크: {cfg.risk}")
    market = MarketData(cfg.exchange, cfg.symbol, cfg.timeframe)
    print(f"현재가 조회... {market.get_price():,.2f}")
    candles = market.get_candles(strategy.warmup + 5)
    print(f"캔들 {len(candles)}개 수신 (전략에 필요한 최소 {strategy.warmup + 1}개) - 마지막: {candles['timestamp'].iat[-1]}")
    print(f"텔레그램 알림: {'켜짐' if Notifier().enabled else '꺼짐(선택 사항)'}")
    if os.environ.get("EXCHANGE_API_KEY"):
        print("API 키 확인 (잔고 조회)...")
        print(make_live_broker(cfg).check_connection())
    return 0


def cmd_run(cfg, args) -> int:
    strategy = get_strategy(cfg)
    if args.live:
        if not args.yes:
            print("⚠" * 20)
            print("실전 모드입니다. 실제 계좌의 돈으로 주문이 나갑니다. 손실이 발생할 수 있습니다.")
            print(f"  거래소 {cfg.exchange} | {cfg.symbol} | 봇 운용 금액 {cfg.capital:,.0f} {cfg.quote_currency} | 전략 {strategy.name}")
            if input("계속하려면 '실전' 이라고 입력하세요: ").strip() != "실전":
                print("취소했습니다.")
                return 1
        broker = make_live_broker(cfg)
        balances = broker.check_connection()
        print(f"API 키 확인 완료. 거래소 잔고: {balances}")
        if balances[cfg.quote_currency] < cfg.capital:
            print(f"⚠ 거래소 {cfg.quote_currency} 잔고가 config 의 capital({cfg.capital:,.0f}) 보다 적습니다.")
        mode = "live"
    else:
        market = MarketData(cfg.exchange, cfg.symbol, cfg.timeframe)
        broker = PaperBroker(market, cfg.costs.fee_pct, cfg.costs.slippage_pct)
        mode = "paper"
        print("모의투자 모드: 실제 시세로 가상 체결합니다. 돈은 나가지 않습니다.")
    bot = TradingBot(cfg, broker, strategy, mode, root=ROOT, notifier=Notifier())
    print("종료하려면 Ctrl+C 를 누르세요. (STOP 이라는 빈 파일을 만들면 보유분 정리 후 종료)")
    bot.run_forever()
    return 0


def cmd_status(cfg, args) -> int:
    for mode in ("paper", "live"):
        st = load_state(ROOT / "state" / f"{mode}_state.json")
        if not st:
            print(f"[{mode}] 기록 없음")
            continue
        print(f"[{mode}] 현금 {st.cash:,.2f} {cfg.quote_currency}")
        if st.position:
            p = st.position
            print(f"   보유: {p.qty:.8g}개, 평균 매수가 {p.entry_price:,.2f}, 보유 중 최고가 {p.highest:,.2f}")
        else:
            print("   보유 포지션 없음")
        r = st.risk
        print(f"   오늘({r.day}) 실현손익 {r.day_pnl:+,.0f}, 매수 {r.trades_today}회, 최고 자산 {r.peak_equity:,.0f}")
        if r.halted:
            print(f"   🛑 정지됨: {r.halt_reason}  (다시 시작하려면 state/{mode}_state.json 을 삭제하세요)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="코인 자동매매", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="설정 파일 경로")
    parser.add_argument("--debug", action="store_true", help="오류 발생 시 상세 내용 출력")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--demo", action="store_true", help="가짜 데이터 사용 (인터넷 불필요)")
        p.add_argument("--csv", help="내 CSV 파일 사용 (열: timestamp,open,high,low,close,volume)")
        p.add_argument("--days", type=int, help="백테스트 기간(일). 기본은 config.yaml 의 backtest.days")

    p = sub.add_parser("backtest", help="백테스트")
    data_args(p)
    p.add_argument("--strategy", choices=STRATEGY_NAMES, help="config.yaml 의 전략 대신 이 전략으로 시험")
    p.add_argument("--chart", action="store_true", help="자산 그래프(PNG) 저장")
    p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("compare", help="전략 비교")
    data_args(p)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("check", help="설정/연결 점검")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("run", help="봇 실행 (기본: 모의투자)")
    p.add_argument("--live", action="store_true", help="⚠ 실전 매매")
    p.add_argument("--yes", action="store_true", help="실전 확인 질문 생략 (서버에서 자동 실행할 때)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("status", help="봇 상태 보기")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        load_env(ROOT / ".env")
        cfg = load_config(args.config)
        if args.command == "run":
            setup_logging()
        return args.func(cfg, args)
    except (ConfigError, ValueError) as exc:
        print(f"\n❌ 설정 오류: {exc}")
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001  초보자에게 긴 오류 대신 안내 메시지를 보여줌
        if args.debug:
            raise
        import ccxt
        if isinstance(exc, (ccxt.NetworkError, ccxt.ExchangeNotAvailable)):
            hint = "인터넷 연결 또는 거래소 서버 상태를 확인하세요. (거래소가 접속을 막는 지역/네트워크일 수도 있습니다)"
        elif isinstance(exc, (ccxt.AuthenticationError, ccxt.PermissionDenied)):
            hint = ".env 의 API 키/시크릿, 키 권한(자산조회+주문), 허용 IP 설정을 확인하세요."
        elif isinstance(exc, ccxt.BadSymbol):
            hint = "config.yaml 의 symbol 이 해당 거래소에 있는 코인인지 확인하세요. (예: BTC/KRW)"
        else:
            hint = "문제가 계속되면 '--debug' 옵션을 붙여 실행한 결과를 Claude 에게 보여주세요."
        print(f"\n❌ 오류: {type(exc).__name__}: {str(exc)[:300]}\n   → {hint}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
