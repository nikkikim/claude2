"""config.yaml 을 읽고 검사합니다. 오타나 이상한 값이 있으면 알기 쉬운 한글 오류를 냅니다."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}

STRATEGY_NAMES = ("volatility_breakout", "ma_cross", "ema_cross", "rsi_reversion")


class ConfigError(Exception):
    """설정이 잘못됐을 때."""


@dataclass
class RiskConfig:
    max_position_pct: float = 50
    risk_per_trade_pct: float = 1
    stop_loss_pct: float = 3
    take_profit_pct: float = 0
    trailing_stop_pct: float = 0
    max_daily_loss_pct: float = 3
    max_drawdown_pct: float = 20
    max_trades_per_day: int = 5
    cooldown_candles: int = 1
    min_order_value: float = 5000


@dataclass
class CostConfig:
    fee_pct: float = 0.05
    slippage_pct: float = 0.05


@dataclass
class Config:
    exchange: str = "upbit"
    symbol: str = "BTC/KRW"
    timeframe: str = "1d"
    timezone: str = "Asia/Seoul"
    capital: float = 1_000_000
    poll_seconds: int = 10
    strategy: str = "volatility_breakout"
    trend_filter_ma: int = 0
    strategy_params: dict = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest_days: int = 365

    @property
    def timeframe_seconds(self) -> int:
        return TIMEFRAME_SECONDS[self.timeframe]

    @property
    def params_for_strategy(self) -> dict:
        return dict(self.strategy_params.get(self.strategy) or {})

    @property
    def quote_currency(self) -> str:
        return self.symbol.split("/")[1]


def _build_section(cls, data: dict | None, section: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{section}' 항목은 '이름: 값' 형태여야 합니다.")
    declared = {f.name: f.type for f in fields(cls)}  # 'float' 또는 'int' (문자열)
    unknown = set(data) - set(declared)
    if unknown:
        raise ConfigError(
            f"'{section}' 에 알 수 없는 항목이 있습니다: {', '.join(sorted(unknown))}\n"
            f"  사용 가능한 항목: {', '.join(sorted(declared))}"
        )
    obj = cls()
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{section}.{key}' 는 숫자여야 합니다. (현재 값: {value!r})")
        setattr(obj, key, int(value) if declared[key] == "int" else float(value))
    return obj


def _validate(cfg: Config) -> None:
    if cfg.timeframe not in TIMEFRAME_SECONDS:
        raise ConfigError(f"timeframe 은 {', '.join(TIMEFRAME_SECONDS)} 중 하나여야 합니다. (현재: {cfg.timeframe})")
    if "/" not in cfg.symbol:
        raise ConfigError(f"symbol 은 'BTC/KRW' 처럼 '/' 로 구분해서 적어주세요. (현재: {cfg.symbol})")
    if cfg.strategy not in STRATEGY_NAMES:
        raise ConfigError(f"strategy 는 {', '.join(STRATEGY_NAMES)} 중 하나여야 합니다. (현재: {cfg.strategy})")
    if cfg.capital <= 0:
        raise ConfigError("capital 은 0보다 커야 합니다.")
    if cfg.poll_seconds < 1:
        raise ConfigError("poll_seconds 는 1 이상이어야 합니다.")
    if cfg.trend_filter_ma < 0:
        raise ConfigError("trend_filter_ma 는 0 이상이어야 합니다.")
    for name in ("max_position_pct", "risk_per_trade_pct", "stop_loss_pct", "take_profit_pct",
                 "trailing_stop_pct", "max_daily_loss_pct", "max_drawdown_pct"):
        value = getattr(cfg.risk, name)
        if not 0 <= value <= 100:
            raise ConfigError(f"risk.{name} 는 0~100 사이여야 합니다. (현재: {value})")
    if cfg.risk.max_position_pct <= 0:
        raise ConfigError("risk.max_position_pct 는 0보다 커야 합니다. (한 번에 매수할 최대 비율)")
    for name in ("fee_pct", "slippage_pct"):
        if not 0 <= getattr(cfg.costs, name) < 10:
            raise ConfigError(f"costs.{name} 값이 이상합니다. (퍼센트 단위로 적어주세요. 0.05 = 0.05%)")
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(cfg.timezone)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"timezone '{cfg.timezone}' 을(를) 찾을 수 없습니다. 예: Asia/Seoul") from exc


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml 문법 오류입니다. 들여쓰기(띄어쓰기)와 ':' 를 확인하세요.\n{exc}") from exc

    top_valid = {"exchange", "symbol", "timeframe", "timezone", "capital", "poll_seconds", "strategy",
                 "trend_filter_ma", "strategy_params", "risk", "costs", "backtest"}
    unknown = set(raw) - top_valid
    if unknown:
        raise ConfigError(f"config.yaml 에 알 수 없는 항목이 있습니다: {', '.join(sorted(unknown))}")

    cfg = Config()
    for key in ("exchange", "symbol", "timeframe", "timezone", "strategy"):
        if key in raw:
            setattr(cfg, key, str(raw[key]))
    for key, typ in (("capital", float), ("poll_seconds", int), ("trend_filter_ma", int)):
        if key in raw:
            try:
                setattr(cfg, key, typ(raw[key]))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"'{key}' 는 숫자여야 합니다. (현재 값: {raw[key]!r})") from exc
    if raw.get("strategy_params") is not None:
        if not isinstance(raw["strategy_params"], dict):
            raise ConfigError("'strategy_params' 항목 형식이 잘못됐습니다.")
        cfg.strategy_params = raw["strategy_params"]
    cfg.risk = _build_section(RiskConfig, raw.get("risk"), "risk")
    cfg.costs = _build_section(CostConfig, raw.get("costs"), "costs")
    bt = raw.get("backtest") or {}
    if "days" in bt:
        cfg.backtest_days = int(bt["days"])
    _validate(cfg)
    return cfg


def load_env(path: str | Path = ".env") -> None:
    """'.env' 파일의 KEY=VALUE 줄을 환경변수로 읽어옵니다. (이미 설정된 값은 덮어쓰지 않음)"""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)
