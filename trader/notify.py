"""텔레그램 알림 (선택 사항). 설정이 없으면 아무것도 하지 않습니다."""
from __future__ import annotations

import logging
import os

log = logging.getLogger("trader")


class Notifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self.token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            import requests

            requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          data={"chat_id": self.chat_id, "text": text}, timeout=10)
        except Exception as exc:  # noqa: BLE001  알림 실패가 매매를 멈추면 안 됨
            log.warning("텔레그램 알림 전송 실패: %s", exc)
