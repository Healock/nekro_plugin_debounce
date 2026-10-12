"""短生命周期的 Agent 回复目标范围。

该模块只保存结构化的发送者元数据，不保存用户消息正文，也不改变 Nekro
频道历史。由于 Nekro 当前的调度接口没有暴露 run_id，scope 只能作为
提示层的安全约束，而不是 exactly-once 的调度关联保证。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


SCOPE_TTL_SECONDS = 180.0


@dataclass(slots=True)
class ReplyScope:
    """一个频道当前可安全关联的回复目标。"""

    chat_key: str
    generation: int
    trigger_source: str
    first_trigger_message_id: str
    representative_message_id: str
    target_sender_id: str
    target_sender_name: str
    platform_userid: str
    created_at: float
    expires_at: float
    state: str = "active"

    def expired(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at

    def cancel(self) -> None:
        self.state = "canceled"
        self.expires_at = min(self.expires_at, time.time())

    def mark_ambiguous(self) -> None:
        self.state = "ambiguous"


class ReplyScopeStore:
    """按 chat_key 保存当前回复范围，并在歧义时默认采取保守策略。"""

    def __init__(self, *, ttl_seconds: float = SCOPE_TTL_SECONDS) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._scopes: dict[str, ReplyScope] = {}

    def get(self, chat_key: str, *, now: float | None = None) -> ReplyScope | None:
        scope = self._scopes.get(chat_key)
        if scope is None:
            return None
        if scope.expired(now):
            self._scopes.pop(chat_key, None)
            return None
        return scope

    def begin(
        self,
        *,
        chat_key: str,
        generation: int,
        trigger_source: str,
        message_id: str,
        sender_id: str,
        sender_name: str,
        platform_userid: str,
        now: float | None = None,
    ) -> ReplyScope:
        now = time.time() if now is None else now
        existing = self.get(chat_key, now=now)
        if existing is not None:
            same_sender = bool(sender_id) and sender_id == existing.target_sender_id
            if same_sender and existing.state == "active":
                existing.generation = generation
                existing.representative_message_id = message_id or existing.representative_message_id
                existing.trigger_source = trigger_source or existing.trigger_source
                existing.expires_at = now + self.ttl_seconds
                return existing
            if existing.state != "ambiguous":
                existing.mark_ambiguous()
            existing.expires_at = now + self.ttl_seconds
            return existing

        scope = ReplyScope(
            chat_key=chat_key,
            generation=generation,
            trigger_source=trigger_source,
            first_trigger_message_id=message_id,
            representative_message_id=message_id,
            target_sender_id=sender_id,
            target_sender_name=sender_name,
            platform_userid=platform_userid,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._scopes[chat_key] = scope
        return scope

    def cancel(self, chat_key: str) -> None:
        scope = self._scopes.get(chat_key)
        if scope is not None:
            scope.cancel()

    def clear(self, chat_key: str) -> None:
        self._scopes.pop(chat_key, None)

    def clear_all(self) -> None:
        for scope in self._scopes.values():
            scope.cancel()

    def render(self, chat_key: str) -> str:
        scope = self._scopes.get(chat_key)
        if scope is not None and scope.state == "active" and scope.expired():
            self._scopes.pop(chat_key, None)
            scope = None
        if scope is None:
            return (
                "[Debounce reply scope]\n"
                '{"state":"missing"}\n'
                "本回合没有可确认的回复对象。只处理明确对 Bot 的当前请求；不要因为某人最近发言、连续发图或位于历史末尾而主动点名、@、追问或评价该成员。"
            )

        payload: dict[str, Any] = {
            "state": scope.state,
            "generation": scope.generation,
            "trigger_source": scope.trigger_source,
        }
        if scope.state == "active":
            payload["target"] = {
                "sender_id": scope.target_sender_id,
                "sender_name": scope.target_sender_name,
                "platform_userid": scope.platform_userid,
            }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if scope.state == "active":
            guidance = (
                "当前回合优先处理明确触发 Bot 的目标消息。其他成员的消息仅作背景；不能因为某人最近发言、连续发图或位于历史末尾就主动点名、@、追问或评价该成员。"
                "只有明确 @Bot、回复 Bot、称呼 Bot，或属于当前目标发送者的续写，才可视为回复对象；不必每次称呼目标姓名。"
            )
        else:
            guidance = (
                "当前回复范围存在歧义或已失效。不要指定、点名或 @ 任何成员，不要把历史中最后发言者当作回复对象；只回答当前明确的 Bot 请求。"
            )
        return f"[Debounce reply scope]\n{encoded}\n{guidance}"


__all__ = ["ReplyScope", "ReplyScopeStore", "SCOPE_TTL_SECONDS"]
