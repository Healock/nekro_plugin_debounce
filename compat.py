"""Nekro 消息对象与 AstrBot 配置语义之间的兼容辅助函数。"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any


SAFE_SEGMENT_TYPES = {"text", "at"}


def segment_type_name(segment: Any) -> str:
    """返回稳定的消息段类型名称。"""

    if isinstance(segment, Mapping):
        value = segment.get("type", "")
    else:
        value = getattr(segment, "type", "")
    value = getattr(value, "value", value)
    return str(value).lower()


def segment_to_dict(segment: Any) -> dict[str, Any]:
    """把 Nekro 消息段转换成可 JSON 序列化的字典。"""

    if isinstance(segment, Mapping):
        return dict(segment)
    model_dump = getattr(segment, "model_dump", None)
    if callable(model_dump):
        try:
            return dict(model_dump(mode="json"))
        except TypeError:
            return dict(model_dump())
    to_dict = getattr(segment, "dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        "type": segment_type_name(segment),
        "text": str(getattr(segment, "text", "")),
    }


def content_data_to_dicts(content_data: Sequence[Any] | None) -> list[dict[str, Any]]:
    return [segment_to_dict(segment) for segment in (content_data or [])]


def message_event_id(message: Any) -> str:
    message_id = str(getattr(message, "message_id", "") or "").strip()
    if not message_id:
        return f"generated-{uuid.uuid4().hex}"
    chat_key = str(getattr(message, "chat_key", "") or "").strip()
    return f"{chat_key}:{message_id}" if chat_key else message_id


def message_id(message: Any) -> str:
    return str(getattr(message, "message_id", "") or "")


def sender_id(message: Any) -> str:
    return str(getattr(message, "sender_id", "") or "")


def sender_name(message: Any) -> str:
    return str(getattr(message, "sender_name", "") or getattr(message, "sender_nickname", "") or "")


def sender_bucket(message: Any) -> str:
    """返回稳定的发送者分桶标识，不改变真实频道标识。"""

    for value in (
        getattr(message, "sender_id", ""),
        getattr(message, "platform_userid", ""),
        getattr(message, "sender_name", ""),
        getattr(message, "sender_nickname", ""),
    ):
        normalized = str(value or "").strip()
        if normalized:
            return normalized
    return "unknown"


def buffer_key(chat_key: str, sender_key: str) -> str:
    """生成仅供插件内部使用的频道与发送者复合键。"""

    return f"{chat_key}\x1f{sender_key}"


def text_compatible(message: Any) -> bool:
    """纯文本和 AT 可以进入完整性分类器。"""

    data = getattr(message, "content_data", None) or []
    return all(segment_type_name(segment) in SAFE_SEGMENT_TYPES for segment in data)


def has_hard_boundary(message: Any) -> bool:
    data = getattr(message, "content_data", None) or []
    return bool(data) and not text_compatible(message)


def usage_scope_matches(message: Any, usage_scope: str) -> bool:
    """保留 AstrBot 的 both/group/private 语义。"""

    if usage_scope == "both":
        return True
    chat_type = getattr(message, "chat_type", "")
    chat_type = str(getattr(chat_type, "value", chat_type)).lower()
    if usage_scope == "group":
        return chat_type == "group"
    if usage_scope == "private":
        return chat_type == "private"
    return False
