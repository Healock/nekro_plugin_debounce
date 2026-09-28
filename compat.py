"""Nekro 消息对象与 AstrBot 配置语义之间的兼容辅助函数。"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any


SAFE_SEGMENT_TYPES = {"text", "at"}
REPLAY_MARKER = "_nekro_plugin_debounce_replay"


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


def restore_segments(data: Sequence[Mapping[str, Any]]) -> list[Any]:
    """尽量恢复成 Nekro 的消息段模型；测试环境没有 Nekro 时保留字典。"""

    try:
        from nekro_agent.schemas.chat_message import segments_from_list

        return segments_from_list([dict(item) for item in data])
    except (ImportError, ModuleNotFoundError, KeyError, TypeError, ValueError):
        return [dict(item) for item in data]


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


def is_replay_message(message: Any) -> bool:
    """判断消息是否由防抖超时流程重新提交。"""

    ext_data = getattr(message, "ext_data", None)
    return isinstance(ext_data, Mapping) and ext_data.get(REPLAY_MARKER) is True


def text_compatible(message: Any) -> bool:
    """纯文本和 AT 可以进入完整性分类器。"""

    data = getattr(message, "content_data", None) or []
    return all(segment_type_name(segment) in SAFE_SEGMENT_TYPES for segment in data)


def has_hard_boundary(message: Any) -> bool:
    data = getattr(message, "content_data", None) or []
    return bool(data) and not text_compatible(message)


def merge_text(parts: Sequence[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def merge_into_message(message: Any, envelopes: Sequence[Any]) -> None:
    """将缓冲内容按顺序写回当前 ChatMessage。"""

    current_text = str(getattr(message, "content_text", "") or "")
    text_parts = [
        str(getattr(item, "text", getattr(item, "content_text", "")) or "")
        for item in envelopes
    ]
    text_parts.append(current_text)
    merged_text = merge_text(text_parts)

    segment_dicts: list[dict[str, Any]] = []
    for item in envelopes:
        segment_dicts.extend(content_data_to_dicts(getattr(item, "content_data", [])))
    segment_dicts.extend(content_data_to_dicts(getattr(message, "content_data", [])))
    merged_segments = restore_segments(segment_dicts)
    message.content_text = merged_text
    message.content_data = merged_segments


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
