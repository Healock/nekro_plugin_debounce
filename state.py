"""防抖插件的运行时状态和持久化模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JournalState(StrEnum):
    """一条 journal 记录的生命周期状态。"""

    PENDING = "pending"
    FLUSHING = "flushing"
    MANUAL_RECOVERY = "manual_recovery"
    ACKED = "acked"


class JournalRecord(BaseModel):
    """可恢复的单条用户消息。"""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    chat_key: str
    generation: int = 0
    sequence: int = 0
    state: JournalState = JournalState.PENDING
    text: str = ""
    content_data: list[dict[str, Any]] = Field(default_factory=list)
    last_message_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    sender_nickname: str = ""
    adapter_key: str = ""
    platform_userid: str = ""
    raw_cq_code: str = ""
    updated_at: float
    timeout_at: float
    retries: int = 0
    error_state: str = ""


class JournalDocument(BaseModel):
    """plugin.store 中保存的完整 JSON 文档。"""

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    records: list[JournalRecord] = Field(default_factory=list)


@dataclass(slots=True)
class MessageEnvelope:
    """缓冲区中的消息快照，不持有 Nekro 运行时对象。"""

    event_id: str
    message_id: str
    chat_key: str
    generation: int
    sequence: int
    text: str
    content_data: list[dict[str, Any]] = field(default_factory=list)
    sender_id: str = ""
    sender_name: str = ""
    sender_nickname: str = ""
    adapter_key: str = ""
    platform_userid: str = ""
    raw_cq_code: str = ""


@dataclass(slots=True)
class ChatBuffer:
    """单频道的内存缓冲。"""

    chat_key: str
    generation: int
    messages: list[MessageEnvelope] = field(default_factory=list)
    last_update: float = 0.0
    timeout_at: float = 0.0

    @property
    def record_ids(self) -> list[str]:
        return [message.event_id for message in self.messages]

    @property
    def text(self) -> str:
        return " ".join(message.text.strip() for message in self.messages if message.text.strip())
