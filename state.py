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


class SemanticState(StrEnum):
    """累计文本当前的语义完整性状态。"""

    INCOMPLETE = "incomplete"
    COMPLETE_NORMAL = "complete_normal"
    COMPLETE_HIGH = "complete_high"


class JournalRecord(BaseModel):
    """可恢复的单条用户消息。"""

    model_config = ConfigDict(extra="ignore")

    event_id: str
    chat_key: str
    sender_bucket: str = ""
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
    received_at: float = 0.0
    updated_at: float
    timeout_at: float
    first_seen_at: float = 0.0
    quiet_deadline: float = 0.0
    max_wait_deadline: float = 0.0
    semantic_complete: bool | None = None
    semantic_probability: float | None = None
    previous_probability: float | None = None
    probability_delta: float | None = None
    semantic_state: SemanticState | None = None
    semantic_checked_at: float = 0.0
    classification_count: int = 0
    selected_wait_seconds: float = 0.0
    classifier_fallback: bool = False
    release_reason: str = ""
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
    sender_bucket: str = ""
    content_data: list[dict[str, Any]] = field(default_factory=list)
    sender_id: str = ""
    sender_name: str = ""
    sender_nickname: str = ""
    adapter_key: str = ""
    platform_userid: str = ""
    raw_cq_code: str = ""
    received_at: float = 0.0

    @property
    def buffer_key(self) -> str:
        return f"{self.chat_key}\x1f{self.sender_bucket}"


@dataclass(slots=True)
class ChatBuffer:
    """单频道的内存缓冲。"""

    chat_key: str
    generation: int
    buffer_key: str = ""
    sender_bucket: str = ""
    messages: list[MessageEnvelope] = field(default_factory=list)
    last_update: float = 0.0
    timeout_at: float = 0.0
    first_seen_at: float = 0.0
    quiet_deadline: float = 0.0
    max_wait_deadline: float = 0.0
    semantic_complete: bool | None = None
    semantic_probability: float | None = None
    previous_probability: float | None = None
    probability_delta: float | None = None
    semantic_state: SemanticState | None = None
    semantic_checked_at: float = 0.0
    classification_count: int = 0
    selected_wait_seconds: float = 0.0
    classifier_fallback: bool = False

    @property
    def record_ids(self) -> list[str]:
        return [message.event_id for message in self.messages]

    @property
    def text(self) -> str:
        return " ".join(message.text.strip() for message in self.messages if message.text.strip())

    @property
    def recent_message_intervals(self) -> list[float]:
        timestamps = [message.received_at for message in self.messages if message.received_at > 0]
        return [later - earlier for earlier, later in zip(timestamps, timestamps[1:]) if later >= earlier]
