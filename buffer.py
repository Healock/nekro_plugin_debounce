"""按频道管理防抖缓冲和 generation。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import Optional

from .state import ChatBuffer, MessageEnvelope


class BufferManager:
    """每个 chat_key 一个锁和一个缓冲区。"""

    def __init__(self) -> None:
        self._buffers: dict[str, ChatBuffer] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._generation_seed: dict[str, int] = {}

    def lock_for(self, chat_key: str) -> asyncio.Lock:
        return self._locks.setdefault(chat_key, asyncio.Lock())

    def get(self, chat_key: str) -> Optional[ChatBuffer]:
        return self._buffers.get(chat_key)

    def has_pending(self, chat_key: str) -> bool:
        buffer = self._buffers.get(chat_key)
        return bool(buffer and buffer.messages)

    def next_position(self, chat_key: str) -> tuple[int, int]:
        buffer = self._buffers.get(chat_key)
        if buffer is None:
            generation = self._generation_seed.get(chat_key, -1) + 1
            return generation, 0
        return buffer.generation, len(buffer.messages)

    def add(self, envelope: MessageEnvelope, timeout_at: float) -> ChatBuffer:
        buffer = self._buffers.get(envelope.chat_key)
        if buffer is None:
            buffer = ChatBuffer(
                chat_key=envelope.chat_key,
                generation=envelope.generation,
                last_update=0.0,
                timeout_at=timeout_at,
            )
            self._buffers[envelope.chat_key] = buffer
        if buffer.generation != envelope.generation:
            raise ValueError("消息 generation 与频道缓冲不一致")
        buffer.messages.append(envelope)
        buffer.last_update = time.time()
        buffer.timeout_at = timeout_at
        self._generation_seed[envelope.chat_key] = max(
            self._generation_seed.get(envelope.chat_key, -1),
            envelope.generation,
        )
        return buffer

    def take(self, chat_key: str, generation: int) -> Optional[ChatBuffer]:
        buffer = self._buffers.get(chat_key)
        if buffer is None or buffer.generation != generation:
            return None
        self._buffers.pop(chat_key, None)
        self._generation_seed[chat_key] = max(self._generation_seed.get(chat_key, -1), generation)
        return buffer

    def clear(self, chat_key: str, generation: int) -> Optional[ChatBuffer]:
        return self.take(chat_key, generation)

    def restore(self, envelopes: Iterable[MessageEnvelope], timeout_at: float) -> None:
        grouped: dict[tuple[str, int], list[MessageEnvelope]] = {}
        for envelope in envelopes:
            grouped.setdefault((envelope.chat_key, envelope.generation), []).append(envelope)
            self._generation_seed[envelope.chat_key] = max(
                self._generation_seed.get(envelope.chat_key, -1),
                envelope.generation,
            )
        for (chat_key, generation), items in grouped.items():
            items.sort(key=lambda item: item.sequence)
            buffer = ChatBuffer(
                chat_key=chat_key,
                generation=generation,
                messages=items,
                last_update=0.0,
                timeout_at=timeout_at,
            )
            self._buffers[chat_key] = buffer

    def snapshot(self) -> dict[str, ChatBuffer]:
        return dict(self._buffers)
