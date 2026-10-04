"""按频道管理防抖缓冲和 generation。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import Optional

from .state import ChatBuffer, MessageEnvelope, SemanticState


class BufferManager:
    """每个 chat_key 一个锁和一个缓冲区。"""

    def __init__(self) -> None:
        self._buffers: dict[str, ChatBuffer] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._generation_seed: dict[str, int] = {}
        # 频道级失效水位覆盖 sender bucket，避免失效后的旧 generation 被新
        # 消息复用。水位只单调增加，重复 invalidate 不会继续跳号。
        self._channel_generation_floor: dict[str, int] = {}

    def lock_for(self, chat_key: str) -> asyncio.Lock:
        return self._locks.setdefault(chat_key, asyncio.Lock())

    def get(self, buffer_key: str) -> Optional[ChatBuffer]:
        buffer = self._buffers.get(buffer_key)
        if buffer is not None:
            return buffer
        matches = [item for item in self._buffers.values() if item.chat_key == buffer_key]
        return matches[0] if len(matches) == 1 else None

    def has_pending(self, buffer_key: str) -> bool:
        buffer = self.get(buffer_key)
        return bool(buffer and buffer.messages)

    def next_position(self, buffer_key: str, chat_key: str | None = None) -> tuple[int, int]:
        buffer = self._buffers.get(buffer_key)
        if buffer is None:
            seed = self._generation_seed.get(buffer_key, -1)
            if chat_key:
                seed = max(seed, self._channel_generation_floor.get(chat_key, -1))
            generation = seed + 1
            return generation, 0
        return buffer.generation, len(buffer.messages)

    def invalidate_chat(self, chat_key: str, generations: Iterable[int] = ()) -> bool:
        """推进频道 generation 水位，返回是否产生了新的失效状态。"""

        candidates = [buffer.generation for buffer in self.for_chat(chat_key)]
        candidates.extend(int(generation) for generation in generations)
        if not candidates:
            return False
        previous = self._channel_generation_floor.get(chat_key, -1)
        floor = max(previous, max(candidates))
        changed = floor > previous
        self._channel_generation_floor[chat_key] = floor
        for buffer in self.for_chat(chat_key):
            self._generation_seed[buffer.buffer_key] = max(
                self._generation_seed.get(buffer.buffer_key, -1),
                floor,
            )
        return changed

    def add(
        self,
        envelope: MessageEnvelope,
        timeout_at: float,
        *,
        first_seen_at: float,
        max_wait_deadline: float,
    ) -> ChatBuffer:
        buffer = self._buffers.get(envelope.buffer_key)
        if buffer is None:
            buffer = ChatBuffer(
                buffer_key=envelope.buffer_key,
                chat_key=envelope.chat_key,
                sender_bucket=envelope.sender_bucket,
                generation=envelope.generation,
                last_update=0.0,
                timeout_at=timeout_at,
                first_seen_at=first_seen_at,
                quiet_deadline=timeout_at,
                max_wait_deadline=max_wait_deadline,
            )
            self._buffers[envelope.buffer_key] = buffer
        if buffer.generation != envelope.generation:
            raise ValueError("消息 generation 与频道缓冲不一致")
        buffer.messages.append(envelope)
        buffer.last_update = time.time()
        buffer.timeout_at = timeout_at
        buffer.quiet_deadline = timeout_at
        buffer.max_wait_deadline = max_wait_deadline
        self._generation_seed[envelope.buffer_key] = max(
            self._generation_seed.get(envelope.buffer_key, -1),
            envelope.generation,
        )
        return buffer

    def take(self, buffer_key: str, generation: int) -> Optional[ChatBuffer]:
        buffer = self._buffers.get(buffer_key)
        if buffer is None or buffer.generation != generation:
            return None
        self._buffers.pop(buffer_key, None)
        self._generation_seed[buffer_key] = max(self._generation_seed.get(buffer_key, -1), generation)
        return buffer

    def clear(self, buffer_key: str, generation: int) -> Optional[ChatBuffer]:
        return self.take(buffer_key, generation)

    def restore(
        self,
        envelopes: Iterable[MessageEnvelope],
        timeout_at: float,
        *,
        buffer_key: str,
        chat_key: str,
        sender_bucket: str,
        first_seen_at: float,
        max_wait_deadline: float,
        semantic_complete: bool | None = None,
        semantic_probability: float | None = None,
        previous_probability: float | None = None,
        probability_delta: float | None = None,
        semantic_state: SemanticState | None = None,
        semantic_checked_at: float = 0.0,
        classification_count: int = 0,
        selected_wait_seconds: float = 0.0,
        classifier_fallback: bool = False,
    ) -> None:
        grouped: dict[tuple[str, int], list[MessageEnvelope]] = {}
        for envelope in envelopes:
            grouped.setdefault((envelope.buffer_key, envelope.generation), []).append(envelope)
            self._generation_seed[envelope.buffer_key] = max(
                self._generation_seed.get(envelope.buffer_key, -1),
                envelope.generation,
            )
        for (restored_buffer_key, generation), items in grouped.items():
            items.sort(key=lambda item: item.sequence)
            first = items[0]
            buffer = ChatBuffer(
                buffer_key=restored_buffer_key or buffer_key,
                chat_key=first.chat_key or chat_key,
                sender_bucket=first.sender_bucket or sender_bucket,
                generation=generation,
                messages=items,
                last_update=0.0,
                timeout_at=timeout_at,
                first_seen_at=first_seen_at,
                quiet_deadline=timeout_at,
                max_wait_deadline=max_wait_deadline,
                semantic_complete=semantic_complete,
                semantic_probability=semantic_probability,
                previous_probability=previous_probability,
                probability_delta=probability_delta,
                semantic_state=semantic_state,
                semantic_checked_at=semantic_checked_at,
                classification_count=classification_count,
                selected_wait_seconds=selected_wait_seconds,
                classifier_fallback=classifier_fallback,
            )
            self._buffers[buffer.buffer_key] = buffer

    def for_chat(self, chat_key: str) -> list[ChatBuffer]:
        return [buffer for buffer in self._buffers.values() if buffer.chat_key == chat_key and buffer.messages]

    def snapshot(self) -> dict[str, ChatBuffer]:
        return dict(self._buffers)
