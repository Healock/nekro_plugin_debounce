"""基于 plugin.store 的 JSON journal。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import Optional, Protocol

from .state import JournalDocument, JournalRecord, JournalState


JOURNAL_STORE_KEY = "debounce_journal_v1"


class JournalError(RuntimeError):
    """journal 读取或写入失败。"""


class StoreLike(Protocol):
    async def get(self, *, chat_key: str, user_key: str, store_key: str) -> Optional[str]: ...

    async def set(self, *, chat_key: str, user_key: str, store_key: str, value: str) -> int: ...


class JournalStore:
    """将全部记录保存为一个版本化 JSON 字符串。"""

    def __init__(self, store: StoreLike) -> None:
        self._store = store
        self._lock = asyncio.Lock()
        self._records: dict[str, JournalRecord] = {}
        self._loaded = False

    async def load(self) -> list[JournalRecord]:
        async with self._lock:
            try:
                raw = await self._store.get(chat_key="", user_key="", store_key=JOURNAL_STORE_KEY)
                document = JournalDocument.model_validate_json(raw) if raw else JournalDocument()
            except Exception as exc:
                raise JournalError(f"读取防抖 journal 失败: {exc}") from exc

            self._records = {record.event_id: record for record in document.records if record.state != JournalState.ACKED}
            self._loaded = True
            return list(self._records.values())

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    async def _save_locked(self) -> None:
        document = JournalDocument(records=list(self._records.values()))
        try:
            await self._store.set(
                chat_key="",
                user_key="",
                store_key=JOURNAL_STORE_KEY,
                value=document.model_dump_json(),
            )
        except Exception as exc:
            raise JournalError(f"写入防抖 journal 失败: {exc}") from exc

    async def append(self, record: JournalRecord) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            existing = self._records.get(record.event_id)
            if existing is not None:
                if existing.model_dump(exclude={"updated_at", "state", "retries", "error_state"}) != record.model_dump(
                    exclude={"updated_at", "state", "retries", "error_state"},
                ):
                    raise JournalError(f"journal event_id 冲突: {record.event_id}")
                return False
            self._records[record.event_id] = record
            try:
                await self._save_locked()
            except Exception:
                self._records.pop(record.event_id, None)
                raise
            return True

    async def transition(
        self,
        event_ids: Iterable[str],
        state: JournalState,
        *,
        error_state: Optional[str] = None,
        increment_retries: bool = False,
    ) -> None:
        await self._ensure_loaded()
        ids = list(event_ids)
        async with self._lock:
            previous: dict[str, JournalRecord] = {
                event_id: record.model_copy(deep=True)
                for event_id in ids
                if (record := self._records.get(event_id)) is not None
            }
            for event_id in ids:
                record = self._records.get(event_id)
                if record is None:
                    continue
                record.state = state
                record.updated_at = time.time()
                if error_state is not None:
                    record.error_state = error_state
                if increment_retries:
                    record.retries += 1
            try:
                await self._save_locked()
            except Exception:
                self._records.update(previous)
                raise

    async def mark_manual_recovery(self, event_ids: Iterable[str], reason: str) -> None:
        await self.transition(event_ids, JournalState.MANUAL_RECOVERY, error_state=reason, increment_retries=True)

    async def mark_flushing(self, event_ids: Iterable[str]) -> None:
        await self.transition(event_ids, JournalState.FLUSHING)

    async def cancel(self, event_ids: Iterable[str], reason: str) -> None:
        """持久化取消状态，供频道失效后阻止旧批次恢复或重放。

        取消操作是幂等的。记录保留在 journal 中作为 durable tombstone，
        因此频道恢复或进程重启时不会把旧批次当作 pending 再次调度。
        """

        await self._ensure_loaded()
        ids = list(event_ids)
        async with self._lock:
            previous: dict[str, JournalRecord] = {
                event_id: record.model_copy(deep=True)
                for event_id in ids
                if (record := self._records.get(event_id)) is not None
            }
            for event_id in ids:
                record = self._records.get(event_id)
                if record is None or record.state in {JournalState.CANCELED, JournalState.ACKED}:
                    continue
                record.state = JournalState.CANCELED
                record.release_reason = reason
                record.error_state = ""
                record.updated_at = time.time()
            try:
                await self._save_locked()
            except Exception:
                self._records.update(previous)
                raise

    async def update_batch(self, event_ids: Iterable[str], **updates: object) -> None:
        """原子更新同一缓冲批次的截止时间和语义状态。"""

        allowed = {
            "timeout_at",
            "first_seen_at",
            "quiet_deadline",
            "max_wait_deadline",
            "semantic_complete",
            "semantic_probability",
            "previous_probability",
            "probability_delta",
            "semantic_state",
            "semantic_checked_at",
            "classification_count",
            "selected_wait_seconds",
            "classifier_fallback",
            "release_reason",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"不支持的 journal 批次字段: {sorted(unknown)}")
        await self._ensure_loaded()
        ids = list(event_ids)
        async with self._lock:
            previous = {
                event_id: record.model_copy(deep=True)
                for event_id in ids
                if (record := self._records.get(event_id)) is not None
            }
            for event_id in ids:
                record = self._records.get(event_id)
                if record is None:
                    continue
                for key, value in updates.items():
                    setattr(record, key, value)
                record.updated_at = time.time()
            try:
                await self._save_locked()
            except Exception:
                self._records.update(previous)
                raise

    async def acknowledge(self, event_ids: Iterable[str]) -> None:
        """先持久化 ACK，再清理记录，避免清理失败造成丢失。"""

        await self._ensure_loaded()
        ids = list(event_ids)
        async with self._lock:
            previous: dict[str, JournalRecord] = {
                event_id: record.model_copy(deep=True)
                for event_id in ids
                if (record := self._records.get(event_id)) is not None
            }
            for event_id in ids:
                record = self._records.get(event_id)
                if record is not None:
                    record.state = JournalState.ACKED
                    record.updated_at = time.time()
            try:
                await self._save_locked()
            except Exception:
                self._records.update(previous)
                raise
            for event_id in ids:
                self._records.pop(event_id, None)
            await self._save_locked()

    async def discard(self, event_ids: Iterable[str]) -> bool:
        """先持久化 ACK，再尽力清除记录；ACK tombstone 不会在启动时恢复。"""

        await self._ensure_loaded()
        ids = list(event_ids)
        async with self._lock:
            previous = {
                event_id: record.model_copy(deep=True)
                for event_id in ids
                if (record := self._records.get(event_id)) is not None
            }
            for event_id in ids:
                record = self._records.get(event_id)
                if record is not None:
                    record.state = JournalState.ACKED
                    record.updated_at = time.time()
            try:
                await self._save_locked()
            except Exception:
                self._records.update(previous)
                raise

            for event_id in ids:
                self._records.pop(event_id, None)
            try:
                await self._save_locked()
            except JournalError:
                return False
            return True

    async def records(self) -> list[JournalRecord]:
        await self._ensure_loaded()
        async with self._lock:
            return list(self._records.values())

    async def records_for(self, chat_key: str, generation: Optional[int] = None) -> list[JournalRecord]:
        records = await self.records()
        return sorted(
            [
                record
                for record in records
                if record.chat_key == chat_key and (generation is None or record.generation == generation)
            ],
            key=lambda record: (record.generation, record.sequence),
        )
