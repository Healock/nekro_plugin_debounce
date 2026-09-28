"""防抖插件生命周期和主状态机。"""

from __future__ import annotations

import time
from typing import Any

from nekro_agent.schemas.signal import MsgSignal

from .buffer import BufferManager
from .classifier import ClassifierAdapter
from .compat import (
    content_data_to_dicts,
    has_hard_boundary,
    is_replay_message,
    merge_into_message,
    merge_text,
    message_event_id,
    message_id,
    REPLAY_MARKER,
    restore_segments,
    sender_id,
    sender_name,
    usage_scope_matches,
)
from .journal import JournalError, JournalStore
from .state import ChatBuffer, JournalRecord, JournalState, MessageEnvelope
from .tasks import TaskManager


MAX_TIMEOUT_STATE_RETRIES = 3
TIMEOUT_RETRY_DELAY_SECONDS = 1.0


class DebounceRuntime:
    def __init__(self, plugin: Any, config: Any) -> None:
        self.plugin = plugin
        self.config = config
        self.buffers = BufferManager()
        self.journal = JournalStore(plugin.store)
        self.classifier = ClassifierAdapter(
            model_type=config.model_type,
            data_dir=plugin.get_plugin_data_dir(),
            logger=plugin.logger,
        )
        self.tasks = TaskManager(self._on_timeout, logger=plugin.logger)
        self._started = False
        self._stopping = False
        self._timeout_state_failures: dict[tuple[str, int], int] = {}

    @property
    def logger(self) -> Any:
        return self.plugin.logger

    async def start(self) -> None:
        if self._started:
            return
        self._stopping = False
        try:
            records = await self.journal.load()
        except JournalError as exc:
            self.logger.exception(f"[Debounce] journal 恢复失败，进入 fail-open: {exc}")
            self._started = True
            return

        pending = [record for record in records if record.state == JournalState.PENDING]
        uncertain = [record for record in records if record.state in {JournalState.FLUSHING, JournalState.MANUAL_RECOVERY}]
        if uncertain:
            self.logger.warning(
                f"[Debounce] 发现 {len(uncertain)} 条不确定 journal，标记人工恢复，不自动重复触发",
            )
            try:
                await self.journal.mark_manual_recovery(
                    [record.event_id for record in uncertain],
                    "startup_uncertain_state",
                )
            except JournalError as exc:
                self.logger.exception(f"[Debounce] 标记人工恢复失败: {exc}")

        grouped: dict[tuple[str, int], list[MessageEnvelope]] = {}
        for record in pending:
            grouped.setdefault((record.chat_key, record.generation), []).append(
                MessageEnvelope(
                    event_id=record.event_id,
                    message_id=record.last_message_id,
                    chat_key=record.chat_key,
                    generation=record.generation,
                    sequence=record.sequence,
                    text=record.text,
                    content_data=record.content_data,
                    sender_id=record.sender_id,
                    sender_name=record.sender_name,
                    sender_nickname=record.sender_nickname,
                    adapter_key=record.adapter_key,
                    platform_userid=record.platform_userid,
                    raw_cq_code=record.raw_cq_code,
                ),
            )
        pending_by_key = {
            key: [record for record in pending if (record.chat_key, record.generation) == key]
            for key in grouped
        }
        for (chat_key, generation), envelopes in grouped.items():
            timeout_at = max(record.timeout_at for record in pending_by_key[(chat_key, generation)])
            self.buffers.restore(envelopes, timeout_at)
            if self.config.timeout_seconds > 0:
                try:
                    self.tasks.schedule(chat_key, generation, timeout_at)
                except Exception as exc:
                    self.logger.exception(f"[Debounce] 恢复 timeout 任务失败: {chat_key}: {exc}")
        self._started = True

    async def stop(self) -> None:
        self._stopping = True
        await self.tasks.cancel_all()
        self._started = False

    async def handle_user_message(self, _ctx: Any, message: Any) -> MsgSignal:
        if self._stopping or not self.config.enabled or not usage_scope_matches(message, self.config.usage_scope):
            return MsgSignal.CONTINUE
        if is_replay_message(message):
            return MsgSignal.CONTINUE

        chat_key = str(getattr(message, "chat_key", "") or "")
        if not chat_key:
            return MsgSignal.CONTINUE
        lock = self.buffers.lock_for(chat_key)
        async with lock:
            current_buffer = self.buffers.get(chat_key)
            if has_hard_boundary(message):
                if current_buffer is None or not current_buffer.messages:
                    return MsgSignal.CONTINUE
                return await self._merge_and_trigger(message, current_buffer, "media_boundary")

            current_text = str(getattr(message, "content_text", "") or "")
            if not current_text.strip() and not getattr(message, "content_data", None):
                return MsgSignal.CONTINUE

            # AstrBot 在已有 pending 时会直接合并下一条文本，避免再次分类导致持续阻塞。
            if current_buffer is not None and current_buffer.messages:
                return await self._merge_and_trigger(message, current_buffer, "pending_text")

            candidate = merge_text(
                [current_text],
            )
            try:
                complete = await self.classifier.is_complete(candidate, float(self.config.send_threshold))
            except Exception as exc:
                self.logger.warning(f"[Debounce] 分类器不可用，当前消息 fail-open: {exc}")
                if current_buffer is None or not current_buffer.messages:
                    return MsgSignal.CONTINUE
                return await self._merge_and_trigger(message, current_buffer, "classifier_failure")

            if complete:
                if current_buffer is None or not current_buffer.messages:
                    return MsgSignal.FORCE_TRIGGER
                return await self._merge_and_trigger(message, current_buffer, "complete")

            generation, sequence = self.buffers.next_position(chat_key)
            event_id = message_event_id(message)
            timeout_at = time.time() + max(0, int(self.config.timeout_seconds))
            record = JournalRecord(
                event_id=event_id,
                chat_key=chat_key,
                generation=generation,
                sequence=sequence,
                text=current_text,
                content_data=content_data_to_dicts(getattr(message, "content_data", [])),
                last_message_id=message_id(message),
                sender_id=sender_id(message),
                sender_name=sender_name(message),
                sender_nickname=str(getattr(message, "sender_nickname", "") or ""),
                adapter_key=str(getattr(message, "adapter_key", "") or ""),
                platform_userid=str(getattr(message, "platform_userid", "") or ""),
                raw_cq_code=str(getattr(message, "raw_cq_code", "") or ""),
                updated_at=time.time(),
                timeout_at=timeout_at,
            )
            try:
                await self.journal.append(record)
            except JournalError as exc:
                self.logger.exception(f"[Debounce] BLOCK_ALL 前 journal 写入失败，当前消息放行: {exc}")
                return MsgSignal.CONTINUE

            envelope = MessageEnvelope(
                event_id=event_id,
                message_id=message_id(message),
                chat_key=chat_key,
                generation=generation,
                sequence=sequence,
                text=current_text,
                content_data=record.content_data,
                sender_id=record.sender_id,
                sender_name=record.sender_name,
                sender_nickname=record.sender_nickname,
                adapter_key=record.adapter_key,
                platform_userid=record.platform_userid,
                raw_cq_code=record.raw_cq_code,
            )
            self.buffers.add(envelope, timeout_at)
            if self.config.timeout_seconds > 0:
                try:
                    self.tasks.schedule(chat_key, generation, timeout_at)
                except Exception as exc:
                    self.logger.exception(f"[Debounce] timeout 创建失败，保留 journal 等待恢复: {exc}")
                    try:
                        await self.journal.transition(
                            [event_id],
                            JournalState.PENDING,
                            error_state="timeout_task_creation_failed",
                            increment_retries=True,
                        )
                    except JournalError as journal_exc:
                        self.logger.exception(f"[Debounce] timeout 失败状态写入失败: {journal_exc}")
            return MsgSignal.BLOCK_ALL

    async def _merge_and_trigger(self, message: Any, buffer: ChatBuffer, reason: str) -> MsgSignal:
        try:
            merge_into_message(message, buffer.messages)
        except Exception as exc:
            self.logger.exception(f"[Debounce] 消息合并失败，当前消息 fail-open，pending 保留: {exc}")
            try:
                await self.journal.mark_manual_recovery(buffer.record_ids, f"merge_failed:{reason}")
                self.buffers.clear(buffer.chat_key, buffer.generation)
                self.tasks.cancel(buffer.chat_key, buffer.generation)
            except JournalError as journal_exc:
                self.logger.exception(f"[Debounce] 合并失败批次无法标记人工恢复: {journal_exc}")
            return MsgSignal.CONTINUE
        self.buffers.clear(buffer.chat_key, buffer.generation)
        self.tasks.cancel(buffer.chat_key, buffer.generation)
        try:
            # 当前回调没有 after-persist API，保守标记人工恢复，禁止启动时自动重复触发。
            await self.journal.mark_manual_recovery(buffer.record_ids, f"outer_persist_unconfirmed:{reason}")
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 合并后更新 journal 失败，保留原记录: {exc}")
        return MsgSignal.FORCE_TRIGGER

    async def _on_timeout(self, chat_key: str, generation: int) -> None:
        lock = self.buffers.lock_for(chat_key)
        async with lock:
            buffer = self.buffers.get(chat_key)
            if buffer is None or buffer.generation != generation or not buffer.messages:
                return
            record_ids = buffer.record_ids
            try:
                await self.journal.mark_flushing(record_ids)
            except JournalError as exc:
                key = (chat_key, generation)
                failures = self._timeout_state_failures.get(key, 0) + 1
                self._timeout_state_failures[key] = failures
                self.logger.exception(f"[Debounce] timeout 状态写入失败，第 {failures} 次: {exc}")
                try:
                    await self.journal.transition(
                        record_ids,
                        JournalState.PENDING,
                        error_state="timeout_state_write_failed",
                        increment_retries=True,
                    )
                except JournalError as journal_exc:
                    self.logger.exception(f"[Debounce] timeout 失败状态写入失败: {journal_exc}")
                if failures <= MAX_TIMEOUT_STATE_RETRIES:
                    retry_at = time.time() + TIMEOUT_RETRY_DELAY_SECONDS * failures
                    try:
                        self.tasks.schedule(chat_key, generation, retry_at)
                    except Exception as retry_exc:
                        self.logger.exception(f"[Debounce] timeout 有限重试创建失败: {retry_exc}")
                else:
                    try:
                        await self.journal.mark_manual_recovery(record_ids, "timeout_state_write_exhausted")
                        self.buffers.clear(chat_key, generation)
                    except JournalError as journal_exc:
                        self.logger.exception(f"[Debounce] timeout 达到重试上限且无法标记人工恢复: {journal_exc}")
                return

            buffer = self.buffers.take(chat_key, generation)
            if buffer is None or not buffer.messages:
                return
            self._timeout_state_failures.pop((chat_key, generation), None)

            try:
                await self._replay_as_human_message(buffer)
            except Exception as exc:
                self.logger.exception(f"[Debounce] timeout 用户消息重放失败，保留人工恢复记录: {exc}")
                try:
                    await self.journal.mark_manual_recovery(record_ids, "timeout_human_replay_failed")
                except JournalError:
                    pass
                return

            try:
                await self.journal.acknowledge(record_ids)
            except JournalError as exc:
                self.logger.exception(f"[Debounce] timeout 已调用但 journal 清理失败: {exc}")

    async def _replay_as_human_message(self, buffer: ChatBuffer) -> None:
        """将超时缓冲重新交给用户消息入口，避免生成 SYSTEM 消息。"""

        from nekro_agent.models.db_chat_channel import DBChatChannel
        from nekro_agent.schemas.chat_message import ChatMessage, ChatType
        from nekro_agent.services.message_service import message_service

        channel = await DBChatChannel.get_channel(chat_key=buffer.chat_key)
        last_message = buffer.messages[-1]
        content_data = []
        for item in buffer.messages:
            content_data.extend(content_data_to_dicts(item.content_data))

        message = ChatMessage(
            message_id=f"debounce-{buffer.generation}-{last_message.message_id or last_message.event_id}",
            sender_id=last_message.sender_id or "0",
            sender_name=last_message.sender_name or "未知用户",
            sender_nickname=last_message.sender_nickname or last_message.sender_name or "未知用户",
            adapter_key=last_message.adapter_key or channel.adapter_key,
            platform_userid=last_message.platform_userid or "0",
            is_tome=0,
            is_recalled=False,
            chat_key=buffer.chat_key,
            chat_type=ChatType(channel.chat_type),
            content_text=buffer.text,
            content_data=restore_segments(content_data),
            raw_cq_code=last_message.raw_cq_code,
            ext_data={REPLAY_MARKER: True},
            send_timestamp=int(time.time()),
        )
        await message_service.push_human_message(
            message=message,
            trigger_agent=True,
            db_chat_channel=channel,
        )


def register_lifecycle(plugin: Any, runtime: DebounceRuntime) -> None:
    @plugin.mount_init_method()
    async def _init() -> None:
        await runtime.start()

    @plugin.mount_cleanup_method()
    async def _cleanup() -> None:
        await runtime.stop()
