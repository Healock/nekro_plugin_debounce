"""防抖插件生命周期和混合状态机。"""

from __future__ import annotations

import time
from typing import Any

from nekro_agent.schemas.signal import MsgSignal

from .buffer import BufferManager
from .classifier import ClassificationResult, ClassifierAdapter
from .compat import (
    REPLAY_MARKER,
    content_data_to_dicts,
    has_hard_boundary,
    is_replay_message,
    merge_into_message,
    merge_message_content,
    message_event_id,
    message_id,
    restore_segments,
    sender_id,
    sender_name,
    usage_scope_matches,
)
from .journal import JournalError, JournalStore
from .state import ChatBuffer, JournalRecord, JournalState, MessageEnvelope, SemanticState
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
            debug_logging=config.debug_logging,
        )
        self.tasks = TaskManager(self._on_timeout, logger=plugin.logger)
        self._started = False
        self._stopping = False
        self._timeout_state_failures: dict[tuple[str, int], int] = {}

    @property
    def logger(self) -> Any:
        return self.plugin.logger

    @property
    def quiet_seconds(self) -> float:
        return max(0.0, float(self.config.timeout_seconds))

    @property
    def high_confidence_quiet_seconds(self) -> float:
        if self.quiet_seconds <= 0:
            return 0.0
        return min(self.quiet_seconds, float(self.config.high_confidence_timeout_seconds))

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

        grouped: dict[tuple[str, int], list[JournalRecord]] = {}
        for record in pending:
            grouped.setdefault((record.chat_key, record.generation), []).append(record)
        for (chat_key, generation), batch in grouped.items():
            batch.sort(key=lambda item: item.sequence)
            envelopes = [
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
                )
                for record in batch
            ]
            first_seen_at = min(record.first_seen_at or record.updated_at for record in batch)
            max_wait_deadline = max(record.max_wait_deadline for record in batch)
            if max_wait_deadline <= 0:
                max_wait_deadline = first_seen_at + float(self.config.max_wait_seconds)
            quiet_deadline = max(record.quiet_deadline or record.timeout_at for record in batch)
            quiet_deadline = min(quiet_deadline or max_wait_deadline, max_wait_deadline)
            latest = batch[-1]
            self.buffers.restore(
                envelopes,
                quiet_deadline,
                first_seen_at=first_seen_at,
                max_wait_deadline=max_wait_deadline,
                semantic_complete=latest.semantic_complete,
                semantic_probability=latest.semantic_probability,
                previous_probability=latest.previous_probability,
                probability_delta=latest.probability_delta,
                semantic_state=latest.semantic_state,
                semantic_checked_at=latest.semantic_checked_at,
                classification_count=latest.classification_count,
                selected_wait_seconds=latest.selected_wait_seconds,
                classifier_fallback=any(record.classifier_fallback for record in batch),
            )
            try:
                self.tasks.schedule(chat_key, generation, quiet_deadline)
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

            buffer = await self._buffer_message(message, current_buffer)
            if buffer is None:
                return MsgSignal.CONTINUE
            if buffer.classifier_fallback:
                return MsgSignal.BLOCK_ALL

            result = await self._classify_buffer(buffer)
            if result is not None and not await self._apply_wait_policy(buffer):
                return MsgSignal.CONTINUE
            return MsgSignal.BLOCK_ALL

    async def _buffer_message(self, message: Any, current_buffer: ChatBuffer | None) -> ChatBuffer | None:
        """持久化一条消息，重置静默窗口但不延长最大等待时间。"""

        chat_key = str(getattr(message, "chat_key", "") or "")
        now = time.time()
        if current_buffer is None or not current_buffer.messages:
            generation, sequence = self.buffers.next_position(chat_key)
            first_seen_at = now
            max_wait_deadline = now + float(self.config.max_wait_seconds)
            classifier_fallback = False
            semantic_complete = None
            semantic_probability = None
            previous_probability = None
            probability_delta = None
            semantic_state = None
            semantic_checked_at = 0.0
            classification_count = 0
        else:
            generation = current_buffer.generation
            sequence = len(current_buffer.messages)
            first_seen_at = current_buffer.first_seen_at or now
            max_wait_deadline = current_buffer.max_wait_deadline or first_seen_at + float(self.config.max_wait_seconds)
            classifier_fallback = current_buffer.classifier_fallback
            semantic_complete = current_buffer.semantic_complete
            semantic_probability = current_buffer.semantic_probability
            previous_probability = current_buffer.previous_probability
            probability_delta = current_buffer.probability_delta
            semantic_state = current_buffer.semantic_state
            semantic_checked_at = current_buffer.semantic_checked_at
            classification_count = current_buffer.classification_count
        quiet_deadline = min(now + self.quiet_seconds, max_wait_deadline)
        event_id = message_event_id(message)
        record = JournalRecord(
            event_id=event_id,
            chat_key=chat_key,
            generation=generation,
            sequence=sequence,
            text=str(getattr(message, "content_text", "") or ""),
            content_data=content_data_to_dicts(getattr(message, "content_data", [])),
            last_message_id=message_id(message),
            sender_id=sender_id(message),
            sender_name=sender_name(message),
            sender_nickname=str(getattr(message, "sender_nickname", "") or ""),
            adapter_key=str(getattr(message, "adapter_key", "") or ""),
            platform_userid=str(getattr(message, "platform_userid", "") or ""),
            raw_cq_code=str(getattr(message, "raw_cq_code", "") or ""),
            updated_at=now,
            timeout_at=quiet_deadline,
            first_seen_at=first_seen_at,
            quiet_deadline=quiet_deadline,
            max_wait_deadline=max_wait_deadline,
            semantic_complete=semantic_complete,
            semantic_probability=semantic_probability,
            previous_probability=previous_probability,
            probability_delta=probability_delta,
            semantic_state=semantic_state,
            semantic_checked_at=semantic_checked_at,
            classification_count=classification_count,
            selected_wait_seconds=self.quiet_seconds,
            classifier_fallback=classifier_fallback,
        )
        try:
            await self.journal.append(record)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] BLOCK_ALL 前 journal 写入失败，当前消息放行: {exc}")
            return None

        envelope = MessageEnvelope(
            event_id=event_id,
            message_id=message_id(message),
            chat_key=chat_key,
            generation=generation,
            sequence=sequence,
            text=record.text,
            content_data=record.content_data,
            sender_id=record.sender_id,
            sender_name=record.sender_name,
            sender_nickname=record.sender_nickname,
            adapter_key=record.adapter_key,
            platform_userid=record.platform_userid,
            raw_cq_code=record.raw_cq_code,
        )
        try:
            buffer = self.buffers.add(
                envelope,
                quiet_deadline,
                first_seen_at=first_seen_at,
                max_wait_deadline=max_wait_deadline,
            )
        except Exception as exc:
            self.logger.exception(f"[Debounce] 消息加入内存缓冲失败，当前消息放行: {exc}")
            try:
                await self.journal.mark_manual_recovery(
                    [event_id] if current_buffer is None else current_buffer.record_ids + [event_id],
                    "buffer_append_failed",
                )
            except JournalError:
                pass
            self.buffers.clear(chat_key, generation)
            self.tasks.cancel(chat_key, generation)
            return None
        buffer.classifier_fallback = classifier_fallback
        buffer.semantic_complete = semantic_complete
        buffer.semantic_probability = semantic_probability
        buffer.previous_probability = previous_probability
        buffer.probability_delta = probability_delta
        buffer.semantic_state = semantic_state
        buffer.semantic_checked_at = semantic_checked_at
        buffer.classification_count = classification_count
        buffer.selected_wait_seconds = self.quiet_seconds
        try:
            await self.journal.update_batch(
                buffer.record_ids,
                first_seen_at=first_seen_at,
                quiet_deadline=quiet_deadline,
                timeout_at=quiet_deadline,
                max_wait_deadline=max_wait_deadline,
                selected_wait_seconds=self.quiet_seconds,
                classifier_fallback=classifier_fallback,
            )
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 保存批次截止时间失败，当前消息放行: {exc}")
            try:
                await self.journal.mark_manual_recovery(buffer.record_ids, "batch_deadline_persist_failed")
            except JournalError:
                pass
            self.buffers.clear(chat_key, generation)
            self.tasks.cancel(chat_key, generation)
            return None
        try:
            self.tasks.schedule(chat_key, generation, quiet_deadline)
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
        return buffer

    async def _classify_buffer(self, buffer: ChatBuffer) -> ClassificationResult | None:
        previous_probability = buffer.semantic_probability
        try:
            result = await self.classifier.classify(
                buffer.text,
                float(self.config.send_threshold),
                float(self.config.high_confidence_threshold),
            )
        except Exception as exc:
            buffer.classifier_fallback = True
            buffer.semantic_complete = None
            buffer.semantic_probability = None
            buffer.previous_probability = previous_probability
            buffer.probability_delta = None
            buffer.semantic_state = None
            buffer.selected_wait_seconds = self.quiet_seconds
            self.logger.warning(f"[Debounce] 当前批次退化为时间防抖: {exc}")
            try:
                await self.journal.update_batch(
                    buffer.record_ids,
                    classifier_fallback=True,
                    semantic_complete=None,
                    semantic_probability=None,
                    previous_probability=previous_probability,
                    probability_delta=None,
                    semantic_state=None,
                    semantic_checked_at=time.time(),
                    selected_wait_seconds=self.quiet_seconds,
                )
            except JournalError as journal_exc:
                self.logger.exception(f"[Debounce] 保存分类器降级状态失败: {journal_exc}")
            return None

        checked_at = time.time()
        state = result.semantic_state or self._semantic_state(result.probability)
        buffer.semantic_complete = result.complete
        buffer.semantic_probability = result.probability
        buffer.previous_probability = previous_probability
        buffer.probability_delta = (
            result.probability - previous_probability if previous_probability is not None else None
        )
        buffer.semantic_state = state
        buffer.semantic_checked_at = checked_at
        buffer.classification_count += 1
        try:
            await self.journal.update_batch(
                buffer.record_ids,
                semantic_complete=result.complete,
                semantic_probability=result.probability,
                previous_probability=buffer.previous_probability,
                probability_delta=buffer.probability_delta,
                semantic_state=state,
                semantic_checked_at=checked_at,
                classification_count=buffer.classification_count,
                classifier_fallback=False,
            )
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 保存语义判定状态失败: {exc}")
        if self.config.debug_logging:
            self.logger.info(
                f"[Debounce] chat={self._debug_chat_key(buffer.chat_key)} "
                f"text_length={len(buffer.text)} messages={len(buffer.messages)} "
                f"previous_probability={self._format_probability(buffer.previous_probability)} "
                f"probability={result.probability:.4f} "
                f"probability_delta={self._format_probability(buffer.probability_delta)} "
                f"threshold={float(self.config.send_threshold):.4f} "
                f"high_threshold={float(self.config.high_confidence_threshold):.4f} "
                f"state={state.value} first_message={len(buffer.messages) == 1} "
                f"elapsed={max(0.0, checked_at - buffer.first_seen_at):.1f}s",
            )
        return result

    def _semantic_state(self, probability: float) -> SemanticState:
        if probability < float(self.config.send_threshold):
            return SemanticState.INCOMPLETE
        if probability >= float(self.config.high_confidence_threshold):
            return SemanticState.COMPLETE_HIGH
        return SemanticState.COMPLETE_NORMAL

    @staticmethod
    def _format_probability(probability: float | None) -> str:
        return "none" if probability is None else f"{probability:.4f}"

    def _wait_seconds_for(self, buffer: ChatBuffer) -> tuple[float, str]:
        if len(buffer.messages) > 1 and buffer.semantic_state == SemanticState.COMPLETE_HIGH:
            return self.high_confidence_quiet_seconds, "semantic_complete_high_confidence_short_wait"
        if buffer.semantic_state == SemanticState.INCOMPLETE:
            return self.quiet_seconds, "semantic_incomplete_wait"
        return self.quiet_seconds, "semantic_complete_normal_wait"

    async def _apply_wait_policy(self, buffer: ChatBuffer) -> bool:
        wait_seconds, status = self._wait_seconds_for(buffer)
        now = time.time()
        deadline = min(now + wait_seconds, buffer.max_wait_deadline)
        buffer.selected_wait_seconds = wait_seconds
        buffer.quiet_deadline = deadline
        buffer.timeout_at = deadline
        try:
            await self.journal.update_batch(
                buffer.record_ids,
                timeout_at=deadline,
                quiet_deadline=deadline,
                selected_wait_seconds=wait_seconds,
            )
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 保存等待窗口失败，当前消息放行: {exc}")
            try:
                await self.journal.mark_manual_recovery(buffer.record_ids, "wait_policy_persist_failed")
            except JournalError:
                pass
            self.buffers.clear(buffer.chat_key, buffer.generation)
            self.tasks.cancel(buffer.chat_key, buffer.generation)
            return False
        try:
            self.tasks.schedule(buffer.chat_key, buffer.generation, deadline)
        except Exception as exc:
            self.logger.exception(f"[Debounce] 更新 timeout 任务失败，保留 journal 等待恢复: {exc}")
        if self.config.debug_logging:
            self.logger.info(
                f"[Debounce] status={status} selected_wait_seconds={wait_seconds:.1f} "
                f"elapsed={max(0.0, now - buffer.first_seen_at):.1f}s",
            )
        return True

    def _debug_chat_key(self, chat_key: str) -> str:
        return f"{chat_key[:4]}...{chat_key[-4:]}" if len(chat_key) > 8 else chat_key

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
        try:
            await self.journal.update_batch(buffer.record_ids, release_reason=reason)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 保存释放原因失败: {exc}")
        self.buffers.clear(buffer.chat_key, buffer.generation)
        self.tasks.cancel(buffer.chat_key, buffer.generation)
        try:
            await self.journal.mark_manual_recovery(buffer.record_ids, f"outer_persist_unconfirmed:{reason}")
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 合并后更新 journal 失败，保留原记录: {exc}")
        if self.config.debug_logging:
            self.logger.info(f"[Debounce] release_reason={reason} text_length={len(getattr(message, 'content_text', '') or '')}")
        return MsgSignal.FORCE_TRIGGER

    async def _on_timeout(self, chat_key: str, generation: int) -> None:
        lock = self.buffers.lock_for(chat_key)
        async with lock:
            buffer = self.buffers.get(chat_key)
            if buffer is None or buffer.generation != generation or not buffer.messages:
                return
            now = time.time()
            if now + 0.01 < buffer.quiet_deadline:
                self.tasks.schedule(chat_key, generation, buffer.quiet_deadline)
                return

            reason: str | None = None
            if buffer.classifier_fallback:
                reason = "classifier_unavailable_time_fallback"
            else:
                result = await self._classify_buffer(buffer)
                finished_at = time.time()
                if finished_at >= buffer.max_wait_deadline:
                    reason = "max_wait_fallback"
                elif result is None:
                    reason = "classifier_unavailable_time_fallback"
                elif result.complete:
                    reason = "quiet_and_complete"
                else:
                    next_deadline = min(finished_at + self.quiet_seconds, buffer.max_wait_deadline)
                    buffer.quiet_deadline = next_deadline
                    buffer.timeout_at = next_deadline
                    buffer.selected_wait_seconds = self.quiet_seconds
                    try:
                        await self.journal.update_batch(
                            buffer.record_ids,
                            timeout_at=next_deadline,
                            quiet_deadline=next_deadline,
                            selected_wait_seconds=self.quiet_seconds,
                        )
                    except JournalError as exc:
                        self.logger.exception(f"[Debounce] 保存下一次静默截止时间失败: {exc}")
                    self.tasks.schedule(chat_key, generation, next_deadline)
                    if self.config.debug_logging:
                        self.logger.info(
                            f"[Debounce] release_reason=semantic_incomplete_wait "
                            f"elapsed={max(0.0, now - buffer.first_seen_at):.1f}s",
                        )
                    return

            await self._flush_after_timeout(buffer, reason or "timeout")

    async def _flush_after_timeout(self, buffer: ChatBuffer, reason: str) -> None:
        record_ids = buffer.record_ids
        try:
            await self.journal.update_batch(record_ids, release_reason=reason)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 保存 timeout 释放原因失败: {exc}")
        try:
            await self.journal.mark_flushing(record_ids)
        except JournalError as exc:
            key = (buffer.chat_key, buffer.generation)
            failures = self._timeout_state_failures.get(key, 0) + 1
            self._timeout_state_failures[key] = failures
            self.logger.exception(f"[Debounce] timeout 状态写入失败，第 {failures} 次: {exc}")
            if failures <= MAX_TIMEOUT_STATE_RETRIES:
                retry_at = time.time() + TIMEOUT_RETRY_DELAY_SECONDS * failures
                self.tasks.schedule(buffer.chat_key, buffer.generation, retry_at)
            else:
                try:
                    await self.journal.mark_manual_recovery(record_ids, "timeout_state_write_exhausted")
                    self.buffers.clear(buffer.chat_key, buffer.generation)
                except JournalError as journal_exc:
                    self.logger.exception(f"[Debounce] timeout 达到重试上限且无法标记人工恢复: {journal_exc}")
            return

        taken = self.buffers.take(buffer.chat_key, buffer.generation)
        if taken is None:
            return
        try:
            await self._replay_as_human_message(taken)
        except Exception as exc:
            self.logger.exception(f"[Debounce] timeout 用户消息重放失败，保留人工恢复记录: {exc}")
            try:
                await self.journal.mark_manual_recovery(record_ids, f"timeout_human_replay_failed:{reason}")
            except JournalError:
                pass
            return

        try:
            await self.journal.acknowledge(record_ids)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] timeout 已调用但 journal 清理失败: {exc}")
        if self.config.debug_logging:
            self.logger.info(f"[Debounce] release_reason={reason} text_length={len(taken.text)}")

    async def _replay_as_human_message(self, buffer: ChatBuffer) -> None:
        """将超时缓冲重新交给用户消息入口，避免生成 SYSTEM 消息。"""

        from nekro_agent.models.db_chat_channel import DBChatChannel
        from nekro_agent.schemas.chat_message import ChatMessage, ChatType
        from nekro_agent.services.message_service import message_service

        channel = await DBChatChannel.get_channel(chat_key=buffer.chat_key)
        last_message = buffer.messages[-1]
        content_text, content_data = merge_message_content(
            [(item.text, item.content_data) for item in buffer.messages],
        )
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
            content_text=content_text,
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
