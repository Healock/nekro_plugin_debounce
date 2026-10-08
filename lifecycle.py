"""防抖插件生命周期和混合状态机。"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from nekro_agent.schemas.signal import MsgSignal

from .buffer import BufferManager
from .classifier import ClassificationResult, ClassifierAdapter
from .compat import (
    buffer_key,
    content_data_to_dicts,
    has_hard_boundary,
    message_event_id,
    message_id,
    sender_id,
    sender_name,
    sender_bucket,
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
        self._preload_task: asyncio.Task[None] | None = None
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
        self._preload_task = asyncio.create_task(self._preload_classifier(), name="debounce-classifier-preload")
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

        grouped: dict[tuple[str, str, int], list[JournalRecord]] = {}
        for record in pending:
            grouped.setdefault(
                (record.chat_key, record.sender_bucket or record.sender_id or "unknown", record.generation),
                [],
            ).append(record)
        for (chat_key, sender_key, generation), batch in grouped.items():
            batch.sort(key=lambda item: item.sequence)
            internal_key = buffer_key(chat_key, sender_key)
            envelopes = [
                MessageEnvelope(
                    event_id=record.event_id,
                    message_id=record.last_message_id,
                    chat_key=record.chat_key,
                    sender_bucket=record.sender_bucket or record.sender_id or sender_key,
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
                    received_at=record.received_at,
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
                buffer_key=internal_key,
                chat_key=chat_key,
                sender_bucket=sender_key,
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
                self.tasks.schedule(internal_key, generation, quiet_deadline)
            except Exception as exc:
                self.logger.exception(f"[Debounce] 恢复 timeout 任务失败: {chat_key}: {exc}")
        self._started = True

    async def _preload_classifier(self) -> None:
        started_at = time.monotonic()
        self.logger.info("[Debounce] 插件初始化阶段开始预加载语义分类器")
        try:
            await self.classifier.preload()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning(
                f"[Debounce] 语义分类器预加载失败，后续批次使用时间防抖: {exc} "
                f"elapsed={time.monotonic() - started_at:.1f}s",
            )
            return
        self.logger.info(f"[Debounce] 语义分类器预加载完成 elapsed={time.monotonic() - started_at:.1f}s")

    async def stop(self) -> None:
        self._stopping = True
        if self._preload_task is not None and not self._preload_task.done():
            self._preload_task.cancel()
            await asyncio.gather(self._preload_task, return_exceptions=True)
        self._preload_task = None
        await self.tasks.cancel_all()
        self._started = False

    async def reset_channel(self, ctx: Any) -> None:
        chat_key = str(getattr(ctx, "chat_key", "") or "")
        if not chat_key:
            return
        async with self.buffers.lock_for(chat_key):
            buffers = self.buffers.for_chat(chat_key)
            generations = [buffer.generation for buffer in buffers]
            try:
                generations.extend(record.generation for record in await self.journal.records_for(chat_key))
            except JournalError as exc:
                self.logger.warning(
                    f"[Debounce] reset 读取历史 generation 失败，使用内存批次水位: "
                    f"chat={self._debug_chat_key(chat_key)}: {exc}",
                )
            if not buffers:
                self.buffers.invalidate_chat(chat_key, generations)
                return
            count = sum(len(buffer.messages) for buffer in buffers)
            for buffer in buffers:
                record_ids = buffer.record_ids
                try:
                    cleaned = await self.journal.discard(record_ids)
                    if not cleaned:
                        self.logger.warning(
                            f"[Debounce] reset 已持久化 ACK，但 journal 物理清理失败: chat={chat_key} "
                            f"sender={buffer.sender_bucket} generation={buffer.generation}",
                        )
                except JournalError as exc:
                    self.logger.exception(
                        f"[Debounce] reset 批次 ACK 失败，尝试标记人工恢复: chat={chat_key} "
                        f"sender={buffer.sender_bucket} generation={buffer.generation}: {exc}",
                    )
                    try:
                        await self.journal.mark_manual_recovery(record_ids, "channel_reset")
                    except JournalError as recovery_exc:
                        self.logger.critical(
                            f"[Debounce] reset journal 无法持久化取消状态，需人工检查: chat={chat_key} "
                            f"sender={buffer.sender_bucket} generation={buffer.generation}: {recovery_exc}",
                        )
                self.tasks.cancel(buffer.buffer_key, buffer.generation)
                self.buffers.clear(buffer.buffer_key, buffer.generation)
            # reset 后推进频道水位，避免新批次复用已取消批次的 generation。
            self.buffers.invalidate_chat(chat_key, generations)
            self.logger.info(
                f"[Debounce] reset 已取消 pending 批次: chat={chat_key} messages={count} "
                f"batches={len(buffers)}",
            )

    async def handle_user_message(self, _ctx: Any, message: Any) -> MsgSignal:
        if self._stopping or not self.config.enabled or not usage_scope_matches(message, self.config.usage_scope):
            return MsgSignal.CONTINUE
        chat_key = str(getattr(message, "chat_key", "") or "")
        if not chat_key:
            return MsgSignal.CONTINUE
        internal_key = buffer_key(chat_key, sender_bucket(message))
        lock = self.buffers.lock_for(chat_key)
        async with lock:
            current_buffer = self.buffers.get(internal_key)
            trigger_match, trigger_source = await self._trigger_match(_ctx, message)
            asyncio.create_task(
                self._observe_trigger_scope(_ctx, message, current_buffer, trigger_match, trigger_source),
                name="debounce-trigger-scope-observation",
            )
            if has_hard_boundary(message):
                if current_buffer is None or not current_buffer.messages:
                    return MsgSignal.CONTINUE
                return await self._merge_and_trigger(message, current_buffer, "media_boundary")

            current_text = str(getattr(message, "content_text", "") or "")
            if not current_text.strip() and not getattr(message, "content_data", None):
                return MsgSignal.CONTINUE

            # 群聊只接管已触发消息，以及同一发送者已经启动的批次。
            # 未触发的首条消息必须继续走核心的正常入库流程，不能被防抖插件阻止。
            if self._is_group_message(message) and current_buffer is None and not trigger_match:
                return MsgSignal.CONTINUE

            buffer = await self._buffer_message(message, current_buffer, internal_key)
            if buffer is None:
                return MsgSignal.CONTINUE
            preload_pending = self._preload_task is not None and not self._preload_task.done()
            preload_failed = self.classifier.load_error is not None
            if not self.classifier.is_ready and (preload_pending or preload_failed):
                buffer.classifier_fallback = True
                try:
                    await self.journal.update_batch(buffer.record_ids, classifier_fallback=True)
                except JournalError as exc:
                    self.logger.exception(f"[Debounce] 保存预加载降级状态失败: {exc}")
                    try:
                        await self.journal.mark_manual_recovery(buffer.record_ids, "preload_state_persist_failed")
                    except JournalError:
                        pass
                    self.buffers.clear(buffer.buffer_key, buffer.generation)
                    self.tasks.cancel(buffer.buffer_key, buffer.generation)
                    return MsgSignal.CONTINUE
                if not await self._apply_wait_policy(buffer):
                    return MsgSignal.CONTINUE
                return MsgSignal.BLOCK_TRIGGER
            if buffer.classifier_fallback:
                return MsgSignal.BLOCK_TRIGGER

            result = await self._classify_buffer(buffer)
            if result is not None and not await self._apply_wait_policy(buffer):
                return MsgSignal.CONTINUE
            return MsgSignal.BLOCK_TRIGGER

    async def _buffer_message(
        self,
        message: Any,
        current_buffer: ChatBuffer | None,
        internal_key: str,
    ) -> ChatBuffer | None:
        """持久化一条消息，重置静默窗口但不延长最大等待时间。"""

        chat_key = str(getattr(message, "chat_key", "") or "")
        sender_key = sender_bucket(message)
        now = time.time()
        if current_buffer is None or not current_buffer.messages:
            generation, sequence = self.buffers.next_position(internal_key, chat_key)
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
            sender_bucket=sender_key,
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
            received_at=now,
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
            self.logger.exception(f"[Debounce] BLOCK_TRIGGER 前 journal 写入失败，当前消息放行: {exc}")
            return None

        envelope = MessageEnvelope(
            event_id=event_id,
            message_id=message_id(message),
            chat_key=chat_key,
            sender_bucket=sender_key,
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
            received_at=record.received_at,
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
            self.buffers.clear(internal_key, generation)
            self.tasks.cancel(internal_key, generation)
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
            self.buffers.clear(internal_key, generation)
            self.tasks.cancel(internal_key, generation)
            return None
        try:
            self.tasks.schedule(internal_key, generation, quiet_deadline)
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

    def _wait_seconds_for(self, buffer: ChatBuffer) -> tuple[float, str, float | None]:
        if len(buffer.messages) > 1 and buffer.semantic_state == SemanticState.COMPLETE_HIGH:
            recent_intervals = buffer.recent_message_intervals[-3:]
            cadence_interval = max(recent_intervals, default=0.0)
            cadence_wait = (
                cadence_interval * float(self.config.cadence_multiplier)
                + float(self.config.cadence_margin_seconds)
                if cadence_interval > 0
                else 0.0
            )
            wait_seconds = max(self.high_confidence_quiet_seconds, cadence_wait)
            return (
                min(wait_seconds, self.quiet_seconds),
                "semantic_complete_high_confidence_adaptive_wait",
                cadence_interval or None,
            )
        if buffer.semantic_state == SemanticState.INCOMPLETE:
            return self.quiet_seconds, "semantic_incomplete_wait", None
        return self.quiet_seconds, "semantic_complete_normal_wait", None

    async def _apply_wait_policy(self, buffer: ChatBuffer) -> bool:
        wait_seconds, status, cadence_interval = self._wait_seconds_for(buffer)
        now = time.time()
        deadline = min(now + wait_seconds, buffer.max_wait_deadline)
        selected_wait_seconds = max(0.0, deadline - now)
        buffer.selected_wait_seconds = selected_wait_seconds
        buffer.quiet_deadline = deadline
        buffer.timeout_at = deadline
        try:
            await self.journal.update_batch(
                buffer.record_ids,
                timeout_at=deadline,
                quiet_deadline=deadline,
                selected_wait_seconds=selected_wait_seconds,
            )
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 保存等待窗口失败，当前消息放行: {exc}")
            try:
                await self.journal.mark_manual_recovery(buffer.record_ids, "wait_policy_persist_failed")
            except JournalError:
                pass
            self.buffers.clear(buffer.buffer_key, buffer.generation)
            self.tasks.cancel(buffer.buffer_key, buffer.generation)
            return False
        try:
            self.tasks.schedule(buffer.buffer_key, buffer.generation, deadline)
        except Exception as exc:
            self.logger.exception(f"[Debounce] 更新 timeout 任务失败，保留 journal 等待恢复: {exc}")
        if self.config.debug_logging:
            self.logger.info(
                f"[Debounce] status={status} cadence_interval_seconds="
                f"{cadence_interval if cadence_interval is not None else 'none'} "
                f"cadence_multiplier={float(self.config.cadence_multiplier):.2f} "
                f"cadence_margin_seconds={float(self.config.cadence_margin_seconds):.1f} "
                f"selected_wait_seconds={selected_wait_seconds:.1f} "
                f"elapsed={max(0.0, now - buffer.first_seen_at):.1f}s",
            )
        return True

    def _debug_chat_key(self, chat_key: str) -> str:
        return f"{chat_key[:4]}...{chat_key[-4:]}" if len(chat_key) > 8 else chat_key

    @staticmethod
    def _is_group_message(message: Any) -> bool:
        chat_type = getattr(message, "chat_type", "")
        chat_type = str(getattr(chat_type, "value", chat_type)).lower()
        return chat_type == "group"

    @staticmethod
    def _shadow_trigger_match(message: Any, preset_name: str = "") -> tuple[bool, str]:
        """按核心的显式触发语义计算消息是否具备启动资格。"""

        if bool(getattr(message, "is_tome", False)):
            return True, "is_tome"
        content = str(getattr(message, "content_text", "") or "")
        name = preset_name.strip()
        if name and name in content:
            return True, "preset_name"
        return False, "none"

    async def _trigger_match(self, ctx: Any, message: Any) -> tuple[bool, str]:
        """读取当前人设名称，计算与核心一致的消息触发资格。"""

        if bool(getattr(message, "is_tome", False)):
            return True, "is_tome"
        current_preset = getattr(ctx, "current_preset", None)
        if not callable(current_preset):
            return False, "preset_unavailable"
        try:
            preset = current_preset()
            if hasattr(preset, "__await__"):
                preset = await preset
            preset_name = str(getattr(preset, "name", "") or "")
        except Exception:
            return False, "preset_unavailable"
        return self._shadow_trigger_match(message, preset_name)

    @staticmethod
    def _shadow_sender_label(message: Any) -> str:
        sender = sender_bucket(message)
        return hashlib.sha256(sender.encode("utf-8")).hexdigest()[:10] if sender else "unknown"

    async def _observe_trigger_scope(
        self,
        ctx: Any,
        message: Any,
        current_buffer: ChatBuffer | None,
        trigger_match: bool | None = None,
        trigger_source: str | None = None,
    ) -> None:
        """记录候选门控结果；此方法不得改变消息、缓冲或返回信号。"""

        if not getattr(self.config, "observe_trigger_scope", False):
            return
        try:
            if trigger_match is None or trigger_source is None:
                matched, source = await self._trigger_match(ctx, message)
            else:
                matched, source = trigger_match, trigger_source
            chat_key = str(getattr(message, "chat_key", "") or "")
            sender = sender_bucket(message)
            pending = self.buffers.for_chat(chat_key)
            same_sender = current_buffer is not None and current_buffer.sender_bucket == sender
            other_senders = {
                buffer.sender_bucket
                for buffer in pending
                if buffer.sender_bucket and buffer.sender_bucket != sender
            }
            decision = "same_sender_continuation" if same_sender else (
                "new_trigger_batch" if matched else "background_or_untriggered"
            )
            if self._is_group_message(message) and not same_sender and not matched:
                behavior = "continued_to_core"
            elif same_sender or matched:
                behavior = "debounce_batch"
            else:
                behavior = "private_debounce_batch"
            self.logger.info(
                f"[Debounce][Observe] chat={self._debug_chat_key(chat_key)} "
                f"sender={self._shadow_sender_label(message)} "
                f"trigger_match={matched} source={source} "
                f"current_sender_batch={same_sender} pending_batches={len(pending)} "
                f"other_sender_pending={len(other_senders)} "
                f"decision={decision} behavior={behavior}",
            )
        except Exception as exc:
            self.logger.warning(f"[Debounce][Observe] 触发范围观测失败，已忽略：{type(exc).__name__}")

    async def _get_release_channel(self, chat_key: str) -> Any:
        """通过 Nekro 公共模型 API 读取释放前的频道状态。"""

        from nekro_agent.models.db_chat_channel import DBChatChannel

        return await DBChatChannel.get_channel(chat_key=chat_key)

    async def invalidate_channel(self, chat_key: str, reason: str = "external_invalidate") -> bool:
        """取消频道内尚未释放的批次并推进 generation 水位。

        该入口供 schedule 等可选协作插件调用。调用失败不会被吞掉，
        由调用方决定是否继续切换频道状态；release gate 仍会重新读取
        DBChatChannel，因此缺少调用方或发生 TOCTOU 时也会 fail-closed。
        """

        chat_key = str(chat_key or "")
        if not chat_key:
            return False
        lock = self.buffers.lock_for(chat_key)
        async with lock:
            buffers = self.buffers.for_chat(chat_key)
            records = await self.journal.records_for(chat_key)
            active_records = [
                record
                for record in records
                if record.state not in {JournalState.CANCELED, JournalState.ACKED}
            ]
            if not buffers and not active_records:
                return False

            record_ids: list[str] = []
            for record in active_records:
                if record.event_id not in record_ids:
                    record_ids.append(record.event_id)
            for buffer in buffers:
                for record_id in buffer.record_ids:
                    if record_id not in record_ids:
                        record_ids.append(record_id)
            try:
                await self.journal.cancel(record_ids, reason)
            except JournalError:
                self.logger.exception(
                    f"[Debounce] 频道失效无法持久化取消状态，保留 pending: chat={chat_key} reason={reason}",
                )
                raise

            generations = [record.generation for record in active_records]
            generations.extend(buffer.generation for buffer in buffers)
            for buffer in buffers:
                self.tasks.cancel(buffer.buffer_key, buffer.generation)
                self.buffers.clear(buffer.buffer_key, buffer.generation)
            self.buffers.invalidate_chat(chat_key, generations)
            self.logger.info(
                f"[Debounce] 频道批次已失效: chat={self._debug_chat_key(chat_key)} "
                f"batches={len(buffers)} records={len(record_ids)} reason={reason}",
            )
            return True

    async def _release_gate(self, buffer: ChatBuffer, reason: str) -> ChatBuffer | None:
        """在频道锁内完成状态、generation 和 journal pending 校验。

        返回被 take 的缓冲表示允许释放；返回 None 表示重复释放、旧
        generation、持久化状态异常或频道已进入 observe/inactive。所有
        延迟路径和媒体边界都必须经过此 gate。
        """

        current = self.buffers.get(buffer.buffer_key)
        if current is None or current.generation != buffer.generation or not current.messages:
            return None
        record_ids = current.record_ids
        records = await self.journal.records_for(current.chat_key, current.generation)
        records_by_id = {record.event_id: record for record in records}
        missing_ids = [record_id for record_id in record_ids if record_id not in records_by_id]
        non_pending_ids = [
            record_id
            for record_id in record_ids
            if record_id in records_by_id and records_by_id[record_id].state != JournalState.PENDING
        ]
        if missing_ids or non_pending_ids:
            self.logger.warning(
                f"[Debounce] release gate journal 不匹配，保留 pending: "
                f"chat={self._debug_chat_key(current.chat_key)} generation={current.generation} "
                f"sender={current.sender_bucket or 'unknown'} missing={len(missing_ids)} "
                f"non_pending={len(non_pending_ids)} expected={len(record_ids)} "
                f"actual={len(records_by_id)}",
            )
            return None

        try:
            channel = await self._get_release_channel(current.chat_key)
        except Exception as exc:
            self.logger.warning(
                f"[Debounce] release gate 无法读取频道状态，保留 pending: "
                f"chat={self._debug_chat_key(current.chat_key)}: {exc}",
            )
            return None
        if channel is None:
            self.logger.warning(
                f"[Debounce] release gate 未找到频道，保留 pending: chat={self._debug_chat_key(current.chat_key)}",
            )
            return None
        if not bool(getattr(channel, "is_active", False)) or bool(getattr(channel, "observe_mode", False)):
            try:
                await self.journal.cancel(record_ids, f"{reason}:channel_inactive_or_observe")
            except JournalError:
                self.logger.exception(
                    f"[Debounce] 频道 inactive/observe 时取消 journal 失败，保留 pending: "
                    f"chat={self._debug_chat_key(current.chat_key)}",
                )
                raise
            self.tasks.cancel(current.buffer_key, current.generation)
            self.buffers.clear(current.buffer_key, current.generation)
            self.buffers.invalidate_chat(current.chat_key, [current.generation])
            self.logger.info(
                f"[Debounce] release gate 丢弃频道批次: chat={self._debug_chat_key(current.chat_key)} "
                f"generation={current.generation} reason={reason}",
            )
            return None

        await self.journal.update_batch(record_ids, release_reason=reason)
        await self.journal.mark_flushing(record_ids)
        taken = self.buffers.take(current.buffer_key, current.generation)
        return taken

    async def _merge_and_trigger(self, _message: Any, buffer: ChatBuffer, reason: str) -> MsgSignal:
        try:
            released = await self._release_gate(buffer, reason)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 媒体边界 release gate 持久化失败，当前消息放行: {exc}")
            return MsgSignal.CONTINUE
        if released is None:
            return MsgSignal.CONTINUE
        try:
            await self.journal.acknowledge(released.record_ids)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] 媒体边界已释放，但 journal 清理失败: {exc}")
        if self.config.debug_logging:
            self.logger.info(
                f"[Debounce] release_reason={reason} dispatch=core_current_message "
                f"pending_messages={len(released.messages)}",
            )
        return MsgSignal.FORCE_TRIGGER

    async def _on_timeout(self, internal_key: str, generation: int) -> None:
        buffer = self.buffers.get(internal_key)
        if buffer is None:
            return
        lock = self.buffers.lock_for(buffer.chat_key)
        async with lock:
            buffer = self.buffers.get(internal_key)
            if buffer is None or buffer.generation != generation or not buffer.messages:
                return
            now = time.time()
            if now + 0.01 < buffer.quiet_deadline:
                self.tasks.schedule(internal_key, generation, buffer.quiet_deadline)
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
                    self.tasks.schedule(internal_key, generation, next_deadline)
                    if self.config.debug_logging:
                        self.logger.info(
                            f"[Debounce] release_reason=semantic_incomplete_wait "
                            f"elapsed={max(0.0, now - buffer.first_seen_at):.1f}s",
                        )
                    return

            await self._flush_after_timeout(buffer, reason or "timeout")

    async def _flush_after_timeout(self, buffer: ChatBuffer, reason: str) -> None:
        try:
            taken = await self._release_gate(buffer, reason)
        except JournalError as exc:
            key = (buffer.buffer_key, buffer.generation)
            failures = self._timeout_state_failures.get(key, 0) + 1
            self._timeout_state_failures[key] = failures
            self.logger.exception(f"[Debounce] timeout release gate 状态写入失败，第 {failures} 次: {exc}")
            if failures <= MAX_TIMEOUT_STATE_RETRIES:
                retry_at = time.time() + TIMEOUT_RETRY_DELAY_SECONDS * failures
                self.tasks.schedule(buffer.buffer_key, buffer.generation, retry_at)
            else:
                try:
                    await self.journal.mark_manual_recovery(buffer.record_ids, "timeout_state_write_exhausted")
                    self.buffers.clear(buffer.buffer_key, buffer.generation)
                except JournalError as journal_exc:
                    self.logger.exception(f"[Debounce] timeout 达到重试上限且无法标记人工恢复: {journal_exc}")
            return
        if taken is None:
            return
        record_ids = taken.record_ids
        try:
            await self._schedule_agent_from_history(taken)
        except Exception as exc:
            self.logger.exception(f"[Debounce] timeout 历史调度失败，保留人工恢复记录: {exc}")
            try:
                await self.journal.mark_manual_recovery(record_ids, f"history_dispatch_failed:{reason}")
            except JournalError:
                pass
            return

        try:
            await self.journal.acknowledge(record_ids)
        except JournalError as exc:
            self.logger.exception(f"[Debounce] timeout 已调度但 journal 清理失败: {exc}")
        if self.config.debug_logging:
            self.logger.info(
                f"[Debounce] release_reason={reason} dispatch=history "
                f"pending_messages={len(taken.messages)}",
            )

    async def _schedule_agent_from_history(self, buffer: ChatBuffer) -> None:
        """从已落库历史调度 Agent，并保留原始触发消息的执行上下文。"""

        from nekro_agent.models.db_chat_channel import DBChatChannel
        from nekro_agent.schemas.agent_ctx import AgentCtx
        from nekro_agent.schemas.chat_message import ChatMessage, ChatType, segments_from_list
        from nekro_agent.services.message_service import message_service

        channel = await DBChatChannel.get_channel(chat_key=buffer.chat_key)
        last_message = buffer.messages[-1]
        message = ChatMessage(
            message_id=last_message.message_id,
            sender_id=last_message.sender_id or "0",
            sender_name=last_message.sender_name or "未知用户",
            sender_nickname=last_message.sender_nickname or last_message.sender_name or "未知用户",
            adapter_key=last_message.adapter_key or channel.adapter_key,
            platform_userid=last_message.platform_userid or "0",
            is_tome=1,
            is_recalled=False,
            chat_key=buffer.chat_key,
            chat_type=ChatType(channel.chat_type),
            content_text=last_message.text,
            content_data=segments_from_list(last_message.content_data),
            raw_cq_code=last_message.raw_cq_code,
            ext_data={},
            send_timestamp=int(last_message.received_at or time.time()),
        )
        ctx = await AgentCtx.create_by_chat_key(chat_key=buffer.chat_key)
        await message_service.schedule_agent_task(
            message=message,
            ctx=ctx,
        )


def register_lifecycle(plugin: Any, runtime: DebounceRuntime) -> None:
    @plugin.mount_init_method()
    async def _init() -> None:
        await runtime.start()

    @plugin.mount_cleanup_method()
    async def _cleanup() -> None:
        await runtime.stop()

    @plugin.mount_on_channel_reset()
    async def _on_channel_reset(ctx: Any) -> None:
        await runtime.reset_channel(ctx)
