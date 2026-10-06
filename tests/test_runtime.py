from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time

import pytest

from nekro_plugin_debounce import DebounceConfig
from nekro_plugin_debounce.journal import JournalError
from nekro_plugin_debounce.lifecycle import DebounceRuntime
from nekro_plugin_debounce.state import JournalRecord
from nekro_plugin_debounce.state import SemanticState


@dataclass
class Message:
    chat_key: str = "chat"
    chat_type: str = "private"
    message_id: str = ""
    sender_id: str = "u1"
    sender_name: str = "User"
    sender_nickname: str = "User"
    is_tome: int = 0
    content_text: str = ""
    content_data: list[dict] = field(default_factory=list)


def test_config_uses_hybrid_defaults() -> None:
    config = DebounceConfig()
    assert config.model_type == "small"
    assert config.send_threshold == 0.8
    assert config.high_confidence_threshold == 0.95
    assert config.timeout_seconds == 10
    assert config.high_confidence_timeout_seconds == 4
    assert config.cadence_multiplier == 1.25
    assert config.cadence_margin_seconds == 0.5
    assert config.max_wait_seconds == 60
    assert config.usage_scope == "both"
    assert config.cancel_on_new_message is True
    assert config.debug_logging is False
    assert config.observe_trigger_scope is False
    assert "debounce_mode" not in DebounceConfig.model_fields


def test_shadow_trigger_match_uses_core_trigger_semantics() -> None:
    from nekro_plugin_debounce import plugin
    from nekro_plugin_debounce.lifecycle import DebounceRuntime

    runtime = DebounceRuntime(plugin, DebounceConfig())
    assert runtime._shadow_trigger_match(Message(content_text="你好", is_tome=1)) == (True, "is_tome")
    assert runtime._shadow_trigger_match(Message(content_text="你好绵绵", is_tome=0), "绵绵") == (
        True,
        "preset_name",
    )
    assert runtime._shadow_trigger_match(Message(content_text="普通群聊", is_tome=0), "绵绵") == (False, "none")


def test_config_rejects_max_wait_shorter_than_quiet_window() -> None:
    with pytest.raises(ValueError, match="最大等待时间"):
        DebounceConfig(timeout_seconds=10, max_wait_seconds=5)


def test_config_rejects_invalid_high_confidence_settings() -> None:
    with pytest.raises(ValueError, match="高置信度阈值"):
        DebounceConfig(send_threshold=0.9, high_confidence_threshold=0.8)
    with pytest.raises(ValueError, match="高置信度静默时间"):
        DebounceConfig(timeout_seconds=2, high_confidence_timeout_seconds=3)
    with pytest.raises(ValueError):
        DebounceConfig(cadence_multiplier=0.9)
    with pytest.raises(ValueError):
        DebounceConfig(cadence_margin_seconds=-0.1)


def test_legacy_config_clamps_missing_short_window() -> None:
    config = DebounceConfig(timeout_seconds=1, max_wait_seconds=5)
    assert config.high_confidence_timeout_seconds == 1


async def _result(complete: bool):
    from nekro_plugin_debounce.classifier import ClassificationResult

    return ClassificationResult(probability=0.9 if complete else 0.1, complete=complete)


async def _capture(buffer, output: list[str]) -> None:
    output.append(buffer.text)


@pytest.mark.asyncio
async def test_complete_message_waits_for_quiet_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))
    runtime.classifier.classify = lambda *_args: _result(True)  # type: ignore[method-assign]
    replayed: list[str] = []
    runtime._replay_as_human_message = lambda buffer: _capture(buffer, replayed)  # type: ignore[method-assign]

    message = Message(message_id="complete-1", content_text="完整句子", content_data=[{"type": "text", "text": "完整句子"}])
    assert (await runtime.handle_user_message(None, message)).name == "BLOCK_ALL"
    assert runtime.buffers.has_pending("chat")
    assert not replayed

    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    buffer.quiet_deadline = time.time() - 1
    buffer.timeout_at = buffer.quiet_deadline
    await runtime._on_timeout("chat", buffer.generation)
    assert replayed == ["完整句子"]
    assert await runtime.journal.records() == []
    await runtime.stop()


@pytest.mark.asyncio
async def test_each_message_reclassifies_accumulated_text() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))
    inputs: list[str] = []

    async def classify(text, *_thresholds):
        inputs.append(text)
        return await _result(False)

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    first = Message(message_id="acc-1", content_text="第一段", content_data=[{"type": "text", "text": "第一段"}])
    second = Message(message_id="acc-2", content_text="第二段", content_data=[{"type": "text", "text": "第二段"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    assert (await runtime.handle_user_message(None, second)).name == "BLOCK_ALL"
    assert inputs == ["第一段", "第一段 第二段"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_group_messages_from_different_senders_use_separate_buffers() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5),
    )
    runtime.classifier.classify = lambda *_args: _result(True)  # type: ignore[method-assign]
    replayed: list[tuple[str, str]] = []

    async def capture(buffer) -> None:
        replayed.append((buffer.sender_bucket, buffer.text))

    runtime._replay_as_human_message = capture  # type: ignore[method-assign]
    first = Message(
        chat_key="group",
        chat_type="group",
        sender_id="user-a",
        message_id="a-1",
        is_tome=1,
        content_text="用户 A 的消息",
        content_data=[{"type": "text", "text": "用户 A 的消息"}],
    )
    second = Message(
        chat_key="group",
        chat_type="group",
        sender_id="user-b",
        message_id="b-1",
        is_tome=1,
        content_text="用户 B 的消息",
        content_data=[{"type": "text", "text": "用户 B 的消息"}],
    )

    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    assert (await runtime.handle_user_message(None, second)).name == "BLOCK_ALL"
    assert len(runtime.buffers.for_chat("group")) == 2

    for buffer in runtime.buffers.for_chat("group"):
        buffer.quiet_deadline = time.time() - 1
        buffer.timeout_at = buffer.quiet_deadline
        await runtime._on_timeout(buffer.buffer_key, buffer.generation)

    assert sorted(replayed) == [("user-a", "用户 A 的消息"), ("user-b", "用户 B 的消息")]
    await runtime.stop()


@pytest.mark.asyncio
async def test_untriggered_group_message_passes_without_creating_buffer() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
    message = Message(
        chat_key="group",
        chat_type="group",
        message_id="background-1",
        content_text="群里的普通消息",
        content_data=[{"type": "text", "text": "群里的普通消息"}],
    )

    assert (await runtime.handle_user_message(None, message)).name == "CONTINUE"
    assert not runtime.buffers.for_chat("group")
    assert await runtime.journal.records() == []
    await runtime.stop()


@pytest.mark.asyncio
async def test_triggered_group_message_starts_batch_and_same_sender_continues() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
    runtime.classifier.classify = lambda *_args: _result(False)  # type: ignore[method-assign]
    first = Message(
        chat_key="group",
        chat_type="group",
        sender_id="user-a",
        message_id="trigger-1",
        is_tome=1,
        content_text="@Bot 请回答",
        content_data=[{"type": "text", "text": "@Bot 请回答"}],
    )
    continuation = Message(
        chat_key="group",
        chat_type="group",
        sender_id="user-a",
        message_id="trigger-2",
        content_text="这是补充说明",
        content_data=[{"type": "text", "text": "这是补充说明"}],
    )

    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    assert (await runtime.handle_user_message(None, continuation)).name == "BLOCK_ALL"
    buffer = runtime.buffers.get("group\x1fuser-a")
    assert buffer is not None
    assert buffer.text == "@Bot 请回答 这是补充说明"
    await runtime.stop()


@pytest.mark.asyncio
async def test_new_message_resets_quiet_deadline_only() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=10, max_wait_seconds=60))
    runtime.classifier.classify = lambda *_args: _result(False)  # type: ignore[method-assign]
    first = Message(message_id="deadline-1", content_text="第一段", content_data=[{"type": "text", "text": "第一段"}])
    second = Message(message_id="deadline-2", content_text="第二段", content_data=[{"type": "text", "text": "第二段"}])
    await runtime.handle_user_message(None, first)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    first_max_wait = buffer.max_wait_deadline
    first_quiet = buffer.quiet_deadline
    await asyncio.sleep(0.02)
    await runtime.handle_user_message(None, second)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    assert buffer.quiet_deadline > first_quiet
    assert buffer.max_wait_deadline == first_max_wait
    await runtime.stop()


@pytest.mark.asyncio
async def test_first_high_confidence_message_uses_normal_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=10, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )
    async def classify_high(*_args):
        from nekro_plugin_debounce.classifier import ClassificationResult

        return ClassificationResult(
            probability=0.99,
            complete=True,
            semantic_state=SemanticState.COMPLETE_HIGH,
        )

    runtime.classifier.classify = classify_high  # type: ignore[method-assign]
    first = Message(message_id="high-first", content_text="第一条", content_data=[{"type": "text", "text": "第一条"}])
    await runtime.handle_user_message(None, first)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    assert buffer.semantic_state == SemanticState.COMPLETE_HIGH
    assert buffer.selected_wait_seconds == 10
    await runtime.stop()


@pytest.mark.asyncio
async def test_later_high_confidence_message_uses_short_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=10, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )

    async def classify(text, _threshold, _high_threshold):
        from nekro_plugin_debounce.classifier import ClassificationResult

        probability = 0.2 if text == "第一条" else 0.99
        return ClassificationResult(
            probability=probability,
            complete=probability >= 0.8,
            semantic_state=SemanticState.INCOMPLETE if probability < 0.8 else SemanticState.COMPLETE_HIGH,
        )

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    first = Message(message_id="high-later-1", content_text="第一条", content_data=[{"type": "text", "text": "第一条"}])
    second = Message(message_id="high-later-2", content_text="第二条", content_data=[{"type": "text", "text": "第二条"}])
    await runtime.handle_user_message(None, first)
    await runtime.handle_user_message(None, second)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    assert buffer.semantic_state == SemanticState.COMPLETE_HIGH
    assert buffer.selected_wait_seconds == 2
    await runtime.stop()


def test_high_confidence_wait_adapts_to_recent_message_cadence() -> None:
    from nekro_plugin_debounce import plugin
    from nekro_plugin_debounce.state import ChatBuffer, MessageEnvelope

    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=15, high_confidence_timeout_seconds=4, max_wait_seconds=60),
    )
    buffer = ChatBuffer(
        chat_key="chat",
        generation=1,
        messages=[
            MessageEnvelope(
                event_id=f"cadence-{index}",
                message_id=f"cadence-{index}",
                chat_key="chat",
                generation=1,
                sequence=index,
                text="part",
                received_at=timestamp,
            )
            for index, timestamp in enumerate((28.0, 31.0, 33.0))
        ],
        semantic_state=SemanticState.COMPLETE_HIGH,
    )

    wait_seconds, status, cadence_interval = runtime._wait_seconds_for(buffer)

    assert status == "semantic_complete_high_confidence_adaptive_wait"
    assert cadence_interval == 3.0
    assert wait_seconds == pytest.approx(4.25)


def test_adaptive_high_confidence_wait_is_capped_by_normal_window() -> None:
    from nekro_plugin_debounce import plugin
    from nekro_plugin_debounce.state import ChatBuffer, MessageEnvelope

    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=5, high_confidence_timeout_seconds=4, max_wait_seconds=60),
    )
    buffer = ChatBuffer(
        chat_key="chat",
        generation=1,
        messages=[
            MessageEnvelope(
                event_id=f"cap-{index}",
                message_id=f"cap-{index}",
                chat_key="chat",
                generation=1,
                sequence=index,
                text="part",
                received_at=timestamp,
            )
            for index, timestamp in enumerate((10.0, 20.0))
        ],
        semantic_state=SemanticState.COMPLETE_HIGH,
    )

    wait_seconds, _, _ = runtime._wait_seconds_for(buffer)

    assert wait_seconds == 5


@pytest.mark.asyncio
async def test_probability_drop_restores_normal_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=10, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )
    probabilities = iter([0.2, 0.99, 0.5])

    async def classify(_text, _threshold, _high_threshold):
        from nekro_plugin_debounce.classifier import ClassificationResult

        probability = next(probabilities)
        state = runtime._semantic_state(probability)
        return ClassificationResult(probability=probability, complete=probability >= 0.8, semantic_state=state)

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    first = Message(message_id="drop-1", content_text="第一条", content_data=[{"type": "text", "text": "第一条"}])
    second = Message(message_id="drop-2", content_text="第二条", content_data=[{"type": "text", "text": "第二条"}])
    third = Message(message_id="drop-3", content_text="第三条", content_data=[{"type": "text", "text": "第三条"}])
    await runtime.handle_user_message(None, first)
    await runtime.handle_user_message(None, second)
    high_buffer = runtime.buffers.get("chat")
    assert high_buffer is not None and high_buffer.selected_wait_seconds == 2
    await runtime.handle_user_message(None, third)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    assert buffer.semantic_state == SemanticState.INCOMPLETE
    assert buffer.semantic_probability == 0.5
    assert buffer.previous_probability == 0.99
    assert buffer.probability_delta == pytest.approx(-0.49)
    assert buffer.selected_wait_seconds == 10
    await runtime.stop()


@pytest.mark.asyncio
async def test_incomplete_timeout_waits_then_max_wait_forces_release() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))
    runtime.classifier.classify = lambda *_args: _result(False)  # type: ignore[method-assign]
    replayed: list[str] = []
    runtime._replay_as_human_message = lambda buffer: _capture(buffer, replayed)  # type: ignore[method-assign]
    message = Message(message_id="wait-1", content_text="未完成", content_data=[{"type": "text", "text": "未完成"}])
    await runtime.handle_user_message(None, message)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None

    buffer.quiet_deadline = time.time() - 1
    await runtime._on_timeout("chat", buffer.generation)
    assert runtime.buffers.has_pending("chat")
    assert not replayed

    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    buffer.quiet_deadline = time.time() - 1
    buffer.max_wait_deadline = time.time() - 1
    await runtime._on_timeout("chat", buffer.generation)
    assert replayed == ["未完成"]
    assert await runtime.journal.records() == []
    await runtime.stop()


@pytest.mark.asyncio
async def test_max_wait_takes_precedence_over_complete_result() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5),
    )
    runtime.classifier.classify = lambda *_args: _result(True)  # type: ignore[method-assign]
    reasons: list[str] = []

    async def capture_reason(buffer, reason):
        reasons.append(reason)

    runtime._flush_after_timeout = capture_reason  # type: ignore[method-assign]
    message = Message(message_id="max-complete", content_text="已完成", content_data=[{"type": "text", "text": "已完成"}])
    await runtime.handle_user_message(None, message)
    buffer = runtime.buffers.get("chat")
    assert buffer is not None
    buffer.quiet_deadline = time.time() - 1
    buffer.max_wait_deadline = time.time() - 1
    await runtime._on_timeout("chat", buffer.generation)
    assert reasons == ["max_wait_fallback"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_classifier_failure_falls_back_to_quiet_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))

    async def fail(*_args):
        raise RuntimeError("model unavailable")

    runtime.classifier.classify = fail  # type: ignore[method-assign]
    replayed: list[str] = []
    runtime._replay_as_human_message = lambda buffer: _capture(buffer, replayed)  # type: ignore[method-assign]
    message = Message(message_id="fallback-1", content_text="降级", content_data=[{"type": "text", "text": "降级"}])
    assert (await runtime.handle_user_message(None, message)).name == "BLOCK_ALL"
    buffer = runtime.buffers.get("chat")
    assert buffer is not None and buffer.classifier_fallback
    buffer.quiet_deadline = time.time() - 1
    await runtime._on_timeout("chat", buffer.generation)
    assert replayed == ["降级"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_media_boundary_merges_without_classifier() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))
    runtime.classifier.classify = lambda *_args: _result(False)  # type: ignore[method-assign]
    first = Message(message_id="media-1", content_text="先说", content_data=[{"type": "text", "text": "先说"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    media = Message(message_id="media-2", content_text="图片", content_data=[{"type": "image", "text": "[图片]"}])
    assert (await runtime.handle_user_message(None, media)).name == "FORCE_TRIGGER"
    assert media.content_text == "先说 图片"
    assert [item["type"] for item in media.content_data] == ["text", "image"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_append_failure_fails_open() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))

    async def fail_append(_record):
        raise JournalError("store unavailable")

    runtime.journal.append = fail_append  # type: ignore[method-assign]
    message = Message(message_id="append-fail", content_text="直接放行", content_data=[{"type": "text", "text": "直接放行"}])
    assert (await runtime.handle_user_message(None, message)).name == "CONTINUE"
    assert not runtime.buffers.has_pending("chat")
    await runtime.stop()


@pytest.mark.asyncio
async def test_start_restores_pending_batch_deadlines() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5))
    first_seen = time.time() - 1
    record = JournalRecord(
        event_id="restore-1",
        chat_key="restore-chat",
        generation=3,
        sequence=0,
        text="待恢复",
        content_data=[{"type": "text", "text": "待恢复"}],
        last_message_id="restore-1",
        updated_at=first_seen,
        timeout_at=first_seen + 1,
        first_seen_at=first_seen,
        quiet_deadline=first_seen + 1,
        max_wait_deadline=first_seen + 5,
    )
    await runtime.journal.append(record)
    await runtime.start()
    restored = runtime.buffers.get("restore-chat")
    assert restored is not None
    assert restored.first_seen_at == first_seen
    assert restored.max_wait_deadline == first_seen + 5
    await runtime.stop()
