from __future__ import annotations

from dataclasses import dataclass, field
import time

import pytest

from nekro_plugin_debounce import DebounceConfig
from nekro_plugin_debounce.journal import JournalError
from nekro_plugin_debounce.lifecycle import DebounceRuntime
from nekro_plugin_debounce.state import JournalRecord


@dataclass
class Message:
    chat_key: str = "chat"
    chat_type: str = "private"
    message_id: str = ""
    sender_id: str = "u1"
    sender_name: str = "User"
    sender_nickname: str = "User"
    content_text: str = ""
    content_data: list[dict] = field(default_factory=list)


def test_config_uses_hybrid_defaults() -> None:
    config = DebounceConfig()
    assert config.model_type == "small"
    assert config.send_threshold == 0.8
    assert config.timeout_seconds == 10
    assert config.max_wait_seconds == 60
    assert config.usage_scope == "both"
    assert config.cancel_on_new_message is True
    assert config.debug_logging is False
    assert "debounce_mode" not in DebounceConfig.model_fields


def test_config_rejects_max_wait_shorter_than_quiet_window() -> None:
    with pytest.raises(ValueError, match="最大等待时间"):
        DebounceConfig(timeout_seconds=10, max_wait_seconds=5)


async def _result(complete: bool):
    from nekro_plugin_debounce.classifier import ClassificationResult

    return ClassificationResult(probability=0.9 if complete else 0.1, complete=complete)


async def _capture(buffer, output: list[str]) -> None:
    output.append(buffer.text)


@pytest.mark.asyncio
async def test_complete_message_waits_for_quiet_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
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
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
    inputs: list[str] = []

    async def classify(text, _threshold):
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
async def test_incomplete_timeout_waits_then_max_wait_forces_release() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
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
async def test_classifier_failure_falls_back_to_quiet_window() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))

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
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
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
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))

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
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=1, max_wait_seconds=5))
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
