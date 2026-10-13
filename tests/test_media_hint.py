from __future__ import annotations

from dataclasses import dataclass, field
import time

import pytest

from nekro_plugin_debounce import DebounceConfig, plugin
from nekro_plugin_debounce.lifecycle import DebounceRuntime
from nekro_plugin_debounce.state import SemanticState


@dataclass
class Message:
    chat_key: str = "media-chat"
    chat_type: str = "private"
    adapter_key: str = "onebot_v11"
    message_id: str = "media-message"
    sender_id: str = "user"
    sender_name: str = "User"
    sender_nickname: str = "User"
    is_tome: int = 1
    content_text: str = "【本地视频挂载成功】"
    content_data: list[dict] = field(default_factory=lambda: [{"type": "image", "text": "[视频]"}])


@pytest.mark.asyncio
async def test_media_hint_uses_max_wait_and_skips_classifier() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5),
    )
    classified = False

    async def classify(*_args):
        nonlocal classified
        classified = True
        raise AssertionError("媒体批次不应调用文本分类器")

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    now = time.time()
    assert runtime.publish_media_hint(
        {
            "version": 1,
            "hint_id": "media-hint-1",
            "adapter_key": "onebot_v11",
            "chat_key": "media-chat",
            "message_id": "media-message",
            "media_handle_id": "abcd1234",
            "media_type": "video",
            "expires_at": now + 120,
        },
    )

    result = await runtime.handle_user_message(None, Message())

    assert result.name == "BLOCK_TRIGGER"
    buffer = runtime.buffers.get("media-chat")
    assert buffer is not None
    assert buffer.media_wait_mode is True
    assert buffer.media_handle_id == "abcd1234"
    assert buffer.semantic_probability == 0.0
    assert buffer.semantic_state == SemanticState.INCOMPLETE
    assert buffer.selected_wait_seconds == pytest.approx(5, abs=0.1)
    assert classified is False
    await runtime.stop()


@pytest.mark.asyncio
async def test_media_timeout_waits_until_max_deadline_without_reclassification() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5),
    )
    runtime.publish_media_hint(
        {
            "version": 1,
            "adapter_key": "onebot_v11",
            "chat_key": "media-chat",
            "message_id": "media-message",
            "media_handle_id": "abcd1234",
            "media_type": "video",
            "expires_at": time.time() + 120,
        },
    )
    await runtime.handle_user_message(None, Message())
    buffer = runtime.buffers.get("media-chat")
    assert buffer is not None
    reclassified = False
    flushed: list[str] = []

    async def classify(*_args):
        nonlocal reclassified
        reclassified = True
        raise AssertionError("media state must not be overwritten by text classification")

    async def flush(_buffer, reason):
        flushed.append(reason)

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    runtime._flush_after_timeout = flush  # type: ignore[method-assign]
    buffer.quiet_deadline = time.time() - 1
    buffer.max_wait_deadline = time.time() + 60
    await runtime._on_timeout(buffer.buffer_key, buffer.generation)
    assert not reclassified
    assert flushed == []

    buffer.max_wait_deadline = time.time() - 1
    await runtime._on_timeout(buffer.buffer_key, buffer.generation)
    assert not reclassified
    assert flushed == ["media_max_wait_fallback"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_media_hint_metadata_survives_journal_round_trip() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig())
    runtime.publish_media_hint(
        {
            "version": 1,
            "adapter_key": "onebot_v11",
            "chat_key": "media-chat",
            "message_id": "media-message",
            "media_handle_id": "abcd1234",
            "media_type": "video",
            "expires_at": time.time() + 120,
        },
    )
    await runtime.handle_user_message(None, Message())
    from nekro_plugin_debounce.journal import JournalStore

    record = (await JournalStore(plugin.store).load())[0]
    assert record.media_wait_mode is True
    assert record.media_handle_id == "abcd1234"
    assert record.media_type == "video"
    assert record.semantic_probability == 0.0
    await runtime.stop()


@pytest.mark.asyncio
async def test_media_batch_restores_wait_mode_and_handle_metadata() -> None:
    plugin.store.data.clear()
    source = DebounceRuntime(plugin, DebounceConfig())
    source.publish_media_hint(
        {
            "version": 1,
            "adapter_key": "onebot_v11",
            "chat_key": "media-chat",
            "message_id": "media-message",
            "media_handle_id": "abcd1234",
            "media_type": "video",
            "expires_at": time.time() + 120,
        },
    )
    await source.handle_user_message(None, Message())

    restored = DebounceRuntime(plugin, DebounceConfig())

    async def preload():
        return None

    restored._preload_classifier = preload  # type: ignore[method-assign]
    await restored.start()
    buffer = restored.buffers.get("media-chat")
    assert buffer is not None
    assert buffer.media_wait_mode is True
    assert buffer.media_handle_id == "abcd1234"
    assert buffer.semantic_probability == 0.0
    await restored.stop()
    await source.stop()


@pytest.mark.asyncio
async def test_expired_media_hint_does_not_claim_media_batch() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5),
    )
    runtime.publish_media_hint(
        {
            "version": 1,
            "adapter_key": "onebot_v11",
            "chat_key": "media-chat",
            "message_id": "media-message",
            "media_handle_id": "expired",
            "expires_at": time.time() - 1,
        },
    )

    result = await runtime.handle_user_message(None, Message())

    assert result.name == "CONTINUE"
    assert runtime.buffers.for_chat("media-chat") == []
    await runtime.stop()


@pytest.mark.asyncio
async def test_media_batch_accepts_same_sender_followup_but_keeps_sender_buckets_separate() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(plugin, DebounceConfig(timeout_seconds=5, max_wait_seconds=5))
    runtime.publish_media_hint(
        {
            "version": 1,
            "adapter_key": "onebot_v11",
            "chat_key": "media-chat",
            "message_id": "media-message",
            "media_handle_id": "abcd1234",
            "media_type": "video",
            "expires_at": time.time() + 120,
        },
    )
    await runtime.handle_user_message(None, Message(chat_type="group"))

    followup = Message(
        chat_type="group",
        message_id="followup",
        content_text="帮我看看这个",
        content_data=[{"type": "text", "text": "帮我看看这个"}],
    )
    await runtime.handle_user_message(None, followup)
    own_buffer = runtime.buffers.get("media-chat\x1fuser")
    assert own_buffer is not None
    assert len(own_buffer.messages) == 2
    assert own_buffer.media_wait_mode
    assert own_buffer.semantic_probability == 0.0

    other_sender = Message(
        chat_type="group",
        message_id="other",
        sender_id="other",
        content_text="无关消息",
        content_data=[{"type": "text", "text": "无关消息"}],
        is_tome=0,
    )
    await runtime.handle_user_message(None, other_sender)
    other_buffer = runtime.buffers.get("media-chat\x1fother")
    assert other_buffer is None
    assert len(own_buffer.messages) == 2
    await runtime.stop()
