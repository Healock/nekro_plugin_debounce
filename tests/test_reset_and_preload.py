from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from nekro_plugin_debounce import DebounceConfig, plugin
from nekro_plugin_debounce.journal import JournalError, JournalStore
from nekro_plugin_debounce.lifecycle import DebounceRuntime
from nekro_plugin_debounce.state import JournalRecord, JournalState


@dataclass
class Message:
    chat_key: str = "chat"
    chat_type: str = "private"
    message_id: str = ""
    sender_id: str = "u1"
    sender_name: str = "User"
    sender_nickname: str = "User"
    content_text: str = "text"
    content_data: list[dict] = field(default_factory=lambda: [{"type": "text", "text": "text"}])


@dataclass
class Context:
    chat_key: str = "chat"


@pytest.mark.asyncio
async def test_reset_cancels_pending_generation_and_new_message_starts_new_batch() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=30, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )
    replayed: list[str] = []

    async def classify(*_args):
        from nekro_plugin_debounce.classifier import ClassificationResult

        return ClassificationResult(probability=0.1, complete=False)

    async def capture(buffer, _reason=""):
        replayed.append(buffer.text)

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    runtime._schedule_agent_from_history = capture  # type: ignore[method-assign]

    first = Message(message_id="before-reset")
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_TRIGGER"
    old_buffer = runtime.buffers.get("chat")
    assert old_buffer is not None

    await runtime.reset_channel(Context())
    assert not runtime.buffers.has_pending("chat")
    await runtime._on_timeout("chat", old_buffer.generation)
    assert replayed == []
    assert await runtime.journal.records() == []
    assert await JournalStore(plugin.store).load() == []

    second = Message(message_id="after-reset", content_text="new", content_data=[{"type": "text", "text": "new"}])
    assert (await runtime.handle_user_message(None, second)).name == "BLOCK_TRIGGER"
    new_buffer = runtime.buffers.get("chat")
    assert new_buffer is not None
    assert new_buffer.generation > old_buffer.generation
    assert new_buffer.text == "new"
    await runtime.stop()


@pytest.mark.asyncio
async def test_reset_persist_failure_marks_batch_for_manual_recovery() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=30, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )

    async def classify(*_args):
        from nekro_plugin_debounce.classifier import ClassificationResult

        return ClassificationResult(probability=0.1, complete=False)

    async def fail_discard(_event_ids):
        raise JournalError("store unavailable")

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    await runtime.handle_user_message(None, Message(message_id="reset-store-failure"))
    runtime.journal.discard = fail_discard  # type: ignore[method-assign]

    await runtime.reset_channel(Context())

    records = await JournalStore(plugin.store).load()
    assert len(records) == 1
    assert records[0].state.value == "manual_recovery"
    assert not runtime.buffers.has_pending("chat")
    await runtime.stop()


@pytest.mark.asyncio
async def test_reset_skips_generation_used_by_older_manual_recovery_record() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=30, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )

    async def classify(*_args):
        from nekro_plugin_debounce.classifier import ClassificationResult

        return ClassificationResult(probability=0.1, complete=False)

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    await runtime.handle_user_message(None, Message(message_id="before-reset"))
    old_buffer = runtime.buffers.get("chat")
    assert old_buffer is not None

    await runtime.journal.append(
        JournalRecord(
            event_id="legacy-manual-recovery",
            chat_key="chat",
            generation=old_buffer.generation + 1,
            sequence=0,
            state=JournalState.MANUAL_RECOVERY,
            sender_id="u1",
            text="legacy",
            updated_at=1.0,
            timeout_at=1.0,
        ),
    )

    await runtime.reset_channel(Context())
    second = Message(message_id="after-reset", content_text="new", content_data=[{"type": "text", "text": "new"}])
    await runtime.handle_user_message(None, second)
    new_buffer = runtime.buffers.get("chat")
    assert new_buffer is not None
    assert new_buffer.generation > old_buffer.generation + 1
    await runtime.stop()


@pytest.mark.asyncio
async def test_reset_without_pending_buffer_still_advances_journal_generation() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=30, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )
    await runtime.journal.append(
        JournalRecord(
            event_id="legacy-manual-recovery",
            chat_key="chat",
            generation=7,
            sequence=0,
            state=JournalState.MANUAL_RECOVERY,
            sender_id="u1",
            text="legacy",
            updated_at=1.0,
            timeout_at=1.0,
        ),
    )

    await runtime.reset_channel(Context())
    await runtime.handle_user_message(None, Message(message_id="after-reset"))
    new_buffer = runtime.buffers.get("chat")
    assert new_buffer is not None
    assert new_buffer.generation == 8
    await runtime.stop()


@pytest.mark.asyncio
async def test_messages_do_not_wait_for_pending_classifier_preload() -> None:
    plugin.store.data.clear()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=30, high_confidence_timeout_seconds=2, max_wait_seconds=60),
    )
    preload_started = asyncio.Event()

    async def preload() -> None:
        preload_started.set()
        await asyncio.Event().wait()

    async def unexpected_classify(*_args):
        raise AssertionError("分类不应在预加载完成前执行")

    runtime.classifier.preload = preload  # type: ignore[method-assign]
    runtime.classifier.classify = unexpected_classify  # type: ignore[method-assign]
    await runtime.start()
    await preload_started.wait()

    result = await runtime.handle_user_message(None, Message(message_id="during-preload"))
    assert result.name == "BLOCK_TRIGGER"
    buffer = runtime.buffers.get("chat")
    assert buffer is not None and buffer.classifier_fallback
    await runtime.stop()
