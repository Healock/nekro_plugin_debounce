from __future__ import annotations

from dataclasses import dataclass, field
import time

import pytest

from nekro_agent.models.db_chat_channel import DBChatChannel
from nekro_agent.services.message_service import message_service

from nekro_plugin_debounce import DebounceConfig
from nekro_plugin_debounce.lifecycle import DebounceRuntime
from nekro_plugin_debounce.state import ChatBuffer, JournalRecord, JournalState, MessageEnvelope


@dataclass
class Message:
    chat_key: str = "gate-chat"
    chat_type: str = "private"
    message_id: str = ""
    sender_id: str = "u1"
    sender_name: str = "User"
    sender_nickname: str = "User"
    content_text: str = "pending"
    content_data: list[dict] = field(default_factory=lambda: [{"type": "text", "text": "pending"}])


def _channel(chat_key: str = "gate-chat"):
    DBChatChannel.channels.clear()
    channel = awaitable_channel(chat_key)
    return channel


def awaitable_channel(chat_key: str):
    # DBChatChannel.get_channel creates the same object that the release gate reads.
    return DBChatChannel.channels.setdefault(chat_key, type("Channel", (), {
        "chat_key": chat_key,
        "is_active": True,
        "observe_mode": False,
        "adapter_key": "test",
        "chat_type": "private",
    })())


async def _ready_runtime(*, complete: bool = True) -> tuple[DebounceRuntime, object]:
    from nekro_plugin_debounce import plugin
    from nekro_plugin_debounce.classifier import ClassificationResult

    plugin.store.data.clear()
    channel = _channel()
    runtime = DebounceRuntime(
        plugin,
        DebounceConfig(timeout_seconds=1, high_confidence_timeout_seconds=1, max_wait_seconds=5),
    )

    async def classify(*_args):
        return ClassificationResult(probability=0.9 if complete else 0.1, complete=complete)

    runtime.classifier.classify = classify  # type: ignore[method-assign]
    return runtime, channel


async def _pending(runtime: DebounceRuntime, message_id: str = "gate-1"):
    result = await runtime.handle_user_message(None, Message(message_id=message_id))
    assert result.name == "BLOCK_TRIGGER"
    buffer = runtime.buffers.get("gate-chat")
    assert buffer is not None
    buffer.quiet_deadline = time.time() - 1
    buffer.timeout_at = buffer.quiet_deadline
    return buffer


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [(True, True), (False, False)])
async def test_timeout_release_gate_discards_observe_or_inactive_batch(status) -> None:
    runtime, channel = await _ready_runtime()
    channel.is_active, channel.observe_mode = status
    replayed: list[str] = []
    runtime._schedule_agent_from_history = lambda buffer: replayed.append(buffer.text)  # type: ignore[method-assign]
    buffer = await _pending(runtime)

    await runtime._on_timeout(buffer.buffer_key, buffer.generation)

    assert replayed == []
    assert runtime.buffers.for_chat("gate-chat") == []
    records = await runtime.journal.records()
    assert len(records) == 1
    assert records[0].state == JournalState.CANCELED
    await runtime.stop()


@pytest.mark.asyncio
async def test_max_wait_and_classifier_fallback_use_release_gate() -> None:
    for fallback in (False, True):
        runtime, channel = await _ready_runtime(complete=not fallback)
        replayed: list[str] = []
        runtime._schedule_agent_from_history = lambda buffer: replayed.append(buffer.text)  # type: ignore[method-assign]
        if fallback:
            async def fail(*_args):
                raise RuntimeError("unavailable")

            runtime.classifier.classify = fail  # type: ignore[method-assign]
        buffer = await _pending(runtime, f"fallback-{fallback}")
        if fallback:
            # The first classification marks this buffer as time fallback.
            assert buffer.classifier_fallback
        buffer.max_wait_deadline = time.time() - 1
        channel.observe_mode = True

        await runtime._on_timeout(buffer.buffer_key, buffer.generation)

        assert replayed == []
        records = await runtime.journal.records()
        assert records[0].state == JournalState.CANCELED
        await runtime.stop()


@pytest.mark.asyncio
async def test_resume_does_not_replay_canceled_batch_and_invalidate_is_idempotent() -> None:
    runtime, channel = await _ready_runtime()
    buffer = await _pending(runtime)
    old_generation = buffer.generation
    channel.observe_mode = True
    assert await runtime.invalidate_channel("gate-chat", "schedule_observe") is True
    assert await runtime.invalidate_channel("gate-chat", "schedule_observe") is False
    channel.observe_mode = False
    channel.is_active = True

    replayed: list[str] = []
    runtime._schedule_agent_from_history = lambda current: replayed.append(current.text)  # type: ignore[method-assign]
    await runtime._on_timeout(buffer.buffer_key, old_generation)
    assert replayed == []
    assert (await runtime.journal.records())[0].state == JournalState.CANCELED

    next_message = Message(message_id="after-resume", content_text="new")
    await runtime.handle_user_message(None, next_message)
    new_buffer = runtime.buffers.get("gate-chat")
    assert new_buffer is not None and new_buffer.generation > old_generation
    await runtime.stop()


@pytest.mark.asyncio
async def test_duplicate_release_and_generation_or_journal_gate_are_safe() -> None:
    runtime, _channel_obj = await _ready_runtime()
    replayed: list[str] = []
    runtime._schedule_agent_from_history = lambda buffer: replayed.append(buffer.text)  # type: ignore[method-assign]
    buffer = await _pending(runtime)

    await runtime._on_timeout(buffer.buffer_key, buffer.generation + 1)
    await runtime._on_timeout(buffer.buffer_key, buffer.generation)
    await runtime._on_timeout(buffer.buffer_key, buffer.generation)
    assert replayed == ["pending"]

    second = await _pending(runtime, "gate-canceled")
    await runtime.journal.cancel(second.record_ids, "test_generation_gate")
    await runtime._on_timeout(second.buffer_key, second.generation)
    assert replayed == ["pending"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_release_gate_ignores_legacy_record_with_same_generation() -> None:
    runtime, _channel_obj = await _ready_runtime()
    replayed: list[str] = []
    runtime._schedule_agent_from_history = lambda buffer: replayed.append(buffer.text)  # type: ignore[method-assign]
    buffer = await _pending(runtime, "current")

    await runtime.journal.append(
        JournalRecord(
            event_id="legacy-same-generation",
            chat_key=buffer.chat_key,
            generation=buffer.generation,
            sequence=0,
            state=JournalState.MANUAL_RECOVERY,
            sender_id="u1",
            text="legacy",
            updated_at=1.0,
            timeout_at=1.0,
        ),
    )

    await runtime._on_timeout(buffer.buffer_key, buffer.generation)

    assert replayed == ["pending"]
    await runtime.stop()


@pytest.mark.asyncio
async def test_media_boundary_uses_the_same_channel_release_gate() -> None:
    runtime, channel = await _ready_runtime(complete=False)
    await _pending(runtime, "media-pending")
    channel.observe_mode = True
    media = Message(
        message_id="media-boundary",
        content_text="图片",
        content_data=[{"type": "image", "text": "[图片]"}],
    )

    result = await runtime.handle_user_message(None, media)

    assert result.name == "CONTINUE"
    assert media.content_text == "图片"
    assert (await runtime.journal.records())[0].state == JournalState.CANCELED
    await runtime.stop()


@pytest.mark.asyncio
async def test_timeout_dispatches_history_without_constructing_user_message() -> None:
    runtime, _channel_obj = await _ready_runtime()
    captured = []

    class _Ctx:
        pass

    async def create_by_chat_key(*, chat_key):
        return _Ctx()

    async def schedule_agent_task(**kwargs):
        captured.append(kwargs)

    from nekro_agent.schemas import agent_ctx

    agent_ctx.AgentCtx.create_by_chat_key = create_by_chat_key  # type: ignore[method-assign]
    message_service.schedule_agent_task = schedule_agent_task  # type: ignore[method-assign]
    buffer = ChatBuffer(
        chat_key="gate-chat",
        buffer_key="gate-chat\x1fu1",
        sender_bucket="u1",
        generation=2,
        messages=[
            MessageEnvelope(
                event_id="raw-1",
                message_id="raw-1",
                chat_key="gate-chat",
                sender_bucket="u1",
                generation=2,
                sequence=0,
                text="raw",
                content_data=[{"type": "text", "text": "raw"}],
                sender_id="u1",
                sender_name="User",
            ),
        ],
    )

    await runtime._schedule_agent_from_history(buffer)

    assert len(captured) == 1
    assert captured[0]["chat_key"] == "gate-chat"
    assert isinstance(captured[0]["ctx"], _Ctx)
    assert captured[0]["trigger_audit"].source.value == "debounce_release"
    assert captured[0]["trigger_audit"].message_id == "raw-1"
    assert captured[0]["trigger_audit"].sender_id == "u1"
    assert captured[0]["trigger_audit"].sender_name == "User"
    assert captured[0]["trigger_audit"].generation == 2
    assert "message" not in captured[0]
    await runtime.stop()
