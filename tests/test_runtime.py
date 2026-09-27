from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nekro_plugin_debounce import DebounceConfig
from nekro_plugin_debounce.lifecycle import DebounceRuntime
from nekro_plugin_debounce.state import JournalRecord
from nekro_plugin_debounce.state import JournalState


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


@pytest.mark.asyncio
async def test_incomplete_then_complete_marks_outer_commit_uncertain() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)
    runtime.classifier.is_complete = lambda *_args: _false_async()  # type: ignore[method-assign]

    first = Message(message_id="m1", content_text="如果明天不下雨", content_data=[{"type": "text", "text": "如果明天不下雨"}])
    signal = await runtime.handle_user_message(None, first)
    assert signal.name == "BLOCK_ALL"

    async def complete(*_args):
        return True

    runtime.classifier.is_complete = complete  # type: ignore[method-assign]
    second = Message(message_id="m2", content_text="我们去爬山吧", content_data=[{"type": "text", "text": "我们去爬山吧"}])
    signal = await runtime.handle_user_message(None, second)
    assert signal.name == "FORCE_TRIGGER"
    assert second.content_text == "如果明天不下雨 我们去爬山吧"
    records = await runtime.journal.records()
    assert records[0].state == JournalState.MANUAL_RECOVERY


async def _false_async(*_args):
    return False


@pytest.mark.asyncio
async def test_media_boundary_merges_without_classifier() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)

    async def incomplete(*_args):
        return False

    runtime.classifier.is_complete = incomplete  # type: ignore[method-assign]
    first = Message(message_id="m3", content_text="先说", content_data=[{"type": "text", "text": "先说"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    media = Message(message_id="m4", content_text="图片", content_data=[{"type": "image", "text": "[图片]"}])
    assert (await runtime.handle_user_message(None, media)).name == "FORCE_TRIGGER"
    assert media.content_text == "先说 图片"
    assert [item["type"] for item in media.content_data] == ["text", "image"]


@pytest.mark.asyncio
async def test_task_creation_failure_keeps_pending_record() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=10)
    runtime = DebounceRuntime(plugin, config)

    async def incomplete(*_args):
        return False

    runtime.classifier.is_complete = incomplete  # type: ignore[method-assign]

    def fail_schedule(*_args, **_kwargs):
        raise RuntimeError("no loop")

    runtime.tasks.schedule = fail_schedule  # type: ignore[method-assign]
    message = Message(message_id="m5", content_text="未完", content_data=[{"type": "text", "text": "未完"}])
    assert (await runtime.handle_user_message(None, message)).name == "BLOCK_ALL"
    record = (await runtime.journal.records())[0]
    assert record.state == JournalState.PENDING
    assert record.error_state == "timeout_task_creation_failed"


@pytest.mark.asyncio
async def test_start_restores_pending_buffer_and_schedules_timeout() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=10)
    runtime = DebounceRuntime(plugin, config)
    record = JournalRecord(
        event_id="restore-1",
        chat_key="restore-chat",
        generation=3,
        sequence=0,
        text="待恢复",
        content_data=[{"type": "text", "text": "待恢复"}],
        last_message_id="restore-1",
        updated_at=1.0,
        timeout_at=9999999999.0,
    )
    await runtime.journal.append(record)
    await runtime.start()
    assert runtime.buffers.has_pending("restore-chat")
    assert ("restore-chat", 3) in runtime.tasks.tasks
    await runtime.stop()
