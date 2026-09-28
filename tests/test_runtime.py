from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nekro_plugin_debounce import DebounceConfig, lifecycle as lifecycle_module
from nekro_plugin_debounce.journal import JournalError
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


def test_compat_config_fields_and_defaults() -> None:
    config = DebounceConfig()
    assert config.model_type == "small"
    assert config.send_threshold == 0.8
    assert config.timeout_seconds == 10
    assert config.usage_scope == "both"
    assert config.cancel_on_new_message is True
    assert config.debounce_mode == "semantic"
    assert config.debug_logging is False


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
async def test_pending_text_triggers_without_second_classification() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)
    calls = 0

    async def incomplete(*_args):
        nonlocal calls
        calls += 1
        return False

    runtime.classifier.is_complete = incomplete  # type: ignore[method-assign]
    first = Message(message_id="pending-1", content_text="第一段", content_data=[{"type": "text", "text": "第一段"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    second = Message(message_id="pending-2", content_text="第二段", content_data=[{"type": "text", "text": "第二段"}])
    assert (await runtime.handle_user_message(None, second)).name == "FORCE_TRIGGER"
    assert second.content_text == "第一段 第二段"
    assert calls == 1


@pytest.mark.asyncio
async def test_time_mode_waits_until_timeout_for_all_messages() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(debounce_mode="time", timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)

    async def fail_classify(*_args):
        raise AssertionError("时间模式不应调用语义分类器")

    runtime.classifier.is_complete = fail_classify  # type: ignore[method-assign]
    first = Message(message_id="time-1", content_text="第一段", content_data=[{"type": "text", "text": "第一段"}])
    second = Message(message_id="time-2", content_text="第二段", content_data=[{"type": "text", "text": "第二段"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"
    assert (await runtime.handle_user_message(None, second)).name == "BLOCK_ALL"
    assert len(runtime.buffers.get("chat").messages) == 2  # type: ignore[union-attr]
    assert len(await runtime.journal.records()) == 2


@pytest.mark.asyncio
async def test_timeout_state_failure_keeps_buffer_and_schedules_retry() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)

    async def incomplete(*_args):
        return False

    runtime.classifier.is_complete = incomplete  # type: ignore[method-assign]
    first = Message(message_id="timeout-1", content_text="等待", content_data=[{"type": "text", "text": "等待"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"

    async def fail_mark(_ids):
        raise JournalError("store unavailable")

    runtime.journal.mark_flushing = fail_mark  # type: ignore[method-assign]
    await runtime._on_timeout("chat", 0)
    assert runtime.buffers.has_pending("chat")
    assert ("chat", 0) in runtime.tasks.tasks
    await runtime.stop()


@pytest.mark.asyncio
async def test_merge_failure_marks_manual_recovery() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)

    async def incomplete(*_args):
        return False

    runtime.classifier.is_complete = incomplete  # type: ignore[method-assign]
    first = Message(message_id="merge-1", content_text="旧文本", content_data=[{"type": "text", "text": "旧文本"}])
    assert (await runtime.handle_user_message(None, first)).name == "BLOCK_ALL"

    def fail_merge(*_args, **_kwargs):
        raise ValueError("invalid content")

    original_merge = lifecycle_module.merge_into_message
    lifecycle_module.merge_into_message = fail_merge
    try:
        second = Message(message_id="merge-2", content_text="新文本", content_data=[{"type": "text", "text": "新文本"}])
        assert (await runtime.handle_user_message(None, second)).name == "CONTINUE"
    finally:
        lifecycle_module.merge_into_message = original_merge
    records = await runtime.journal.records()
    assert records[0].state == JournalState.MANUAL_RECOVERY


@pytest.mark.asyncio
async def test_append_failure_fails_open() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)

    async def incomplete(*_args):
        return False

    async def fail_append(_record):
        raise JournalError("store unavailable")

    runtime.classifier.is_complete = incomplete  # type: ignore[method-assign]
    runtime.journal.append = fail_append  # type: ignore[method-assign]
    message = Message(message_id="append-fail", content_text="直接放行", content_data=[{"type": "text", "text": "直接放行"}])
    assert (await runtime.handle_user_message(None, message)).name == "CONTINUE"
    assert not runtime.buffers.has_pending("chat")


@pytest.mark.asyncio
async def test_classifier_failure_fails_open() -> None:
    from nekro_plugin_debounce import plugin

    plugin.store.data.clear()
    config = DebounceConfig(timeout_seconds=0)
    runtime = DebounceRuntime(plugin, config)

    async def fail_classify(*_args):
        raise RuntimeError("model unavailable")

    runtime.classifier.is_complete = fail_classify  # type: ignore[method-assign]
    message = Message(message_id="classifier-fail", content_text="直接放行", content_data=[{"type": "text", "text": "直接放行"}])
    assert (await runtime.handle_user_message(None, message)).name == "CONTINUE"


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
