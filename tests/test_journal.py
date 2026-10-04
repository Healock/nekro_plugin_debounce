from __future__ import annotations

import time

import pytest

from nekro_plugin_debounce.journal import JournalStore
from nekro_plugin_debounce.state import JournalRecord, JournalState


def make_record(event_id: str = "m1") -> JournalRecord:
    return JournalRecord(
        event_id=event_id,
        chat_key="chat",
        generation=2,
        sequence=0,
        text="hello",
        content_data=[{"type": "text", "text": "hello"}],
        last_message_id="m1",
        updated_at=time.time(),
        timeout_at=time.time() + 10,
        received_at=123.5,
    )


def test_journal_record_without_received_at_uses_legacy_default() -> None:
    record = make_record().model_dump(exclude={"received_at"})

    restored = JournalRecord.model_validate(record)

    assert restored.received_at == 0.0


class Store:
    def __init__(self) -> None:
        self.value: str | None = None

    async def get(self, **_kwargs):
        return self.value

    async def set(self, **kwargs):
        self.value = kwargs["value"]
        return 1


@pytest.mark.asyncio
async def test_journal_round_trip_and_ack() -> None:
    backing = Store()
    journal = JournalStore(backing)
    record = make_record()
    assert await journal.append(record)
    assert not await journal.append(record)
    loaded = await JournalStore(backing).load()
    assert loaded[0].event_id == "m1"
    assert loaded[0].received_at == 123.5
    await journal.mark_flushing(["m1"])
    assert (await journal.records())[0].state == JournalState.FLUSHING
    await journal.mark_manual_recovery(["m1"], "test")
    assert (await journal.records())[0].state == JournalState.MANUAL_RECOVERY
    await journal.acknowledge(["m1"])
    assert await journal.records() == []


@pytest.mark.asyncio
async def test_journal_restores_pending_state() -> None:
    backing = Store()
    first = JournalStore(backing)
    await first.append(make_record())
    second = JournalStore(backing)
    records = await second.load()
    assert records[0].state == JournalState.PENDING


@pytest.mark.asyncio
async def test_cancel_is_durable_and_idempotent() -> None:
    backing = Store()
    journal = JournalStore(backing)
    await journal.append(make_record())

    await journal.cancel(["m1"], "schedule_observe")
    await journal.cancel(["m1"], "schedule_observe")

    records = await JournalStore(backing).load()
    assert len(records) == 1
    assert records[0].state == JournalState.CANCELED
    assert records[0].release_reason == "schedule_observe"


@pytest.mark.asyncio
async def test_discard_leaves_safe_ack_tombstone_when_cleanup_save_fails() -> None:
    class CleanupFailStore(Store):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def set(self, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise OSError("temporary cleanup failure")
            self.value = kwargs["value"]
            return 1

    backing = CleanupFailStore()
    journal = JournalStore(backing)
    await journal.append(make_record())

    assert not await journal.discard(["m1"])
    assert await journal.records() == []
    assert await JournalStore(backing).load() == []
