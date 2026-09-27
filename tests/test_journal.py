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
    )


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
