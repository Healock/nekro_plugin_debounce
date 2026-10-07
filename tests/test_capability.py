from types import SimpleNamespace

import pytest

from nekro_plugin_debounce.capability import DebounceBridge


@pytest.mark.asyncio
async def test_empty_channel_is_confirmed() -> None:
    bridge = DebounceBridge(SimpleNamespace(invalidate_channel=_return_false))

    result = await bridge.invalidate_channel("chat")

    assert result.available is True
    assert result.success is True
    assert result.confirmed is True
    assert result.batch_count == 0
    assert result.reason == "no_pending_batch"


@pytest.mark.asyncio
async def test_invalidation_failure_is_not_confirmed() -> None:
    bridge = DebounceBridge(SimpleNamespace(invalidate_channel=_raise_error))

    result = await bridge.invalidate_channel("chat")

    assert result.available is True
    assert result.success is False
    assert result.confirmed is False
    assert result.reason == "invalidation_failed:RuntimeError"


async def _return_false(_chat_key: str, _reason: str) -> bool:
    return False


async def _raise_error(_chat_key: str, _reason: str) -> bool:
    raise RuntimeError("test failure")
