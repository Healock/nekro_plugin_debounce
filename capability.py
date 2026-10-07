"""为协作插件提供稳定的频道失效能力。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CAPABILITY_VERSION = 1


@dataclass(frozen=True)
class ChannelInvalidationResult:
    """频道失效请求的明确结果。"""

    available: bool
    success: bool
    confirmed: bool
    batch_count: int
    reason: str


@dataclass(frozen=True)
class DebounceBridge:
    """供 Schedule 获取的防抖协作能力。"""

    runtime: Any
    version: int = CAPABILITY_VERSION

    async def invalidate_channel(
        self,
        chat_key: str,
        reason: str = "external_invalidate",
    ) -> ChannelInvalidationResult:
        try:
            invalidated = await self.runtime.invalidate_channel(chat_key, reason)
        except Exception as exc:
            return ChannelInvalidationResult(
                available=True,
                success=False,
                confirmed=False,
                batch_count=0,
                reason=f"invalidation_failed:{type(exc).__name__}",
            )

        return ChannelInvalidationResult(
            available=True,
            success=True,
            confirmed=True,
            batch_count=1 if invalidated else 0,
            reason="invalidated" if invalidated else "no_pending_batch",
        )
