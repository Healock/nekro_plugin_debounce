"""注册 Nekro 用户消息回调。"""

from __future__ import annotations

from typing import Any


def register_matcher(plugin: Any, runtime: Any) -> None:
    @plugin.mount_on_user_message()
    async def on_user_message(ctx: Any, message: Any):
        return await runtime.handle_user_message(ctx, message)


__all__ = ["register_matcher"]
