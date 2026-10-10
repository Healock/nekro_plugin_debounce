"""防抖 timeout 任务注册与清理。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


TimeoutCallback = Callable[[str, int], Awaitable[None]]


class TaskManager:
    def __init__(self, callback: TimeoutCallback, logger: Any = None) -> None:
        self._callback = callback
        self._logger = logger
        self._tasks: dict[tuple[str, int], asyncio.Task[None]] = {}

    def schedule(self, chat_key: str, generation: int, timeout_at: float) -> asyncio.Task[None]:
        key = (chat_key, generation)
        current_task = asyncio.current_task()
        existing = self._tasks.get(key)
        if existing is not None and existing is not current_task:
            self.cancel(chat_key, generation)
        coroutine = self._run(key, timeout_at)
        try:
            task = asyncio.create_task(coroutine, name=f"debounce-timeout:{chat_key}:{generation}")
        except Exception:
            coroutine.close()
            raise
        self._tasks[key] = task
        return task

    async def _run(self, key: tuple[str, int], timeout_at: float) -> None:
        try:
            delay = max(0.0, timeout_at - time.time())
            await asyncio.sleep(delay)
            await self._callback(*key)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._logger is not None:
                self._logger.exception(f"[Debounce] timeout 任务失败: {key[0]} generation={key[1]}")
        finally:
            if self._tasks.get(key) is asyncio.current_task():
                self._tasks.pop(key, None)

    def cancel(self, chat_key: str, generation: int) -> None:
        task = self._tasks.pop((chat_key, generation), None)
        if task is not None and not task.done():
            task.cancel()

    async def cancel_all(self) -> None:
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
