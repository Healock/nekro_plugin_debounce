from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class _Logger:
    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def exception(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Store:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, **kwargs: Any) -> str | None:
        return self.data.get(kwargs["store_key"])

    async def set(self, **kwargs: Any) -> int:
        key = kwargs["store_key"]
        existed = key in self.data
        self.data[key] = kwargs["value"]
        return int(existed)


class _ConfigBase(BaseModel):
    pass


class _Plugin:
    def __init__(self, **_kwargs: Any) -> None:
        self.logger = _Logger()
        self.store = _Store()
        self._data_dir = Path(tempfile.mkdtemp(prefix="debounce-test-"))

    def mount_config(self):
        return lambda cls: cls

    def get_config(self, config_cls: type[_ConfigBase] = _ConfigBase):
        return config_cls()

    def mount_on_user_message(self):
        return lambda func: func

    def mount_init_method(self):
        return lambda func: func

    def mount_cleanup_method(self):
        return lambda func: func

    def get_plugin_data_dir(self) -> Path:
        return self._data_dir


def _dynamic_import_pkg(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("测试未启用动态依赖导入")


def pytest_configure() -> None:
    if "nekro_agent" in sys.modules:
        return
    root = types.ModuleType("nekro_agent")
    root.__path__ = []  # type: ignore[attr-defined]
    api = types.ModuleType("nekro_agent.api")
    api.__path__ = []  # type: ignore[attr-defined]
    plugin_api = types.ModuleType("nekro_agent.api.plugin")
    plugin_api.ConfigBase = _ConfigBase
    plugin_api.NekroPlugin = _Plugin
    plugin_api.dynamic_import_pkg = _dynamic_import_pkg
    schemas = types.ModuleType("nekro_agent.schemas")
    schemas.__path__ = []  # type: ignore[attr-defined]
    signal = types.ModuleType("nekro_agent.schemas.signal")

    from enum import Enum

    class MsgSignal(Enum):
        FORCE_TRIGGER = -1
        CONTINUE = 0
        BLOCK_TRIGGER = 1
        BLOCK_ALL = 2

    signal.MsgSignal = MsgSignal
    sys.modules.update(
        {
            "nekro_agent": root,
            "nekro_agent.api": api,
            "nekro_agent.api.plugin": plugin_api,
            "nekro_agent.schemas": schemas,
            "nekro_agent.schemas.signal": signal,
        },
    )
