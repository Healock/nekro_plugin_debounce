from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from enum import Enum

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

    def mount_on_channel_reset(self):
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

    class MsgSignal(Enum):
        FORCE_TRIGGER = -1
        CONTINUE = 0
        BLOCK_TRIGGER = 1
        BLOCK_ALL = 2

    signal.MsgSignal = MsgSignal
    models = types.ModuleType("nekro_agent.models")
    models.__path__ = []  # type: ignore[attr-defined]
    db_chat_channel = types.ModuleType("nekro_agent.models.db_chat_channel")

    @dataclass
    class _Channel:
        chat_key: str
        is_active: bool = True
        observe_mode: bool = False
        adapter_key: str = "test"
        chat_type: str = "private"

    class DBChatChannel:
        channels: dict[str, _Channel] = {}

        @classmethod
        async def get_channel(cls, *, chat_key: str) -> _Channel:
            return cls.channels.setdefault(chat_key, _Channel(chat_key=chat_key))

    db_chat_channel.DBChatChannel = DBChatChannel
    chat_message = types.ModuleType("nekro_agent.schemas.chat_message")

    class ChatType(str, Enum):
        PRIVATE = "private"
        GROUP = "group"

    @dataclass
    class ChatMessage:
        message_id: str
        sender_id: str
        sender_name: str
        sender_nickname: str
        adapter_key: str
        platform_userid: str
        is_tome: int
        is_recalled: bool
        chat_key: str
        chat_type: ChatType
        content_text: str
        content_data: list[Any]
        raw_cq_code: str
        ext_data: dict[str, Any]
        send_timestamp: int

    chat_message.ChatMessage = ChatMessage
    chat_message.ChatType = ChatType
    chat_message.segments_from_list = lambda data: data
    services = types.ModuleType("nekro_agent.services")
    services.__path__ = []  # type: ignore[attr-defined]
    schemas_agent_ctx = types.ModuleType("nekro_agent.schemas.agent_ctx")

    class AgentCtx:
        @classmethod
        async def create_by_chat_key(cls, *, chat_key: str, **_kwargs: Any) -> "AgentCtx":
            instance = cls()
            instance.chat_key = chat_key
            return instance

    schemas_agent_ctx.AgentCtx = AgentCtx
    message_service_module = types.ModuleType("nekro_agent.services.message_service")

    class _MessageService:
        async def push_human_message(self, **_kwargs: Any) -> None:
            return None

        async def schedule_agent_task(self, **_kwargs: Any) -> None:
            return None

    message_service_module.message_service = _MessageService()
    sys.modules.update(
        {
            "nekro_agent": root,
            "nekro_agent.api": api,
            "nekro_agent.api.plugin": plugin_api,
            "nekro_agent.schemas": schemas,
            "nekro_agent.schemas.signal": signal,
            "nekro_agent.models": models,
            "nekro_agent.models.db_chat_channel": db_chat_channel,
            "nekro_agent.schemas.chat_message": chat_message,
            "nekro_agent.schemas.agent_ctx": schemas_agent_ctx,
            "nekro_agent.services": services,
            "nekro_agent.services.message_service": message_service_module,
        },
    )
