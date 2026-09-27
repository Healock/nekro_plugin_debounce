"""NekroAgent AstrBot 兼容消息防抖插件。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nekro_agent.api.plugin import ConfigBase, NekroPlugin


plugin = NekroPlugin(
    name="消息防抖",
    module_name="nekro_plugin_debounce",
    description="使用 ONNX 完整性模型合并连续消息，兼容 AstrBot 防抖配置语义。",
    version="0.1.0",
    author="Healock",
    url="https://github.com/Healock/nekro_plugin_debounce",
    support_adapter=[],
)


@plugin.mount_config()
class DebounceConfig(ConfigBase):
    model_type: Literal["small", "normal"] = Field(
        default="small",
        title="模型类型",
        description="small: 轻量模型；normal: 标准模型。",
    )
    send_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        title="完整性概率阈值",
        description="达到该概率后发送合并消息。",
    )
    timeout_seconds: int = Field(
        default=10,
        ge=0,
        title="缓存超时时间（秒）",
        description="设为 0 表示不自动超时发送。",
    )
    enabled: bool = Field(default=True, title="启用消息防抖")
    usage_scope: Literal["both", "group", "private"] = Field(
        default="both",
        title="使用场景",
        description="兼容 AstrBot 的 both/group/private 语义。",
    )
    cancel_on_new_message: bool = Field(
        default=True,
        title="新消息时取消旧回复",
        description="保留 AstrBot 配置字段；Nekro v0.1.0 不伪造 Agent 取消。",
    )
    debug_mode: bool = Field(default=False, title="调试日志")


config = plugin.get_config(DebounceConfig)

from .lifecycle import DebounceRuntime, register_lifecycle  # noqa: E402
from .matcher import register_matcher  # noqa: E402

runtime = DebounceRuntime(plugin, config)
register_matcher(plugin, runtime)
register_lifecycle(plugin, runtime)

__all__ = ["DebounceConfig", "config", "plugin", "runtime"]
