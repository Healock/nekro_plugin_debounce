"""NekroAgent 消息防抖插件。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from nekro_agent.api.plugin import ConfigBase, NekroPlugin


plugin = NekroPlugin(
    name="消息防抖",
    module_name="nekro_plugin_debounce",
    description="结合语义完整性和静默时间窗口合并连续消息。",
    version="0.4.3",
    author="Healock",
    url="https://github.com/Healock/nekro_plugin_debounce",
    support_adapter=[],
    allow_sleep=False,
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
    high_confidence_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        title="高置信度阈值",
        description="达到该概率后，后续消息使用较短的静默观察时间。",
    )
    timeout_seconds: int = Field(
        default=10,
        ge=0,
        title="缓存超时时间（秒）",
        description="最后一条消息后的静默观察时间。",
    )
    high_confidence_timeout_seconds: int = Field(
        default=4,
        ge=1,
        title="高置信度静默时间（秒）",
        description="后续消息达到高置信度完整时使用的最短静默观察时间。",
    )
    cadence_multiplier: float = Field(
        default=1.25,
        ge=1.0,
        title="发送节奏等待倍率",
        description="高置信度等待时，将近期消息间隔乘以该倍率估算静默时间。",
    )
    cadence_margin_seconds: float = Field(
        default=0.5,
        ge=0.0,
        title="发送节奏额外等待（秒）",
        description="在近期消息间隔估算值上增加的缓冲时间。",
    )
    max_wait_seconds: int = Field(
        default=60,
        ge=1,
        title="最大等待时间（秒）",
        description="从第一条消息开始计算的最大等待时间。",
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
        description="保留 AstrBot 配置字段；Nekro v0.2.0 不伪造 Agent 取消。",
    )
    debug_logging: bool = Field(
        default=False,
        title="启用调试日志",
        description="开启后记录完整性概率、近期发送间隔、等待状态和释放原因。",
    )

    @model_validator(mode="after")
    def validate_waiting_window(self) -> "DebounceConfig":
        if self.max_wait_seconds < self.timeout_seconds:
            raise ValueError("最大等待时间必须大于或等于静默观察时间")
        if self.high_confidence_threshold < self.send_threshold:
            raise ValueError("高置信度阈值必须大于或等于完整性概率阈值")
        if self.timeout_seconds > 0 and self.high_confidence_timeout_seconds > self.timeout_seconds:
            if "high_confidence_timeout_seconds" in self.model_fields_set:
                raise ValueError("高置信度静默时间不能大于普通静默时间")
            self.high_confidence_timeout_seconds = self.timeout_seconds
        return self


config = plugin.get_config(DebounceConfig)

from .lifecycle import DebounceRuntime, register_lifecycle  # noqa: E402
from .matcher import register_matcher  # noqa: E402

runtime = DebounceRuntime(plugin, config)
register_matcher(plugin, runtime)
register_lifecycle(plugin, runtime)


async def invalidate_channel(chat_key: str, reason: str = "external_invalidate") -> bool:
    """可选地取消频道 pending 防抖批次，供其他插件在切换状态前调用。"""

    return await runtime.invalidate_channel(chat_key, reason)

__all__ = ["DebounceConfig", "config", "invalidate_channel", "plugin", "runtime"]
