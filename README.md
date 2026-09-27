# NekroAgent 消息防抖（AstrBot 兼容迁移版）

版本：`0.1.0`

本插件将 AstrBot 消息防抖的主要用户体验移植到 NekroAgent：连续发送的文本先按频道缓冲，由 ONNX 完整性模型判断是否已经说完；未完整消息不会触发 Agent，完整消息合并后只触发一次。

## 兼容配置

保留 AstrBot 原字段和默认值：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model_type` | `small` | `small` 或 `normal` |
| `send_threshold` | `0.8` | 完整概率阈值 |
| `timeout_seconds` | `10` | 超时自动发送，`0` 表示不超时 |
| `enabled` | `true` | 是否启用 |
| `usage_scope` | `both` | `both`、`group`、`private` |
| `cancel_on_new_message` | `true` | 字段保留；Nekro v0.1.0 不取消运行中的 Agent |

现有 Nekro 配置不会自动导入 AstrBot 配置文件。`_conf_schema.json` 的 `0.8/10` 是迁移默认值；AstrBot README 与源码中的 `0.5/30` fallback 不作为新配置默认值。

## 消息边界

- `mount_on_user_message` 能看到的命令、`is_tome` 和显式 @ 不会在插件内额外绕过；上游已经消费、没有进入回调的命令不由本插件处理。
- 使用 `ChatMessage.chat_key` 做频道级缓冲，不按 sender 分桶。群聊合并后原生 sender 元数据可能代表最后一条消息。
- 已有 pending 文本时，下一条可分类文本直接合并并触发，保持 AstrBot 的连续输入行为；该条不会再次运行完整性分类器。
- 纯文本和 AT 可进入分类器。图片、语音、视频、文件、Forward、卡片、戳一戳等非文本段是硬边界：没有 pending 时原样放行，有 pending 时按顺序合并到当前 `ChatMessage`，保留 `content_data` 并直接触发。

## Journal 与故障处理

默认使用 `plugin.store` 保存 JSON journal。返回 `BLOCK_ALL` 前必须成功写入；存储、模型或合并失败时 fail-open。journal 已写入但 timeout 任务创建失败时保留 `pending` 记录，等待下次消息或启动恢复。timeout 状态写入失败时保留内存缓冲并进行有限重试，超过上限后转人工恢复。启动时恢复 `pending` 记录，`flushing` 与不确定状态标记为人工恢复，禁止自动重复触发。合并失败的批次也会标记为人工恢复，不会留下可自动重放的 `pending` 状态。

Nekro 公共 API 没有当前用户消息的 after-persist 回调，因此完整消息路径在合并后保留 `MANUAL_RECOVERY` 记录；状态写入失败时不能声称外层消息已经完成持久化确认。记录可能长期增长，需要后续人工清理或回收。v0.1.0 不承诺 exactly-once。超时文本通过 `push_system(..., trigger_agent=True)` 发送，角色为 `SYSTEM`；调用异常视为不确定状态，不自动重试。

## 模型与依赖

ONNX Runtime、Transformers、NumPy 和 ModelScope 仅在首次需要分类时惰性导入，使用 Nekro 的 `dynamic_import_pkg` 管理。导入插件时不会联网、安装依赖或加载模型。模型目录为：

```text
<plugin data dir>/models/<model_type>/model.onnx
<plugin data dir>/models/<model_type>/tokenizer/
```

模型缺失时首次使用会尝试从 ModelScope 下载到插件专属缓存；部署也可以提前放置模型文件。

## Nekro API 差异

Nekro v0.1.0 不实现 AstrBot 的 ProviderRequest 改写、真实用户消息伪造、运行中 Agent 取消、旧回复丢弃和媒体消息重放。`cancel_on_new_message` 仅为配置兼容字段，不代表已提供取消能力。

## 验证边界

本仓库包含状态机、分类器数学适配、journal 序列化/恢复/幂等、timeout 和媒体边界测试。实际验证结果：`python -m compileall -q .` 通过，`python -m pytest -q` 为 17 passed，`git diff --check` 通过。未运行 Docker、真实 Nekro、OneBot、ONNX 推理或模型下载，因此不能据此声称插件已在真实部署中加载或运行通过。
