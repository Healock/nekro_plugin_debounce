# NekroAgent 消息防抖

版本：`0.4.1`

本插件移植自 AstrBot 插件 `astrbot_plugin_debounce`。

- 原作者：`advent259141`
- 原仓库：[advent259141/astrbot_plugin_debounce](https://github.com/advent259141/astrbot_plugin_debounce)

连续发送的文本先按频道缓冲。插件同时观察累计文本的语义完整性和最后一条消息后的静默时间：静默窗口结束且语义完整时触发；语义持续不完整时最多等待到硬上限，再强制触发一次。

## 兼容配置

保留 AstrBot 原字段和默认值：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model_type` | `small` | `small` 或 `normal` |
| `send_threshold` | `0.8` | 完整概率阈值；值越高，越不容易判定为已说完。 |
| `high_confidence_threshold` | `0.95` | 后续消息进入短静默窗口所需的高置信度阈值。 |
| `timeout_seconds` | `10` | 最后一条消息后的静默观察时间，单位为秒；`0` 表示立即进入 timeout 判断 |
| `high_confidence_timeout_seconds` | `2` | 后续累计文本达到高置信度完整时使用的短静默时间。 |
| `max_wait_seconds` | `60` | 从第一条消息开始计算的最大等待时间，单位为秒 |
| `enabled` | `true` | 是否启用 |
| `usage_scope` | `both` | `both`、`group`、`private` |
| `cancel_on_new_message` | `true` | 字段保留；Nekro v0.4.0 不取消运行中的 Agent |
| `debug_logging` | `false` | 开启后在日志中记录概率变化、阈值、等待窗口和释放原因 |

每条文本消息都会基于当前频道的累计文本重新分类。第一条消息始终使用普通静默窗口；后续消息只有在达到 `high_confidence_threshold` 时才使用较短的静默窗口，其余情况使用普通窗口。静默结束时会重新分类，概率下降时会恢复普通等待，直到语义完整或达到 `max_wait_seconds` 后强制触发。模型加载或推理失败时，当前批次退化为普通时间防抖。

累计文本的概率不是单调递增的进度值。新增内容可能引入新的未完成语义，因此插件始终使用最新一次分类结果，不保留历史最高概率。例如概率从 `0.9` 降到 `0.5` 时，当前状态会随之回退。

`high_confidence_threshold` 和模型输出的 softmax 概率没有经过可靠性校准。`0.95` 是进入短等待的工程阈值，不代表模型实际有 95% 的正确率；启用 `debug_logging` 后，应结合真实服务器日志观察概率分布和误拆分情况。

旧配置中的 `debounce_mode` 已废弃。插件不再在时间模式和语义模式之间二选一；残留字段不会阻止配置加载，也不参与运行逻辑。

现有 Nekro 配置不会自动导入 AstrBot 配置文件。`_conf_schema.json` 的 `0.8/10` 是迁移默认值；AstrBot README 与源码中的 `0.5/30` fallback 不作为新配置默认值。

## 消息边界

- `mount_on_user_message` 能看到的命令、`is_tome` 和显式 @ 不会在插件内额外绕过；上游已经消费、没有进入回调的命令不由本插件处理。
- 使用 `ChatMessage.chat_key` 做频道级缓冲，不按 sender 分桶。群聊合并后原生 sender 元数据可能代表最后一条消息。
- 已有 pending 文本时，下一条可分类文本会合并到累计文本并再次运行完整性分类器。
- 纯文本和 AT 可进入分类器。图片、语音、视频、文件、Forward、卡片、戳一戳等非文本段是硬边界：没有 pending 时原样放行，有 pending 时按顺序合并到当前 `ChatMessage`，保留 `content_data` 并直接触发。

## Journal 与故障处理

默认使用 `plugin.store` 保存 JSON journal。返回 `BLOCK_ALL` 前必须成功写入；存储、模型或合并失败时 fail-open。journal 已写入但 timeout 任务创建失败时保留 `pending` 记录，等待下次消息或启动恢复。timeout 状态写入失败时保留内存缓冲并进行有限重试，超过上限后转人工恢复。启动时恢复 `pending` 记录，`flushing` 与不确定状态标记为人工恢复，禁止自动重复触发。合并失败的批次也会标记为人工恢复，不会留下可自动重放的 `pending` 状态。

Nekro 公共 API 没有当前用户消息的 after-persist 回调，因此完整消息路径在合并后保留 `MANUAL_RECOVERY` 记录；状态写入失败时不能声称外层消息已经完成持久化确认。记录可能长期增长，需要后续人工清理或回收。v0.4.0 不承诺 exactly-once。超时文本会通过 Nekro 当前的用户消息处理入口重新提交，避免把用户内容写成 `SYSTEM` 消息；该路径依赖 Nekro 内部消息服务，调用异常视为不确定状态，不自动重试。

## 模型与依赖

ONNX Runtime、Transformers、NumPy 和 ModelScope 在插件初始化阶段开始惰性导入，使用 Nekro 的 `dynamic_import_pkg` 管理；插件模块导入阶段不会联网、安装依赖或加载模型。预加载在后台执行，不阻塞 Nekro 启动。预加载完成前收到的消息使用时间防抖，不会在消息回调中等待首次依赖安装或模型下载。`model_type` 会选择预设的 ModelScope 仓库；也可以提前放置模型文件。模型目录为：

```text
<plugin data dir>/models/<model_type>/model.onnx
<plugin data dir>/models/<model_type>/tokenizer/
```

模型缺失时插件初始化会尝试从 ModelScope 下载到插件专属缓存。依赖安装或下载可能在首次启动时消耗较多网络和磁盘资源。预加载失败时插件保持可用并退化为时间防抖。本地静态测试不能证明真实服务器的模型下载或 ONNX 推理成功。

频道执行 `/reset` 时，插件会取消该频道尚未释放的防抖批次，避免重置后旧消息延迟触发。

## Nekro API 差异

Nekro v0.4.1 不实现 AstrBot 的 ProviderRequest 改写、通用用户消息伪造、运行中 Agent 取消、旧回复丢弃和媒体消息重放。超时重放仅用于恢复本插件自己阻塞的文本批次；`cancel_on_new_message` 仅为配置兼容字段，不代表已提供取消能力。

## 验证边界

本仓库包含混合状态机、分级等待、概率回退、分类器数学适配、journal 序列化/恢复/幂等、timeout 和媒体边界测试。未运行 Docker、真实 Nekro、OneBot、ONNX 推理或模型下载，因此不能据此声称插件已在真实部署中加载或运行通过。
