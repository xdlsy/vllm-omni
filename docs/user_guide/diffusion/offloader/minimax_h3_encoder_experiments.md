# MiniMax-H3 Encoder 优化：NPU 验证清单与计划

本轮以降低 `encode_prompt` 总耗时为首要目标，优先优化 `encoder_offload` 和
`device_memory_release`，要求 Encoder 输出逐位一致，最终 pipeline 端到端耗时不回退。
八项开关默认全部关闭。**目前没有 NPU 性能实测结论**；CPU 生命周期测试不能证明 NPU
正确性或加速效果。服务器上必须重新采集同窗口基线，不能直接用历史耗时判定收益。

## 1. 适用范围与限制

适用于未量化的 MiniMax-H3 Qwen3-VL Encoder，已接入 pipeline 组件卸载和模型级
sequential offload。保持原来的 Encoder 组件卸载配置；如果 Encoder 从不进入载入、卸载
生命周期，这些开关不会被执行。本轮不修改编码数学计算、TP 通信、DiT、VAE 或最终产物编码。

**每个实验配置都要重启服务**：开关在首次 Encoder 迁移前读取一次。所有 Encoder rank、
所有节点上的 worker 必须收到相同配置，不能部分 rank 开启、部分关闭。

不支持原有 Encoder layerwise offload、HSDP、量化权重以及跨卸载分组共享存储的情况，
遇到不支持的配置会报错。这些实验不用于训练。
CPU master 和权重常驻要求参数及 **buffer 都不改变**；首次迁移后不能重新加载权重、
原地修改参数或状态 buffer。有可变推理状态时，应使用保留回写的配置。
仅设置 `requires_grad=False` 不能证明状态不变。

## 2. 八项开关及实际行为

以下开关名统一加前缀 `VLLM_OMNI_ENCODER_`。
H2D 指 CPU→NPU，D2H 指 NPU→CPU。

| 开关 | 默认值 | 开启方式与行为 |
| --- | --- | --- |
| `FROZEN_MASTER` | `0` | 设为 `1`：对最终加载完成的权重保留一次 CPU 快照；卸载时重新绑定 CPU 存储，不回写设备权重。 |
| `CACHE_RETENTION` | `0` | 设为 `1`：缓存不超过设备容量 25%、物理空闲不少于 5% 时保留分配器缓存；越界、读取信息失败或异常时清理。单独开启不取消权重回写。 |
| `PINNED_BUFFERS` | `0` | 设为 `1`：一次分配、反复使用 pinned CPU 缓冲区。单独开启仍保留 D2H；与 CPU master 的组合另行测量。 |
| `RESIDENT_LAYERS` | `0` | `-1` 表示整个 Encoder 常驻；正数 N 表示前 N 个文本层常驻。常驻组首次载入后不再重复 H2D、D2H。 |
| `PACKED_TRANSPORT` | `0` | 设为 `1`：每个卸载组打包为对齐的连续字节缓冲区，按组传输，保留参数共享关系、stride 和原始 offset。 |
| `GROUP_SIZE` | `0` | `0` 为整模型；`1` 为逐 block；N>1 为每组至多 N 个连续 block。视觉、文本分别分组，非 block 模块保留至组件退出。 |
| `PREFETCH` | `0` | 设为 `1`：当前组计算期间，在 copy stream 预取下一组，消费前等待完成事件。必须同时设 `PINNED_BUFFERS=1`、`GROUP_SIZE>=1`。 |
| `REUSE_BUFFERS` | `0` | 设为 `1`：保留显式设备缓冲区，每次载入仍从 CPU 更新内容；不能与权重常驻组合。 |

三种设备存储必须区分：

- 分配器缓存：没有活跃张量使用、可供分配器再次分配的空间。
- 权重常驻：活跃设备存储，直接复用原有权重内容。
- 设备缓冲区复用：活跃设备存储，每次载入重新写入内容。

后两种即使参数已重新绑定到 CPU，也继续占用设备容量。复用缓冲区属于**各组独立持有**，
不是所有组共享两个槽位；各组都运行后，总容量可能达到整个 Encoder 的大小。
因此它不是低显存卸载方案，也不代表已经支持 CUDA/NPU 图捕获。
常驻和缓冲区复用都必须通过后续 DiT、VAE 和端到端峰值显存检查。
整模型常驻（`-1`）不能与分组组合；部分文本层常驻可以与分组组合。
打包初始化期间还会临时同时占用原始与打包后的宿主存储。

## 3. 启动配置与逐项对照

将 `SERVICE_CMD` 设置成**现有且已跑通的服务启动命令数组**，保留原来的 NPU、TP 和
卸载参数。下面不是新的模型启动命令模板，只是在原命令前切换实验开关。
不同实验之间保持请求、采样、prompt、分辨率、seed、步数、并发和权重完全一致。
性能计时请求不要开启 profiler；profiling 单独采集。

```bash
# 仅展示数组写法： SERVICE_CMD=(bash /absolute/path/to/existing-launch.sh)
launch_arm() (
    arm="$1"
    shift
    unset VLLM_OMNI_ENCODER_FROZEN_MASTER VLLM_OMNI_ENCODER_CACHE_RETENTION
    unset VLLM_OMNI_ENCODER_PINNED_BUFFERS VLLM_OMNI_ENCODER_RESIDENT_LAYERS
    unset VLLM_OMNI_ENCODER_PACKED_TRANSPORT VLLM_OMNI_ENCODER_GROUP_SIZE
    unset VLLM_OMNI_ENCODER_PREFETCH VLLM_OMNI_ENCODER_REUSE_BUFFERS
    case "$arm" in
        baseline) ;;
        frozen) export VLLM_OMNI_ENCODER_FROZEN_MASTER=1 ;;
        cache) export VLLM_OMNI_ENCODER_CACHE_RETENTION=1 ;;
        pinned) export VLLM_OMNI_ENCODER_PINNED_BUFFERS=1 ;;
        resident-all) export VLLM_OMNI_ENCODER_RESIDENT_LAYERS=-1 ;;
        resident-prefix) export VLLM_OMNI_ENCODER_RESIDENT_LAYERS=4 ;;
        packed) export VLLM_OMNI_ENCODER_PACKED_TRANSPORT=1 ;;
        layer) export VLLM_OMNI_ENCODER_GROUP_SIZE=1 ;;
        blocks) export VLLM_OMNI_ENCODER_GROUP_SIZE=4 ;;
        prefetch-control|prefetch)
            export VLLM_OMNI_ENCODER_PINNED_BUFFERS=1
            export VLLM_OMNI_ENCODER_GROUP_SIZE=1
            if [ "$arm" = prefetch ]; then export VLLM_OMNI_ENCODER_PREFETCH=1; fi
            ;;
        reuse) export VLLM_OMNI_ENCODER_REUSE_BUFFERS=1 ;;
        *) echo "Unknown arm: $arm" >&2; exit 2 ;;
    esac
    exec "$@"
)
launch_arm baseline "${SERVICE_CMD[@]}"
# 每轮停止服务后，只选择下面一种配置，重放完全相同的请求：
# launch_arm frozen "${SERVICE_CMD[@]}"
# launch_arm cache "${SERVICE_CMD[@]}"
# launch_arm pinned "${SERVICE_CMD[@]}"
# launch_arm resident-all "${SERVICE_CMD[@]}"
# launch_arm resident-prefix "${SERVICE_CMD[@]}"
# launch_arm packed "${SERVICE_CMD[@]}"
# launch_arm layer "${SERVICE_CMD[@]}"
# launch_arm blocks "${SERVICE_CMD[@]}"
# launch_arm prefetch-control "${SERVICE_CMD[@]}"
# launch_arm prefetch "${SERVICE_CMD[@]}"
# launch_arm reuse "${SERVICE_CMD[@]}"
```

| 实验名 | 验证内容 | 对照 |
| --- | --- | --- |
| `baseline` | 全关基线 | 每轮重新测量 |
| `frozen` | CPU master 取消回写 | 全关基线 |
| `cache` | 有界缓存保留 | 全关基线 |
| `pinned` | pinned 缓冲区复用，保留回写 | 全关基线 |
| `resident-all` / `resident-prefix` | 全部常驻 / 前 4 个文本层常驻 | 全关基线；检查设备容量 |
| `packed` | 合并传输 | 全关基线 |
| `layer` / `blocks` | 逐 block / 每组 4 个 block 卸载 | 全关基线 |
| `prefetch-control` | pinned＋逐 block，不预取 | 预取的直接控制组 |
| `prefetch` | 在上述控制组基础上开启预取 | `prefetch-control`，并同时与全关基线比较 |
| `reuse` | 复用设备缓冲区，但重复传输内容 | 全关基线；检查设备容量 |

建议顺序：CPU master → 有界缓存 → pinned → 容量允许时测试常驻 → 合并传输 → 分组 →
预取 → 设备缓冲区复用。先逐项验证，再测组合。**每项与自己的控制组比较，不能只与上一个实验比较。**
先用单 Encoder rank 做功能快验，再恢复原始 Encoder TP 配置验收；不同 TP 配置的耗时不能直接比较。

## 4. NPU 验收步骤

1. **固定环境与请求**：记录 commit、工作区差异、权重标识、torch/torch_npu、vLLM、CANN、
   驱动、卡号、拓扑、TP、空闲内存和其他进程。固定输入、seed、产物编码及同步边界；
   确认 HTTP 200、响应字节数和对应请求日志。
2. **预热与冷启动分开**：profiler 外至少预热 5 个完整请求；开启编译时至少 10 个。
   单独记录首次快照、pin、打包等初始化成本，不能只报告稳态而隐藏冷启动代价。
3. **同窗口 A/B/A**：每种配置至少测量 3 个请求；每次重启使用相同预热和测量次数。
   保存原始样本，评估漂移和噪声。差异落在噪声范围内时增加样本，不能判为有效；
   不使用跨天绝对耗时作为收益证据。
4. **先计时、再单独采 profiling**：A/B 使用相同阶段测点与必要的设备同步。
   分组会把权重迁移移入 `encoder_forward`，所以必须比较完整 `encode_prompt`，
   不能仅因出口卸载变短就认定加速。profiler 采样请求不计入性能均值。
5. **记录内存并测到 pipeline 结束**：每个边界记录各 rank 的 RSS/PSS、pinned/锁页内存、
   设备 allocated/reserved、物理空闲以及单请求峰值。可参考平台统计和
   `/proc/PID/status` 中的 `VmLck`/`VmPin`；不可获取时明确标记，不将其默认为零。
   继续测量 DiT、VAE 和端到端；发生 OOM 或端到端回退，不予采纳。
6. **输出一致性**：在 `result_to_cpu`/分发之前，使用相同采集路径保存各 rank 的 Encoder
   隐藏张量。基线重跑、优化重跑、基线与优化之间都要求 dtype、shape、逐位结果一致；
   保留原始字节比较，包含 NaN payload。最终图片/视频文件 MD5 必须不变，并检查基线
   本身是否有坏图、坏帧；MD5 相同不能单独证明基线健康。任何差异都阻止验收。
7. **稳定性与异常清理**：覆盖纯文本、图片、视频、长短序列、变形状、重复及交替请求，
   至少连续 20 轮。覆盖无视觉输入、末组不足整组、多 Encoder rank。检查受控 OOM、
   拷贝失败、组内前向失败、预取未消费时取消请求，以及 sequential offload 关闭后重新开启。
   确认清理正确、下一请求正常、无 pinned/设备内存持续增长、无重复钩子。
   不在共享生产服务器上注入致命设备故障。

## 5. 阶段耗时记录表

保持下列阶段列名。单位为秒。参考值来自此前测试，**不是本分支的测量结果**；
验收时填写服务器上同窗口的新基线与实验值。

| 阶段 | 历史参考值（s） | 新基线 A（s） | 实验 B（s） | 重复基线 A（s） |
| --- | ---: | --- | --- | --- |
| `input_prepare_and_distribute` | 0.656797 | | | |
| `encoder_load` | 0.614717 | | | |
| `encoder_forward` | 0.339266 | | | |
| `result_to_cpu` | 0.228486 | | | |
| `encoder_offload` | 1.132678 | | | |
| `device_memory_release` | 0.524752 | | | |
| `result_distribute` | 0.027588 | | | |
| `encode_prompt` | 3.524284 | | | |

同时记录 DiT、VAE、产物编码和 pipeline 端到端耗时。
仅 `encoder_offload` 或 `device_memory_release` 下降，不等于通过验收。

## 6. 用 profiling 确认优化实际生效

开启后会记录日志 `encoder residency options=... lifecycle=... transport=...`。
对累计 H2D/D2H 字节数、拷贝次数、`device_allocations`、`buffer_reuses` 取逐请求差值。
单独开启缓存保留时没有 stager 传输计数，需要看实际分配器统计与拷贝事件。
计数按底层字节存储传输统计，包含打包对齐，不等于参数个数；必须与 NPU memcpy 事件
交叉核对。计数器不能证明重叠或加速。

| 方案 | 必须观察到的变化 |
| --- | --- |
| CPU master | 首次快照后，权重 D2H 字节数、次数消失；激活的 `result_to_cpu` 仍存在。 |
| 有界缓存 | 目标释放区间内 `FreePhysical`/`UnmapMem` 减少；超过边界或信息不可读时仍清理。检查开销是否移到后续阶段。 |
| pinned | 初始化后宿主缓冲区地址与 pinned 占用稳定；CPU master 关闭时仍有正常 D2H。 |
| 常驻 | 选中组的设备地址和分配容量保留，首次加载后不再 H2D/D2H；未选中组仍迁移。 |
| 合并传输 | memcpy 次数减少、单次块变大，总字节数与打包大小对应；共享存储和非连续张量检查通过，双向都统计。 |
| 分组 | 每组首 block 前载入、末 block 后卸载；同时活跃组数量受限，并计入非 block 模块和常驻组。 |
| 预取 | 在 copy/compute stream 中看到下一组传输与当前组计算重叠，下一组消费前拷贝完成；组尾同步仍存在，必须测实际净收益。 |
| 设备缓冲区复用 | 各组首次申请后不再重复分配，地址稳定、`buffer_reuses` 增长，但每次载入仍有 H2D；区别于分配器缓存和免拷贝常驻。 |

## 7. CPU 验证结果与尚未验证的内容

```bash
python3 tools/run_encoder_residency_cpu_tests.py
```

此入口使用真实 PyTorch 张量、视图、共享存储、存储重绑定和前向计算；替代了 vLLM
包初始化、日志和平台 stream/event 操作。Encoder 与 sequential 生命周期方法从源码
载入，不导入完整模型 kernel；同时运行原有共享 stager 测试。
本地 CPU 主机无法验证设备锁页分配，因此 pin 分配也使用模拟。

本次本地结果：Python 3.13 / torch 2.10.0 CPU，**42 项通过**，其中新增实验 25 项、
原有共享 stager 17 项。Ruff、格式、SPDX、测试标记、禁止导入、设备调用和补丁空白检查通过。

本地未安装 vLLM，正常 `python3 -m pytest --maxfail=1` 无法初始化
`tests.helpers.fixtures.config`。完整 pre-commit、mypy、Markdown lint 本地不可用，
仍须在完整目标环境执行。

**CPU 测试不能证明 NPU DMA/event 时序、分布式通信、完整模型输出一致性、容量或性能。**
必须完成正常仓库测试和以上 NPU 验收后，才决定是否采纳或更改默认值。
