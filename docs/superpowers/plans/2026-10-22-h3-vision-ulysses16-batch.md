# Qwen视觉编码 Ulysses16 + ViT组Batch 开发与验证计划

> For agentic workers: 使用 `superpowers:executing-plans` 按任务执行；本计划不派发子代理。以下复选框记录实施进度，计划交付不代表实现或验收通过。

**Goal:** 在个人分支实现视觉全栈Ulysses16与组Batch，固定六图负载，先证明相对原实现L1逐位一致，再证明视觉编码及DiT前总耗时降低。

**Architecture:** 每个视觉rank持有完整视觉权重，仅处理本地token；Attention前后以All-to-All转换token/head布局，同长度图按组Batch计算，各图上下文保持独立。视觉最终输出及全部DeepStack按原顺序汇合，交给原文本TP8；DiT USP8、VAE tile8及其数值路径保持基线设置。

**Tech Stack:** 当前个人分支PyTorch/torch_npu、HCCL、原SDPA及经过逐位验证的NPU后端、CPU pytest、原生CPU/NPU Level1 profiling。

**Spec:** 本文件的设计约束；固定用例 `runs/20261020_h3_ref2va_prompt/baselines/ref2va-turbo-8card-v1/case.json`（路径相对主仓 `D:/CodeHub/vllm-omni`）。旧 `vision-ulysses8-ring2-design.md` 仅作为索引、merger与通信布局参考，本期不实现Ring。

## 1. 全局约束与当前状态

- 工作树：`D:/CodeHub/vllm-omni/.worktrees/h3-p0-profiling`；个人分支：`codex/pre-encoding-opt-p0`。
- 2026-10-22只读核对：HEAD及本地feature指针均为 `717f8a3499fb7dc6b3128b58a8baeacded1a7bbb`；工作树有15个已暂存、未提交P0移植文件。实施前重新核对并保存索引/工作树差异与源文件摘要，不能覆盖或丢弃这些改动。
- 本期在上述个人分支继续开发；P0与视觉优化分开记录/提交，不向feature自动回写。每次修改或合入 `feature/pre-encoding-opt` 前必须通知用户并取得明确批准。
- 本轮交付计划；不启动模型、不占用远端设备、不更改集成分支。本计划不是16卡资源已获授权或已就绪的证明。
- 固定 `H3-REF2VA-TURBO-8CARD-v1`：原六图、原顺序、原提示词、Turbo文件及scale1.0、seed42、4步、flow_shift6/audio3、时长请求5s、16:9、现有INT8/RainFusion/DLO全部沿用case.json。
- 输入采用已固定主线预处理，**不恢复旧参考短边2048**；请求省略short_edge/fps，固定代码默认输出短边768、24FPS。既有成功产物为1344×768、124帧。
- 固定用例文件与manifest保持不可变；16卡只是独立实验臂的资源/视觉并行配置，不把v1改成16卡基线。
- 所有新增特性默认关闭；保留可单独关闭的SP与Batch开关。开关、回退和实际参与rank必须留痕。
- 本期不加入Ring、图缓存、跨请求视觉缓存、分辨率/token裁剪、量化变更、Qwen/VAE并发或文本CP2。
- 正确性按 `accuracy-gate` 的L1执行：跨实现逐元素 `torch.equal`，差异再小也不能自动降为L2；正式收益按 `perf-gate` 同窗验收。
- 最终目标平台为A2；A5仅作为临时开发与功能验证环境。并行、Attention后端及ACL Graph等候选均以A2可用性为约束，不将A5特有内核或收益作为交付依据。回归A2后，在同一A2环境分别运行未优化控制臂与优化臂，重新完成关键张量L1、完整六图和同窗性能验收；不要求A2与A5之间逐位相同，不以A5视频摘要替代A2同机控制臂。固定六图及数值配置不变，原A5冻结基线保留，A2建立独立平台运行记录。图在A2重新预热/捕获，不搬运A5已捕获的运行时图；通信入图与具体算子支持按A2实际软件栈验证。

## 2. 设计与审查重点

### 2.1 数据流

```text
固定六图预处理及全局grid/位置/段边界
  → 每段按完整2×2 merger组切分到P个视觉rank
  → 本地PatchEmbed/位置选择、LN/QKV/RoPE
  → All-to-All：本地token×16heads → 全段token×(16/P)heads
  → 恢复segment-major顺序，按同长度图分组Batch Attention
  → 恢复原图顺序，逆All-to-All → 本地token×16heads
  → 本地输出投影/残差/MLP，下一层继续保留token分片
  → 本地merger及全部DeepStack merger
  → 按全局token索引汇合，仅向文本TP8分发所需完整特征
  → 原文本编码 → 原VAE/DiT后续流程
```

- 当前视觉QKV/MLP已经packed；组Batch主要消减逐图Attention调用。不能把六张图连成一条全互注意力序列。
- 以部署config和实际输入核实heads、head_dim、层数及DeepStack索引，不将旧profile参数硬编码进通用路径。P必须整除视觉head数；P=16目标下每rank为1head，完整视觉权重不做TP切分。
- 原始RoPE/插值坐标、scale、dtype与操作顺序保持不变。全局坐标按原索引选择，不能按局部grid重新插值；物理kernel维度填充不能改模型scale。
- CPU侧每请求生成段长、分组与恢复映射，逐层复用；不引入跨请求metadata缓存。首次实现优先同长度组Batch，异长组分别计算；varlen作为独立后续子任务，只有同长度路径不能满足目标时再启用与单独对拍。
- 每图merger组不跨rank。非整除时以完整4-token组分配；通信buffer可填充，但padding不得参与有效token的Attention，原valid Q/K/V顺序不变。先验证无填充子集，再独立验证非整除实际六图；不默认padding无损。
- 五个重点失败面：非整除/极小/空分片；多图乱序或串图；DeepStack索引错位；某rank提前退出导致collective挂起；默认关闭路径或offload生命周期被改变。任务2～5逐项覆盖。

### 2.2 16卡进程与后续8卡的接入边界

源码当前用 `rank < text_encoder_tp_size` 同时限定整个编码器的加载和执行，且runtime world_size由DiT等并行维度推导。简单设置num_gpus16/usp8可能推导成DP2，不能作为本计划实现。

目标采用**显式stage角色**：16个视觉参与者，0～7继续承载原文本/DiT/VAE，8～15仅承载视觉角色。请求协议为PREPARE→VISION→FEATURE_HANDOFF→主8卡后续执行→DONE/ERROR；辅助rank在主8卡阶段等待有界状态消息，不进入DiT/VAE或其collective。

任务1必须验证并锁定启动、worker角色与stage子组的改动面；H3的后续8卡路径不能继续隐式使用16卡默认world group。若现有runtime无法小范围承接，先交付独立16rank视觉编码原型与证据，记录服务接入阻塞及具体扩展方案，**不以DiT USP16、DP2或张量回放冒充原配置端到端验收**。

### 2.3 内部接口与文件职责（拟新增，不是已有API）

| 文件，相对个人工作树 | 职责/接口 |
| --- | --- |
| `vllm_omni/diffusion/models/minimax_h3/vision_parallel.py`，新增 | `VisionShardPlan`保存原段长、有效索引、4-token组归属与通信布局；`build_vision_plan(lengths: tuple[int, ...], sp_size: int, merge_size: int) -> VisionShardPlan`；索引scatter/gather、正逆A2A及恢复映射 |
| `vllm_omni/diffusion/models/minimax_h3/vision_batch.py`，新增 | `VisionBatchPlan`保存同长度分组与逆映射；`build_batch_plan(lengths: tuple[int, ...]) -> VisionBatchPlan`；`run_grouped_attention(q, k, v, *, plan, scale)`使用统一BSHD约定，显式转入后端布局 |
| `vllm_omni/diffusion/models/minimax_h3/encoder.py`，修改 | VisionAttention/Block/Model接收视觉上下文；拆分视觉和文本加载、驻留与调用；merger与全部DeepStack恢复；保持旧构造/默认forward兼容 |
| `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py`，修改 | 独立vision group、输入分发、16→text8特征交接和错误协议；P0区间及实际生效计数 |
| `vllm_omni/diffusion/data.py`、`vllm_omni/diffusion/distributed/parallel_state.py`，按任务1审计修改 | 独立视觉配置、组生命周期、角色和合法性检查；禁止静默推导DP2 |
| `vllm_omni/diffusion/worker/diffusion_worker.py`及实际executor启动文件 | 任务1列出精确启动调用链后补stage角色接入；修改面先写入实施记录，不做全框架重构 |
| `tests/diffusion/models/minimax_h3/test_minimax_h3_vision_parallel.py`，新增 | token/head/segment索引、非整除、merger/DeepStack、错误与关闭路径 |
| `tests/diffusion/models/minimax_h3/test_minimax_h3_vision_batch.py`，新增 | 同长度组、恢复顺序、图间隔离、后端数值、回退和空输入 |
| `tests/diffusion/models/minimax_h3/test_minimax_h3_vision_stage.py`，新增 | 16视觉/8后续角色、组成员、请求顺序、异常收敛、重复请求、offload |
| `benchmarks/minimax_h3/vision_encoding_probe.py`，新增 | 固定六图真实视觉权重的隔离重放，输出完整比对清单、通信计数、阶段计时；不能代替HTTP验收 |

配置首版用H3局部门控：`VLLM_OMNI_H3_VISION_SP_SIZE=1/2/4/8/16`、`VLLM_OMNI_H3_VISION_BATCH_MODE=off/equal_length`，默认1/off。worker启动资源必须显式匹配，环境变量不会自动创造16个rank；后续如转正式CLI需另做配置贯通与非法组合测试。

现有 `distributed/qkv_a2a.py` 含CUDA/Triton与对称内存路径，不能直接视为NPU实现。首版使用经实际torch_npu/HCCL确认的公开collective；先把交换与数值做对，再考虑合并QKV通信、异步流或布局融合。

## 3. 开发任务与阶段出口

### 任务1：锁定基线、真实形状和runtime接入面

- [ ] 保存当前个人分支HEAD、P0差异、输入/权重/镜像与源码manifest；对冻结case的manifest重新校验。
- [ ] 从固定六图当前预处理获得每图grid、有效段长、merger组与全部DeepStack层索引。26176/6544只作已有读数校验，不代替实际六图分布。
- [ ] 列出executor→worker→init_world_group→H3编码→DiT的精确组归属，给出16视觉/8后续的启动与错误状态图；审核所有默认world group引用。
- [ ] 写负例测试：SP16但仅8rank、heads不整除、辅助rank错误进入DiT、world16被解析成DP2必须明确拒绝。
- [ ] 输出 `runtime-contract.md`、`shape-contract.json`、`source-manifest.json`。出口：配置与角色接入可实现且数值范围不扩大；尚不测优化性能。

### 任务2：建立逐位对拍载体与纯索引协议

- [ ] 先写失败测试 `test_roundtrip_preserves_segment_head_order`、`test_uneven_merge_groups_covered_once`、`test_deepstack_restores_global_indices`，实现任务2的最小索引函数后重跑通过。
- [ ] 使用可识别图/token/head的整数数据验证scatter→正A2A→逆A2A→gather逐位恒等，覆盖P=1/2/4/8/16、空片段、非整除和媒体乱序。
- [ ] 隔离载体接受 `--case <case.json> --sp-size P --batch-mode off|equal_length --mode correctness|timing --output <new-run-dir>`，拒绝覆盖已有run目录。
- [ ] 原实现同配置重跑两次，验证视觉输入、每层Attention/Block、最终视觉特征、全部DeepStack逐位稳定；若原实现不稳定，先定位确定性问题，不以浮点容差放行。
- [ ] 出口：每个比对点有shape/dtype/逐位结果、首个不等位置、max_abs_diff及张量摘要。CPU测试仅证明索引逻辑，不能作为NPU数值通过证据。

### 任务3：单独实现ViT同长度组Batch

- [ ] 写失败测试 `test_batch_isolates_images`、`test_batch_restores_original_order`、`test_batch_off_preserves_reference`；按组打包、一次SDPA后逆映射，异长组分别执行。
- [ ] 先SP1验证，原QKV/MLP数学路径保持原样，排除SP影响；若六图没有可合并同长度组，记录命中为0，不声称Batch有效。
- [ ] 固定输入/权重/dtype/scale，对真实六图的单Attention→Block→整塔→全部merger输出逐位对拍。
- [ ] 如果Batch改变内核/累加次序而不逐位，先定位首处分歧，保持原token和数值语义寻求等价实现；两轮有依据的最小修复后仍不通过，记录阻塞与回退，不进行该臂性能验收。
- [ ] 出口：Batch路径真实命中、正确性L1通过、默认off回归通过。若引入varlen，单列新测试/新实验臂，不能混入本任务结果。

### 任务4：单独实现全视觉栈Ulysses

- [ ] Batch关闭，从P=2→4→8验证真实NPU通信，再在真实16rank环境验证P=16；无16卡时停在“8卡开发验证完成、16卡未验证”。
- [ ] 每层只交换所需QKV/O，层间hidden持续本地分片；不能每层gather完整hidden再计算完整塔。
- [ ] 先验证无padding段，再验证实际六图的非整除处理；严格保留K/V有效序列和图边界。所有rank在进入collective前统一决议失败/回退，禁止单rank自行返回。
- [ ] 对拍本地投影、RoPE、正A2A后QKV、Attention、逆A2A、MLP、每个Block、最终视觉/全部DeepStack；检查改变GEMM行数是否已导致首处分歧。
- [ ] 输出真实rank清单、每层token数/heads、collective组ID/次数/字节、peak memory，证明16卡均参与有效计算且没有重复完整vision。
- [ ] 出口：SP独立L1通过。A2A是纯置换并不能保证整塔逐位；任何投影/Attention数值失败均为未通过，不用“数学等价”替代。

### 任务5：组合、服务接入与完整六图正确性门

- [ ] 组合SP与Batch，重新验证同一批比对点，不把两个单项通过推定为组合通过。
- [ ] 接入独立视觉加载/驻留、16视觉→text8交接与stage角色；后8rank不加载/执行文本或DiT/VAE，原8卡路径保持原组成员及数值配置。
- [ ] 比较原/新路径的原序最终视觉与全部DeepStack、文本hidden、条件张量、初始noise/latent、每步去噪latent以及解码前输出，均要求torch.equal；同时检查请求内RNG状态/调用顺序。
- [ ] 每种配置至少两次相同请求，确认自身稳定；对通过的配置检查完整视频字节数及SHA256与同窗原路径一致。HTTP200/媒体一致不能替代中间张量L1。
- [ ] 覆盖连续请求、错误后下一请求、offload开关生命周期；输入为空/只有一图/极小异长图用作回归，不替代固定六图验收。
- [ ] 出口： `correctness.json`、逐比对点摘要、视频SHA、实际执行计数全部通过，才可开放对应臂的性能测量。

### 任务6：同窗性能与profiling归因

- [ ] 所有臂使用任务5已通过的同一份源码快照，仅切门控；同一16卡预约窗口，臂之间串行、独立服务、清理自有进程并确认资源。
- [ ] 每臂固定两次4步预热（profiler外），之后5次热态完整六图请求；记录每次结果，报告中位数、均值、范围、离散度。历史18.155538s不能作本窗分母。
- [ ] 保留以下实验矩阵；服务接入未完成时只对隔离视觉载体执行对应矩阵，明确不作HTTP/DiT前总耗时结论。

| 臂 | runtime资源 | 视觉SP | 组Batch | 用途 |
| --- | --- | ---: | --- | --- |
| A | 原8卡服务，额外8卡保持空闲 | 1，原8rank重复视觉路径 | off | 同窗固定v1控制臂 |
| A0 | 新16参与者框架，辅助8rank不算vision | 1，保持原视觉路径 | off | 证明stage接入本身无损，并测框架新增开销 |
| B | 同A0，后续仍原8卡 | 1 | equal_length | Batch单项净收益 |
| C | 同A0，后续仍原8卡 | 16 | off | U16单项净收益 |
| D | 同A0，后续仍原8卡 | 16 | equal_length | 目标组合，最终相对A验收 |

- [ ] 测量三层：完整视觉阶段（含scatter、重排、merger、全部特征汇合和交接）、pipeline进入到首DiT进入的同口径墙钟、完整HTTP墙钟。视觉局部内核加速不能替代DiT前总耗时收益。
- [ ] 每rank/每图/每层分解compute、A2A、重排、padding、汇合、空闲与峰值显存，parent/child区间用互斥或union口径，不累加重叠时间。Host P0不等于设备完成时间，独立trace核实真实先后。
- [ ] 正式计时关闭张量dump、调试同步和原生profiler；保持各臂同等轻量计时。隔离载体采用起止同步的完整阶段墙钟，HTTP按原请求边界，不能为候选臂单独插入barrier。
- [ ] 根据当前窗口控制臂的实际重复离散与前后漂移估计噪声地板，不复用旧环境固定百分比。收益>3×地板才直接采纳；接近地板追加A→D→A，各段仍5次热请求；仍不可分辨则不宣称收益。
- [ ] 只有L1通过的臂才做性能实验；至少对A/D各采一次独立CPU/NPU Level1/PipeUtilization全阶段trace，核对实际16视觉/8后续rank与四族有效利用率，采样请求不入性能均值。
- [ ] 指标必须同时给出HTTP状态、视频字节/SHA、阶段账与门控命中。净收益=本窗A耗时−D耗时；额外报D相对A0及B/C贡献，不将单项加速比相乘。

### 任务7：收口与分支交付

- [ ] 运行新增CPU测试、现有H3 encoder/parallel/collectives/offload/DLO测试、P0测试、ruff、格式与diff检查；缺依赖或平台跳过单列，不声称全套通过。
- [ ] 整理配置、源码manifest、运行环境/拓扑、correctness、timings、profile、视频与日志。性能/精度证据与对应提交或源码快照一一关联。
- [ ] 只在个人分支形成可审查diff/提交，不修改feature。报告“哪个区间耗时、为何耗时、优化后减少了什么”，明确收益、未通过项和回退门控。
- [ ] 最终提交用户审阅；合入feature前另行通知并取得明确批准。

## 4. 本地验证入口与产物

以下命令在个人工作树、具备当前项目依赖的环境执行；新增文件在对应任务实现后才存在。先验证测试确实失败，再实现到通过。

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_minimax_h3_vision_parallel.py tests/diffusion/models/minimax_h3/test_minimax_h3_vision_batch.py tests/diffusion/models/minimax_h3/test_minimax_h3_vision_stage.py -q
python -m pytest tests/diffusion/models/minimax_h3/test_minimax_h3_encoder_sdpa.py tests/diffusion/models/minimax_h3/test_minimax_h3_parallel.py tests/diffusion/models/minimax_h3/test_minimax_h3_collectives.py tests/diffusion/models/minimax_h3/test_minimax_h3_offload.py tests/diffusion/models/minimax_h3/test_minimax_h3_dlo_lifecycle.py tests/profile/test_h3_p0.py -q
git diff --check
```

ruff与格式检查按最终实际改动文件清单运行，不全仓自动修复。NPU验证命令待任务1记录真实容器、路径、拓扑和可用卡后生成；本计划不提供未经验证的 `num_gpus16 --usp8` 启动命令。

本地实验产物统一到主仓 `runs/20261020_h3_ref2va_prompt/vision-ulysses16-batch/<host>/<run-id>/`；每次目录独立。至少包括 `runtime-contract.md`、`shape-contract.json`、`source-manifest.json`、`correctness.json`、`timings.jsonl`、`profile/`、`media/`、`logs/`、`report.md`、源/目标包SHA及清理回执。case/权重/媒体不外网上传，内部SSH回传后检查摘要与完整性。

## 5. 资源、预算与停止条件

- 可先在172或183已授权的物理0～7卡完成SP1/2/4/8与Batch验证；运行前核实匹配镜像/源码/权重、磁盘和拓扑，健康OK、AICore利用率0且无NPU进程，间隔10秒确认两次。
- 真U16需要16张可用、拓扑明确的设备。不能把8卡虚拟成16rank的数值/性能结果，也不能将两台机器各8卡自动拼为已获授权的跨机16卡资源。缺资源时继续本地/8卡开发并明确16卡验证未完成。
- 每次NPU正确性或性能窗口最长90分钟；服务启动1350秒、每请求1800秒、start_profile120秒、stop_profile300秒，均受总90分钟优先约束；单次监控等待不超过60秒。开始新臂前估计剩余预算，不足则保留已完成结果、结束窗口。
- 不创建新的等待自动任务；现有完成后的自动任务保持暂停。只清理本次自有PID/PGID，不干预其他任务；共享卡出现外部负载则该样本受干扰，不用于干净性能验收。
- 数值分歧先定位首点，禁止提前跑该臂正式性能；只因网络/设备瞬态失败可在预算内有依据重试未完成组。稳定复现同错转定位；不能确认清理、资源/空间不足或到时即停止并保留现场。
- 性能不增益：先按trace核实通信、重排、padding、内核效率与汇合的净成本，再做有证据的小范围改进；不能通过更换六图、缩图、缓存编码结果或改变DiT/VAE配置制造收益。

## 6. 本轮计划自检

- 已覆盖个人分支/feature批准边界、不可变六图基线、两项独立开关及组合复验。
- 已覆盖索引/位置/图间隔离/merger/DeepStack、异长非整除、错误收敛与offload。
- 已将16视觉/8后续runtime接入列为前置硬门；没有假设现有USP参数自动控制视觉，也没有把8卡结果视作16卡证明。
- 已将逐位、机制、性能按顺序分门，旧时间与新控制臂分开，profiling计时不进入性能统计。
- 未实施模型代码，未创建或移动任何Git分支，未提交/推送，未运行NPU；本计划中的测试通过项均尚待执行。

## 7. 开发前复用审计（2026-10-22）

用户要求先核查Qwen3-VL已有优化，已完成当前H3、标准vLLM及官方vLLM-Ascend源码/PR审计。详见主仓`runs/20261020_h3_ref2va_prompt/vision-optimization-research.md`的2026-10-22更新。

- 当前六图已经一次VisionModel调用，PatchEmbed/QKV/MLP已处理拼接token；本计划的Batch严格指尚逐段执行的Attention，不能宣称减少六次完整ViT调用。
- 标准vLLM已有视觉TP/按图DP、packed varlen、位置索引缓存及CPU元数据准备；H3自定义类不自动继承。MMEncoderAttention的SDPA fallback仍可能逐段执行，必须验证实际后端调用。
- Ascend当前main已有TND FIA与ViT ACL Graph；历史7104可借鉴请求内CPU元数据复用，但早期6204的统一QKV padding已不等同当前main代码。图执行保持后置，不自动加入本期。
- Ascend PR14574仍open且未合并，是TP+SP混合候选；其MLP带ReduceScatter且fused策略未实现，不能当作纯Ulysses16开关。参考索引/padding/DeepStack恢复，保留H3现有数值路径和独立视觉组要求。
- 实施顺序保持先任务1/2配置与逐位载体；请求内段长复用、同长度Attention Batch、变长/FIA与SP分别留痕、单项验证，不同时改变后端及并行后再归因。固定基线与feature审批边界不变。

## 2026-10-22 实施检查点（配置与对拍载体）

P0已独立提交783101992；本轮新增仅属于Ulysses准备工作。真实六图CPU配置核实完成：段长8160/8160/880/660/660/7656，总26176，merger6544，16heads×72dim，27层及DeepStack8/16/24。新增CPU分片协议、角色负例和真实权重SP1/off对拍载体；不启用实际SP/Batch，不改模型forward。

任务1配置/源码接入审计已完成；服务角色协议仍待实现。任务2载体代码与CPU测试已完成，NPU同配置重跑未通过环境门：183容器设备namespace冲突，h3-lsy旧vLLM导入当前源码失败，172有他人任务。不得将任务2的NPU出口标记完成；后续先解除环境阻塞，再进入优化实现/性能。详细运行记录在主工作区runs/20261020_h3_ref2va_prompt/vision-ulysses16-batch/183/vision-contract-20261022/runtime-contract.md。

Ruling: 将本阶段载体作为单独提交，不引入尚未验证的真实collective、Batch或服务角色修改；原因是需要先建立可靠对拍基准，环境失败不应驱动业务数学路径变更。代价是NPU出口与U16仍未完成。

## 2026-10-22 后续检查点（取代上文环境阻塞状态）

183新自有兼容容器设备访问已恢复，SP1同配置123点通过，载体提交5b25760fa。隔离Ulysses已实现并验证SP2各rank123点逐位通过。SP4首层Attention失败；单卡重放定位到短段少head调用形状差异，默认关闭的短段重复head诊断选项第一层通过，全层尚待验证。183卡4出现他人任务后停止新NPU启动；CPU17测试及独立代码复核完成。任务4仍进行中，SP8/16、Batch、服务集成、A2和性能出口均未完成。详见主仓vision-ulysses16-batch/183/ulysses-sp2-20261022/report.md。
最新出口：短段min-heads=8候选在183 A5固定六图完成SP4与SP8各两次123点/rank逐位一致，重跑摘要相同；本地回传校验与独立证据复核完成。用户授权共享窗口只用于正确性，未测性能。下一阶段仍需Batch独立对拍、服务角色接入、真实SP16与A2回归。
