# Budget Agent

**固定推理预算下的 Coding Agent：上下文管理与代码修复评测。**

GitHub：[amzing2334-jpg/budget-agent](https://github.com/amzing2334-jpg/budget-agent)（私有）。代码、测试、文档及选定的 v5 实验记录已纳入版本管理；模型权重、运行环境和云端连接材料保留在本地。仓库范围与历史产物位置见 [仓库说明](docs/REPOSITORY.md)。

**2026-10-28 第十轮：真实参考资源复核。** Featurize公开页已显示符合原方案的4090主机候选（14核/60.1GB，1.87元/小时），但当前会话未登录，旧账单与实际分配未核验。源码包一致性、部署脚本静态检查和官方模型价格已复核；未改Agent代码、未租用或调用模型。见[执行前状态与阻塞条件](docs/REFERENCE_PREFLIGHT_2026-10-28.md)。

**2026-10-28 第九轮：子测试执行证据与奖励准入。** 修复混合通过/失败与跳过子测试被误判未评分的问题；在线与离线共用判据，全部跳过仍无奖励，旧记录不反推。真实CPU子进程与固定TokenPolicy验证奖励1/0/未评分，未训练。CPU **781项通过、13项跳过**，真实OpenJudge **142项通过**。见[验收与剩余边界](docs/SUBTEST_EXECUTION_VALIDATION.md)。

**2026-10-27 第八轮：评分结论与测试明细一致性。** 顶层成功、评分状态和测试明细矛盾时保持未评分，不进入成功率、配对胜负或 OpenJudge 通过项；非法明细不再使批次中断。59份历史产物判定与哈希不变。CPU **761项通过、13项跳过**，真实OpenJudge **122项通过**。见[复现、官方需求与边界](docs/VERIFIER_CONSISTENCY_VALIDATION.md)。本轮没有模型效果或新费用。

**2026-10-27 第七轮：请求级预算审计修复。** 缺失/非法请求记录保留为unknown；明确单请求超额不能被正常汇总标志掩盖。59份历史产物结论不变，合成损坏批次完整落盘。CPU **724项通过、12项跳过**，真实OpenJudge **84项通过**。见[复现、官方需求复核与边界](docs/REQUEST_LEDGER_VALIDATION.md)。本轮没有模型效果或新费用。

**2026-10-27 第六轮：补齐失败证据重呈对照。** 新增`reflection_policy=evidence`，在相同总预算下比较普通历史、确定性错误摘录和生成反思；原文有来源/版本/偏移，所有再输入正常计费，没有额外生成调用。CPU **684项通过、11项跳过**，真实OpenJudge **43项通过**。三条件流程已具备，真实模型收益尚未验证；见[设计与官方需求复核](docs/FEEDBACK_CONTROL_VALIDATION.md)和[固定动作演示](runs/iteration-20261027-05/demo-final/report/report.html)。

**2026-10-27：新增企业工程候选首版。** 基于固定源码研读 Reflexion、AgentScope OpenJudge、Claw-Eval 和腾讯 AIG，接入有总预算和次数上限的失败反思、严格工具/文件权限判决、固定公开测试命令、独立质量/预算/授权三态验收，以及真实 OpenJudge 异步适配。默认策略保持关闭反思；候选配置显式启用，不能把单次合同通过当生产认证。完整 CPU **656 项通过、11 项跳过**，专用真实 OpenJudge 集成 **42 项通过**，没有模型推理或训练。见[设计、源码映射与生产缺口](docs/ENTERPRISE_AGENT_DESIGN.md)、[可查看的脚本化正负演示](runs/enterprise-contract-v1/demo-final/report/report.html)和[验收回执](runs/enterprise-contract-v1/cpu-validation.json)。

**2026-10-27 第五轮：修复未完成评分被当作负奖励，以及首请求失败导致整批配对中断。** verifier 缺依赖、提前退出、零测试或全部跳过现在明确为未评分，不生成可训练 Episode；正常测试失败仍给 0 分。旧格式结果重新报告时使用同一判据，原始证据不改。配对保留缺失模型/用量来源的失败记录及排除原因，已知模型冲突仍拒绝。CPU **561 项通过、8 项跳过**；详见[复现、验收和能力边界](docs/GRADING_COMPLETENESS_VALIDATION.md)。云端仍无可用实例，本轮没有真实模型请求或训练。

**2026-10-27 第四轮：模型请求禁止隐式重定向。** 本地双HTTP服务复现了默认urllib在301/302/303后转发Authorization并接受跳转目标用量的问题；现在所有自动重定向均被拒绝，用量未知时保存记录并中止批量。正常请求不变，完整CPU套件529项通过、8项跳过，见[复现与验收](docs/API_REDIRECT_VALIDATION.md)。真实参考试跑已获本轮预算及配置授权，百炼专用Key已准备，但云主机分配504超时；[接入记录](docs/REFERENCE_PREFLIGHT_2026-10-27.md)与后续[迭代记录](docs/ITERATION_LOG.md)保留实际状态。尚无本轮真实模型修复结果。

**2026-10-27 第三轮：修复批量用量异常后的继续调度，准备固定模型参考试跑。** 未知、未对账或实际超额的请求现在会中止后续任务/策略/重复次数并保存部分报告；正常已知预算耗尽仍可继续对照。新增显式思考模式参数与 `qwen3.8-max-0902` 非思考候选配置，尚未调用真实服务。CPU 套件 516 项通过、8 项跳过，见[验收](docs/BATCH_STOP_VALIDATION.md)；下一轮具体模型、任务、云端资源和待授权费用见[参考方案](docs/REFERENCE_PILOT_2026-10-27.md)。

**2026-10-27 迭代：新增 `mask-read` 有界源码保留候选。** 在 mask 的全部动作和最近两条完整观察之外，利用明确的额外输入额度保留当前连续工作区版本的旧源码整行，按实际路径/行去重并保留来源。`read_memory_tokens=0` 与 mask 请求完全相同；默认 1024 是输入增量上限，所有内容照常计入总预算。历史前缀回放仅用于测可见性与成本，不是新模型实验或修复收益。详见[实现与验收](docs/READ_MEMORY_VALIDATION.md)和[本轮行业复核](docs/INDUSTRY_REVIEW_2026-10-27.md)。

**2026-10-26 迭代：新增可关闭的可见源码重读控制，以及 `strategy / decision_policy / loop_policy` 单因素配对评测。** 控制器只对实际请求中已经完整可见的相同行内容反馈，给模型一次纠正机会后才停止；反馈正常计费，原始轨迹和独立评分保留。历史七次尝试回放仅发现三次符合该窄判据的读取，三条 mask 轨迹均不触发，因此不声称解决旧停滞或提高修复率。实现、验收和复现命令见 [本轮验证](docs/READ_LOOP_VALIDATION.md)；后续每四小时工作从[迭代记录](docs/ITERATION_LOG.md)接续，岗位依据见[行业复核](docs/INDUSTRY_REVIEW_2026-10-26.md)。

当前主线是**真实仓库修复、可靠交互与可信评测**。v5 根据[笔记研读与项目回看](docs/FEISHU_NOTE_CAREER_ANALYSIS_2026-10-22.md)，加入有范围/分页的文件工具、可定位的编辑失败反馈、动作输出额度下限、离线行为诊断，以及固定版本 mini-SWE-agent 外部参考入口。后续实测又加入 `mask` 观察压缩和可选预算决策提示，分别进行同源码对照；详见 [v5 后续验收](docs/VALIDATION_V5_FOLLOWUP.md)。默认 `history_format=chat`，旧表示通过 `transcript` 保留。

**v5 新增的七次真实云端尝试仍未证明修复收益。** 固定 Qwen3-4B 的内部动作预留 1/128 对照和真实 mini-SWE 外部参考均为空补丁；同源码 `recent/mask` 配对把调用数从 8 增至 13，但没有产生编辑或测试。后续 `mask × direct/budget-aware` 配对也均无补丁。旧观察占用预算与“重复获取已读源码”已由原始轨迹定位，不能把工具调用更多写成能力提高。七次均为同一道已知开发题，不是七道独立题。最终 CPU 回归 376 项通过、8 项跳过；另有固定上游 6 项 CPU 集成通过。A4000 已在授权时限内归还并由控制台核验。完整输入、补丁、独立官方评分与分析见 [云端产物清单](runs/v5-followup-20261022-01/cloud-comparison.md) 和[失败分析](runs/v5-followup-20261022-01/CLOUD_FAILURE_ANALYSIS.md)。

**2026-10-22 的 v4 云端验证已完成，但尚未交付真实修复。** Qwen2.5-Coder-7B 与 Qwen3-4B 各进行 transcript/chat 两次开发诊断，四条轨迹均没有产生补丁，官方记录为跳过测试的空补丁，而非测试通过。评分环境的正确/错误补丁对照通过。全部原始记录已回传，实例已归还。本轮暴露了占位路径误用、重复歧义编辑和目录输出浪费预算等具体问题；详见 [v4 验证记录](docs/VALIDATION_V4.md) 与[两模型失败分析](docs/V4_FAILURE_ANALYSIS.md)。这些结果不支持历史格式提高修复率，也不支持开始 Agentic RL 训练。

v3 已在云端固定真实 7B 模型执行一个开发任务并接入官方评分，但尚无有效修复。失败归因推动局部编辑工具与本轮历史协议修订；实际结果见 [v3 云端验证记录](docs/VALIDATION_V3.md)。这些功能的存在不等于结构化上下文更好。

v2 新增开源模型 LoRA、SFT、真实多轮采样、独立测试奖励和精简 GRPO 参数更新。首版的三种上下文策略、统一推理预算、本地／Docker 执行环境、SWE-bench 接口和 CSV / Markdown / HTML 报告继续保留。

v2 交付是**可训练的研究工程基础设施**。v1 离线演示使用预先写好的修复动作；v2 已使用真实 0.5B 模型验证 SFT，RL 更新核心通过自动化测试。小型同模板合成评测从 base 的 1/6 到 SFT 的 6/6，不能归因于 RL，也不代表 SWE-bench 能力。RL 试跑发现一处评分输出误读，受影响更新已标为无效，代码已修复。运行按用户要求中止，没有最终 RL 检查点或训练后评测，待云端复验。

**本机不再执行训练；后续训练仅在用户提供的云端 GPU 上启动。** 云端建议先用单卡 RTX 4090 24GB，针对 1.5B–3B 模型做 LoRA 和短上下文验证。详见 [训练说明](docs/TRAINING.md) 和 [v2 验证记录](docs/VALIDATION_V2.md)。

## 在当前电脑运行

已经在本目录创建 `.venv`，并安装项目、pytest 与可选 tiktoken。本轮 CPU 检查与保存产物见 [v5 后续验收](docs/VALIDATION_V5_FOLLOWUP.md)。没有在本机执行模型推理或训练；历史训练验证另见 v2 记录。PowerShell：

```powershell
cd C:\找工作\budget-agent
.venv\Scripts\python.exe -m budget_agent doctor
.venv\Scripts\python.exe -m budget_agent demo --output runs/my-first-demo
.venv\Scripts\python.exe -m pytest -q
```

`demo` 在 3 个自建小型 Python 仓库上分别运行 `full`、`summary`、`structured`，共 9 次脚本化修复。执行记录、补丁和独立测试结果保存在输出目录。HTML 报告位于 `runs/my-first-demo/report/report.html`，可以直接打开，无需网络。

查看本次已运行的结果：

- [离线流程报告](runs/smoke-v1/report/report.html)
- [小上下文压力验证报告](runs/context-stress-v1/report/report.html)
- [Docker 流程验证报告](runs/docker-smoke-v1/report/report.html)
- [首版验证记录](docs/VALIDATION.md)

已有结果目录保留历史运行。正式实验请每次使用新目录，避免无意将重复尝试混入样本。

在其他机器安装（Python >= 3.11、Git）：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell 则使用 .venv\Scripts\Activate.ps1，或直接运行其中的 python.exe
python -m pip install -e ".[dev]"
python -m budget_agent demo --output runs/demo
```

当前验证过的开发依赖版本记录在 `requirements-dev.lock.txt`；需要重建同样的依赖版本时先安装该文件，再执行 `pip install -e . --no-deps`。

原生 Windows 支持本地 smoke 和 Docker Desktop Linux 容器。真实 SWE-bench 的官方评分建议在 Linux / WSL2 上运行。

## 上下文策略与预算

| 策略 | 行为 | 对照意义 |
|---|---|---|
| `full` | 保留完整历史；超限明确停止 | 本项目自己的不压缩基线，未声称复现 mini-swe-agent |
| `recent` | 保留完整任务与最近连续交互；最新观察过长时明确截取 | 可运行的简单滑动窗口基线，区别于 observation masking |
| `mask` | 始终保留全部动作与来源标记，只保留最近 2 个事件的完整观察；旧观察可 recall | 提前减少重复输入，无额外摘要调用；放不下时明确停止 |
| `mask-read` | 在 mask 上有界恢复当前连续版本的旧源码整行，去重并保留来源 | 额外输入需计费；待验证是否改善修复，不能仅凭覆盖增加声称更优 |
| `summary` | 历史放不下时，调用同一模型生成滚动摘要 | 摘要输入和输出同样消耗预算 |
| `structured` | 优先保留近期事件、失败证据、修改与记忆；按需读取原始记录 | 确定性策略，不额外调用摘要模型 |

结构化记忆有事件 ID、观察时的代码版本、来源版本与 stale 标记。旧测试结果在修改代码后不能被当成当前证据。`remember` 存工作假设与来源，`recall` 可以重新读取归档输出。所有策略保留完整 issue；连 issue 都放不下时，任务会明确停止。

三个主要限制分别是：

- `context_tokens`：一次调用的输入与预留输出上限。
- `total_tokens`：整个任务累计输入＋输出，包括重复发送的历史和摘要调用。
- `max_steps`：Agent 工具决策步数上限，摘要调用另计入模型调用数与 token。

v5 另设 `min_action_output_tokens=128`：剩余额度无法容纳输入和最低动作输出额度时，调用前停止。它不强迫生成 128 个 token，也不保证长编辑完整；可以显式设为 1 做对照。摘要和动作仍从同一总预算扣除。历史结果的配置与原始轨迹保持原样。

`decision_policy=direct` 保留原始工具提示；可选 `budget-aware` 在每次输入中加入当前剩余总 token、步数、工作区是否变化、连续浏览次数，以及复现—编辑—验证的简短提示。这些文字照常计入同一预算，不增加免费模型调用，不限制工具选择，也不会将工作区变化当作正确修复。该因素与上下文策略分开比较；新策略只有经过真实补丁与独立评分，才能声称提升能力。

查看轨迹中的重复失败、重复成功观察、协议错误和输出额度压力：

```bash
python -m budget_agent diagnose --input runs/your-run --output runs/new-diagnostics.json
```

输出文件必须是新路径；该命令不运行模型。新运行的结果和 HTML/Markdown 报告自动包含行为统计，旧结果缺失统计时显示 `unavailable`。成功执行工具不能当作真实修复分数。

`loop_policy=off` 是默认对照；`--loop-policy recover` 仅处理同一连续工作区版本内、实际模型请求中完整可见的源码重读。反馈一次后继续连续重读时以 `stalled` 停止；原始观察不改写，独立评分继续执行。隐藏后的源码恢复、部分新增行、截断行、shell 命令和 recall 不按该判据阻止。它不能检测所有停滞，更不强制模型编辑。单独比较这个因素可使用 `python -m budget_agent.paired --runs RUNS --factor loop_policy --baseline off --candidate recover --output paired.json`，其余配置必须相同。

外部参考使用[真正的 mini-SWE-agent 上游循环](docs/MINI_SWE_BASELINE.md)，默认仅打印计划，云端 `--execute` 才运行。它与内部策略分别分组，不能充作纯上下文消融。

本地计数用于调用前检查；服务端返回的 `usage` 用于实际对账。默认 `byte` 是 UTF-8 字节加消息开销的保守估算，**不是模型 tokenizer，也不是收费 token**。实际用量超过预留时，先记录真实用量，再停止后续执行；不能保证远端一次请求绝不会超额。请求超时、响应损坏或缺少用量时停止，记为 `usage_unknown`，不自动重试可能已经计费的请求。

兼容API客户端也不跟随HTTP重定向，包括同源地址迁移。收到3xx时按未知用量停止，凭证不会随Location转发；需要迁移端点时先核对供应商地址，再显式设置 `base_url`。

`provider=transformers` 使用固定 commit 的 tokenizer 和 chat template，同一份输入 token ID 用于计数与生成，生成数量按实际 ID 计数（含结束 token）。每个任务/策略重置 PyTorch RNG；这不保证跨设备逐位复现。模型加载一次，各次尝试使用独立上下文与容器。云端配置示例为 [cloud-qwen7b.toml](configs/cloud-qwen7b.toml)，7B 模型是受控诊断基线，不代表当前最强模型。安装仅推理依赖可使用 `pip install -e ".[inference]"`，无需安装 LoRA/RL 包。

配对分析命令为 `python -m budget_agent.paired --runs RUNS --baseline recent --candidate structured --output paired.json`。它拒绝不完整任务配对、不同模型/预算/代码指纹和重复尝试；未评分、基础设施错误和预算违规保留在记录中，不能作为有效胜负。未知模型费用仍为空，租卡与准备环境成本需另计。

配置实际 tokenizer 后可安装并使用：

```bash
python -m pip install -e ".[tokens]"
python -m budget_agent run --config configs/api.example.toml --tokenizer tiktoken:cl100k_base
```

只有模型确实使用该编码时才应选择它；聊天序列化开销仍是估算。价格未配置时费用显示未知；填写单价后得到配置价格下的估算费用，尚未区分缓存折扣等计费细节。

## 后续接入真实模型

首版支持 **OpenAI 兼容的非流式 Chat Completions HTTP 接口**。服务必须返回文本 `choices[0].message.content`，以及整数 `usage.prompt_tokens` / `usage.completion_tokens`。这不表示已验证所有模型供应商兼容性；不支持 Responses、Anthropic 原生接口或流式响应。

支持供应商明确提供的 `enable_thinking` 时，可在 TOML 设置 `api_enable_thinking = false`，或使用 `--no-api-enable-thinking`；未设置则不发该字段。该参数不是通用 OpenAI 标准，也不表示实现了思维链保存/重放。需要约束完整输出时使用供应商支持的 `api_max_tokens_field = "max_completion_tokens"`；实际用量仍必须对账。

复制 `configs/api.example.toml`，填写准确模型版本和 API 根地址。密钥仅从环境变量读取，不写进配置或报告。PowerShell 7 可在不回显的情况下输入：

```powershell
$env:LLM_API_KEY = Read-Host "API key" -MaskInput
.venv\Scripts\python.exe -m budget_agent run --config configs/my-model.toml --task slugify-digits --output runs/model-pilot-01
```

项目不自动读取 `.env`；`.env.example` 只列出变量名称。默认真实模型使用 Docker；首次需要运行 Linux 容器引擎并准备镜像：

```bash
docker pull python:3.11-slim
```

本次已经启动 Docker Desktop，并验证 `python:3.11-slim` 的执行和独立评分。正式实验建议把配置中的镜像 tag 换成不可变 digest，并固定模型版本。

本地模型可使用 `http://127.0.0.1:端口/v1`，localhost 不要求密钥。若供应商要求 `max_completion_tokens`，在配置中修改 `api_max_tokens_field`。模型 API 的 temperature 支持情况也需按实际服务核实。

本地执行不是沙箱，真实模型必须显式加 `--allow-local-execution` 才能在宿主运行 shell。Docker 模式仅挂载任务仓库；独立评分在 Agent 停止后使用另一份干净目录进行。源文件改变不能通过修改公开测试来冒充修复。

## 执行对照实验

```bash
python -m budget_agent compare --config configs/my-model.toml --tasks tasks/local.jsonl --split dev --strategies full summary structured --repeat 2 --output runs/model-comparison-01
python -m budget_agent report --input runs/model-comparison-01 --output runs/model-comparison-01/report
```

`seed` 当前只标记重复次数，不发送给 API，不保证随机采样可重现。各策略使用同一任务与配置；报告会按请求模型、实际模型、预算、任务类型和 split 分组，并核验任务指纹与重复标签。超额、未知用量、任务不匹配会明确标记，不能当作有效固定预算比较。

小上下文检查可以直接离线运行：

```bash
python -m budget_agent compare --config configs/smoke.toml --task slugify-digits --context-tokens 4096 --output runs/context-check
```

这里出现的策略结果差异仅是**预定动作在不同上下文管理流程中的软件行为**，不能推断真实模型哪种策略更好。

## SWE-bench 接入

已准备 [5 个真实 SWE-bench Lite 开发任务](tasks/swebench-pilot.jsonl)，以及[来源和 SHA256](tasks/swebench-pilot.provenance.json)。它们按照数据集顺序选取，没有根据本项目表现筛选。源数据来自官方 `test` split；本项目将这 5 个任务标记为 `dev` 用来调试，不能再当作独立最终测试集。

标准补丁、隐藏测试 patch、FAIL_TO_PASS、PASS_TO_PASS、hints_text 均不会进入任务清单。已完成 Astropy 单题的真实模型开发诊断、固定镜像准备和官方评分对照；其他清单任务尚未完成批量真实评测。

可以先选择其中一个 task ID，预拉取它的 manifest `image`，再开始推理。SWE-bench 镜像可能明显大于内置 smoke 镜像；先验证一个任务，再扩展。

```bash
python -m budget_agent run --config configs/my-model.toml --tasks tasks/swebench-pilot.jsonl --task astropy__astropy-12907 --strategy structured --output runs/swe-structured-01
python -m budget_agent swe-export --input runs/swe-structured-01 --output runs/swe-structured-01/predictions.jsonl
```

任务 ID 以清单实际内容为准；长 issue 可能需要增加上下文预算，并选择正确的 tokenizer。各策略必须采用同样的新预算。

在独立 Linux 评分环境中安装官方 harness（固定 `swebench==5.0.2`；云端单题链路已验证，本机只做 CPU 软件测试）：

```bash
python -m pip install -e ".[swebench]"
# 默认仅显示官方命令参数，不运行
python -m budget_agent swe-evaluate --predictions runs/swe-structured-01/predictions.jsonl
# 准备好官方环境后执行；每次使用新的 run-id
python -m budget_agent swe-evaluate --predictions runs/swe-structured-01/predictions.jsonl --run-id pilot-structured-001 --execute
python -m budget_agent swe-attach --runs runs/swe-structured-01 --reports logs/evaluation/pilot-structured-001
python -m budget_agent report --input runs/swe-structured-01 --output runs/swe-structured-01/report
```

导出时拒绝混合策略或同任务多个尝试，避免悄悄挑最好结果。回填评分核验官方 `patch.diff`。导入官方空补丁时同时指定 `--summary-path` 和 `--predictions-path`，并保留导出生成的 `.manifest.json`。空补丁显示 `official_empty_patch`、`resolved=null`、测试未执行；基础设施异常也保持 `resolved=null`，不冒充模型失败。真正未导入评分时仍显示 `pending_official_evaluation`。

下载另一段固定子集或导入已有标准 JSON / JSONL：

```bash
python scripts/prepare_swebench_pilot.py --offset 5 --count 30 --split test --output tasks/swebench-eval.jsonl
python -m budget_agent swe-import --input path/to/raw.jsonl --output tasks/custom.jsonl --split dev
```

下载脚本不会覆盖已有清单。完整评测约定见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

## 代码入口与运行产物

```text
src/budget_agent/
  runner.py       模型 → 动作 → 环境 → 独立评分
  budget.py       调用前预留、实际用量对账、费用计量
  context.py      full / summary / structured
  models.py       HTTP 模型适配、动作解析、脚本 smoke
  environment.py 文件工具、命令执行、Docker、独立评分
  tasks.py        任务清单、脚本化验证动作
  report.py       分组对照与离线报告
  swebench.py     数据脱敏、补丁导出、官方评分回填
  cli.py          命令行入口
```

每次运行生成独立目录，包含 `result.json`、`calls.jsonl`、`trajectory.jsonl`、`patch.diff` 和任务 workspace。`calls.jsonl` 保存实际发送的上下文、模型返回与用量；`trajectory.jsonl` 保存动作与观察。单条工具输出有 100 KB 捕获限制，截断会注明；超长 SWE-bench 补丁会报错，不会静默导出残缺补丁。

建议阅读顺序：`schema.py → runner.py → context.py → budget.py → environment.py`。训练模块在 `learning/`，历史有效性限制见 v2 记录。当前先做同模型历史格式对照和独立参考模型诊断，再扩展真实任务与上下文研究；云端命令见 [v4 方案](docs/IMPLEMENTATION_V4.md)。

参考：[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)、[SWE-bench 官方评测](https://www.swebench.com/SWE-bench/guides/evaluation/)、[AutoCompact](https://arxiv.org/abs/2610.02163)。首版采用独立实现的轻量执行循环，没有直接依赖或复制 mini-swe-agent 源码，也没有完成与其原版的实验比较。
