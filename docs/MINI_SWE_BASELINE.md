# mini-SWE-agent 外部参考入口

这个入口运行真正的上游 `DefaultAgent`。本项目只适配模型调用、token 预算、既有 Docker 环境及结果导出，不另写一个相似循环冒充 mini-SWE-agent。它是**受相同 token 上限约束的外部 harness 参考**，不能与内部 `recent/summary/structured` 对照合并为上下文策略消融。

当前仅支持一个 SWE-bench 任务、已有 `transformers` CUDA 后端和 Docker。默认只打印计划；`--help` 和计划模式不导入 mini-SWE、Torch、Transformers，不下载模型、不启动容器、不租卡。执行入口限制为 Linux。除默认的接口替身测试外，已完成固定上游真实循环的 CPU 集成；其中模型和环境仍为明确的替身，因此不构成代码修复能力测量。

## 固定上游

- 版本：`2.4.6`
- Git SHA：`a83fcae82d2a08f0ee0c688f9d137b3566c097f8`
- 上游循环：[agents/default.py](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/agents/default.py)
- 原始提示和观测模板：[swebench_backticks.yaml](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/swebench_backticks.yaml)
- 原始文本动作解析：[actions_text.py](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/utils/actions_text.py)
- 原始提交判定：[environments/docker.py](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/docker.py)

执行时检查 checkout 的完整 SHA、包内已跟踪和未跟踪改动、实际导入路径及包版本，拒绝用另一份安装或修改后的上游代码。结果另外保存原始 YAML 的 SHA-256。

v2 上游默认使用原生 tool calls；现有本地 HF 后端只有文本消息，所以这里明确选择其官方 textbased 配置。动作仍为上游的单个 `mswea_bash_command` 代码块，不使用本项目的 JSON 文件工具。此结果不应标成 mini-SWE 默认原生工具版本的成绩。

接口已对照上述完整 SHA 的源码复核，未仅依赖测试替身：

| 上游调用点 | 固定版本的实际契约 | 本入口 |
| --- | --- | --- |
| `DefaultAgent` 的模型调用 | `format_message(**kwargs)` 返回 dict；`query(messages)` 返回带 `extra.actions` 的消息；观测格式函数接收三个位置参数并返回 `list[dict]` | 对应方法签名和返回值一致，元数据不进入实际模型 prompt |
| 模板与保存 | 两个对象提供 `get_template_vars()` 和 `serialize()`，返回 dict；`serialize` 深度合并 agent、model、env 结果 | 使用普通可 JSON 序列化的 dict，不返回 JSON 字符串 |
| `run(task)` | 返回最后一条 exit 消息的 `extra` dict，正常退出包含 `exit_status` 和 `submission` | 直接使用该 submission，不假定返回 tuple |
| 异常流 | `InterruptAgentFlow(*messages)` 保存消息 dict 组成的 tuple；`FormatError` 和 `LimitsExceeded` 沿用此接口 | 传入一个消息 dict；补充格式错误的原始响应和成本后重新抛出 |
| 轨迹保存 | `AgentConfig.output_path: Path \| None`；循环每步的 `finally` 调用 `save(output_path)` | 传入 `Path`，不传不存在的 save 参数 |
| 提交检测 | 固定版本 `_check_finished(self, output)` 函数体不访问 `self`，只检查输出和 returncode 后抛出 `Submitted` | 调用上游未绑定方法并传 `None`；完整 SHA 检查防止未来接口变化被静默接受 |

`KeyboardInterrupt` 在上游不是普通 `Exception`，但仍经过每步 `finally` 保存轨迹。本入口在模型调用尚未返回时记录一行 `usage_unknown`，保留未核对的 reservation，并由外层保存 `interrupted` 结果；这种原始轨迹可能还没有 exit 消息，以外层结果及账本中的中断标记为准。

“真实上游循环 + 假模型/假环境”的 CPU 集成无需 Torch、Transformers、模型权重或 Docker 服务。核心循环的依赖是 Jinja2、Pydantic v2、python-dotenv、platformdirs、Rich；模板读取需 PyYAML。完整执行本入口的 `load_upstream` 还会导入上游 `LitellmTextbasedModelConfig`，因此需 LiteLLM 及其依赖；上游重试工具另需 Tenacity。实际执行记录见下文。

## 真实上游 CPU 集成记录（2026-10-22）

在新的临时 venv 中，使用公开仓库 checkout 的固定 SHA，真正执行 `load_upstream()` 和未修改的 `minisweagent.agents.default.DefaultAgent`。模板、动作解析、格式错误处理、步骤终止、提交异常、轨迹序列化及保存均来自该 checkout。`ScriptedCPUBackend` 只返回预置文本和合成使用量；`ScriptedCPUEnvironment` 只返回预置输出，不执行 Bash 命令。环境来源明确写为 `cpu_test_double`。

单独选择新增集成测试的首跑：**6 passed, 13 deselected in 5.55s**。加入“推理模块没有导入”的断言后，完整适配器文件复跑：**19 passed in 4.52s**，其中 6 项是真实上游集成，13 项是原有契约测试。随后补充上游查询计数与原始中断状态断言，6 项集成再次通过（**5.08s**）。项目原 `.venv` 未安装这些依赖，运行同一个测试文件为 **13 passed, 6 skipped**；未设置 `MINI_SWE_UPSTREAM_REPO` 时会明确跳过集成项。

| 真实上游集成场景 | 实际退出状态 | 已核对模型调用 | 预置环境动作 |
| --- | --- | ---: | ---: |
| 一次格式错误、恢复读取、显式提交、导出 predictions | `Submitted` | 3 | 2 |
| 连续三次格式错误 | `RepeatedFormatError` | 3 | 0 |
| 一步后触发上游 step limit | `LimitsExceeded` | 1 | 1 |
| 首次调用前预算不足，仍保存上游轨迹 | `budget_exhausted` | 0 | 0 |
| 模型边界抛出 `ModelError` | `ModelError` | 0；保留 1 次未知请求 | 0 |
| 模型边界抛出 `KeyboardInterrupt` | `interrupted` | 0；保留 1 次未知请求 | 0 |

正常提交项另外核对：上游 trajectory 中记录的是原始 `DefaultAgent` 类型，格式错误与观测元数据被保留，调用模型时只发送 role/content，提交补丁与现有 exporter 生成的 predictions 完全一致，并生成原有 manifest。两项失败调用都有且仅有一行 `usage_unknown` 账本，保存的 budget 仍标记 unreconciled。此次未发现需要修改适配器的上游接口错误。

集成还确认了一个计数差异：上游在进入 `model.query()` 前就增加 `n_calls`，所以预算检查拒绝首次生成时，它的原始 `model_stats.api_calls` 是 1，但真实模型边界调用为 0。实际调用与 token 使用量以本项目 `calls.jsonl` 和 `Budget` 为准，不能把上游该字段当作已计费请求数。

实际环境为 Windows、Python **3.13.5**。直接依赖及关键传递依赖版本：

```text
litellm==1.75.5.post2
pydantic==2.14.1
Jinja2==3.1.6
PyYAML==6.0.3
python-dotenv==1.2.4
rich==15.0.0
platformdirs==4.13.5
pytest==9.1.1
tenacity==9.2.1
openai==3.36.1
```

该临时环境共有 60 个 distribution（包含 pip 与传递依赖），`pip check` 返回 `No broken requirements found`。通过包发现检查确认 **torch、transformers、peft、accelerate、datasets 均未安装**；六项集成测试结束时还断言模型与训练模块没有导入。LiteLLM 仅供导入官方配置类，没有调用它的模型接口。没有下载权重、模型推理、训练、Docker 启动、云端租用或付费模型调用。

可复现操作如下。PowerShell 变量 `taskCpuRoot` 指向新建的临时目录，`taskProject` 指向本项目；不要复用项目的 `.venv`。

```powershell
$taskProject = 'C:\找工作\budget-agent'
$taskCpuRoot = Join-Path $env:TEMP ('mini-swe-cpu-integration-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskCpuRoot
& "$taskProject\.venv\Scripts\python.exe" -m venv "$taskCpuRoot\venv"
git clone --no-checkout --filter=blob:none https://github.com/SWE-agent/mini-swe-agent.git "$taskCpuRoot\upstream"
git -C "$taskCpuRoot\upstream" checkout --detach a83fcae82d2a08f0ee0c688f9d137b3566c097f8
& "$taskCpuRoot\venv\Scripts\python.exe" -m pip install 'litellm==1.75.5.post2' 'pydantic==2.14.1' 'Jinja2==3.1.6' 'PyYAML==6.0.3' 'python-dotenv==1.2.4' 'rich==15.0.0' 'platformdirs==4.13.5' 'pytest==9.1.1' 'tenacity==9.2.1' 'openai==3.36.1'
$env:MINI_SWE_UPSTREAM_REPO = "$taskCpuRoot\upstream"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& "$taskCpuRoot\venv\Scripts\python.exe" -m pytest "$taskProject\tests\test_mini_swe_baseline.py" -k real_upstream -vv --basetemp "$taskCpuRoot\test-artifacts" --junitxml "$taskCpuRoot\integration.xml"
```

最终最小证据集已持久保存到项目 [upstream-integration/](../runs/v5-followup-20261022-01/upstream-integration/REPLAY.md)：[receipt](../runs/v5-followup-20261022-01/upstream-integration/receipt.json)、[最终测试日志](../runs/v5-followup-20261022-01/upstream-integration/integration-validated.log)、[JUnit](../runs/v5-followup-20261022-01/upstream-integration/integration-validated.xml)、[完整依赖冻结](../runs/v5-followup-20261022-01/upstream-integration/requirements-observed.txt)、六场景原始轨迹与账本，以及可重放命令。没有复制 venv、模型或密钥。结果按原字节复制为 `cpu-result.json`，避开真实模型报告扫描的 `result.json`；内部原始临时路径保留。预算检查阶段没有发出请求的场景没有账本行，receipt 对此作了说明。

原运行目录位于 Windows `%TEMP%\mini-swe-cpu-integration-6fa90f23501b4f42b4946718963ee9c6`；持久证据不依赖它继续存在。执行前后 checkout SHA 均一致，包源目录无改动；实际读取 YAML 的 SHA-256 为 `05aa713bbbeb50af86a8890b21261a97d6c0e59258ba83841deac18e3645528b`。本目录属于 CPU 接口证据，不能计入真实模型修复成绩。

这一验证确认 Windows CPU 上的 Python 接口链路可以运行。Linux GPU 后端、真实 Docker task image、实际模型输出分布和官方评分仍需在服务器上另行验证；本轮没有产生外部基线修复率。

## 云端准备与运行

仅在已获授权的 Linux GPU 服务器上进行。下面假定已有本项目的云端推理环境、固定模型权重和预拉取的任务镜像；此入口不负责安装、拉取镜像、训练或租卡。

在云端创建独立的上游 checkout，然后安装其依赖；执行器会再次验证 SHA。`/path/to` 是需替换的服务器目录。

```bash
git clone --branch v2.4.6 --depth 1 https://github.com/SWE-agent/mini-swe-agent.git /path/to/mini-swe-agent
git -C /path/to/mini-swe-agent rev-parse HEAD
python -m pip install -e /path/to/mini-swe-agent
```

输出的 SHA 必须等于上面的固定值。依赖是这个 checkout 的 mini-SWE 包（含 PyYAML、Jinja、Pydantic、LiteLLM 等）和已有本项目的推理栈；没有额外模型服务、vLLM 服务或训练依赖。适配器直接调用已有 HF 模型，**不经过 LiteLLM 的请求重试或费用估算**。

先打印计划：

```bash
python scripts/run_mini_swe_baseline.py \
  --tasks tasks/swebench-pilot.jsonl \
  --task astropy__astropy-12907 \
  --split dev \
  --config configs/cloud-qwen7b.toml \
  --upstream-repo /path/to/mini-swe-agent \
  --output /path/to/runs/mini-reference-12907
```

确认是在已授权的服务器上运行后，对同一命令追加 `--execute`。输出目录必须不存在。每次只执行一个任务，其他任务使用独立目录；不要覆盖或反复筛选同一任务的最佳结果。此命令没有自动租卡或释放服务的能力，云端时长和费用仍由外层部署流程控制。

计划沿用配置中的模型完整 revision、dtype、温度、seed、context/total/max-output token 上限、行动输出下限、步骤和工具超时。内部 context 策略在这个入口不生效；结果明确记为 `upstream_full_history`。例如现有 Qwen 7B 配置是 8192 context、32000 累计输入加输出、768 单次最大输出、20 次模型查询。新行动输出下限默认为 128。

## 预算、隔离和评分边界

每次调用先对**实际发送的完整 chat-template 输入**计数并预留输出，再调用同一个 HF backend。动作格式错误的输出也先计入预算；不重试模型请求、不补发免费修复调用，不使用我们的历史裁剪策略。完整历史放不进 context 或剩余预算时结束，不能悄悄截断历史。模型调用失败而使用量无法核对时保留 `usage_unknown` / `unreconciled_request`，不能算作有效预算成绩。

上游 dollar `cost_limit` 被设为 0，即禁用该计数器，严格预算由现有 `Budget` 承担。没有配置 token 单价时结果的费用为 `null`；上游轨迹里为兼容接口而保留的零 cost 不表示云端免费。GPU 租金、墙钟时间与 token 预算是不同指标。相同 token 上限也不代表不同 harness 的一次动作具有相同工作量：Bash 可以执行组合命令。

任务从现有脱敏清单读取；只把 `task.issue` 交给上游。容器复用本项目已有的网络关闭、CPU/内存/PID 限制、capability 限制与独立镜像检查，不挂载数据集、隐藏测试、API 密钥或评分目录。Bash 使用既有环境的 `bash -lc`，记录实际 image ID 和 checkout。这与上游原生环境配置的 `bash -c` 有差别，属于明确记录的环境适配，不是完全原样运行的官方配置。

**提交语义保持上游原样。** 只有上游 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 提交协议产生的 `submission` 才写入 `patch.diff`；即使已经编辑源码，若预算用尽或格式错误导致没有提交，补丁仍为空。不会用工作区 diff 替代。过长提交被环境截断时直接报错，不评分一个截断补丁。内部 agent 会在结束时采集工作区 diff，因此外部比较必须同时展示退出状态和空提交数量。

每次输出包含：

- `plan.json`：任务文件哈希、配置、模型 revision、上游 SHA。
- 任务子目录中的 `mini.traj.json`：上游保存的原始轨迹；启动前失败可能没有该文件。
- `calls.jsonl`：实际输入、输出、逐次 token 账本。
- `result.json` 与 `patch.diff`：既有评分契约，包含独立 `harness`、上游 SHA、环境与代码来源。
- `predictions.jsonl` 及既有 `.manifest.json`：由原来的 `export_predictions` 生成，未新建另一套 manifest 系统。

评分仍用独立的官方 SWE-bench harness；先查看命令计划：

```bash
python -m budget_agent.cli swe-evaluate \
  --predictions /path/to/runs/mini-reference-12907/predictions.jsonl \
  --run-id mini-reference-12907 \
  --instance-ids astropy__astropy-12907
```

真正评分在外层独立评分目录运行，并使用新的 run ID。之后复用 `swe-attach` 的补丁哈希核对与显式 aggregate summary / predictions 路径机制。空补丁仍为 `official_empty_patch` / `resolved=null` / `tests_executed=false`，不把未执行测试写成测试失败。官方数据和测试不能回流 agent 容器。

`config` 中加入外部 harness、上游 SHA、提交与 context 语义，`source_fingerprint` 另外包含适配脚本和上游 SHA。因此现有报告不会把它和内部 context 策略聚为同一比较组。不要放松 `paired.py` 的条件检查以强行比较；这类参考应该做独立描述性结果表。

## v5 应冻结的实验边界

建议先以同一固定 Qwen checkpoint、同一任务清单、同一镜像、greedy decoding、seed 42 和现有 8192/32000/768/20 预算做以下组。总 token 或步骤上限若以后增加，应另开预算实验，不能拿它替代本轮。

| 实验 | 只改变什么 | 必须保持什么 | 结果解释 |
| --- | --- | --- | --- |
| 交互契约诊断 | v4 工具/提示与 v5 分页、定位反馈和提示 | 同一个内部 context 策略、相同预算，行动输出下限都固定为 1 | 是整个交互契约的影响，不能单独归因给分页；跨版本不能走严格 paired，旧结果只作历史描述 |
| 预算末尾诊断 | 行动输出下限 1 → 128 | 都使用 v5 工具、相同 context 策略，其余预算一致 | 避免残余几 token 无效调用的影响，不是 context 收益 |
| 内部 context 对照 | recent / summary / structured 等既定策略 | 相同 v5 代码、工具、行动输出下限 128、模型、任务和预算 | 可复用现有严格 paired；summary 调用必须计费计 token |
| 外部参考 | mini-SWE textbased harness | 同一 checkpoint、任务、精确 token 上限与隔离资源 | 整体 harness 参考；完整历史、Bash 和提交协议同时不同，不能归因给 context |

温度为零时反复跑 seed 并不是独立统计样本。先记录全体预定任务的结果、未评分、基础设施失败和空提交，再分析成功率、token 与耗时。现有 `swebench-pilot.jsonl` 只有 5 个 Astropy 任务，来自官方 Lite test 后划为**本项目 dev**；它是开发诊断集，不是官方 dev，也不能代表跨仓库泛化。正式结论需要在运行前固定新的多仓库留出清单，复用现有脱敏导入；已经调过的 pilot 任务不应再次宣称为未见测试。

### 持续集成与现有真实基线（2026-11-02）

GitHub Actions 在已有的 Ubuntu / Python 3.13 作业检出上述上游提交，使用独立 venv 运行六个真实 `DefaultAgent` 场景。依赖版本来自 10 月 22 日归档的环境，模型响应和命令输出均为预置数据。本地复现时，在项目根目录用干净 venv 安装 `python -m pip install -r requirements-mini-swe-contract.txt`，将 `MINI_SWE_UPSTREAM_REPO` 设为上述干净 checkout 的路径，然后运行 `python -m pytest -q tests/test_mini_swe_baseline.py -k real_upstream`。提交用例实际将导出的预测交给 Git，核对应用后的 CRLF 和末行无 LF。[CI 配置](../.github/workflows/tests.yml)

使用标有 `model_metadata.test_double=true` 的后端时，结果写为 `scripted_smoke`。离线报告和评分适配器按同一规则解释历史文件，保留原始记录。真实云端参考已在 Astropy 开发题上运行过一次。Qwen3-4B-Instruct-2507 发起 6 次调用、用了 26,800 token，因剩余预算不足结束。官方记录为空补丁，未执行评分测试，`resolved=null`。[原始结果](../runs/cloud-v5-a4000-20261022-01/mini-baseline/astropy__astropy-12907__mini-swe-textbased/result.json) · [本次样本审计](ITERATION_LOG.md#mini-evidence-20261102)
