# 代码审查与修复（2026-11-05）

分支 `review/2026-11-05`，基线为 `main@db5bd43`。本轮只在 CPU 上做软件验证，没有运行模型、Docker 或训练。

## 阅读范围

- **逐行读过**：`src/budget_agent/` 下的 runner、context、budget、environment、models、hf_model、schema、tasks、verification、security、observations、loop_control、reflection、evaluation、openjudge_adapter、swebench、paired、oci_cache、diagnostics，以及 `learning/` 的全部 6 个文件。
- **只看了结构**：report.py 和 cli.py（函数清单、参数入口），scripts/。
- **测试**：只读了与本轮改动相关的部分，即 test_environment、test_security、test_edit_file、test_runner、test_budget、test_verifier_execution。
- **测试结果**：基线 900 passed、13 skipped。本轮结束后 930 passed、13 skipped，外加 1 个 strict xfail（见 F2）。

## 本轮修复

每一项都先在基线上复现，再修改，然后补上回归测试。

| 编号 | 级别 | 问题 | 复现（基线） | 修复 | 回归测试 |
|---|---|---|---|---|---|
| F1 | P0 奖励可信度 | 评分进程用 `python -c` 启动，cwd 就是候选代码目录，仓库根目录的文件可以遮蔽评分器自己 import 的标准库 | 默认编辑策略下（没有 `editable` 白名单），在仓库根目录放一个 20 行的 `unittest.py`，bug 没修也得到 `resolved=True` | 改用 `python -I -B`，评分器先 import 自己需要的模块，之后才把候选目录加进 `sys.path`。仓库模块仍然优先于 site-packages | `test_grader_isolation.py` |
| F2 | P0 奖励可信度 | 解析时取**第一行**带标记的判定。候选代码在 import 时可以从 `/proc/self/cmdline` 读到标记，抢先打印一行伪造的判定 | 只改允许编辑的 `slugify.py`，bug 保留，结果仍是 `resolved` | 标记必须恰好出现一次。出现两次判为 `invalid_patch`：奖励记 0，而不是抛异常让训练中断 | 同上。"伪造后立即 `os._exit`"的变体仍然能骗过评分，已标为 strict xfail，见下文"仍未解决的风险" |
| F3 | P1 正确性 | Docker 下 `patch()` 先执行 `git add -N`，等于改写了 agent 的 git index。只要 agent 新建过文件（比如复现脚本），调用一次 `diff` 工具，文件内容没变，revision 指纹也会变。后果是旧观察被误标为 stale、read-loop 分段被重置、mask-read 的记忆被清空。agent 自己执行 `git add` 也会改变指纹。工作区里出现嵌套 git 仓库（比如 `git clone` 了一个依赖）时，旧的指纹代码会抛 `IsADirectoryError`，整个 run 以 `environment_error` 结束 | 用一个真实 git 仓库运行原来那段容器内代码，三种情况都能复现 | 新增 `_GIT_SNAPSHOT`：把工作区 stage 到一个临时 index（`GIT_INDEX_FILE`）里。revision 改用 `git write-tree`，它只由内容决定；patch 改用 `git diff --cached <base>`；git 的 stderr 不再混进补丁 | `test_git_snapshot.py`：在真实 git 仓库上执行同一段容器内代码，不需要 Docker |
| F4 | P1 agent 质量与评分 | 输出只保留前 100 KB。pytest 的失败摘要和 traceback 都在末尾，所以 agent 看不到。更严重的一点：评分器把判定行打印在最后，验证器自身输出一旦超过 100 KB，判定行就被截掉，本来通过的结果变成 `grading_error` | 验证器打印 250 KB：基线返回 `grading_error`，本分支返回 `resolved` | 改为首尾各保留 50 KB，中间标注省略了多少字节。`output_truncated` 的语义不变，补丁被截断时依然拒绝导出 | `test_output_capture.py` |
| F5 | P2 性能 | 每一步都要算两次工作区指纹：步首一次，动作后一次。步首这次和上一步的结果完全相同。只读工具执行完也重新算，而在 Docker 下每算一次就是一次容器 exec | 脚本化 demo 每个 run 调用 17 次 | 上一步的指纹直接沿用。只在写文件、编辑、`run`、`public_test` 之后才重新计算。结果里新增 `workspace_fingerprints` 字段，记录调用次数和耗时 | `test_runner.py` 拿每步的真实指纹逐一比对，并检查调用次数。demo 从 17 次降到 4 次。用 6 条保存下来的 Astropy 真实轨迹回算，从 132 次降到 12 次。这 6 条几乎都在读代码，编辑多的轨迹节省会少一些 |
| F6 | P2 性能 | 上下文策略靠二分查找截取长度，每试一次都把完整的 message 列表重新 tokenize 一遍 | 下表 | 对 tokenizer 计数按内容摘要做缓存：每条约几十字节，上限 65,536 条，满了就清空。byte 计数不走缓存 | `test_budget.py`。另外在 5 种策略下确认，缓存前后构造出的输入逐字节相同 |
| F7 | P3 | `search` 会把二进制文件当文本扫描（比如 astropy 的 FITS 测试数据），结果全是噪声 | | 前 8 KB 里出现 NUL 字节就跳过该文件。本地和容器两条路径都生效 | `test_edit_file.py` |
| F8 | P3 | 本地和 Docker 两套评分器各写了一遍"复制评分树、应用改动"的逻辑；编辑权限规则也写了两遍 | | 抽出 `_changed_files`、`_write_grading_tree`、`_editable` 三个函数。容器内的那份规则必须自带（因为要序列化进容器），所以加了一个测试，保证两份规则判定一致 | `test_container_edit_policy_matches_grading_policy` |
| F9 | P3 | `__version__` 是 0.1.0，pyproject 里是 0.2.0；cli.py 有一个没用到的 import | | 统一为 0.2.0，删掉多余 import | |
| F10 | P1 正确性 | F3 的临时 index 若用 `copyfile` 复制，index 文件的 mtime 会被刷新，git 判断"racily clean"条目的机制随之失效：agent 在 1 秒内把文件改回同样大小的内容时，指纹可能察觉不到 | 在真实 git 仓库上反复构造这种情形：`copyfile` 在 20 次里漏掉 1 次；在全量测试中表现为偶发失败 | 改用 `copy2`，保留 index 的 mtime | 同一构造重复 20 次，漏检 0 次；快照测试连跑 40 遍，全部通过 |
| F11 | 新功能 | 真实仓库任务需要两样东西：每个任务各自的解释器，以及容忍测试产物的评分 | | 任务元数据可以设置 `exec_env`（把任务 venv 前置到 PATH，`{workspace}` 替换为本次 attempt 的副本路径）；评分可以设置 `grading_scope: editable_only`（受保护文件的改动只记录、不应用，参照 SWE-bench 重置测试文件的做法） | `test_task_exec_env.py`；SWE-smith 端到端检查：gold 修复留下 `.coverage` 和复现脚本时，仍判为 resolved，并记录这两个被忽略的文件 |

F6 的测量方法：`python scripts/bench_context_build.py --counter tiktoken:cl100k_base`，用合成的 30 步轨迹，每 5 步构造一次上下文，共 6 次，统计总耗时。表中是 3 次运行的中位数。机器是共享的，噪声较大，有一次基线 structured 跑出 24 s。

| 策略 | main | 本分支 | 倍数 |
|---|---|---|---|
| recent | 637 ms | 47 ms | 约 14× |
| structured | 8372 ms | 378 ms | 约 22× |
| mask-read | 132 ms | 33 ms | 约 4× |

byte 计数器本来就快（9 ms 以内），这次没有变化。`provider=transformers` 的计数走 `apply_chat_template`，这次没有优化，见 R9。

## 后续进展（2026-11-06）

- **补丁篡改监控**（PR #4，`integrity.patch_signals`）：扫描补丁新增行里的篡改模式，包括进程退出（含 `_os._exit` 这类别名写法）、`/proc/self`、栈帧和 gc 内省、对 builtins 和 `sys.modules` 的改写、对测试框架的 monkeypatch、pytest 钩子、tracing 钩子，以及新增启动钩子文件、改动测试文件。扫描结果只记录进 `result.json`，不影响评分；SFT 数据构建时默认排除被标记的样本。
- **SWE-smith 任务断网**（PR #2，`network_policy: deny`）：防止 agent 直接下载上游原始文件来"修复"注入的 bug。
- **评测与训练共用一条路径**（PR #2）：rollout 记录采样时的 token ID 和 behavior logprob，GRPO 直接在这些产物上训练。

## 仍未解决的风险：评分在进程内执行

候选代码和评分器跑在同一个 Python 进程里，所以它有很多办法篡改结论：伪造判定行后立即 `os._exit(0)`；替换 `json.dumps` 或 `unittest.TestResult.wasSuccessful`；顺着解释器栈帧把标记找出来。在同一个进程里，这些都防不住。所以当前的判定只能证明"诚实的修复是否正确"，不能抵抗主动作弊。如果要把它当作 RL 奖励，至少还需要三件事：

1. 对函数级任务改用**进程外的 I/O 判定**：候选代码在子进程里只负责计算输出，期望值只保存在父进程中。
2. 在补丁中扫描篡改特征（`os._exit`、`/proc/self`、`sys._getframe`、给 builtins、json、unittest 打补丁、新增 `sitecustomize.py` 或 `conftest.py`）。只做标记，交人工抽查，不直接改分。
3. 训练过程中持续混入"只含 bug、不含修复"的金丝雀任务。它们一旦拿到正奖励，立刻停止训练。

## 未修改的建议（按优先级）

- **R1 训练环境和评测环境是两套实现。** `learning/rollout.py` 有自己的 system prompt，工具说明里没有 edit_file 和 public_test，remember 和 recall 会被直接拒绝。观察格式和预算逻辑也都不同。这样训练出来的策略，会在另一套协议下接受评测。建议先抽出一个 `AgentEnv.reset()/step()` 接口，让 runner 和 rollout 都通过它运行。
- **R2 rollout 无法扩展。** `HFPolicy.sample` 每生成一个 token 就做一次 Python 前向；episode 一条接一条地跑；每次 GRPO 更新只用 1 道题、4 条轨迹；整个仓库没有任何并发（HTTP 客户端是同步的，也没有用 asyncio）。建议用 vLLM 或 SGLang 的 OpenAI 兼容服务加 asyncio 并发生成 rollout，环境放进独立的进程池或容器池。
- **R3 GRPO 实现上的问题。** 一组奖励全部相同时直接丢弃，相应的生成算力就浪费了（可以参考 DAPO 的 dynamic sampling）。`old_log_probs` 来自带 KV cache 的 bf16 采样，与训练时重算的值之间存在 training-inference mismatch，目前只记录了 `max_ratio_deviation`，没有做 TIS/MIS 之类的修正。没有监控 KL 和 entropy。遇到未评分的 episode 直接抛异常，整个训练就中断了；应该改成 mask 掉这条或单独计数。
- **R4 多轮 token 化。** 每一步都把完整对话重新渲染一遍，再按（prompt, completion）逐步训练，token 开销是 O(T²)。将来如果改成一条拼接序列加 loss mask，就必须记录 token-in-token-out，否则 re-tokenize 会让训练用的 token 和采样时的不一致。
- **R5 工具协议。** 目前只支持"每轮输出一个 JSON 对象"这一种文本协议，不支持原生 function calling，也只有一种 harness。业界会在多种工具调用格式和多种 harness 上训练（Qwen3-Coder-Next 用了 21 种模板），因为单一 scaffold 学到的能力很难迁移到别的 scaffold。
- ~~**R6 `run_task` 圈复杂度 51，大约 290 行。**~~ 已在 PR #3 完成：拆成 `_Attempt` 类，`runner.py` 中每个函数的复杂度不超过 10。scripted 演示的 9 次运行，结果逐字段一致。原来的说明保留如下：** 内部的嵌套闭包会读取循环变量（`step`、`revision`）。`_repository_tool` 的复杂度是 40。建议拆成一个 `Attempt` 类，内部工具（remember、recall）改用注册表分发。
- **R7 代码排版。** 有 164 行超过 120 个字符，最长一行 1516 个字符，review 和 diff 都很难看。建议引入 ruff（先启用 F/E9/B 规则，再逐步加上 E501），放进 CI。
- **R8 Docker 资源和镜像。** 资源限制写死成 `--cpus 2 --memory 4g --pids-limit 256`，应该做成可配置的。SWE-bench 镜像名用的是 `:latest`，没有固定 digest，与仓库其他地方坚持 digest 的做法不一致。
- **R9 HF 计数开销。** `HFTokenCounter.messages` 在二分查找的每次试探中都要重新套一遍 chat template。建议按消息增量计数，或者直接在 rollout 侧维护 token id 序列。
- **R10 投入比例。** evaluation、paired、report、diagnostics 四个文件合计 1024 行，按字符数约占源码的四分之一，几乎都是审计和对账逻辑。而仓库里真实模型的有效修复数目前还是 0。最近两周的提交大多是同一个 SWE-bench 实例的评分对照。建议把精力转向任务规模、rollout 吞吐和真实训练结果。
