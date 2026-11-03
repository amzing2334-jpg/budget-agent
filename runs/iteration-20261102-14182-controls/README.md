# Astropy 14182 评分环境验收

在开始 `direct` 与 `workspace-feedback` 的付费对照前，先检验评分器能否在这道题上区分标准修复和没有修改源码的负控。本机已核对数据、镜像元数据、补丁可应用性和评分器将生成的脚本。本目录尚无 14182 的容器测试成绩。

### 本地可复现的准备

从[固定提交](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/resolve/b0dde1093fe417d83b7184254edf8199c1f0dff5/data/test-00000-of-00001.parquet)下载 test parquet。用安装了 PyArrow 的 Python 运行以下命令。已验证的组合为 Python 3.13.5、PyArrow 23.0.1。输出目录的父目录应已存在。

```bash
case_dir=runs/iteration-20261102-14182-controls
python "$case_dir/prepare_grader.py" --parquet /path/to/test.parquet --output /private/new-14182-inputs
```

程序校验输入与输出哈希，生成三份仅供评分器使用的文件。`dataset.json` 保留发布方原始行，`pinned-dataset.json` 只替换其镜像引用，`negative.jsonl` 提交一份新增无关标记文件的补丁。求解器只读取 [agent-task.jsonl](agent-task.jsonl)。它保留原脱敏题面，并使用与 `pinned-dataset.json` 相同的 Linux/amd64 manifest 摘要。标准修复和隐藏测试不放入 Agent 工作区。[固定材料及 SHA-256](14182-spec.json)

### 经授权后在 Linux 评分主机执行

先在专用主机准备 `swebench==5.0.2` 环境、私有输入和已核定的按时归还安排。在该主机拉取规格中 `solver.pinned_image` 的完整摘要引用，然后运行：

```bash
# STOP_EPOCH 是本次主机归还定时器和执行器硬停止时刻中较早的一个（UTC Unix 秒）。
# 由编排器从本次已核验的时限传入，不沿用上一次的时间戳。
bash runs/iteration-20261102-14182-controls/run_controls.sh \
  /absolute/grading-venv/bin/python /private/new-14182-inputs /private/new-14182-run \
  "${STOP_EPOCH:?set from the current verified deadline}"
```

脚本要求使用新结果目录。它先核对评分文件、wheel 的安装版本、该题的实际镜像引用、平台以及 harness 生成的脚本，再在同一个固定镜像上分别运行正负对照。每组生成新的 `run_id`，并发为 1，单次官方测试超时为 600 秒。镜像的 10 个压缩层合计 1,137,501,157 字节，这个数只用于估算传输量。

每次启动官方评分器前检查主机时钟。正控前剩余时间必须 **大于 1,320 秒**（两侧各 600 秒，另留 120 秒归档）；负控前按实耗重算，须 **大于 720 秒**。每次在当前结果目录的 `control-admission.jsonl` 记录时间和判定。余量不足时退出码为 75；保留已有证据，不启动该侧，也不写 `controls-verified.json`。不要原地重试；旧的三参数调用须更新，已归档的试跑包仍为旧版本。只有评分入口成功退出且本次通过回执核验成功，编排器才能继续，并须在启动模型前另行核对预算和时间。

此检查只做启动决策。600 秒是容器内测试超时，容器准备、评分、归档也会耗时；120 秒是本项目预留，尚无数据证明它覆盖所有负载。独立的中断、归还定时器和证据取回流程仍须生效，入口不会替调用者安装定时器、延长租约或验证传入的截止时间。

正控进程返回后，脚本先用 `swe-check-gold` 读取当前运行的 `report.json` 和相邻的 `patch.diff`、`eval.sh`。报告须对应冻结题目，补丁、评分脚本和目标集合须匹配冻结摘要，正控须已应用、已解决且两类目标均无失败。通过后才检查负控时间并启动负控。正控缺报告或核验失败时，保留当前目录，按标准错误和实例日志定位原因，在新运行目录重做正控。

两个报告返回后，`swe-check-controls` 重新读取双方原件。负控须报告已应用、未解决、至少一个修复目标失败，且没有原有回归目标失败。只有本次整组命令成功退出并生成新的 `controls-verified.json`，才完成主机验收。回执记录两份原始报告的哈希。各目标的通过或失败按官方解析口径解释，具体测试进程是否执行可查实例测试日志。

也可以只读复核已有报告。以下操作不调用 Docker 或模型：

```bash
python -m budget_agent swe-check-gold --spec "$case_dir/14182-spec.json" \
  --gold-report /private/gold/astropy__astropy-14182/report.json

python -m budget_agent swe-check-controls --spec "$case_dir/14182-spec.json" \
  --gold-report /private/gold/astropy__astropy-14182/report.json \
  --negative-report /private/negative/astropy__astropy-14182/report.json \
  --output /private/new-control-receipt.json
```

<a id="cloud-cache-14182"></a>

### 在后续主机复用本次云端镜像

2026-11-03 的主机已把完整 OCI 布局保存到该账户已有的持久盘。换主机后先核实挂载目录、Docker daemon 的 containerd socket 和命名空间。以下入口针对当时的 Docker 29.1.4、Linux/amd64、`moby` 命名空间及该缓存的单平台索引。保持缓存只读，直至导入结束。导入完成仍须运行上面的两项官方对照。[首次云端记录](cloud-preflight-20261103.json)

```bash
layout=/home/featurize/work/budget-agent-assets/docker/14182-4a1da544
bash runs/iteration-20261102-14182-controls/restore_image.sh "$layout" /absolute/grading-venv/bin/python
```

恢复脚本依据仓库中冻结的[平台 manifest](platform-manifest.json) 检查挂载盘。先列出缺失或长度错误的文件，大小齐全时才逐文件计算 SHA-256。完整缓存会多顺序读取约 1.14 GB 用于核验，之后 tar 只发送 2 个布局文件和 12 个所需对象。脚本还逐字节比对[本地布局索引](cache-index.json)，以固定 containerd 将安装的名称。重新格式化过的索引会被拒绝。这里的本地单平台索引与注册表的多平台索引用途不同。

诊断时可单独运行 `python -m budget_agent swe-cache-files --layout "$layout" --spec "$case_dir/14182-spec.json" --manifest "$case_dir/platform-manifest.json"`。该命令只读磁盘，在整组校验通过后输出可归档的相对路径。文件缺失或损坏时按报错的摘要修复，再用同一命令复核。

首次正控在归还前应用了标准补丁，尚无最终判分。下轮使用新的目录，不复用本次未完成的官方运行。

### 下一组真实策略对照

本题属于项目开发集。评分环境验收通过且本轮预算已获授权后，求解侧可共用 `configs/reference-qwen38-max.toml`、本目录的单题清单及 `--total-tokens 64000`，分别指定 `--decision-policy direct` 和 `--decision-policy workspace-feedback`，每侧一次、各用新运行目录。使用上述固定评分数据为两份补丁各设新的官方 `run_id`，将实际评分材料与本次对照回执核对后再解释成败。记录两侧是否交付补丁、首次工作区变化消耗的请求及 token，以及最终独立评分。
