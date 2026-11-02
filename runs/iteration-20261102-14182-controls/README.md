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
bash runs/iteration-20261102-14182-controls/run_controls.sh \
  /absolute/grading-venv/bin/python /private/new-14182-inputs /private/new-14182-run
```

脚本要求使用新结果目录。它先核对评分文件、wheel 的安装版本、该题的实际镜像引用、平台以及 harness 生成的脚本，再在同一个固定镜像上分别运行正负对照。每组生成新的 `run_id`，并发为 1，单次官方测试超时为 600 秒。主机的绝对停止时刻需覆盖整个实验，由租用方按预算独立控制。镜像的 10 个压缩层合计 1,137,501,157 字节，这个数只用于估算传输量。

两个报告返回后，脚本调用 `swe-check-controls`。正控须报告已应用、已解决、两类目标均无失败。负控须报告已应用、未解决、至少一个修复目标失败，且没有原有回归目标失败。两份补丁、最终执行脚本和目标集合均须命中预先冻结的摘要。只有当前命令成功退出并生成新的 `controls-verified.json`，才完成本次主机验收。回执记录了两份原始报告的哈希。各目标的通过/失败指官方解析口径，具体测试进程是否执行仍可查原始日志。

也可以只读复核已有报告。以下操作不调用 Docker 或模型：

```bash
python -m budget_agent swe-check-controls --spec "$case_dir/14182-spec.json" \
  --gold-report /private/gold/astropy__astropy-14182/report.json \
  --negative-report /private/negative/astropy__astropy-14182/report.json \
  --output /private/new-control-receipt.json
```

<a id="cloud-cache-14182"></a>

### 在后续主机复用本次云端镜像

2026-11-03 的主机已把完整 OCI 布局保存到该账户已有的持久盘。Docker 29.1.4 的 containerd 存储使用以下步骤将本地镜像与求解清单中的规范引用关联起来。换主机后先核实挂载目录、Docker daemon 的 containerd socket 和命名空间。以下命令适用于本次核实的环境。导入完成仍须运行上面的两项官方对照。[首次云端记录](cloud-preflight-20261103.json)

```bash
set -euo pipefail
layout=/home/featurize/work/budget-agent-assets/docker/14182-4a1da544
repo=swebench/sweb.eval.x86_64.astropy_1776_astropy-14182
digest=sha256:4a1da544b0a2d7bb43266093db899f77c60700f76b84497e6c66138cb30d8663
ctr=(sudo ctr -a /run/docker/containerd/containerd.sock -n moby)
tar -C "$layout" -cf - oci-layout index.json blobs | "${ctr[@]}" images import --local --platform linux/amd64 -
if ! sudo docker image inspect "$repo@$digest" >/dev/null 2>&1; then
    ref="docker.io/$repo:14182-verified"
    "${ctr[@]}" images ls | awk -v n="$ref" -v d="$digest" '$1 == n && $3 == d {ok=1} END {exit !ok}'
    "${ctr[@]}" images tag --local "$ref" "docker.io/$repo@$digest"
fi
sudo docker image inspect "$repo@$digest" --format '{{.Os}}/{{.Architecture}} {{json .RepoDigests}}'
```

首次正控在归还前应用了标准补丁，尚无最终判分。下轮使用新的目录，不复用本次未完成的官方运行。

### 下一组真实策略对照

本题属于项目开发集。评分环境验收通过且本轮预算已获授权后，求解侧可共用 `configs/reference-qwen38-max.toml`、本目录的单题清单及 `--total-tokens 64000`，分别指定 `--decision-policy direct` 和 `--decision-policy workspace-feedback`，每侧一次、各用新运行目录。使用上述固定评分数据为两份补丁各设新的官方 `run_id`，将实际评分材料与本次对照回执核对后再解释成败。记录两侧是否交付补丁、首次工作区变化消耗的请求及 token，以及最终独立评分。
