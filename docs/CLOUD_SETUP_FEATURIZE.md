> 本文保留首次 preflight 的历史状态。后续真实 7B 推理、SWE-bench 评分、局部编辑与持久化进展见 [v3 云端验证记录](VALIDATION_V3.md)。

# Featurize 云端配置与验证记录

检查日期：2026-10-21 至 2026-10-22（北京时间）。本次只完成租用、依赖配置和基础设施验证，未下载基础模型、未启动 SFT 或 RL，也没有在本地电脑训练。

## 实例与费用

| 项目 | 实际配置 |
|---|---|
| 平台 | Featurize |
| 实例 ID | `900eebaa-209e-4f59-b9d0-b375cfb05b4f` |
| GPU | 1 × NVIDIA GeForce RTX 4090，24 GB |
| CPU | 14 核 AMD EPYC 7453 |
| 内存 / 磁盘 | 控制台显示 64.4 GB / 401.0 GB |
| 系统镜像 | 官方 GPU Docker，Ubuntu 22.04.4 |
| 计费 | 按量，1.87 元/小时 |
| 租用开始 | 2026-10-21 23:44，控制台时间 |
| 计划归还 | 2026-10-22 01:34（北京时间） |

已创建 `budget-agent-auto-release-v2.timer`，调用平台 CLI 归还上述实例。核验结果为 `ActiveState=active`，下一次触发为 `2026-10-21 17:34:00 UTC`。服务以 `featurize` 用户运行，使用权限为 0600 的本机环境文件传入平台原有认证环境；环境文件没有复制到项目或本地。

计划使用约 110 分钟，按当前单价估算约 3.43 元；加上此前矩池云约 0.22 元，约 3.65 元，实际以平台账单为准。定时器尚未到触发时间，**已设置不等于已归还**，尚无最终归还与停止计费回执。

## 连接与目录

在这台 Windows 电脑上连接（`-F NUL` 避开现有 SSH 配置文件权限问题）：

```powershell
ssh -F NUL -p <控制台端口> <控制台用户名>@<控制台主机>
```

连接参数和密码以平台当前实例的“SSH 连接”窗口为准。这里使用占位符；历史实例的连接信息不再作为可执行配置保留。

```bash
source /home/featurize/budget-agent-env/bin/activate
cd /home/featurize/work/budget-agent
```

- 项目、有效 SFT adapter 和验证结果：`/home/featurize/work/budget-agent`。
- Python 虚拟环境：`/home/featurize/budget-agent-env`，位于实例本地磁盘。
- `/home/featurize/work` 是平台持久工作目录，本轮检查占用约 39 MB，未扩容；模型缓存和虚拟环境不放进该目录。
- 归还后，实例本地虚拟环境和 Docker 镜像缓存需要重新配置；持久目录的实际保留规则见平台文档。

## 已验证的依赖

| 组件 | 版本 / 结果 |
|---|---|
| Python | 3.11.8 |
| NVIDIA driver | 580.95.05 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Transformers | 5.4.0 |
| PEFT | 0.19.0 |
| Accelerate | 1.13.0 |
| Docker server | 29.1.4 |
| 项目安装 | `pip install -e /home/featurize/work/budget-agent` 成功 |

实际 GPU 检查包含 `torch.cuda.is_available()`、`torch.cuda.is_bf16_supported()`、GPU 上的 BF16 矩阵乘法与结果断言，全部通过。这不包含模型前向、反向或优化器更新。

依赖安装方式如下，已在云端执行成功：

```bash
python -m venv /home/featurize/budget-agent-env
source /home/featurize/budget-agent-env/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.4.0 peft==0.19.0 accelerate==1.13.0
pip install -e /home/featurize/work/budget-agent
```

本次解析到的完整安装包列表在 `runs/featurize-preflight-01/setup/pip-freeze.txt`；这是一份环境快照，没有声称所有间接依赖已经建立长期兼容保证。

## Docker 与项目烟测

`docker run --rm hello-world` 成功。直接从 Docker Hub 下载固定 Python 镜像的层时遇到 EOF，改用 DaoCloud 官方镜像服务取得同一 SHA256 摘要。下载完成后，原始 Docker Hub 引用也成功拉取，并在本地解析到同一镜像内容，无须修改项目默认镜像配置。

固定摘要：

```text
sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
```

成功运行的项目命令：

```bash
cd /home/featurize/work/budget-agent
PYTHONPATH=src python -m budget_agent run \
  --config configs/smoke.toml \
  --task slugify-digits \
  --environment docker \
  --docker-image m.daocloud.io/docker.io/library/python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce \
  --output runs/featurize-preflight-01
```

结果为 `finished`、`resolved=true`，独立评分运行 3 个测试，0 失败、0 错误，推理预算未超限。工具执行与评分使用 Docker。此运行使用 `scripted` provider，只证明执行、修复、评分和报告链路可用，**不证明 LLM 的代码修复能力或 RL 收益**。

验证报告、轨迹、补丁、评分结果与环境证据已回传本地：

- `runs/featurize-preflight-01/report/report.html`
- `runs/featurize-preflight-01/report/report.md`
- `runs/featurize-preflight-01/setup/`
- `train_outputs/featurize-validation-20261021.tar.gz`

后续模型验证仍需下载固定 revision 的 Qwen2.5-Coder-0.5B-Instruct。旧 `runs/learning-v2/grpo` 结果依旧无效；后续 RL 必须从有效 SFT adapter 重新生成云端轨迹，不能沿用旧 RL 结果。

## 官方参考

- [工作目录与持久化](https://docs.featurize.cn/docs/manual/work)
- [CLI](https://docs.featurize.cn/docs/manual/cli)
- [实例租用](https://docs.featurize.cn/docs/manual/instance-rent)
- [DaoCloud 官方镜像服务与摘要校验](https://github.com/DaoCloud/public-image-mirror)
