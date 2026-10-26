# GitHub 仓库与实验记录范围

2026-10-26 按用户要求，将本项目完整 Git 历史上传到 `amzing2334-jpg/budget-agent` 私有仓库，并将其设为本地主要远程 `origin`。

- 当前仓库：https://github.com/amzing2334-jpg/budget-agent
- 本地目录：`C:\找工作\budget-agent`
- 当前提交身份仅在本仓库配置：`amzing2334-jpg <234151135+amzing2334-jpg@users.noreply.github.com>`。
- `.gitattributes` 禁止自动换行转换，保留源码指纹和实验文件的原始字节。

2026-10-24 最初在 `Frank081202/budget-agent` 建立首个提交及 `main` 分支。旧仓库继续保留，本地以 `frank-original` 远程引用它；已有提交及其原始作者保持不变。此账号选择仅针对 Coding Agent 项目，不改变工作区中其他项目的归属。

## 纳入版本管理的内容

代码、配置示例、测试、任务清单、运行脚本、依赖约定、CI 和项目文档完整入库。另显式纳入以下已检查的轻量实验记录，保持原始目录布局：

- `runs/cloud-v5-a4000-20261022-01/`：上一轮七次真实模型尝试的请求、轨迹、结果、补丁与报告。
- `runs/v5-followup-20261022-01/` 顶层分析文档、分析脚本、CPU 检查日志/回执与离线诊断。
- 上述 followup 的 `upstream-integration/` 集成报告、重放说明、依赖清单及六个合成场景的原始接口轨迹。
- `runs/iteration-20261026-01/` 中选定的 CPU 验收回执/日志、历史可见源码审计，以及 `scripted-final/` 的预设动作请求、轨迹、控制日志和报告；这部分属于软件验收，不是新的真实模型实验。早期开发试跑和 workspace 副本仍不上传。
- `runs/cloud-v5-a4000-20261022-01-receipts/` 中选定的评分附加摘要、评分控制摘要、计数与实验计划。

原始实验结果未因迁移改写。七次尝试均来自同一道已知开发题且补丁为空，不构成总体基准成绩；软件测试通过也不等于真实修复能力。详见 [v5 验收](VALIDATION_V5_FOLLOWUP.md)。

## 继续保留在本地的内容

虚拟环境、模型权重、训练检查点、安装包、缓存、源码压缩归档、云端连接/释放脚本、控制台截图以及其余未选定的运行目录不上传。本地原件全部保留；`.gitignore` 仍默认忽略新的 `runs/` 产物，只有明确检查过的证据文件才单独入库。

历史文档中的未选定运行目录、`cloud-receipts/` 及项目外独立评分目录链接属于本地证据引用，在 GitHub 或全新克隆中不一定可用。结果中的绝对路径也是原始运行时记录，不是新电脑上的执行入口。云端实例已归还，文档中的历史连接参数不能作为当前可用服务器。

## 克隆与验证

需要当前私有仓库的访问权限：

```bash
git clone https://github.com/amzing2334-jpg/budget-agent.git
cd budget-agent
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,tokens]"
python -m pytest -q
```

这条安装和测试路线只使用轻量开发依赖，不安装可选训练栈，不启动真实模型推理或训练。真实模型执行和云端租用仍须遵守项目 AGENTS.md 与既有预算约定。
