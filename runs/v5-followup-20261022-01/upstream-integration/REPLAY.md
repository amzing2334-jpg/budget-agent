# 固定 mini-SWE 上游的 CPU 集成证据

本目录保存 2026-10-22 实际运行的 6 个上游集成场景。真实执行固定 SHA 的 `DefaultAgent`、模板、动作解析、异常流、提交判定和轨迹保存。模型返回文本、使用量和环境命令输出都是预置替身；没有模型推理、训练或 Docker 启动。

- [receipt.json](receipt.json)：固定 SHA、Python、6 项通过、替身边界、依赖检查及逐场景状态。
- [integration-validated.log](integration-validated.log) 与 [JUnit XML](integration-validated.xml)：最终 6 项通过，旧的 13 项契约测试被 deselect。
- [requirements-observed.txt](requirements-observed.txt)：独立临时 venv 的完整依赖冻结。
- `cases/`：六场景的上游 `mini.traj.json`、`cpu-result.json`、已有的 `calls.jsonl` 和补丁；正常提交还保存 exporter 输出。

`cpu-result.json` 是原始 `result.json` 的逐字节副本，改名只为避免真实模型报告递归扫描后误收这些合成测试。内部路径字段保留生成时的临时路径。首次预算检查拒绝生成的场景没有 `calls.jsonl`，因为没有发出模型请求；receipt 已注明。这些文件是接口运行证据，不是 SWE-bench 修复成绩。

## 重放

以下 PowerShell 命令在**新临时 venv** 中安装冻结的 CPU 依赖，再只选择六项真实上游集成。项目既有 `.venv` 只用作创建新环境的 Python，不向其中安装包。不要修改已经保存的证据目录。

```powershell
$taskProject = 'C:\找工作\budget-agent'
$taskEvidence = Join-Path $taskProject 'runs\v5-followup-20261022-01\upstream-integration'
$taskCpuRoot = Join-Path $env:TEMP ('mini-swe-cpu-replay-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskCpuRoot
& "$taskProject\.venv\Scripts\python.exe" -m venv "$taskCpuRoot\venv"
git clone --no-checkout --filter=blob:none https://github.com/SWE-agent/mini-swe-agent.git "$taskCpuRoot\upstream"
git -C "$taskCpuRoot\upstream" checkout --detach a83fcae82d2a08f0ee0c688f9d137b3566c097f8
& "$taskCpuRoot\venv\Scripts\python.exe" -m pip install -r "$taskEvidence\requirements-observed.txt"
$env:MINI_SWE_UPSTREAM_REPO = "$taskCpuRoot\upstream"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& "$taskCpuRoot\venv\Scripts\python.exe" -m pytest "$taskProject\tests\test_mini_swe_baseline.py" -k real_upstream -vv --basetemp "$taskCpuRoot\test-artifacts" --junitxml "$taskCpuRoot\integration.xml"
& "$taskCpuRoot\venv\Scripts\python.exe" -m pip check
```

原始运行使用 Windows、Python 3.13.5；上游 SHA 与适配脚本、测试文件的 SHA-256 见 receipt。若之后修改了项目代码，可通过这两个文件哈希判断是否仍是同一适配实现。测试中会执行实际上游加载与版本校验，并检查没有导入模型或训练模块。
