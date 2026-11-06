#!/usr/bin/env bash
# Read-only inventory of a shared AutoDL instance. Changes nothing.
set -u
section() { printf '\n===== %s =====\n' "$1"; }
section identity
hostname; cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME)='; uname -r; date
section "cpu / memory quota"
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null
free -g | head -2; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
section gpu
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
nvidia-smi 2>/dev/null | grep -E 'CUDA Version' | head -1
section disks
df -h / /root/autodl-tmp /root/autodl-fs /autodl-fs 2>/dev/null
section "top-level layout (sizes, 60s cap)"
ls -la /root | head -40
timeout 60 du -sh /root/autodl-tmp/* 2>/dev/null | sort -h | tail -25
ls -la /root/autodl-fs 2>/dev/null | head -20
section "running work (do not touch)"
ps -eo pid,user,etime,pcpu,pmem,args --sort=-pcpu | head -25
tmux ls 2>&1 | head; screen -ls 2>&1 | head -5
crontab -l 2>&1 | head
section "python / conda"
which python python3 conda pip uv git 2>&1
python --version 2>&1; python3 --version 2>&1
conda env list 2>&1 | head -20
pip config list 2>&1 | head; cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null | head -20
cat ~/.condarc 2>/dev/null | head -20
python -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())" 2>&1 | tail -1
section "network helpers"
ls -la /etc/network_turbo 2>&1; env | grep -iE '^(HF_|HUGGINGFACE|MODELSCOPE|http_proxy|https_proxy)' 2>/dev/null
timeout 8 curl -sI https://hf-mirror.com 2>&1 | head -1; timeout 8 curl -sI https://www.modelscope.cn 2>&1 | head -1
timeout 8 curl -sI https://github.com 2>&1 | head -1; timeout 8 curl -sI https://pypi.org/simple/ 2>&1 | head -1
section "isolation-relevant"
which docker bwrap firejail 2>&1; id; getent passwd | awk -F: '$3>=1000{print $1,$3,$6}' | head
stat -c '%A %U %n' /root /root/autodl-tmp 2>/dev/null
