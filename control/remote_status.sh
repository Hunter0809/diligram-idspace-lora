set -eu
cd /home/mr_jiang/qkr
date -Is
echo '--- markers ---'
ls -l runs/idspace_e3/train.started runs/idspace_e3/train.completed runs/idspace_e3/launcher.pid 2>/dev/null || true
echo '--- launch log ---'
ls -l logs/idspace_e3_launch.log 2>/dev/null || true
tail -20 logs/idspace_e3_launch.log 2>/dev/null || true
echo '--- lock and script ---'
ls -l runs/idspace_e3/.e3.lock 2>/dev/null || true
cat runs/idspace_e3/launcher.pid 2>/dev/null || true
ps -eo pid,etime,pcpu,pmem,state,args | grep -E 'idspace_e3_prepare|idspace_e3_launch|torchrun|train.py' | grep -v grep || true
printf 'prepared_files='
find runs/idspace_e3/prepared -type f 2>/dev/null | wc -l
ls -lh runs/idspace_e3/prepared 2>/dev/null | tail -8 || true
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
