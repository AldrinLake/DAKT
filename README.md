申请集群
salloc --partition=gpu-3090-01 --gres=gpu:6 --mem=120G --cpus-per-task=24 --time=999:00:00 -J task
salloc --partition=gpu-4090-01 --gres=gpu:8 --mem=120G --cpus-per-task=24 --time=999:00:00 -J task
salloc --partition=gpu-A10-01 --gres=gpu:10 --mem=120G --cpus-per-task=24 --time=999:00:00 -J task
查看分配到的节点
echo $SLURM_JOB_NODELIST


salloc --partition=gpu-3090-01 --gres=gpu:6 --mem=120G --cpus-per-task=24 --time=12:00:00 -J train_3090 \
bash -lc 'JOBID=${SLURM_JOB_ID:-manual}; OUT=$HOME/salloc_job_${JOBID}.info; port=$(python - <<PY
import socket
s=socket.socket(); s.bind(("0.0.0.0",0)); p=s.getsockname()[1]; s.close(); print(p)
PY
); nohup code-server --bind-addr 0.0.0.0:${port} --auth password > ~/code-server-${JOBID}.log 2>&1 & echo "${HOSTNAME}:${port}" > ${OUT}; echo "Started: $(cat ${OUT})"; sleep 1; cat ${OUT}'


查看所有用户的作业
squeue

查看实时资源占用（谁在用哪块 GPU）
squeue -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"

看剩余可用 GPU 的一行命令
sinfo -o "%20P %10a %10l %10D %10t %20N"

