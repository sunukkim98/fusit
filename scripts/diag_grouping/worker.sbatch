#!/bin/bash
#SBATCH --partition=slave
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=12:00:00

# One GPU worker for the grouping diagnostic (fusit.diag_grouping), on scripts/table1's
# pattern: it walks the units in UNITS and claims each with an atomic mkdir before running it,
# so workers on GPUs of any speed divide the work without a fixed assignment. A unit whose
# claim exists but has no .done marker was interrupted; its output file is a checkpoint, and
# resubmitting after clearing that claim resumes it (see submit.sh --retry).
#
#   UNITS    space-separated "task:dataset:num_shards:shard" list, in claim order. task is a
#            fusit.diag_grouping gen task (source, single, <grouping>_s<seed>) or eval
#   ROOT     output root, e.g. results/diag_grouping
#   N_ITEMS  optional: items per dataset (smoke runs)

set -u
cd /home/sunuk/fusit
PY=/home/sunuk/miniconda3/envs/dpfusion-repro/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Every model is already in the shared HF cache; see scripts/table1/worker.sbatch.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NITEMS_ARG=""; [ -n "${N_ITEMS:-}" ] && NITEMS_ARG="--n-items $N_ITEMS"

echo "worker $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for unit in $UNITS; do
  IFS=: read -r task ds nshards shard <<< "$unit"
  claim=$ROOT/claims/${task}_${ds}_${shard}
  mkdir -p "$ROOT/claims"
  mkdir "$claim" 2>/dev/null || continue
  echo "$SLURM_JOB_ID $(hostname)" > "$claim/owner"
  echo "[$(date +%T)] $task $ds shard $shard/$nshards start"
  if [ "$task" = eval ]; then
    RUN="$PY -m fusit.diag_grouping eval --root $ROOT --dataset $ds --num-shards $nshards --shard-index $shard"
  else
    RUN="$PY -m fusit.diag_grouping gen --root $ROOT --task $task --dataset $ds --num-shards $nshards --shard-index $shard $NITEMS_ARG"
  fi
  if $RUN; then
    touch "$claim/done"
    echo "[$(date +%T)] $task $ds shard $shard done"
  else
    echo "[$(date +%T)] $task $ds shard $shard FAILED (claim kept, output is a checkpoint)"
  fi
done
echo "[$(date +%T)] no units left to claim"
