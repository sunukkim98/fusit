#!/bin/bash
# Submit the grouping diagnostic (fusit.diag_grouping) for one dataset across the usable GPUs.
#
#   bash scripts/diag_grouping/submit.sh synthetic|synthpai [--smoke] [--retry]
#
# Two waves per dataset. Generation units run fusit.main -- tags reused from results/table1,
# DP-Fusion paraphrases, attack -- for source grouping at the caps table1 lacks, single
# grouping at every cap and, on synthetic, the seed 1-3 variance runs. Eval units (judge,
# lexical scores, perplexity) are submitted with afterany on every generation worker, since
# they score whatever generation produced.
#
# --smoke runs 20 items per dataset under results/diag_grouping/smoke with fewer shards.
# --retry releases every unfinished claim of that dataset first, including ones a running
# worker holds; only use it once no worker of that dataset is left in the queue.
#
# As scripts/table1: SynthPAI profiles peak at ~20 GB, so SynthPAI units go to full RTX 4090s;
# the 24 GB MIG slices on node6 take synthetic only.
set -eu
cd /home/sunuk/fusit
PY=/home/sunuk/miniconda3/envs/dpfusion-repro/bin/python
DS=${1:-}; shift || true
case $DS in synthetic|synthpai) ;; *) echo "usage: $0 synthetic|synthpai [--smoke] [--retry]"; exit 2 ;; esac
ROOT=results/diag_grouping; N_ITEMS=""; RETRY=0
for a in "$@"; do
  case $a in --smoke) ROOT=results/diag_grouping/smoke; N_ITEMS=20 ;; --retry) RETRY=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done
P=dg; [ -n "$N_ITEMS" ] && P=dgs
PREFIX=${P}_${DS}

# shard counts and task names come from fusit.diag_grouping, so the two cannot drift
read -r NSH VAR_TASKS VAR_NSH VAR_DS <<< "$($PY -c "
from fusit.diag_grouping import SHARDS, VARIANCE, gen_tasks
print(SHARDS['$DS'], ','.join(k for k, t in gen_tasks().items() if t.get('variance')), VARIANCE['num_shards'], VARIANCE['dataset'])")"
VAR_TASKS=${VAR_TASKS//,/ }
if [ -n "$N_ITEMS" ]; then NSH=2; VAR_NSH=1; fi

if [ $RETRY = 1 ]; then
  if squeue -u "$USER" -h -o %j | grep -q "^${PREFIX}_"; then
    echo "workers of $PREFIX are still queued; refusing --retry"; exit 1
  fi
  for c in $ROOT/claims/*_${DS}_*; do
    [ -e "$c" ] || continue
    [ -e "$c/done" ] || { echo "releasing unfinished claim $(basename $c)"; rm -rf "$c"; }
  done
fi
mkdir -p $ROOT/logs $ROOT/claims
[ -e $ROOT/config.json ] || $PY -m fusit.diag_grouping config --root $ROOT ${N_ITEMS:+--n-items $N_ITEMS}
# the tree may be uncommitted: keep the code this submission runs next to its results
SNAP=$ROOT/code/$(date +%Y%m%d_%H%M%S)_$DS
mkdir -p $SNAP && git diff HEAD > $SNAP/uncommitted.diff && git rev-parse HEAD > $SNAP/HEAD \
  && cp src/fusit/main.py src/fusit/diag_grouping.py scripts/diag_grouping/*.s* $SNAP/

units() {  # task nshards
  for i in $(seq 0 $(($2 - 1))); do printf "%s:%s:%d:%d " $1 $DS $2 $i; done
}
GEN_UNITS="$(units source $NSH)$(units single $NSH)"
if [ "$DS" = "$VAR_DS" ]; then
  for t in $VAR_TASKS; do GEN_UNITS+="$(units $t $VAR_NSH)"; done
fi
EVAL_UNITS="$(units eval $NSH)"

JOBS=()
submit() {  # wave node count units [dependency]
  local wave=$1 node=$2 count=$3 units=$4 dep=${5:-}
  for _ in $(seq 1 $count); do
    JOBS+=($(sbatch --parsable --job-name=${PREFIX}_${wave}_$node --nodelist=$node ${dep:+--dependency=$dep} \
      --output=$ROOT/logs/%x_%j.out --error=$ROOT/logs/%x_%j.err \
      --export=ALL,ROOT=$ROOT,UNITS="$units",N_ITEMS="$N_ITEMS" scripts/diag_grouping/worker.sbatch))
  done
}
placement() {  # wave units [dependency]
  if [ -n "$N_ITEMS" ]; then
    [ $DS = synthetic ] && submit $1 node6 4 "$2" "${3:-}"
    submit $1 node4 2 "$2" "${3:-}"
  elif [ $DS = synthetic ]; then
    submit $1 node6 4 "$2" "${3:-}"; submit $1 node2 3 "$2" "${3:-}"; submit $1 node4 3 "$2" "${3:-}"; submit $1 node3 1 "$2" "${3:-}"
  else
    submit $1 node2 4 "$2" "${3:-}"; submit $1 node4 4 "$2" "${3:-}"; submit $1 node3 1 "$2" "${3:-}"
  fi
}

placement gen "$GEN_UNITS"
GEN_JOBS=$(IFS=:; echo "${JOBS[*]}")
JOBS=()
placement eval "$EVAL_UNITS" "afterany:$GEN_JOBS"
EVAL_JOBS=$(IFS=:; echo "${JOBS[*]}")

echo "$(date '+%F %T') $DS root=$ROOT head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) gen=$GEN_JOBS eval=$EVAL_JOBS" \
  | tee -a $ROOT/submissions.log
