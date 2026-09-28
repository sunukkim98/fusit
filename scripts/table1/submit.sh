#!/bin/bash
# Submit the Table 1 run across every usable GPU.
#
#   bash scripts/table1/submit.sh [trace|dpfusion|tracerps|utility|perplexity|floor]           first submission (default trace)
#   bash scripts/table1/submit.sh [trace|dpfusion|tracerps|utility|perplexity|floor] --retry   clear unfinished claims, resubmit
#
# --retry releases EVERY unfinished claim, including ones a running worker holds; only use it
# once no worker of that variant is left in the queue.
#
# SynthPAI profiles run to 13.5k characters and peak at ~20 GB, so SynthPAI units are only
# claimed by full RTX 4090s; the MIG slices on node6 (24 GB, a quarter of the compute) take
# the short synthetic comments. 4090 workers take SynthPAI first, then help with synthetic.
set -eu
cd /home/sunuk/fusit
VARIANT=trace
case "${1:-}" in trace|dpfusion|tracerps|utility|perplexity|floor) VARIANT=$1; shift ;; esac
case $VARIANT in trace) OUT=results/table1 P=t1t ;; dpfusion) OUT=results/table1_dpfusion P=t1d ;; tracerps) OUT=results/table1_tracerps P=t1r ;; utility) OUT=results/table1_utility P=t1u ;; perplexity) OUT=results/table1_perplexity P=t1p ;; floor) OUT=results/diag_floor P=t1f ;; esac

SYNTHPAI_SHARDS=30   # ~10 profiles each
SYNTHETIC_SHARDS=20  # ~26 comments each
NSHARDS="synthpai:$SYNTHPAI_SHARDS synthetic:$SYNTHETIC_SHARDS"

if [ "${1:-}" = "--retry" ]; then
  if squeue -u "$USER" -h -o %j | grep -q "^${P}_"; then
    echo "workers of variant $VARIANT are still queued; refusing --retry"; exit 1
  fi
  for c in $OUT/claims/*; do
    [ -e "$c/done" ] || { echo "releasing unfinished claim $(basename $c)"; rm -rf "$c"; }
  done
fi
mkdir -p $OUT/logs $OUT/claims $OUT/synthpai $OUT/synthetic

synthpai_units=$(for i in $(seq 0 $((SYNTHPAI_SHARDS-1))); do printf "synthpai:%d " $i; done)
synthetic_units=$(for i in $(seq 0 $((SYNTHETIC_SHARDS-1))); do printf "synthetic:%d " $i; done)

submit() {  # node count units
  local node=$1 count=$2 units=$3
  for _ in $(seq 1 $count); do
    sbatch --job-name=${P}_$node --nodelist=$node \
      --output=$OUT/logs/%x_%j.out --error=$OUT/logs/%x_%j.err \
      --export=ALL,VARIANT=$VARIANT,UNITS="$units",NSHARDS="$NSHARDS" scripts/table1/worker.sbatch
  done
}

submit node2 4 "$synthpai_units $synthetic_units"
submit node4 4 "$synthpai_units $synthetic_units"
submit node3 1 "$synthpai_units $synthetic_units"
submit node6 4 "$synthetic_units"
