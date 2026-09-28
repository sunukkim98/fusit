#!/bin/bash
#SBATCH --partition=slave
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=12:00:00
#SBATCH --output=results/table1/logs/%x_%j.out
#SBATCH --error=results/table1/logs/%x_%j.err

# One GPU worker for the Table 1 run. It walks the work units in UNITS (dataset:shard pairs)
# and claims each with an atomic mkdir on the shared filesystem before running it, so any
# number of workers on GPUs of any speed divide the work without a fixed assignment -- a
# fast card simply claims more units. A unit whose claim exists but has no .done marker was
# interrupted; its output file is a checkpoint, and resubmitting after clearing that claim
# resumes it (see submit.sh --retry).
#
#   UNITS    space-separated "dataset:shard" list, in claim order
#   NSHARDS  "dataset:N" pairs giving each dataset's shard count
#   VARIANT  trace (default): TRACE cues -> DP-Fusion, results/table1
#            dpfusion: DP-Fusion as the paper runs it with a real tagger -- Presidio +
#            BERT-NER at threshold 0.5, single group -- results/table1_dpfusion
#            tracerps: the TRACE and TRACE-RPS baselines (fusit.baselines) with the same
#            models -- results/table1_tracerps
#            utility: utility of every output of the three runs above (fusit.utility) --
#            results/table1_utility
#            perplexity: DP-Fusion's teacher-forced perplexity for the same outputs
#            (fusit.perplexity) -- results/table1_perplexity

set -u
cd /home/sunuk/fusit
PY=/home/sunuk/miniconda3/envs/dpfusion-repro/bin/python
VARIANT=${VARIANT:-trace}
MODULE=fusit.main
case "$VARIANT" in
  trace)
    OUT=results/table1
    VARIANT_ARGS="--tagger-model NousResearch/Llama-2-7b-chat-hf --sources ner cot att"
    ;;
  dpfusion)
    OUT=results/table1_dpfusion
    # DP-Fusion Appendix A.17-A.18: Presidio driving dslim/bert-base-NER at threshold 0.5,
    # all private entities, single-group DP-Fusion. no_defense and x_priv are skipped:
    # the first is already scored by the trace run with the same attacker, and with NER as
    # the only source the second is identical to ner_redaction.
    VARIANT_ARGS="--sources ner --precedence ner --ner-backend presidio --ner-score-threshold 0.5 --attack-conditions ner_redaction dp_fusion"
    ;;
  tracerps)
    OUT=results/table1_tracerps
    MODULE=fusit.baselines
    ;;
  utility)
    OUT=results/table1_utility
    MODULE=fusit.utility
    ;;
  perplexity)
    OUT=results/table1_perplexity
    MODULE=fusit.perplexity
    ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 2 ;;
esac
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Every model is already in the shared HF cache. node6 has no route to huggingface.co, and
# without this each from_pretrained stalls on retried HEAD requests before falling back.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nshards_of() { for p in $NSHARDS; do [ "${p%%:*}" = "$1" ] && echo "${p##*:}"; done; }

echo "[$VARIANT] worker $SLURM_JOB_ID on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for unit in $UNITS; do
  ds=${unit%%:*}; shard=${unit##*:}
  claim=$OUT/claims/${ds}_${shard}
  mkdir -p "$OUT/claims"
  mkdir "$claim" 2>/dev/null || continue
  echo "$SLURM_JOB_ID $(hostname)" > "$claim/owner"
  echo "[$(date +%T)] $ds shard $shard/$(nshards_of $ds) start"

  k=10; [ "$ds" = synthpai ] && k=30   # experiment plan 2.1: 10 synthetic / 30 synthpai
  SHARD_ARGS="--dataset $ds --num-shards $(nshards_of $ds) --shard-index $shard"
  MODELS="--paraphrase-model Qwen/Qwen2.5-7B-Instruct --attacker-model NousResearch/Meta-Llama-3.1-8B-Instruct --k-att $k"
  case "$MODULE" in
    fusit.main)      RUN="$PY -m fusit.main $VARIANT_ARGS --alpha-beta 0.01 0.1 $MODELS $SHARD_ARGS --n-items 0" ;;
    fusit.baselines) RUN="$PY -m fusit.baselines --tagger-model NousResearch/Llama-2-7b-chat-hf --max-rounds ${MAX_ROUNDS:-5} $MODELS $SHARD_ARGS --n-items 0" ;;
    fusit.utility)   RUN="$PY -m fusit.utility $SHARD_ARGS" ;;
    fusit.perplexity) RUN="$PY -m fusit.perplexity $SHARD_ARGS" ;;
  esac
  if $RUN \
      --output "$OUT/$ds/shard_${shard}.jsonl"; then
    touch "$claim/done"
    echo "[$(date +%T)] $ds shard $shard done"
  else
    echo "[$(date +%T)] $ds shard $shard FAILED (claim kept, output is a checkpoint)"
  fi
done
echo "[$(date +%T)] no units left to claim"
