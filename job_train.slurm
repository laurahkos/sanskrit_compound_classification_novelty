#!/bin/bash --login
# Use sbatch job_train.slurm <cond1|cond1S|cond2|cond2S|cond3|cond3S|cond3B|cond3BS> <seed>
# trains both byt5 models and the classifier, caches per-record predictions on both test sets,
# runs the kāraka axis from the caches, then deletes the checkpoints of fully cached models.
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 12
#SBATCH -t 96:00:00

module purge
module load libs/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

COND=${1:?usage: sbatch job_train.slurm <cond> <seed>}
SEED=${2:?usage: sbatch job_train.slurm <cond> <seed>}
cd "${PROJECT_DIR:-$HOME/iscls_code}"

VCD_DIR=data/vcd
SACTI_DIR=data/sacti_dedup_context
VCD_TEST=$VCD_DIR/type/test.jsonl
SACTI_TEST=$SACTI_DIR/test.jsonl
RUN_DIR=${RUNS_ROOT:-runs}/$COND/seed$SEED
CACHE_DIR=${CACHE_ROOT:-cache}/$COND/seed$SEED
KARAKA_DIR=${KARAKA_ROOT:-karaka}/$COND/seed$SEED
mkdir -p "$RUN_DIR" "$CACHE_DIR" "$KARAKA_DIR"

CTX=(); MAX_LEN=64; BATCH=32; ACCUM=1
if [[ $COND == *S ]]; then CTX=(--use-sentence); MAX_LEN=256; BATCH=16; ACCUM=2; fi

SACTI_TRAIN=(--conll "$SACTI_DIR/train.jsonl" --conll-dev "$SACTI_DIR/dev.jsonl")
case ${COND%S} in
    cond1)  DATA=(--vcd-data "$VCD_DIR") ;;
    cond2)  DATA=("${SACTI_TRAIN[@]}") ;;
    cond3)  DATA=(--vcd-data "$VCD_DIR" "${SACTI_TRAIN[@]}" --mix-ratio 0.5) ;;
    cond3B) DATA=(--vcd-data "$VCD_DIR" "${SACTI_TRAIN[@]}" --mix-ratio 0.5 --mix-mode balanced) ;;
    *) echo "unknown condition: $COND" >&2; exit 1 ;;
esac

{
    echo "date: $(date -Is)  job: ${SLURM_JOB_ID:-none}  cond: $COND  seed: $SEED"
    echo "byt5: epochs=15 micro_batch=$BATCH grad_accum=$ACCUM max_len=$MAX_LEN"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -c 'from type_classifier import CONLL_FINE_TO_VCD as m; print("sacti_map:", sorted(m.items()))'
} > "$RUN_DIR/job_info.txt"

python byt5_type_eval.py "${DATA[@]}" "${CTX[@]}" --output "$RUN_DIR" --seed "$SEED" \
    --epochs 15 --batch-size "$BATCH" --grad-accum "$ACCUM" --max-len "$MAX_LEN"

TAGS=(google_byt5-base buddhist-nlp_byt5-sanskrit classifier)
ROOTS=("$RUN_DIR/google_byt5-base" "$RUN_DIR/buddhist-nlp_byt5-sanskrit" "$RUN_DIR/classifier-baseline")
CKPTS=("${ROOTS[0]}/best" "${ROOTS[1]}/best" "${ROOTS[2]}/best_model.pt")

CACHED=()
for i in "${!TAGS[@]}"; do
    tag=${TAGS[$i]}
    if python cache_predictions.py --checkpoint "${CKPTS[$i]}" --eval-data "$VCD_TEST" --output "$CACHE_DIR/${tag}_vcd.json" &&
       python cache_predictions.py --checkpoint "${CKPTS[$i]}" --eval-data "$SACTI_TEST" --output "$CACHE_DIR/${tag}_sacti.json"; then
        CACHED[$i]=1
        for f in trainer_state.json best/trainer_state.json epoch_log.jsonl; do
            if [ -f "${ROOTS[$i]}/$f" ]; then cp "${ROOTS[$i]}/$f" "$CACHE_DIR/${tag}_$(basename "$f")"; fi
        done
    else
        CACHED[$i]=0
        rm -f "$CACHE_DIR/${tag}_vcd.json" "$CACHE_DIR/${tag}_sacti.json"
        echo "$tag: caching failed, checkpoint kept" >&2
    fi
done

KARAKA_ARGS=()
for i in "${!TAGS[@]}"; do
    if [ "${CACHED[$i]}" = 1 ]; then KARAKA_ARGS+=(--model "${TAGS[$i]}=$CACHE_DIR/${TAGS[$i]}_sacti.json"); fi
done
if [ ${#KARAKA_ARGS[@]} -gt 0 ]; then
    python karaka_axis_t3.py --eval-data "$SACTI_TEST" "${KARAKA_ARGS[@]}" \
        --output "$KARAKA_DIR/karaka_axis.json" > "$KARAKA_DIR/karaka_axis.log" 2>&1 ||
        echo "WARNING: kāraka axis failed; re-run from the caches (see $KARAKA_DIR/karaka_axis.log)" >&2
fi

for i in "${!TAGS[@]}"; do
    if [ "${CACHED[$i]}" = 1 ]; then rm -rf "${ROOTS[$i]}"; fi
done
