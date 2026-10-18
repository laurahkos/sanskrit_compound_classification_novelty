#!/bin/bash --login
# use: sbatch job_posthoc.slurm
#SBATCH -p multicore
#SBATCH -n 4
#SBATCH -t 12:00:00

module purge
set -uo pipefail   # no -e: one failed analysis must not stop the rest
cd "${PROJECT_DIR:-$HOME/iscls_code}"

CACHE_ROOT=${CACHE_ROOT:-cache}
KARAKA_ROOT=${KARAKA_ROOT:-karaka}
OUT=${POSTHOC_ROOT:-posthoc}
N_BOOT=${N_BOOT:-2000}
VCD_DIR=data/vcd
SACTI_DIR=data/sacti_dedup_context
VCD_TEST=$VCD_DIR/type/test.jsonl
SACTI_TEST=$SACTI_DIR/test.jsonl
CONDS=(cond1 cond1S cond2 cond2S cond3 cond3S cond3B cond3BS)
TAGS=(classifier google_byt5-base buddhist-nlp_byt5-sanskrit)
mkdir -p "$OUT"/{stratify,compare/arch,compare/context,compare/regime,compare/sampling,karaka,ambiguity,logs}
FAILED=()

run() {   # run <name> <cmd...>: log, record failure, never abort
    local name=$1; shift
    if "$@" > "$OUT/logs/${name//\//__}.log" 2>&1; then echo "ok    $name"
    else echo "FAIL  $name"; FAILED+=("$name"); fi
}
rubric_for() {
    case ${1%S} in
        cond1) echo "$VCD_DIR" ;;
        cond2) echo "$SACTI_DIR/train.jsonl" ;;
        cond3|cond3B) echo "$VCD_DIR $SACTI_DIR/train.jsonl" ;;
    esac
}
eval_for() { [ "$1" = vcd ] && echo "$VCD_TEST" || echo "$SACTI_TEST"; }
has_cache() { compgen -G "$CACHE_ROOT/$1/seed*/$2_$3.json" > /dev/null; }   # cond tag dom
schemes_for() { [ "$1" = vcd ] && echo "5class merged" || echo "merged"; }
merge_flag() { [ "$1" = merged ] && echo "--merge-exo"; }

echo "== stratification"
for cond in "${CONDS[@]}"; do
    for seed_dir in "$CACHE_ROOT/$cond"/seed*; do
        [ -d "$seed_dir" ] || continue
        seed=$(basename "$seed_dir")
        mkdir -p "$OUT/stratify/$cond/$seed"
        for tag in "${TAGS[@]}"; do
            for dom in vcd sacti; do
                [ -f "$seed_dir/${tag}_$dom.json" ] || continue
                for scheme in 5class merged; do
                    run "stratify_${cond}_${seed}_${tag}_${dom}_$scheme" \
                        python stratify_eval.py --dataset "$dom" --predictions "$seed_dir/${tag}_$dom.json" \
                            --train-data $(rubric_for "$cond") --eval-data "$(eval_for $dom)" $(merge_flag $scheme) \
                            --output "$OUT/stratify/$cond/$seed/${tag}_${dom}_$scheme.json"
                done
            done
        done
    done
done

compare() {   # compare <name> <dom> <rubric> <scheme> <A> <B> [<A> <B> ...]
    local name=$1 dom=$2 rubric=$3 scheme=$4; shift 4
    local pairs=()
    while [ $# -ge 2 ]; do
        has_cache "${1%%:*}" "${1#*:}" "$dom" && has_cache "${2%%:*}" "${2#*:}" "$dom" && pairs+=(--pair "$1" "$2")
        shift 2
    done
    [ ${#pairs[@]} -gt 0 ] || { echo "skip  $name"; return; }
    run "$name" python seed_compare.py --cache-root "$CACHE_ROOT" --dataset "$dom" --eval-data "$(eval_for $dom)" \
        --train-data $rubric $(merge_flag $scheme) --n-bootstrap "$N_BOOT" "${pairs[@]}" \
        --output "$OUT/compare/$name.json"
}

echo "== architectures within each condition"
for cond in "${CONDS[@]}"; do
    for dom in vcd sacti; do
        for scheme in $(schemes_for $dom); do
            compare "arch/${cond}_${dom}_$scheme" "$dom" "$(rubric_for "$cond")" "$scheme" \
                "$cond:buddhist-nlp_byt5-sanskrit" "$cond:google_byt5-base" \
                "$cond:google_byt5-base" "$cond:classifier" \
                "$cond:buddhist-nlp_byt5-sanskrit" "$cond:classifier"
        done
    done
done

echo "== context (condXS vs condX)"
for base in cond1 cond2 cond3 cond3B; do
    for dom in vcd sacti; do
        for scheme in $(schemes_for $dom); do
            args=(); for tag in "${TAGS[@]}"; do args+=("${base}S:$tag" "$base:$tag"); done
            compare "context/${base}_${dom}_$scheme" "$dom" "$(rubric_for $base)" "$scheme" "${args[@]}"
        done
    done
done

echo "== joint vs single-corpus training; balanced vs concatenated joint training"
for ctx in "" S; do
    for dom in vcd sacti; do
        if [ "$dom" = vcd ]; then single=cond1$ctx; else single=cond2$ctx; fi
        for scheme in $(schemes_for $dom); do
            for joint in cond3 cond3B; do
                args=(); for tag in "${TAGS[@]}"; do args+=("$joint$ctx:$tag" "$single:$tag"); done
                compare "regime/${joint}${ctx}_vs_${single}_${dom}_$scheme" "$dom" "$(rubric_for $single)" "$scheme" "${args[@]}"
            done
            args=(); for tag in "${TAGS[@]}"; do args+=("cond3B$ctx:$tag" "cond3$ctx:$tag"); done
            compare "sampling/cond3B${ctx}_vs_cond3${ctx}_${dom}_$scheme" "$dom" "$(rubric_for cond3)" "$scheme" "${args[@]}"
        done
    done
done

echo "== kāraka axis"
present=(); for c in "${CONDS[@]}"; do compgen -G "$CACHE_ROOT/$c/seed*/*_sacti.json" > /dev/null && present+=("$c"); done
[ ${#present[@]} -gt 0 ] && run karaka_t3 python karaka_axis_t3.py --eval-data "$SACTI_TEST" --cache-root "$CACHE_ROOT" \
    --conds "${present[@]}" --n-bootstrap "$N_BOOT" --output "$OUT/karaka/karaka_t3.json"

echo "== context x type ambiguity"
for dom in sacti vcd; do
    if [ "$dom" = vcd ]; then pool=("$VCD_DIR"/type/{train,val,test}.jsonl); bases=(cond1 cond3 cond3B)
    else pool=("$SACTI_DIR"/{train,dev,test}.jsonl); bases=(cond2 cond3 cond3B); fi
    conds=(); for b in "${bases[@]}"; do has_cache "$b" "*" "$dom" && has_cache "${b}S" "*" "$dom" && conds+=("$b"); done
    [ ${#conds[@]} -gt 0 ] && run "ambiguity_$dom" python context_ambiguity.py --cache-root "$CACHE_ROOT" --dataset "$dom" \
        --eval-data "$(eval_for $dom)" --pool "${pool[@]}" --conds "${conds[@]}" --merge-exo \
        --n-bootstrap "$N_BOOT" --output "$OUT/ambiguity/$dom.json"
done

echo "== summary tables"
run summary python summarise_results.py --stratify-root "$OUT/stratify" --karaka-root "$KARAKA_ROOT" \
    --cache-root "$CACHE_ROOT" --sacti-test "$SACTI_TEST" --out "$OUT/summary"

[ ${#FAILED[@]} -eq 0 ] && echo "all done: $OUT/" || { echo "failed:"; printf '  %s\n' "${FAILED[@]}"; }
