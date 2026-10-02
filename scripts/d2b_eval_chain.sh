#!/usr/bin/env bash
# D2b evaluation run chain (docs/d2b_eval.md): CUT3R on 169 sequences,
# then the variability runs of all three pipelines on the 15-sequence
# subset, region IoU with matched random baselines, aggregation.
# One GPU; several CPU-bound driver processes share it (as in the
# MASt3R-SLAM evaluation). Every step is resumable: rerun the script.
#
# Usage (detached): tmux new -d -s d2b "scripts/d2b_eval_chain.sh <gpu_index>"
set -u
GPU="$1"
NPROC=6
REPO=/data1_ycao/chua/projects/mrsp
cd "$REPO"
PY=scratch/.venv/bin/python
OUT=results/d2b_eval
V=results/pipelines/variability
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { echo "[$(stamp)] $*" | tee -a logs/d2b_chain.log; }
SUBSET=$($PY -c "import sys; sys.path.insert(0, 'scripts'); from cut3r_run_corpus import variability_subset; print(' '.join(variability_subset()))")
SEQ_ARGS=""; for s in $SUBSET; do SEQ_ARGS="$SEQ_ARGS --sequence $s"; done
note "chain start, GPU $GPU, $NPROC processes; nvidia-smi:"; nvidia-smi >> logs/d2b_chain.log

note "START cut3r main evaluation"
for i in $(seq 0 $((NPROC - 1))); do
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_pipeline_sequence.py --pipeline cut3r --all \
    --shard-index $i --shard-total $NPROC --out-root $OUT/per_sequence/cut3r \
    > logs/d2b_main_cut3r_shard$i.log 2>&1 &
done
wait
note "END cut3r main evaluation; ok manifests: $(grep -l '"status": "ok"' $OUT/per_sequence/cut3r/*/MANIFEST.json 2>/dev/null | wc -l)"

note "START variability evaluations"
JOBS=""
for p in endodac mast3r_slam cut3r; do for k in 1 2 3 4 5; do JOBS="$JOBS $p:seed$k"; done; done
JOBS="$JOBS mast3r_slam:rerun_unperturbed"
run_job() {
  p="${1%%:*}"; r="${1##*:}"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_pipeline_sequence.py --pipeline $p $SEQ_ARGS \
    --pred-root $V/$p/$r --out-root $OUT/variability/$p/$r > logs/d2b_variability_${p}_$r.log 2>&1
  echo "[$(stamp)] variability $p $r rc=$?" >> logs/d2b_chain.log
}
export -f run_job stamp; export GPU PY SEQ_ARGS V OUT
echo $JOBS | tr ' ' '\n' | xargs -P $NPROC -I{} bash -c 'run_job {}'
note "END variability evaluations"

note "START region IoU, three pipelines"
$PY scripts/pipeline2_region_iou.py --workers 16 --out-dir $OUT/region_iou \
  --root endodac=results/d1/per_sequence --root mast3r_slam=results/pipeline2_eval/per_sequence/mast3r_slam \
  --root cut3r=$OUT/per_sequence/cut3r > logs/d2b_region_iou.log 2>&1
note "END region IoU rc=$?"

note "START region IoU, variability runs"
ROOTS=""
for j in $JOBS; do p="${j%%:*}"; r="${j##*:}"; ROOTS="$ROOTS --root ${p}__$r=$OUT/variability/$p/$r"; done
$PY scripts/pipeline2_region_iou.py --workers 15 --out-dir $OUT/variability_region_iou --cell fully_predicted:0.25 \
  $SEQ_ARGS --full-names $ROOTS > logs/d2b_variability_region_iou.log 2>&1
note "END region IoU variability rc=$?"

note "START aggregation"
$PY scripts/d2b_aggregate.py > logs/d2b_aggregate.log 2>&1
note "END aggregation rc=$?"
note "CHAIN DONE"
