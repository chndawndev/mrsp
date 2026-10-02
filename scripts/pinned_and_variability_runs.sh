#!/usr/bin/env bash
# CUT3R pinned primary run (169 sequences), then the variability runs of
# docs/eval_protocol.md "2026-10-02: Numerical run-to-run variability" for
# CUT3R, EndoDAC and MASt3R-SLAM: 15-sequence subset, noise std 1e-6 on the
# 0-1 image scale, seeds 1 to 5. Inference only, no evaluation metric.
# One GPU, one job at a time. Every step is resumable (rerun the script).
#
# Usage (detached): tmux new -d -s pinvar "scripts/pinned_and_variability_runs.sh <gpu_index>"
set -u
GPU="$1"
REPO=/data1_ycao/chua/projects/mrsp
cd "$REPO"
V=results/pipelines/variability
SIGMA=1e-6
CUT3R_PY=/data1_ycao/chua/miniforge3/envs/cut3r/bin/python
ENDODAC_PY=/data1_ycao/chua/miniforge3/envs/endodac/bin/python
CPU_PY=scratch/.venv/bin/python
# the protocol's subset, from the archives (scripts/cut3r_run_corpus.py::variability_subset)
SUBSET=$($CPU_PY -c "import sys; sys.path.insert(0, 'scripts'); from cut3r_run_corpus import variability_subset; print(' '.join(variability_subset()))")
SEQ_ARGS=""; for s in $SUBSET; do SEQ_ARGS="$SEQ_ARGS --sequence $s"; done
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
step() {  # step <log name> <command...>
  local name="$1"; shift
  echo "[$(stamp)] START $name" | tee -a logs/pinvar_chain.log
  "$@" > "logs/$name.log" 2>&1
  echo "[$(stamp)] END $name rc=$? free_disk=$(df -BG --output=avail $REPO | tail -1)" | tee -a logs/pinvar_chain.log
}
echo "[$(stamp)] chain start, GPU $GPU, subset: $SUBSET" | tee -a logs/pinvar_chain.log
nvidia-smi >> logs/pinvar_chain.log

CUDA_VISIBLE_DEVICES=$GPU step cut3r_run_corpus $CUT3R_PY scripts/cut3r_run_corpus.py
for k in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$GPU step variability_cut3r_seed$k $CUT3R_PY scripts/cut3r_run_corpus.py \
    --variability-subset --noise-sigma $SIGMA --noise-seed $k --out-root $V/cut3r/seed$k
done
for k in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$GPU step variability_endodac_seed$k $ENDODAC_PY scripts/endodac_inference.py \
    $SEQ_ARGS --noise-sigma $SIGMA --noise-seed $k --out-root $V/endodac/seed$k
done
for k in 1 2 3 4 5; do
  step variability_mast3r_seed$k $CPU_PY scripts/mast3r_slam_run_corpus.py \
    $SEQ_ARGS --noise-sigma $SIGMA --noise-seed $k --out-root $V/mast3r_slam/seed$k --run-tag var_seed$k --gpu $GPU
done
echo "[$(stamp)] CHAIN DONE" | tee -a logs/pinvar_chain.log
