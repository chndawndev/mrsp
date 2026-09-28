#!/usr/bin/env python
"""MASt3R-SLAM Stage 2 (docs/pipelines/mast3r_slam.md): per-frame run.

Copied entry script, NOT an in-place edit of the vendored `main.py`
(scratch/pipelines/MASt3R-SLAM/main.py, read in full before writing this
file) -- CLAUDE.md-safe per this task's own explicit instruction. The
only changes from the vendored driver:
  1. `InstrumentedFrameTracker` (scripts/mast3r_slam_perframe_tracker.py)
     used in place of `FrameTracker`.
  2. Always headless (no --no-viz flag needed; visualization is never
     started, matching Stage 1's run).
  3. Right after every `tracker.track(frame)` call, saves that frame's
     camera-frame Z depth and confidence (at MASt3R's internal
     resolution) to `depth/{frame_id:04d}.npz`.
  4. After the run and after the existing save_traj/save_reconstruction/
     save_keyframes calls, dumps every keyframe's FINAL Sim3 pose and
     FINAL Z/confidence (overwriting that frame_id's tracking-time
     dump -- see docs/pipelines/mast3r_slam.md Stage 2 step 1's note on
     why keyframe depth is only correct once fully fused) and writes
     `per_frame_tracking.csv` from the instrumented tracker's records.

Must be run with CUDA_VISIBLE_DEVICES already exported in the shell
(scripts/gpu_status.py pattern) and cwd = the MASt3R-SLAM clone, e.g.:

    source .../conda.sh && conda activate mast3r-slam && \\
    export CUDA_VISIBLE_DEVICES=<idx> && \\
    cd scratch/pipelines/MASt3R-SLAM && \\
    python <repo>/scripts/mast3r_slam_run_perframe.py \\
        --dataset <repo>/scratch/mast3r_slam/c1_cecum_t1_v1 \\
        --config config/base.yaml --save-as c1_cecum_t1_v1_perframe \\
        --out-dir <repo>/results/pipelines/mast3r_slam_perframe/c1_cecum_t1_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import lietorch
import numpy as np
import torch
import torch.multiprocessing as mp

from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval_mod
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker  # noqa: F401 (kept for run_backend's relocalization path, which uses FactorGraph, not FrameTracker)
from mast3r_slam.visualization import WindowMsg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mast3r_slam_perframe_tracker import InstrumentedFrameTracker  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame, add_after_query=False, k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx, kf_idx, config["reloc"]["min_match_frac"], is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame, add_after_query=True, k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")
        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        kf_idx = []
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame, add_after_query=True, k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)
        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(kf_idx, frame_idx, config["local_opt"]["min_match_frac"])
        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


def save_frame_depth(depth_dir: Path, frame_id: int, X_canon, C, img_shape) -> None:
    shape_flat = img_shape.reshape(-1)
    h, w = int(shape_flat[0]), int(shape_flat[1])
    z = X_canon[..., 2].detach().cpu().numpy().reshape(h, w).astype(np.float32)
    conf = C.detach().cpu().numpy().reshape(h, w).astype(np.float32)
    np.savez_compressed(depth_dir / f"{frame_id:04d}.npz", z=z, conf=conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="config/base.yaml")
    ap.add_argument("--save-as", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    depth_dir = out_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"

    load_config(args.config)
    log(f"dataset={args.dataset}")
    log(f"config={config}")

    manager = mp.Manager()
    main2viz = new_queue(manager, True)
    viz2main = new_queue(manager, True)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    model = load_mast3r(device=device)
    model.share_memory()

    use_calib = config["use_calib"]
    K = None

    if dataset.save_results:
        save_dir, seq_name = eval_mod.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = InstrumentedFrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()
    t_start = time.time()

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break
        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue
        if not last_msg.is_paused:
            states.unpause()
        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]

        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            save_frame_depth(depth_dir, frame.frame_id, frame.X_canon, frame.C, frame.img_shape)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
            # X_canon/C are set by InstrumentedFrameTracker.track()'s
            # frame.update_pointmap() call before the match_frac skip
            # check, so this is valid even for a skipped/try_reloc frame.
            save_frame_depth(depth_dir, frame.frame_id, frame.X_canon, frame.C, frame.img_shape)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)
            save_frame_depth(depth_dir, frame.frame_id, frame.X_canon, frame.C, frame.img_shape)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            log(f"FPS: {FPS}")
        i += 1

    elapsed = time.time() - t_start
    log(f"loop done, elapsed={elapsed:.2f}s, n_frames={i}")

    if dataset.save_results:
        save_dir, seq_name = eval_mod.prepare_savedir(args, dataset)
        eval_mod.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval_mod.save_reconstruction(save_dir, f"{seq_name}.ply", keyframes, last_msg.C_conf_threshold)
        eval_mod.save_keyframes(save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes)

    # Dump every keyframe's FINAL state (pose + fully-fused depth/confidence),
    # overwriting that frame_id's tracking-time depth dump -- see module
    # docstring point 4.
    n_kf = len(keyframes)
    log(f"dumping {n_kf} keyframes' final state")
    with open(out_dir / "keyframes_final.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "s"])
        for idx in range(n_kf):
            kf = keyframes[idx]
            vec = kf.T_WC.data.detach().cpu().numpy().reshape(-1).tolist()
            writer.writerow([kf.frame_id] + vec)
            save_frame_depth(depth_dir, kf.frame_id, kf.X_canon, kf.C, kf.img_shape)

    with open(out_dir / "per_frame_tracking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id", "ref_keyframe_frame_id", "T_ref_to_frame_json",
            "match_frac", "match_frac_k", "skipped",
        ])
        for r in tracker.records:
            writer.writerow([
                r["frame_id"], r["ref_keyframe_frame_id"],
                json.dumps(r["T_ref_to_frame"]), r["match_frac"], r["match_frac_k"], r["skipped"],
            ])

    manifest = {
        "n_frames_total": i, "n_keyframes": n_kf, "elapsed_seconds": elapsed,
        "n_tracking_records": len(tracker.records),
        "h": h, "w": w, "dataset": args.dataset, "save_as": args.save_as,
    }
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"manifest: {manifest}")

    print("done")
    backend.join()


if __name__ == "__main__":
    main()
