"""Frame-at-a-time CUT3R inference through the vendored recurrent path
(docs/pipelines/cut3r.md, Stage 2). Imported by scripts/cut3r_run.py and
scripts/cut3r_run_corpus.py; the vendored code is not edited.

The loop that is executed is the vendored
ARCroco3DStereo.forward_recurrent (scratch/pipelines/CUT3R/src/dust3r/
model.py), verbatim, under the same no_grad + autocast(enabled=False)
context as the vendored src/dust3r/inference.py::inference_recurrent.
Two things differ from calling inference_recurrent on the whole list, and
neither touches a computed value:

  1. inference_recurrent moves every view to the GPU before the loop.
     Here the list handed to forward_recurrent moves each view to the GPU
     when the loop reaches it (same keys, same ignore set), so input
     tensors do not accumulate on the GPU.
  2. forward_recurrent keeps every frame's head output on the GPU until
     the end, then inference_recurrent moves all of them to the CPU. Here
     model._downstream_head is wrapped (instance attribute, vendored file
     untouched) so that each frame's output is moved to the CPU as soon
     as it is produced. forward_recurrent does not read a head output
     again after appending it to its result list.

What still accumulates on the GPU inside forward_recurrent is its own
all_state_args list (one state tuple per frame).

`recurrent_plain` (inference_recurrent on the whole list, nothing
wrapped) is kept for checking that the wrapper changes no value.
"""
from __future__ import annotations

import torch

IGNORE_KEYS = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}  # as inference_recurrent


class LazyDeviceViews:
    """Sequence of views; each is moved to `device` when iterated over."""

    def __init__(self, views: list[dict], device: str):
        self.views, self.device = views, device

    def __len__(self) -> int:
        return len(self.views)

    def __iter__(self):
        for view in self.views:
            out = {}
            for name, val in view.items():
                if name in IGNORE_KEYS:
                    out[name] = val
                elif isinstance(val, (tuple, list)):
                    out[name] = [x.to(self.device, non_blocking=True) for x in val]
                else:
                    out[name] = val.to(self.device, non_blocking=True)
            yield out


@torch.no_grad()
def run_recurrent_streaming(views: list[dict], model, device: str, keep, batched_encoder: bool = False) -> list:
    """Returns [keep(pred_i on CPU) for every frame]. `keep` selects what is
    retained per frame (e.g. Z and pose only).

    batched_encoder=True is the confirmation test of
    docs/noise_sensitivity.md only: the image encoder is called ONCE on all
    frames, as the parallel path does (model._encode_views ->
    _encode_image on every image), and forward_recurrent's per-frame
    _encode_image calls are answered from that batch, in order. Everything
    after the encoder is forward_recurrent's own loop. It needs the
    parallel path's memory."""
    from src.dust3r.utils.device import to_cpu

    kept = []
    original_head = model._downstream_head
    if batched_encoder:
        with torch.cuda.amp.autocast(enabled=False):
            imgs = torch.cat([v["img"] for v in views], 0).to(device)
            shapes = torch.cat([v["true_shape"] for v in views], 0).to(device)
            img_out, img_pos, _ = model._encode_image(imgs, shapes)
            del imgs
        calls = {"i": 0}

        def encode_from_batch(image, true_shape):
            i = calls["i"]
            calls["i"] += 1
            if image.shape[0] != 1:
                raise RuntimeError("expected one frame per forward_recurrent step")
            return [o[i:i + 1] for o in img_out], img_pos[i:i + 1], None

        model._encode_image = encode_from_batch

    def head_then_offload(*args, **kwargs):
        res = to_cpu(original_head(*args, **kwargs))
        kept.append(keep(res))
        return None  # forward_recurrent only appends this to its result list

    model._downstream_head = head_then_offload
    try:
        with torch.cuda.amp.autocast(enabled=False):
            model.forward_recurrent(LazyDeviceViews(views, device), device, ret_state=True)
    finally:
        del model._downstream_head  # instance attribute off; the class method is back
        if batched_encoder:
            del model._encode_image
    if len(kept) != len(views):
        raise RuntimeError(f"{len(kept)} predictions for {len(views)} views")
    return kept
