# Training-data audit: C3VD exposure of candidate pipelines

Method: read each method's paper and official repository (training
commands, dataset preparation docs, checkpoint notices), and one level of
upstream pretrained weights. "None" means no C3VD-family data is
disclosed in these sources. Checked 2026-10-01.

This is a literature record, not a measurement.

| method | training data per official sources | C3VD | source |
|---|---|---|---|
| EndoDAC (used) | SCARED; Depth Anything backbone | none | https://arxiv.org/abs/2405.08672, https://github.com/BeileiCui/EndoDAC |
| MASt3R-SLAM (used) | MASt3R checkpoint: ~14 public general-domain datasets plus one undisclosed internal dataset; the checkpoint notice also lists Naver indoor localization datasets | none disclosed | https://github.com/naver/mast3r (training command, CHECKPOINTS_NOTICE) |
| CUT3R | 32 general-domain datasets, listed in docs/preprocess.md | none | https://arxiv.org/abs/2501.12387, https://github.com/CUT3R/CUT3R |
| MonST3R | PointOdyssey, TartanAir, Spring, Waymo; DUSt3R init | none | https://github.com/Junyi42/monst3r |
| DROID-SLAM | TartanAir (per paper; not re-verified in repo) | none | https://github.com/princeton-vl/DROID-SLAM |
| EndoDAV | evaluated on SCARED and Hamlyn; training set not stated explicitly in the sources read | likely none, unconfirmed | https://papers.miccai.org/miccai-2025/0288-Paper1355.html |
| Endo3R | README lists C3VD among its training datasets | YES | https://github.com/wrld/Endo3R |
| ColonAdapter | README uses C3VD as the training data path example; training code not released | likely YES | https://github.com/JayJiang99/ColonAdapter |

Decision: Endo3R and ColonAdapter are not used as benchmark pipelines.
