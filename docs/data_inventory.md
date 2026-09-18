# C3VDv2 Data Inventory

Dataset root: `/data1_ycao/chua/datasets/C3VDv2` (read-only). Generated 2026-09-17.

## 1. Top-level contents

| Item | Size | Notes |
|---|---|---|
| `registered_videos/` | 741G | 191 `.zip` + 2 pre-extracted dirs (`c1_transverse1_t1_v1`, `c2_transverse1_t1_v1`) = 193 sequences |
| `simulated_screening_videos/` | 168G | 12 `.zip` archives |
| `deformation_videos/` | 11G | 15 `.zip` archives |
| `camera_calibration/` | 5.0G | 2 `.zip` + raw `.avi`/frames/calib outputs |
| `3D_models/` | 992M | `colon0/colon1/colon2`, each with `lumen(s)/` and `molds/` |
| `C3VDv2_Data_Summary_Sheet_v1.xlsx` | 28K | index (see §4) |
| `camera_intrinsics.txt` | 4.0K | omnidirectional model (width/height/cx/cy/a0-a4/c/d/e) |
| `README.md` | 12K | full methodology, very detailed — see below |
| `_manifests/download_cmds.sh`, `_logs/` | 60K / 0 | download bookkeeping, empty log dir |

**Total: 925G. Archive count: 261** (191 registered + 12 screening + 15 deformation + 2 calibration + [2 registered dirs already extracted, not counted as archives]).

## 2. Representative archive (no extraction)

`registered_videos/c1_cecum_t1_v1.zip` — 2.3G compressed, 5,024,270,923 B (4.7G) uncompressed, **1309 files**. Top entries: `coverage_mesh.obj`, `pose.txt`, then per-frame files under `diffuse/`, `rgb/`, `depth/`, `normals/`, `optical_flow/`, `occlusions/`.

## 3. Extracted sequence: `c1_cecum_t1_v1` (→ `scratch/c1_cecum_t1_v1`, 5.5G)

**Tree (depth 3, 218 frames/modality):**
```
c1_cecum_t1_v1/
├── coverage_mesh.obj        pose.txt
├── rgb/NNNN.png              (218)
├── depth/NNNN_depth.tiff     (218)
├── normals/NNNN_normals.tiff (218)
├── optical_flow/NNNN_flow.tiff (217 — frame N flow is relative to N-1, so no 0000)
├── occlusions/NNNN_occlusion.png (218 — **PNG, not `.tiff` as README states**)
└── diffuse/NNNN_diffuse.png  (218)
```
`NNNN` = 4-digit zero-padded frame index.

**Per-modality inspection (frame 0000, `depth`/`normals`/`optical_flow` verified with `tifffile` — **PIL/`file` misreads these TinyTIFFWriter TIFFs as 8-bit; they are truly 16-bit/channel**):

| Modality | Format | Size | Bit depth | Observed range |
|---|---|---|---|---|
| rgb | PNG, RGB | 1350×1080 | 8-bit/ch | 0–255 |
| depth | TIFF (TinyTIFFWriter), grayscale | 1350×1080 | uint16 | 0–65535 (raw; README: linear scale of 0–100mm) |
| normals | TIFF, RGB | 1350×1080 | uint16/ch | R 0–65534, G 0–65534, B 0–60125 |
| optical_flow | TIFF, RGB | 1350×1080 | uint16/ch | R(x) 0–32974, G(y) 0–32857, B unused (0) |
| occlusions | PNG, grayscale | 1350×1080 | uint8 | binary {0, 255} |
| diffuse | PNG, grayscale | 1350×1080 | uint8 | 0–255 |

**`pose.txt`** (218 lines, one per frame), first 3 lines verbatim:
```
-0.419637,-0.314624,-0.850948,0,-0.247586,0.942343,-0.22616,0,0.873904,0.115453,-0.472498,0,-265.339,24.9915,-400.877,1
-0.419655,-0.314626,-0.850986,0,-0.247588,0.942342,-0.226164,0,0.873867,0.115448,-0.472577,0,-265.344,24.9899,-400.878,1
-0.419682,-0.314629,-0.851043,0,-0.247591,0.942342,-0.22617,0,0.87381,0.115442,-0.472696,0,-265.349,24.9913,-400.879,1
```
16 comma-separated values = flattened row-major 4×4 camera-to-world matrix.

**`coverage_mesh.obj`:** 350,595 vertices, 699,908 faces, no `vn` lines. Exactly **2 distinct `vt` entries**: `vt 0.0 0.0` (index 1) and `vt 1.0 1.0` (index 2) — used as per-face coverage flags (`f v/vt v/vt v/vt`), not per-vertex UVs. Face usage: index 1 (observed) → 635,753 faces; index 2 (unobserved) → 64,155 faces. Matches README's "vt=1 observed / vt=2 unobserved" (1-indexed reference into the vt list, not the literal value).

**README / license / calibration files:** none inside the sequence archive itself. At the dataset root: `README.md` (methodology, very thorough), `camera_intrinsics.txt`, and `camera_calibration/` (checkerboard + vicalib calibration videos/outputs). No separate LICENSE file found.

## 4. Index (`C3VDv2_Data_Summary_Sheet_v1.xlsx`)

**1 sheet: `release_v1`** — 192 rows × 17 columns → saved as `docs/release_v1.csv`.

Columns: `Colon, Segment, Phantom Number, Video Number, Video Name, Camera Speed, Edge Enhancement, Brightness, Debris, Deformation, Open End Visible, Tags, Comments, Qualitative Score, Quantitative Score, Youtube Preview URL ID, Total Frames`

Categorical column distinct values (counts):
- **Colon**: c1 (106), c2 (86)
- **Segment** (10): transverse1 28, rectum 26, ascending 25, cecum 24, transverse2 23, descending 18, sigmoid1 14, sigmoid2 13, sigmoid 13, full 8
- **Phantom Number**: t4 52, t1 51, t3 47, t2 42
- **Video Number**: v1 63, v2 57, v3 57, v4 15
- **Edge Enhancement**: 3.0 (113), 2.0 (68), 1.0 (8), 0.0 (1), 5.0 (1), NaN (1)
- **Debris**: no 115, yes 77
- **Deformation**: no 169, yes 15, NaN 8
- **Open End Visible**: yes 131, no 56, "No" 4, "no " 1 — ⚠️ inconsistent casing/whitespace, treat as 2 real categories
- **Qualitative Score**: 1.0 (85), 3.0 (44), 2.0 (40), NaN (23)

`Tags` (78 distinct, free text, `;`-separated), `Comments` (133 distinct, free text), `Camera Speed`/`Quantitative Score`/`Total Frames` numeric, `Video Name`/`Youtube Preview URL ID` unique per row (192).

Example rows:
| Colon | Segment | Phantom | Video | Name | Speed | Edge | Bright | Debris | Deform | OpenEnd | Tags | QualScore | QuantScore | Frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | ascending | t1 | v1 | c1_ascending_t1_v1 | 20 | 2 | 5 | no | no | no | zigzag | 3.0 | -1.4156 | 282 |
| c1 | ascending | t1 | v2 | c1_ascending_t1_v2 | 15 | 2 | 5 | no | no | no | loop | 3.0 | -1.2866 | 454 |
| c1 | ascending | t2 | v1 | c1_ascending_t2_v1 | 17 | 2 | 6 | no | no | no | zigzag | 2.0 | 0.2027 | 374 |

## 5. Cross-check: index vs. archives

All **192/192 index `Video Name` entries have a matching archive/dir** across `registered_videos`, `deformation_videos`, `simulated_screening_videos`. No missing sequences.

**26 archives exist on disk but are not in the index**, all `c0_*` (legacy original C3VD v1 data, re-packaged): `c0_cecum_t{1-4}_v{1-3}` (8), `c0_transverse_t{1-4}_v{1-3}` (10), `c0_sigmoid_t{1-3}_v{1-2}` (4), `c0_descending_t4_v1` (1), `c0_full_t{1-4}_v1` (4) = 26, matching README's "22 registered short videos + 4 long simulated screening videos" for c0. This is expected scope, not a data error — the summary sheet documents only the C3VDv2 (`c1`/`c2`) release.

## Open questions for you
- Extract more sequences (e.g. a `v3`-debris or `v4`-deformation or a `simulated_screening_videos` example) before deciding on a loader/pipeline design?
- Should `c0_*` legacy sequences be included in downstream use, given they're undocumented in the index sheet?
